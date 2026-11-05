"""Shallow-circuit Ising ground states - parameter-shift gradient approach."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.optimize as opt
import time

_ANGLE_CACHE = {}

def _build_hamiltonian(n, coupling, field):
    """Build the Ising Hamiltonian H = -J sum Z_i Z_{i+1} - h sum X_i."""
    dim = 1 << n
    H = np.zeros((dim, dim), dtype=float)
    
    I = np.eye(2, dtype=float)
    X = np.array([[0, 1], [1, 0]], dtype=float)
    Z = np.array([[1, 0], [0, -1]], dtype=float)
    
    def kron_list(ops):
        res = ops[0]
        for op in ops[1:]:
            res = np.kron(res, op)
        return res

    for i in range(n - 1):
        ops = [I] * n
        ops[i] = Z
        ops[i+1] = Z
        H -= coupling * kron_list(ops)
        
    for i in range(n):
        ops = [I] * n
        ops[i] = X
        H -= field * kron_list(ops)
        
    return H

def _circuit_state_fast(theta, n, layers):
    """Compute the statevector using bit-index arithmetic for speed."""
    dim = 1 << n
    state = np.zeros(dim, dtype=float)
    state[0] = 1.0
    
    # Precompute CZ masks
    cz_masks = []
    for i in range(n - 1):
        mask_i = 1 << i
        mask_ip1 = 1 << (i + 1)
        indices = np.arange(dim)
        mask = (indices & mask_i) & (indices & mask_ip1)
        cz_masks.append(mask)
    
    for l in range(layers):
        # Apply Ry to each qubit
        for i in range(n):
            t = theta[l, i]
            c = np.cos(t / 2.0)
            s = np.sin(t / 2.0)
            
            # Split by bit i
            mask_i = 1 << i
            indices = np.arange(dim)
            bit_set = (indices & mask_i) != 0
            bit_unset = ~bit_set
            
            # For states with bit i = 0: new_amplitude = c * old
            # For states with bit i = 1: new_amplitude = -s * old (from |0>)
            # Actually: Ry(t)|0> = cos(t/2)|0> + sin(t/2)|1>
            # Ry(t)|1> = -sin(t/2)|0> + cos(t/2)|1>
            
            old0 = state[bit_unset].copy()
            old1 = state[bit_set].copy()
            
            state[bit_unset] = c * old0 - s * old1
            state[bit_set] = s * old0 + c * old1
        
        # Apply CZ between adjacent pairs
        if l < layers - 1:
            for i in range(n - 1):
                state[cz_masks[i]] *= -1.0
    
    return state

def _energy_from_state(state, H):
    return float(np.dot(state, H @ state))

def _exact_ground_energy(n, coupling, field):
    """Get exact ground energy via diagonalization."""
    H = _build_hamiltonian(n, coupling, field)
    eigenvalues = np.linalg.eigvalsh(H)
    return eigenvalues[0]

def _mean_field_init(n, coupling, field):
    """
    Mean-field / variational product state initialization.
    """
    theta = np.zeros((3, n))
    
    ratio = field / coupling
    if ratio < 2.0:
        t_opt = np.arcsin(min(ratio / 2.0, 1.0))
    else:
        t_opt = np.pi / 2
    
    theta[0] = t_opt * 0.5
    theta[1] = t_opt * 0.3
    theta[2] = t_opt * 0.2
    
    return theta

def _circuit_energy(theta_flat, n, H):
    """Energy of the circuit state."""
    theta = theta_flat.reshape((3, n))
    state = _circuit_state_fast(theta, n, 3)
    return _energy_from_state(state, H)

def _energy_gradient(theta_flat, n, H):
    """
    Compute exact gradient using parameter-shift rule.
    For Ry(t) gates, dE/dt = (E(t+pi/2) - E(t-pi/2)) / 2
    """
    theta = theta_flat.reshape((3, n))
    grad = np.zeros_like(theta)
    
    for l in range(3):
        for i in range(n):
            theta_plus = theta.copy()
            theta_plus[l, i] += np.pi / 2.0
            theta_minus = theta.copy()
            theta_minus[l, i] -= np.pi / 2.0
            
            e_plus = _circuit_energy(theta_plus.flatten(), n, H)
            e_minus = _circuit_energy(theta_minus.flatten(), n, H)
            
            grad[l, i] = (e_plus - e_minus) / 2.0
    
    return grad.flatten()

def angles(n, coupling, field, layers):
    key = (n, coupling, field, layers)
    if key in _ANGLE_CACHE:
        return _ANGLE_CACHE[key]
    
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    H = _build_hamiltonian(n, coupling, field)
    E0 = _exact_ground_energy(n, coupling, field)
    
    # Mean-field initialization
    theta_mf = _mean_field_init(n, coupling, field)
    
    # Also try simple heuristics
    theta_simple = np.zeros((3, n))
    theta_simple[-1] = np.pi / 2  # Put all rotation in last layer
    
    theta_uniform = np.full((3, n), np.pi / 4)
    
    # Evaluate initial candidates
    candidates = [theta_mf, theta_simple, theta_uniform]
    
    best_theta = None
    best_energy = float('inf')
    
    for tc in candidates:
        if time.time() > deadline:
            break
        e = _circuit_energy(tc.flatten(), n, H)
        if e < best_energy:
            best_energy = e
            best_theta = tc.copy()
    
    # Also add random starts
    rng = np.random.default_rng(12345)
    for _ in range(15):
        if time.time() > deadline - 10:
            break
        tc = rng.uniform(-np.pi, np.pi, size=(3, n))
        e = _circuit_energy(tc.flatten(), n, H)
        if e < best_energy:
            best_energy = e
            best_theta = tc.copy()
    
    # Add perturbations around best
    for _ in range(10):
        if time.time() > deadline - 10:
            break
        tc = best_theta + rng.normal(0, 0.3, size=(3, n))
        e = _circuit_energy(tc.flatten(), n, H)
        if e < best_energy:
            best_energy = e
            best_theta = tc.copy()
    
    # Define objective and gradient functions
    def objective(x):
        return _circuit_energy(x, n, H)
    
    def gradient(x):
        return _energy_gradient(x, n, H)
    
    # Strategy 1: L-BFGS-B with exact gradients
    if time.time() < deadline - 15:
        try:
            res = opt.minimize(
                objective,
                best_theta.flatten(),
                jac=gradient,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-16, 'gtol': 1e-14}
            )
            if res.fun < best_energy:
                best_energy = res.fun
                best_theta = res.x.reshape((3, n))
        except Exception:
            pass
    
    # Strategy 2: More random restarts with gradient-based optimization
    num_restarts = 0
    while time.time() < deadline - 20:
        tc = best_theta + rng.normal(0, 0.5, size=(3, n))
        try:
            res = opt.minimize(
                objective,
                tc.flatten(),
                jac=gradient,
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-13}
            )
            if res.fun < best_energy:
                best_energy = res.fun
                best_theta = res.x.reshape((3, n))
        except Exception:
            pass
        num_restarts += 1
        if num_restarts >= 15:
            break
    
    # Final polish with tighter tolerances
    if time.time() < deadline - 5:
        try:
            res = opt.minimize(
                objective,
                best_theta.flatten(),
                jac=gradient,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-16, 'gtol': 1e-15}
            )
            if res.fun < best_energy:
                best_energy = res.fun
                best_theta = res.x.reshape((3, n))
        except Exception:
            pass
    
    # Ensure output is finite
    best_theta = np.asarray(best_theta, dtype=float)
    if not np.all(np.isfinite(best_theta)):
        best_theta = np.zeros((3, n))
        best_theta[-1] = np.pi / 2
    
    _ANGLE_CACHE[key] = best_theta
    return best_theta

# EVOLVE-BLOCK-END
