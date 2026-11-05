"""Shallow-circuit Ising ground states - direct analytical approach."""
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

def _apply_cz_vectorized(state_flat, n):
    """Apply CZ gates on adjacent pairs (0,1), (1,2), ..., (n-2,n-1) in ascending order.
    CZ flips sign when both qubits are |1>.
    """
    dim = 1 << n
    indices = np.arange(dim)
    for i in range(n - 1):
        mask_i = 1 << i
        mask_ip1 = 1 << (i + 1)
        # Both bits set
        both_set = (indices & mask_i) != 0
        both_set &= (indices & mask_ip1) != 0
        state_flat[both_set] *= -1.0
    return state_flat

def _circuit_state(theta, n, layers):
    """Compute the statevector of the 3-layer Ry/CZ circuit using bit-index arithmetic."""
    dim = 1 << n
    state = np.zeros(dim, dtype=float)
    state[0] = 1.0  # |0...0>
    
    for l in range(layers):
        for i in range(n):
            t = theta[l, i]
            c = np.cos(t / 2.0)
            s = np.sin(t / 2.0)
            mask = 1 << i
            # Indices where bit i is 0
            idx0 = np.arange(dim)
            idx0 = idx0[(idx0 & mask) == 0]
            idx1 = idx0 | mask
            # Ry: |0> -> c|0> + s|1>, |1> -> -s|0> + c|1>
            old0 = state[idx0].copy()
            old1 = state[idx1].copy()
            state[idx0] = c * old0 - s * old1
            state[idx1] = s * old0 + c * old1
        
        if l < layers - 1:
            _apply_cz_vectorized(state, n)
            
    return state

def _energy_from_state(state, H):
    return float(state @ (H @ state))

def _energy(theta_flat, n, coupling, field, H, E0):
    theta = theta_flat.reshape((3, n))
    state = _circuit_state(theta, n, 3)
    return _energy_from_state(state, H)

def _exact_ground_state(n, coupling, field):
    """Get exact ground state and energy via diagonalization."""
    H = _build_hamiltonian(n, coupling, field)
    eigenvalues, eigenvectors = np.linalg.eigh(H)
    E0 = eigenvalues[0]
    psi0 = eigenvectors[:, 0]
    return psi0, E0

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
    state = _circuit_state(theta, n, 3)
    return _energy_from_state(state, H)

def angles(n, coupling, field, layers):
    key = (n, coupling, field, layers)
    if key in _ANGLE_CACHE:
        return _ANGLE_CACHE[key]
    
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    H = _build_hamiltonian(n, coupling, field)
    psi0, E0 = _exact_ground_state(n, coupling, field)
    
    def objective(x):
        theta = x.reshape((3, n))
        state = _circuit_state(theta, n, 3)
        return _energy_from_state(state, H)
    
    # Generate diverse initial candidates
    rng = np.random.default_rng(42)
    candidates = []
    
    # Mean-field init
    theta_mf = _mean_field_init(n, coupling, field)
    candidates.append(theta_mf)
    
    # Simple heuristic: all rotation in last layer
    theta_simple = np.zeros((3, n))
    theta_simple[-1] = np.pi / 2
    candidates.append(theta_simple)
    
    # Uniform
    theta_uniform = np.full((3, n), np.pi / 4)
    candidates.append(theta_uniform)
    
    # More random starts
    for _ in range(20):
        candidates.append(rng.uniform(-np.pi, np.pi, size=(3, n)))
    
    # Evaluate all candidates
    best_theta = None
    best_energy = float('inf')
    
    for tc in candidates:
        if time.time() > deadline - 5:
            break
        e = objective(tc.flatten())
        if e < best_energy:
            best_energy = e
            best_theta = tc.copy()
    
    # L-BFGS-B from best
    if time.time() < deadline - 15:
        try:
            res = opt.minimize(
                objective,
                best_theta.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            if res.fun < best_energy:
                best_energy = res.fun
                best_theta = res.x.reshape((3, n))
        except Exception:
            pass
    
    # Powell polish
    if time.time() < deadline - 10:
        try:
            res = opt.minimize(
                objective,
                best_theta.flatten(),
                method='Powell',
                options={'maxiter': 1000, 'xtol': 1e-12, 'ftol': 1e-15}
            )
            if res.fun < best_energy:
                best_energy = res.fun
                best_theta = res.x.reshape((3, n))
        except Exception:
            pass
    
    # Additional random restarts with optimization
    restart_budget = 8
    for r in range(restart_budget):
        if time.time() > deadline - 5:
            break
        tc = best_theta + rng.normal(0, 0.5, size=(3, n))
        try:
            res = opt.minimize(
                objective,
                tc.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-13}
            )
            if res.fun < best_energy:
                best_energy = res.fun
                best_theta = res.x.reshape((3, n))
        except Exception:
            pass
    
    # Final tight polish
    if time.time() < deadline - 3:
        try:
            res = opt.minimize(
                objective,
                best_theta.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 800, 'ftol': 1e-16, 'gtol': 1e-14}
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
