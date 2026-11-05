"""Combined program: vectorized state evolution with parameter-shift gradients and multi-stage optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


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


def _apply_cz_vectorized(state, n):
    """Apply CZ gates on adjacent pairs in ascending order. Modifies state in place."""
    dim = 1 << n
    indices = np.arange(dim)
    for i in range(n - 1):
        mask_i = 1 << i
        mask_ip1 = 1 << (i + 1)
        both_set = (indices & mask_i) != 0
        both_set &= (indices & mask_ip1) != 0
        state[both_set] *= -1.0
    return state


def _circuit_state_vectorized(theta, n, layers):
    """Compute the statevector of the circuit using vectorized bit-index arithmetic."""
    dim = 1 << n
    state = np.zeros(dim, dtype=float)
    state[0] = 1.0
    
    for l in range(layers):
        for i in range(n):
            t = theta[l, i]
            c = np.cos(t / 2.0)
            s = np.sin(t / 2.0)
            mask = 1 << i
            idx0 = np.arange(dim)
            idx0 = idx0[(idx0 & mask) == 0]
            idx1 = idx0 | mask
            old0 = state[idx0].copy()
            old1 = state[idx1].copy()
            state[idx0] = c * old0 - s * old1
            state[idx1] = s * old0 + c * old1
        
        if l < layers - 1:
            _apply_cz_vectorized(state, n)
            
    return state


def _energy_from_state(state, H):
    """Compute <H> for a real state vector using matrix multiplication."""
    return float(state @ (H @ state))


def _exact_ground_energy(n, coupling, field):
    """Compute exact ground state energy by diagonalizing H."""
    H = _build_hamiltonian(n, coupling, field)
    eigvals = np.linalg.eigvalsh(H)
    return np.real(eigvals[0])


def _energy_vectorized(thetas, n, H, layers):
    """Compute circuit energy using vectorized state evolution and matrix multiplication."""
    state = _circuit_state_vectorized(thetas, n, layers)
    return _energy_from_state(state, H)


def _param_shift_gradient(thetas, n, H, layers, e0, n_spins):
    """
    Compute gradient of (E - e0)/n using the parameter-shift rule.
    For Ry(theta) = cos(theta/2) I - sin(theta/2) X,
    d/dtheta <psi|H|psi> = (E(theta + pi/2) - E(theta - pi/2)) / 2.
    """
    grad = np.zeros_like(thetas)
    shift = np.pi / 2.0
    
    for l in range(layers):
        for i in range(n):
            tp = thetas.copy()
            tm = thetas.copy()
            tp[l, i] += shift
            tm[l, i] -= shift
            
            ep = _energy_vectorized(tp, n, H, layers)
            em = _energy_vectorized(tm, n, H, layers)
            
            # Gradient of (E - e0)/n = (E_plus - E_minus) / (2 * n)
            grad[l, i] = (ep - em) / (2.0 * n_spins)
    
    return grad


def _heuristic_init(n, field, layers):
    """
    Physics-inspired initial guess.
    For the transverse-field Ising model, the ground state has significant X-component.
    The optimal single-qubit rotation angle roughly satisfies tan(theta) ~ field/coupling.
    """
    theta = np.zeros((layers, n))
    
    if field > 0.01:
        # Angle that balances Z-alignment (coupling) and X-alignment (field)
        base_angle = np.arctan2(field, 1.0)
    else:
        base_angle = np.pi / 2
    
    # Last layer: primary rotation toward ground state
    theta[layers - 1] = base_angle
    
    # Middle layer: moderate correction
    theta[layers - 2] = 0.5 * base_angle
    
    # First layer: small correction
    theta[layers - 3] = 0.25 * base_angle
    
    return theta


def angles(n, coupling, field, layers):
    start_time = time.time()
    deadline = 135.0  # Leave 15s margin for 150s limit
    
    # Build Hamiltonian once
    H = _build_hamiltonian(n, coupling, field)
    
    # Precompute exact ground energy
    e0 = _exact_ground_energy(n, coupling, field)
    
    def objective(params):
        elapsed = time.time() - start_time
        if elapsed > deadline:
            return 1e10
        thetas = params.reshape(layers, n)
        e = _energy_vectorized(thetas, n, H, layers)
        return (e - e0) / n
    
    def objective_grad(params):
        elapsed = time.time() - start_time
        if elapsed > deadline:
            return 1e10, np.zeros_like(params)
        thetas = params.reshape(layers, n)
        e = _energy_vectorized(thetas, n, H, layers)
        val = (e - e0) / n
        g = _param_shift_gradient(thetas, n, H, layers, e0, n)
        return val, g.flatten()
    
    # Generate diverse initializations
    n_params = layers * n
    inits = []
    
    # Physics-inspired heuristic
    inits.append(_heuristic_init(n, field, layers).flatten())
    
    # Mean-field: pi/2 for all (maximally mixed X)
    inits.append(np.full(n_params, np.pi / 2.0))
    
    # All zeros
    inits.append(np.zeros(n_params))
    
    # Structured inits: last layer only
    for frac in [0.3, 0.5, 0.7, 1.0]:
        arr = np.zeros(n_params)
        arr[-n:] = frac * np.pi / 2.0
        inits.append(arr)
    
    # Random inits with fixed seed for reproducibility
    rng = np.random.RandomState(42)
    for _ in range(15):
        inits.append(rng.uniform(-np.pi, np.pi, size=n_params))
    
    # Symmetric inits: same angle for all qubits in a layer
    for _ in range(5):
        layer_angles = rng.uniform(-np.pi, np.pi, size=layers)
        inits.append(np.repeat(layer_angles, n))
    
    # Alternating inits
    for _ in range(3):
        a = rng.uniform(-np.pi, np.pi)
        b = rng.uniform(-np.pi, np.pi)
        arr = np.zeros(n_params)
        for l in range(layers):
            for i in range(n):
                arr[l * n + i] = a if i % 2 == 0 else b
        inits.append(arr)
    
    # Sort by objective value
    init_objs = []
    for x in inits:
        try:
            init_objs.append((objective(x), x))
        except Exception:
            pass
    init_objs.sort(key=lambda t: t[0])
    
    best_params = init_objs[0][1]
    best_val = init_objs[0][0]
    
    # Bounds for L-BFGS-B
    bounds = [(-np.pi, np.pi)] * n_params
    
    # Multi-stage optimization
    n_restarts = 10
    for r in range(n_restarts):
        elapsed = time.time() - start_time
        if elapsed > deadline - 25.0:
            break
        
        if r < len(init_objs):
            x_start = init_objs[r][1]
        else:
            x_start = init_objs[0][1] + rng.randn(n_params) * 0.5
        
        # Use tighter tolerance for better initial points
        if r < 5:
            options = {'maxiter': 400, 'maxfun': 1200, 'gtol': 1e-10}
        else:
            options = {'maxiter': 200, 'maxfun': 600, 'gtol': 1e-9}
        
        try:
            res = minimize(
                objective_grad,
                x_start,
                method='L-BFGS-B',
                jac=True,
                bounds=bounds,
                options=options
            )
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # High-precision polish with tighter tolerance
    elapsed = time.time() - start_time
    if elapsed < deadline - 15.0:
        try:
            res = minimize(
                objective_grad,
                best_params,
                method='L-BFGS-B',
                jac=True,
                bounds=bounds,
                options={'maxiter': 600, 'maxfun': 2000, 'gtol': 1e-12}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Nelder-Mead for final refinement (derivative-free, can escape shallow local minima)
    elapsed = time.time() - start_time
    if elapsed < deadline - 8.0:
        try:
            res = minimize(
                objective,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-10}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Final very tight L-BFGS-B polish
    elapsed = time.time() - start_time
    if elapsed < deadline - 3.0:
        try:
            res = minimize(
                objective_grad,
                best_params,
                method='L-BFGS-B',
                jac=True,
                bounds=bounds,
                options={'maxiter': 400, 'maxfun': 1500, 'gtol': 1e-13}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    return best_params.reshape(layers, n)


# EVOLVE-BLOCK-END
