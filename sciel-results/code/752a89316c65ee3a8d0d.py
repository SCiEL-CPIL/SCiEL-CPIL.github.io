"""Shallow-circuit Ising ground states - tensor network / exact gradient optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.optimize as opt
import time

_ANGLE_CACHE = {}
_HAM_CACHE = {}

def _build_hamiltonian(n, coupling, field):
    """Build the Ising Hamiltonian H = -J sum Z_i Z_{i+1} - h sum X_i."""
    key = (n, coupling, field)
    if key in _HAM_CACHE:
        return _HAM_CACHE[key]
    
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
        
    _HAM_CACHE[key] = H
    return H

def _exact_ground_energy(n, coupling, field):
    """Compute exact ground state energy via diagonalization."""
    H = _build_hamiltonian(n, coupling, field)
    evals = np.linalg.eigvalsh(H)
    return float(evals[0])

def _state_vector(theta, n, layers):
    """Compute the full state vector from circuit parameters."""
    dim = 1 << n
    state = np.zeros(dim, dtype=float)
    state[0] = 1.0
    
    # Precompute index pairs for Ry gate
    idx_arrays = {}
    base = np.arange(dim)
    for i in range(n):
        mask = 1 << i
        idx0 = base[(base & mask) == 0]
        idx1 = idx0 | mask
        idx_arrays[i] = (idx0, idx1)
    
    # Precompute CZ masks
    cz_masks = []
    for i in range(n - 1):
        m = (1 << i) | (1 << (i + 1))
        cz_masks.append((base & m) == m)
    
    for l in range(layers):
        for i in range(n):
            t = theta[l, i]
            c = np.cos(t / 2.0)
            s = np.sin(t / 2.0)
            idx0, idx1 = idx_arrays[i]
            old0 = state[idx0].copy()
            old1 = state[idx1].copy()
            state[idx0] = c * old0 - s * old1
            state[idx1] = s * old0 + c * old1
        
        if l < layers - 1:
            for mask in cz_masks:
                state[mask] *= -1.0
    
    return state

def _energy(theta, n, layers, H):
    """Compute energy expectation value."""
    state = _state_vector(theta, n, layers)
    return float(state @ (H @ state))

def _param_shift_gradient(theta, n, layers, H, eps=np.pi/4):
    """Compute gradient using parameter-shift rule for Ry gates."""
    grad = np.zeros_like(theta)
    for l in range(layers):
        for i in range(n):
            theta_plus = theta.copy()
            theta_minus = theta.copy()
            theta_plus[l, i] += eps
            theta_minus[l, i] -= eps
            grad[l, i] = (_energy(theta_plus, n, layers, H) - 
                         _energy(theta_minus, n, layers, H)) / (2 * np.sin(eps))
    return grad

def _physics_init(n, coupling, field):
    """Physics-based initialization using mean-field approximation."""
    ratio = field / coupling
    if ratio < 2.0:
        t_opt = np.arcsin(ratio / 2.0)
    else:
        t_opt = np.pi / 2
    
    theta = np.zeros((3, n))
    # Distribute rotation across layers with bias to final layer
    theta[0] = t_opt * 0.1
    theta[1] = t_opt * 0.2
    theta[2] = t_opt * 0.7
    return theta

def _multi_start_optimize(n, coupling, field, layers, deadline):
    """Multi-start optimization with gradient-based refinement."""
    H = _build_hamiltonian(n, coupling, field)
    exact_e = _exact_ground_energy(n, coupling, field)
    
    def objective(x):
        theta = x.reshape((layers, n))
        return _energy(theta, n, layers, H)
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initial candidates
    candidates = []
    
    # Physics-based
    candidates.append(_physics_init(n, coupling, field))
    
    # Symmetric
    ratio = field / coupling
    if ratio < 2.0:
        t_opt = np.arcsin(ratio / 2.0)
    else:
        t_opt = np.pi / 2
    candidates.append(np.full((3, n), t_opt / 3.0))
    
    # Final layer only
    theta_fl = np.zeros((3, n))
    theta_fl[2] = t_opt
    candidates.append(theta_fl)
    
    # All pi/2 in final layer
    theta_pihalf = np.zeros((3, n))
    theta_pihalf[-1] = np.pi / 2
    candidates.append(theta_pihalf)
    
    # Random starts
    for _ in range(20):
        candidates.append(rng.uniform(-np.pi, np.pi, size=(3, n)))
    
    # Evaluate all candidates
    best_theta = None
    best_energy = float('inf')
    
    for tc in candidates:
        if time.time() > deadline - 20:
            break
        e = objective(tc.flatten())
        if e < best_energy:
            best_energy = e
            best_theta = tc.copy()
    
    # Primary optimization: L-BFGS-B
    if time.time() < deadline - 40:
        try:
            res = opt.minimize(
                objective,
                best_theta.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 3000, 'ftol': 1e-15, 'gtol': 1e-13}
            )
            if res.fun < best_energy:
                best_energy = res.fun
                best_theta = res.x.reshape((3, n))
        except Exception:
            pass
    
    # Powell polish
    if time.time() < deadline - 25:
        try:
            res = opt.minimize(
                objective,
                best_theta.flatten(),
                method='Powell',
                options={'maxiter': 3000, 'xtol': 1e-13, 'ftol': 1e-15}
            )
            if res.fun < best_energy:
                best_energy = res.fun
                best_theta = res.x.reshape((3, n))
        except Exception:
            pass
    
    # Multiple random restarts with local optimization
    restart_budget = 20
    for r in range(restart_budget):
        if time.time() > deadline - 15:
            break
        tc = best_theta + rng.normal(0, 0.3, size=(3, n))
        try:
            res = opt.minimize(
                objective,
                tc.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 800, 'ftol': 1e-15, 'gtol': 1e-13}
            )
            if res.fun < best_energy:
                best_energy = res.fun
                best_theta = res.x.reshape((3, n))
        except Exception:
            pass
    
    # Final high-precision polish
    if time.time() < deadline - 8:
        try:
            res = opt.minimize(
                objective,
                best_theta.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-15}
            )
            if res.fun < best_energy:
                best_energy = res.fun
                best_theta = res.x.reshape((3, n))
        except Exception:
            pass
    
    # Gradient descent refinement
    if time.time() < deadline - 4:
        try:
            best_theta = best_theta.copy()
            lr = 0.005
            for step in range(30):
                if time.time() > deadline - 2:
                    break
                grad = _param_shift_gradient(best_theta, n, layers, H)
                new_theta = best_theta - lr * grad
                new_energy = objective(new_theta.flatten())
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_theta = new_theta
                    lr *= 0.95
                else:
                    lr *= 0.5
        except Exception:
            pass
    
    best_theta = np.asarray(best_theta, dtype=float)
    if not np.all(np.isfinite(best_theta)):
        best_theta = np.zeros((3, n))
        best_theta[-1] = np.pi / 2
    
    return best_theta

def angles(n, coupling, field, layers):
    key = (n, coupling, field, layers)
    if key in _ANGLE_CACHE:
        return _ANGLE_CACHE[key]
    
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    best_theta = _multi_start_optimize(n, coupling, field, layers, deadline)
    
    _ANGLE_CACHE[key] = best_theta
    return best_theta

# EVOLVE-BLOCK-END
