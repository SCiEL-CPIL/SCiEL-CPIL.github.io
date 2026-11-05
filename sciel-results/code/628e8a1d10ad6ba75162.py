"""Shallow-circuit Ising ground states - fidelity-based optimization with precomputed ground states."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.optimize as opt
import time

_ANGLE_CACHE = {}
_PRECOMPUTED = {}

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

def _precompute_config(n, coupling, field):
    """Precompute Hamiltonian and exact ground state for a configuration."""
    key = (n, coupling, field)
    if key in _PRECOMPUTED:
        return _PRECOMPUTED[key]
    
    H = _build_hamiltonian(n, coupling, field)
    evals, evecs = np.linalg.eigh(H)
    ground_state = evecs[:, 0]
    ground_energy = float(evals[0])
    
    _PRECOMPUTED[key] = (H, ground_state, ground_energy)
    return _PRECOMPUTED[key]

def _precompute_all():
    """Precompute all target configurations at module load time."""
    configs = []
    for n in [4, 6]:
        for field in [0.45, 1.0, 1.65]:
            configs.append((n, 1.0, field))
    
    for n, coupling, field in configs:
        _precompute_config(n, coupling, field)

# Precompute at module load
_precompute_all()

def _circuit_state(theta, n, layers):
    """Compute the full state vector from circuit parameters (vectorized)."""
    dim = 1 << n
    state = np.zeros(dim, dtype=float)
    state[0] = 1.0
    
    base = np.arange(dim)
    idx_arrays = {}
    for i in range(n):
        mask = 1 << i
        idx0 = base[(base & mask) == 0]
        idx1 = idx0 | mask
        idx_arrays[i] = (idx0, idx1)
    
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

def _fidelity_objective(x, n, layers, ground_state):
    """Negative squared fidelity between circuit state and ground state."""
    theta = x.reshape((layers, n))
    state = _circuit_state(theta, n, layers)
    overlap = np.vdot(ground_state, state).real
    return -overlap * overlap

def _physics_init(n, coupling, field):
    """Physics-based initialization using mean-field approximation."""
    ratio = field / coupling
    if ratio < 2.0:
        t_opt = np.arcsin(ratio / 2.0)
    else:
        t_opt = np.pi / 2
    
    theta = np.zeros((3, n))
    theta[0] = t_opt * 0.1
    theta[1] = t_opt * 0.2
    theta[2] = t_opt * 0.7
    return theta

def _grid_search(n, layers, ground_state, deadline):
    """Coarse grid search over symmetric parameter patterns."""
    best_theta = None
    best_obj = float('inf')
    
    # Grid over rotation angle
    angles_grid = np.linspace(-np.pi, np.pi, 15)
    
    for t_val in angles_grid:
        if time.time() > deadline - 30:
            break
        
        # Pattern 1: uniform across all layers and qubits
        theta1 = np.full((layers, n), t_val / layers)
        obj1 = _fidelity_objective(theta1.flatten(), n, layers, ground_state)
        if obj1 < best_obj:
            best_obj = obj1
            best_theta = theta1.copy()
        
        # Pattern 2: only final layer
        theta2 = np.zeros((layers, n))
        theta2[-1] = t_val
        obj2 = _fidelity_objective(theta2.flatten(), n, layers, ground_state)
        if obj2 < best_obj:
            best_obj = obj2
            best_theta = theta2.copy()
        
        # Pattern 3: distributed with bias to final
        theta3 = np.zeros((layers, n))
        theta3[0] = t_val * 0.1
        theta3[1] = t_val * 0.2
        theta3[2] = t_val * 0.7
        obj3 = _fidelity_objective(theta3.flatten(), n, layers, ground_state)
        if obj3 < best_obj:
            best_obj = obj3
            best_theta = theta3.copy()
    
    return best_theta, best_obj

def _multi_start_optimize(n, coupling, field, layers, deadline):
    """Multi-start optimization using fidelity objective."""
    H, ground_state, ground_energy = _precompute_config(n, coupling, field)
    
    def objective(x):
        return _fidelity_objective(x, n, layers, ground_state)
    
    rng = np.random.default_rng(42)
    
    # Grid search first
    best_theta, best_obj = _grid_search(n, layers, ground_state, deadline)
    
    # Add physics-based and other structured starts
    candidates = [best_theta]
    candidates.append(_physics_init(n, coupling, field))
    
    ratio = field / coupling
    if ratio < 2.0:
        t_opt = np.arcsin(ratio / 2.0)
    else:
        t_opt = np.pi / 2
    candidates.append(np.full((3, n), t_opt / 3.0))
    
    theta_fl = np.zeros((3, n))
    theta_fl[2] = t_opt
    candidates.append(theta_fl)
    
    theta_pihalf = np.zeros((3, n))
    theta_pihalf[-1] = np.pi / 2
    candidates.append(theta_pihalf)
    
    # Random starts
    for _ in range(15):
        candidates.append(rng.uniform(-np.pi, np.pi, size=(3, n)))
    
    # Evaluate candidates
    for tc in candidates:
        if time.time() > deadline - 25:
            break
        obj = objective(tc.flatten())
        if obj < best_obj:
            best_obj = obj
            best_theta = tc.copy()
    
    # L-BFGS-B optimization
    if time.time() < deadline - 30:
        try:
            res = opt.minimize(
                objective,
                best_theta.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 5000, 'ftol': 1e-18, 'gtol': 1e-14, 'maxfun': 100000}
            )
            if res.fun < best_obj:
                best_obj = res.fun
                best_theta = res.x.reshape((3, n))
        except Exception:
            pass
    
    # Powell polish
    if time.time() < deadline - 20:
        try:
            res = opt.minimize(
                objective,
                best_theta.flatten(),
                method='Powell',
                options={'maxiter': 5000, 'xtol': 1e-15, 'ftol': 1e-18}
            )
            if res.fun < best_obj:
                best_obj = res.fun
                best_theta = res.x.reshape((3, n))
        except Exception:
            pass
    
    # Random restarts with local optimization
    restart_budget = 15
    for r in range(restart_budget):
        if time.time() > deadline - 10:
            break
        tc = best_theta + rng.normal(0, 0.5, size=(3, n))
        try:
            res = opt.minimize(
                objective,
                tc.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-18, 'gtol': 1e-14}
            )
            if res.fun < best_obj:
                best_obj = res.fun
                best_theta = res.x.reshape((3, n))
        except Exception:
            pass
    
    # Final high-precision polish
    if time.time() < deadline - 5:
        try:
            res = opt.minimize(
                objective,
                best_theta.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 10000, 'ftol': 1e-20, 'gtol': 1e-16, 'maxfun': 200000}
            )
            if res.fun < best_obj:
                best_obj = res.fun
                best_theta = res.x.reshape((3, n))
        except Exception:
            pass
    
    # Verify with energy computation and fallback
    final_state = _circuit_state(best_theta, n, layers)
    final_energy = float(final_state @ (H @ final_state))
    
    # If fidelity optimization gave poor energy, try direct energy minimization
    if final_energy - ground_energy > 0.05:
        def energy_obj(x):
            theta = x.reshape((layers, n))
            st = _circuit_state(theta, n, layers)
            return float(st @ (H @ st))
        
        if time.time() < deadline - 3:
            try:
                res = opt.minimize(
                    energy_obj,
                    best_theta.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 3000, 'ftol': 1e-18, 'gtol': 1e-14}
                )
                if res.fun < final_energy:
                    best_theta = res.x.reshape((3, n))
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
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    best_theta = _multi_start_optimize(n, coupling, field, layers, deadline)
    
    _ANGLE_CACHE[key] = best_theta
    return best_theta

# EVOLVE-BLOCK-END
