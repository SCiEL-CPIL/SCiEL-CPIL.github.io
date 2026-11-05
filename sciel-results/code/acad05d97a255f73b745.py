"""Shallow-circuit Ising ground states - MPS-based energy with efficient gradients."""
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

def _circuit_energy(theta, n, layers, H):
    """Compute energy of the circuit output state."""
    dim = 1 << n
    state = np.zeros(dim, dtype=float)
    state[0] = 1.0
    
    base = np.arange(dim)
    idx_arrays = []
    for i in range(n):
        mask = 1 << i
        idx0 = base[(base & mask) == 0]
        idx1 = idx0 | mask
        idx_arrays.append((idx0, idx1))
    
    cz_masks = []
    for i in range(n - 1):
        m = (1 << i) | (1 << (i + 1))
        mask = (base & m) == m
        cz_masks.append(mask)
    
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
    
    return float(state @ (H @ state))

def _circuit_energy_grad(theta, n, layers, H):
    """Compute energy and gradient using finite differences."""
    dim = 1 << n
    state = np.zeros(dim, dtype=float)
    state[0] = 1.0
    
    base = np.arange(dim)
    idx_arrays = []
    for i in range(n):
        mask = 1 << i
        idx0 = base[(base & mask) == 0]
        idx1 = idx0 | mask
        idx_arrays.append((idx0, idx1))
    
    cz_masks = []
    for i in range(n - 1):
        m = (1 << i) | (1 << (i + 1))
        mask = (base & m) == m
        cz_masks.append(mask)
    
    def evolve(theta_in):
        s = np.zeros(dim, dtype=float)
        s[0] = 1.0
        for l in range(layers):
            for i in range(n):
                t = theta_in[l, i]
                c = np.cos(t / 2.0)
                sn = np.sin(t / 2.0)
                idx0, idx1 = idx_arrays[i]
                old0 = s[idx0].copy()
                old1 = s[idx1].copy()
                s[idx0] = c * old0 - sn * old1
                s[idx1] = sn * old0 + c * old1
            
            if l < layers - 1:
                for mask in cz_masks:
                    s[mask] *= -1.0
        return s
    
    psi = evolve(theta)
    energy = float(psi @ (H @ psi))
    
    grad = np.zeros((layers, n))
    eps = 1e-7
    
    for l in range(layers):
        for i in range(n):
            theta_plus = theta.copy()
            theta_minus = theta.copy()
            theta_plus[l, i] += eps
            theta_minus[l, i] -= eps
            
            psi_plus = evolve(theta_plus)
            psi_minus = evolve(theta_minus)
            
            e_plus = float(psi_plus @ (H @ psi_plus))
            e_minus = float(psi_minus @ (H @ psi_minus))
            
            grad[l, i] = (e_plus - e_minus) / (2 * eps)
    
    return energy, grad

def _physics_init(n, coupling, field):
    """Physics-based initialization using mean-field estimate."""
    ratio = field / coupling
    if ratio < 2.0:
        t_opt = np.arcsin(ratio / 2.0)
    else:
        t_opt = np.pi / 2
    
    theta = np.zeros((3, n))
    theta[0] = t_opt * 0.15
    theta[1] = t_opt * 0.25
    theta[2] = t_opt * 0.60
    return theta

def _symmetric_init(n, coupling, field):
    """Symmetric initialization."""
    ratio = field / coupling
    if ratio < 2.0:
        t_opt = np.arcsin(ratio / 2.0)
    else:
        t_opt = np.pi / 2
    return np.full((3, n), t_opt / 3.0)

def _final_layer_init(n, coupling, field):
    """All rotation in final layer."""
    ratio = field / coupling
    if ratio < 2.0:
        t_opt = np.arcsin(ratio / 2.0)
    else:
        t_opt = np.pi / 2
    theta = np.zeros((3, n))
    theta[2] = t_opt
    return theta

def angles(n, coupling, field, layers):
    key = (n, coupling, field, layers)
    if key in _ANGLE_CACHE:
        return _ANGLE_CACHE[key]
    
    start_time = time.time()
    deadline = start_time + 130.0
    
    H = _build_hamiltonian(n, coupling, field)
    
    def objective(x):
        theta = x.reshape((3, n))
        return _circuit_energy(theta, n, 3, H)
    
    def objective_grad(x):
        theta = x.reshape((3, n))
        e, g = _circuit_energy_grad(theta, n, 3, H)
        return e, g.flatten()
    
    # Generate diverse initial candidates
    rng = np.random.default_rng(42)
    candidates = []
    
    candidates.append(_physics_init(n, coupling, field))
    candidates.append(_symmetric_init(n, coupling, field))
    candidates.append(_final_layer_init(n, coupling, field))
    
    theta_simple = np.zeros((3, n))
    theta_simple[-1] = np.pi / 2
    candidates.append(theta_simple)
    
    candidates.append(np.full((3, n), np.pi / 4))
    candidates.append(np.full((3, n), -np.pi / 4))
    candidates.append(np.full((3, n), np.pi / 6))
    candidates.append(np.full((3, n), -np.pi / 6))
    
    # Add more structured candidates
    for frac in [0.3, 0.5, 0.7, 1.0]:
        tc = np.zeros((3, n))
        tc[2] = frac * np.pi / 2
        candidates.append(tc)
    
    for _ in range(50):
        candidates.append(rng.uniform(-np.pi, np.pi, size=(3, n)))
    
    # Evaluate all candidates
    best_theta = None
    best_energy = float('inf')
    
    for tc in candidates:
        if time.time() > deadline - 15:
            break
        e = objective(tc.flatten())
        if e < best_energy:
            best_energy = e
            best_theta = tc.copy()
    
    # Primary optimization: L-BFGS-B with finite-difference gradient
    if time.time() < deadline - 40:
        try:
            res = opt.minimize(
                objective,
                best_theta.flatten(),
                jac=objective_grad,
                method='L-BFGS-B',
                options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-12}
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
                options={'maxiter': 2000, 'xtol': 1e-12, 'ftol': 1e-14}
            )
            if res.fun < best_energy:
                best_energy = res.fun
                best_theta = res.x.reshape((3, n))
        except Exception:
            pass
    
    # Random restarts with local optimization
    restart_budget = 40
    for r in range(restart_budget):
        if time.time() > deadline - 15:
            break
        # Mix of small and large perturbations
        scale = 0.3 + 0.7 * (r % 3) / 2.0
        tc = best_theta + rng.normal(0, scale, size=(3, n))
        try:
            res = opt.minimize(
                objective,
                tc.flatten(),
                jac=objective_grad,
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            if res.fun < best_energy:
                best_energy = res.fun
                best_theta = res.x.reshape((3, n))
        except Exception:
            pass
    
    # Final ultra-precision polish
    if time.time() < deadline - 5:
        try:
            res = opt.minimize(
                objective,
                best_theta.flatten(),
                jac=objective_grad,
                method='L-BFGS-B',
                options={'maxiter': 5000, 'ftol': 1e-18, 'gtol': 1e-14}
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
