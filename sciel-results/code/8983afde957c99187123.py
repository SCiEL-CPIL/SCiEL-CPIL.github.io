# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, least_squares, differential_evolution

_H_cache = {}
_e0_cache = {}
_phase_cache = {}
_idx_cache = {}

def _build_H(n, coupling, field):
    key = (n, coupling, field)
    if key in _H_cache:
        return _H_cache[key]
    dim = 1 << n
    H = np.zeros((dim, dim), dtype=np.float64)
    # Z terms
    for i in range(n - 1):
        for state in range(dim):
            bi = (state >> i) & 1
            bj = (state >> (i + 1)) & 1
            zz_val = 1.0 if bi == bj else -1.0
            H[state, state] += -coupling * zz_val
    # X terms
    for i in range(n):
        for state in range(dim):
            flipped = state ^ (1 << i)
            H[state, flipped] += -field
    _H_cache[key] = H
    return H

def _exact_ground_energy(n, coupling, field):
    key = (n, coupling, field)
    if key in _e0_cache:
        return _e0_cache[key]
    H = _build_H(n, coupling, field)
    eigvals = np.linalg.eigvalsh(H)
    e0 = eigvals[0]
    _e0_cache[key] = e0
    return e0

def _get_phases(n):
    """
    Precompute phase vectors for CZ gates.
    CZ_{i, i+1} applies -1 to states where both i and i+1 are 1.
    """
    key = n
    if key in _phase_cache:
        return _phase_cache[key]
    dim = 1 << n
    phases = []
    for i in range(n - 1):
        mask1 = 1 << i
        mask2 = 1 << (i + 1)
        phase = np.ones(dim, dtype=np.float64)
        idx = np.arange(dim)
        mask = ((idx & mask1) != 0) & ((idx & mask2) != 0)
        phase[mask] = -1.0
        phases.append(phase)
    _phase_cache[key] = phases
    return phases

def _get_idx_pairs(n):
    """
    Precompute index pairs for Ry gates.
    Returns dict q -> (idx_0, idx_1)
    """
    key = n
    if key in _idx_cache:
        return _idx_cache[key]
    dim = 1 << n
    idx_pairs = {}
    for q in range(n):
        mask = 1 << q
        idx_0 = np.arange(dim)[(np.arange(dim) & mask) == 0]
        idx_1 = idx_0 + mask
        idx_pairs[q] = (idx_0, idx_1)
    _idx_cache[key] = idx_pairs
    return idx_pairs

def _circuit_state_fast(n, thetas, layers):
    """
    Vectorized application of the Ry/CZ circuit.
    thetas: (layers, n) array of rotation angles.
    """
    dim = 1 << n
    psi = np.zeros(dim, dtype=np.complex128)
    psi[0] = 1.0 + 0j
    
    phases = _get_phases(n)
    idx_pairs = _get_idx_pairs(n)

    for layer in range(layers):
        # Apply Ry to all qubits
        for q in range(n):
            theta = thetas[layer, q]
            c = np.cos(theta / 2.0)
            s = np.sin(theta / 2.0)
            
            idx_0, idx_1 = idx_pairs[q]
            
            v0 = psi[idx_0]
            v1 = psi[idx_1]
            
            psi[idx_0] = c * v0 - s * v1
            psi[idx_1] = s * v0 + c * v1

        # Apply CZ between layers (except after the last layer)
        if layer < layers - 1:
            for i in range(n - 1):
                psi *= phases[i]

    return psi

def _circuit_energy(n, coupling, field, thetas, layers):
    H = _build_H(n, coupling, field)
    psi = _circuit_state_fast(n, thetas, layers)
    return np.real(np.vdot(psi, H @ psi))

def _exact_ground_state(n, coupling, field):
    key = (n, coupling, field)
    if key in _e0_cache:
        return _e0_cache[key]
    H = _build_H(n, coupling, field)
    eigvals, eigvecs = np.linalg.eigh(H)
    e0 = eigvals[0]
    psi0 = eigvecs[:, 0]
    # Fix global phase: make the component with largest magnitude real and positive
    max_idx = np.argmax(np.abs(psi0))
    if np.real(psi0[max_idx]) < 0:
        psi0 = -psi0
    elif np.imag(psi0[max_idx]) != 0:
        psi0 = psi0 / psi0[max_idx]
        psi0 = psi0 * np.abs(psi0[max_idx])
    _e0_cache[key] = (e0, psi0)
    return _e0_cache[key]

def _state_residuals(params, n, coupling, field, layers, psi_target):
    """
    Compute residuals for least-squares state matching.
    params: flattened symmetry-reduced parameters
    Returns: real and imaginary parts of (psi_circuit - psi_target)
    """
    thetas = _symmetry_mapped_params(params, n, layers)
    psi = _circuit_state_fast(n, thetas, layers)
    diff = psi - psi_target
    return np.concatenate([np.real(diff), np.imag(diff)])

def _objective_full(params, n, coupling, field, layers, e0):
    thetas = params.reshape((layers, n))
    e_circuit = _circuit_energy(n, coupling, field, thetas, layers)
    return (e_circuit - e0) / n

def _symmetry_mapped_params(params, n, layers):
    """
    Maps a vector of length layers * n/2 to a full symmetric matrix of shape (layers, n).
    Assumes symmetry: theta[layer, i] == theta[layer, n-1-i]
    """
    half_n = n // 2
    thetas = np.zeros((layers, n))
    for l in range(layers):
        for i in range(half_n):
            val = params[l * half_n + i]
            thetas[l, i] = val
            thetas[l, n - 1 - i] = val
    return thetas

def _objective_sym(params, n, coupling, field, layers, e0):
    thetas = _symmetry_mapped_params(params, n, layers)
    e_circuit = _circuit_energy(n, coupling, field, thetas, layers)
    return (e_circuit - e0) / n

def _heuristic_init(n, coupling, field, layers):
    """
    Generate heuristic initial parameters based on physics intuition.
    Returns list of parameter vectors (symmetry-reduced if n is even).
    """
    inits = []
    half_n = n // 2
    
    # 1. Zero angles (|0...0>)
    inits.append(np.zeros(layers * half_n))
    
    # 2. Pi/2 angles (|+...+>)
    inits.append(np.full(layers * half_n, np.pi / 2))
    
    # 3. Pi angles (|1...1>)
    inits.append(np.full(layers * half_n, np.pi))
    
    # 4. Physics-inspired: Transverse field rotation
    ratio = field / coupling if coupling != 0 else field
    theta_est = np.arctan(ratio) * 0.8
    inits.append(np.full(layers * half_n, theta_est))
    
    # 5. Alternating small angles
    inits.append(np.full(layers * half_n, 0.5))
    inits.append(np.full(layers * half_n, 1.0))
    
    # 6. Random restarts
    np.random.seed(42)
    for _ in range(5):
        inits.append(np.random.uniform(0, np.pi, size=(layers * half_n)))
        
    return inits

def angles(n, coupling, field, layers):
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin for 150s limit
    
    # Get exact ground energy AND ground state
    e0, psi_target = _exact_ground_state(n, coupling, field)
    
    # Use symmetry if n is even (which it is for 4 and 6)
    use_symmetry = (n % 2 == 0)
    if use_symmetry:
        dim_params = layers * (n // 2)
        obj_func = lambda p: _objective_sym(p, n, coupling, field, layers, e0)
        map_func = lambda p: _symmetry_mapped_params(p, n, layers)
        inits = _heuristic_init(n, coupling, field, layers)
    else:
        dim_params = layers * n
        obj_func = lambda p: _objective_full(p, n, coupling, field, layers, e0)
        map_func = lambda p: p.reshape((layers, n))
        inits = [np.zeros(dim_params)]
        inits.append(np.full(dim_params, np.pi/2))
        np.random.seed(42)
        for _ in range(10):
             inits.append(np.random.uniform(0, np.pi, size=(dim_params)))

    best_x = inits[0]
    best_val = obj_func(best_x)
    
    # Evaluate all inits first to find the best starting point
    for x_init in inits:
        if time.time() > deadline:
            break
        val = obj_func(x_init)
        if val < best_val:
            best_val = val
            best_x = x_init

    # Strategy 1: Energy-based optimization (L-BFGS-B) with multiple starts
    candidates_to_optimize = [best_x]
    
    seen = set()
    for x_init in inits:
        if tuple(np.round(x_init, 5)) in seen:
            continue
        seen.add(tuple(np.round(x_init, 5)))
        candidates_to_optimize.append(x_init)
        if len(candidates_to_optimize) >= 8:
            break

    for x_init in candidates_to_optimize:
        if time.time() > deadline:
            break
        
        try:
            res = minimize(obj_func, x_init, method='L-BFGS-B',
                           options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50})
            
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass

    # Strategy 2: State-vector matching using least_squares
    remaining = deadline - time.time()
    if remaining > 20.0:
        try:
            x_ls_init = best_x.copy()
            
            def residuals_ls(params):
                return _state_residuals(params, n, coupling, field, layers, psi_target)
            
            res_ls = least_squares(residuals_ls, x_ls_init, method='trf',
                                   ftol=1e-15, xtol=1e-15, gtol=1e-15,
                                   max_nfev=300)
            
            val_ls = obj_func(res_ls.x)
            if val_ls < best_val:
                best_val = val_ls
                best_x = res_ls.x.copy()
        except Exception:
            pass
    
    # Strategy 3: Nelder-Mead for local refinement
    remaining = deadline - time.time()
    if remaining > 10.0:
        try:
            res_nm = minimize(obj_func, best_x, method='Nelder-Mead',
                             options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True})
            if res_nm.fun < best_val:
                best_val = res_nm.fun
                best_x = res_nm.x.copy()
        except Exception:
            pass
    
    # Final high-precision polish with L-BFGS-B
    remaining = deadline - time.time()
    if remaining > 5.0:
        try:
            res = minimize(obj_func, best_x, method='L-BFGS-B',
                           options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 100})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass
    
    # Second polish with state matching if time allows
    remaining = deadline - time.time()
    if remaining > 5.0:
        try:
            def residuals_ls2(params):
                return _state_residuals(params, n, coupling, field, layers, psi_target)
            
            res_ls2 = least_squares(residuals_ls2, best_x, method='trf',
                                    ftol=1e-16, xtol=1e-16, gtol=1e-16,
                                    max_nfev=200)
            val_ls2 = obj_func(res_ls2.x)
            if val_ls2 < best_val:
                best_val = val_ls2
                best_x = res_ls2.x.copy()
        except Exception:
            pass

    # Map back to full shape if symmetry was used
    full_thetas = map_func(best_x)
    
    # Ensure finite values
    if not np.all(np.isfinite(full_thetas)):
        return np.zeros((layers, n))
        
    return full_thetas
# EVOLVE-BLOCK-END
