# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

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
    idx = np.arange(dim)
    for i in range(n - 1):
        mask1 = 1 << i
        mask2 = 1 << (i + 1)
        phase = np.ones(dim, dtype=np.float64)
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
    idx = np.arange(dim)
    for q in range(n):
        mask = 1 << q
        idx_0 = idx[(idx & mask) == 0]
        idx_1 = idx_0 + mask
        idx_pairs[q] = (idx_0, idx_1)
    _idx_cache[key] = idx_pairs
    return idx_pairs

def _circuit_state_fast(n, thetas, layers):
    """
    Vectorized application of the Ry/CZ circuit using real arithmetic.
    thetas: (layers, n) array of rotation angles.
    Returns a real numpy array of shape (dim,).
    """
    dim = 1 << n
    psi = np.zeros(dim, dtype=np.float64)
    psi[0] = 1.0
    
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
            
            # R_y = [[c, -s], [s, c]]
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
    # Energy is <psi|H|psi>. Since psi is real, this is psi @ (H @ psi)
    Hpsi = H @ psi
    return np.dot(psi, Hpsi)

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
    thetas = _symmetry_mapped_params(params, n, coupling, field, layers)
    e_circuit = _circuit_energy(n, coupling, field, thetas, layers)
    return (e_circuit - e0) / n

def angles(n, coupling, field, layers):
    start_time = time.time()
    deadline = start_time + 145.0 # 5s safety margin for 150s limit
    
    e0 = _exact_ground_energy(n, coupling, field)
    
    # Use symmetry if n is even
    use_symmetry = (n % 2 == 0)
    if use_symmetry:
        dim_params = layers * (n // 2)
        obj_func = lambda p: _objective_sym(p, n, coupling, field, layers, e0)
        map_func = lambda p: _symmetry_mapped_params(p, n, layers)
    else:
        dim_params = layers * n
        obj_func = lambda p: _circuit_energy(n, coupling, field, p.reshape((layers, n)), layers) - e0
        obj_func = lambda p: (obj_func(p) / n) # Correct objective definition
        # Re-defining properly for non-symmetric case
        def obj_full(p):
             t = p.reshape((layers, n))
             e = _circuit_energy(n, coupling, field, t, layers)
             return (e - e0) / n
        obj_func = obj_full
        map_func = lambda p: p.reshape((layers, n))

    # Generate diverse initial points
    # We want to cover the space well. 
    # Heuristics: 
    # 1. Zeros
    # 2. Pi/2
    # 3. Random points
    
    inits = []
    inits.append(np.zeros(dim_params))
    inits.append(np.full(dim_params, np.pi / 2))
    inits.append(np.full(dim_params, np.pi))
    
    # Random restarts
    # Use a fixed seed for reproducibility of the search path, but different for each run if possible? 
    # The task implies we return the best we find.
    np.random.seed(42)
    # Add 20 random restarts to ensure good coverage
    for _ in range(20):
        inits.append(np.random.uniform(-np.pi, np.pi, size=(dim_params,)))
        
    # Shuffle inits to ensure we don't just stick to the first few
    np.random.shuffle(inits)

    best_x = None
    best_val = np.inf

    # Phase 1: Global-ish search using Nelder-Mead
    # Nelder-Mead is derivative-free and robust for low dimensions.
    # We run it on multiple inits.
    nm_options = {
        'maxiter': 500,
        'xatol': 1e-8,
        'fatol': 1e-10,
        'adaptive': True
    }
    
    # We will process inits one by one, checking time.
    # To be efficient, we might limit the number of full NM runs if they are slow.
    # NM on 12-18 dims is fast (milliseconds).
    
    for x_init in inits:
        if time.time() > deadline - 10.0: # Leave 10s for polish
            break
            
        try:
            res = minimize(obj_func, x_init, method='Nelder-Mead', options=nm_options)
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            continue

    # If no valid result yet, fallback to a simple init
    if best_x is None:
        best_x = inits[0]
        best_val = obj_func(best_x)

    # Phase 2: Local Polish using L-BFGS-B
    # L-BFGS-B uses gradients (finite diff) and can converge very tightly.
    # We run it on the best result from Phase 1.
    # We might also run it on the top 3 results from Phase 1 if we kept track, 
    # but for now, just the best one is usually sufficient if NM was good.
    
    # Let's keep track of top candidates to polish
    # We didn't store all, so let's just polish the best one found so far.
    # Actually, let's re-evaluate the top few inits quickly with a faster local method?
    # No, NM already did a local search.
    
    lbfgs_options = {
        'maxiter': 1000,
        'ftol': 1e-15,
        'gtol': 1e-12,
        'maxls': 100
    }
    
    try:
        res = minimize(obj_func, best_x, method='L-BFGS-B', options=lbfgs_options)
        if res.fun < best_val:
            best_val = res.fun
            best_x = res.x.copy()
    except Exception:
        pass

    # Phase 3: Final High-Precision Polish
    # If time permits, run another pass with even tighter tolerances.
    remaining = deadline - time.time()
    if remaining > 2.0:
        lbfgs_options_tight = {
            'maxiter': 500,
            'ftol': 1e-16,
            'gtol': 1e-13,
            'maxls': 50
        }
        try:
            res = minimize(obj_func, best_x, method='L-BFGS-B', options=lbfgs_options_tight)
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass

    # Map back to full shape
    full_thetas = map_func(best_x)
    
    # Ensure finite values
    if not np.all(np.isfinite(full_thetas)):
        return np.zeros((layers, n))
        
    return full_thetas
# EVOLVE-BLOCK-END
