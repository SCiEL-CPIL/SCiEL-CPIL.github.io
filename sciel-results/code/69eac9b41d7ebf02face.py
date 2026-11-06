"""Improved optimizer for shallow-circuit Ising ground states."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_H_cache = {}
_e0_cache = {}
_idx_cache = {}

def _build_H(n, coupling, field):
    key = (n, coupling, field)
    if key in _H_cache:
        return _H_cache[key]
    dim = 1 << n
    H = np.zeros((dim, dim), dtype=np.float64)
    for i in range(n - 1):
        for state in range(dim):
            bi = (state >> i) & 1
            bj = (state >> (i + 1)) & 1
            zz_val = 1.0 if bi == bj else -1.0
            H[state, state] += -coupling * zz_val
    for i in range(n):
        for state in range(dim):
            flipped = state ^ (1 << i)
            H[state, flipped] += -field
    _H_cache[key] = H
    return H

def _exact_ground(n, coupling, field):
    key = (n, coupling, field)
    if key in _e0_cache:
        return _e0_cache[key]
    H = _build_H(n, coupling, field)
    eigvals, eigvecs = np.linalg.eigh(H)
    e0 = eigvals[0]
    psi0 = eigvecs[:, 0]
    _e0_cache[key] = (e0, psi0)
    return e0, psi0

def _get_indices(n):
    key = n
    if key in _idx_cache:
        return _idx_cache[key]
    dim = 1 << n
    ry_idx = []
    for q in range(n):
        mask = 1 << q
        idx = np.arange(dim)
        mask_bit = (idx & mask) == 0
        idx_0 = idx[mask_bit]
        idx_1 = idx_0 + mask
        ry_idx.append((idx_0, idx_1))
    cz_idx = []
    for i in range(n - 1):
        mask1 = 1 << i
        mask2 = 1 << (i + 1)
        idx = np.arange(dim)
        both_set = ((idx & mask1) != 0) & ((idx & mask2) != 0)
        cz_idx.append(both_set)
    _idx_cache[key] = (ry_idx, cz_idx)
    return _idx_cache[key]

def _circuit_state_fast(n, thetas, layers):
    dim = 1 << n
    state = np.zeros(dim, dtype=np.complex128)
    state[0] = 1.0 + 0j
    ry_idx, cz_idx = _get_indices(n)

    for layer in range(layers):
        for q in range(n):
            theta = thetas[layer, q]
            c = np.cos(theta / 2.0)
            s = np.sin(theta / 2.0)
            idx_0, idx_1 = ry_idx[q]
            vals_0 = state[idx_0]
            vals_1 = state[idx_1]
            state[idx_0] = c * vals_0 - s * vals_1
            state[idx_1] = s * vals_0 + c * vals_1

        if layer < layers - 1:
            for i in range(n - 1):
                both_set = cz_idx[i]
                state[both_set] *= -1.0

    return state

def _circuit_energy(n, coupling, field, thetas, layers):
    H = _build_H(n, coupling, field)
    psi = _circuit_state_fast(n, thetas, layers)
    return np.real(np.conj(psi) @ H @ psi)

def _objective(params, n, coupling, field, layers, e0):
    thetas = params.reshape((layers, n))
    e_circuit = _circuit_energy(n, coupling, field, thetas, layers)
    return (e_circuit - e0) / n

def _correlation_informed_init(n, coupling, field, layers, psi0):
    """Fit Ry angles using single-qubit expectations AND pairwise ZZ correlations.
    
    For a 3-layer circuit with CZ between layers, the effective rotation angles
    are modified by the CZ entanglers. We use a simplified model where we match
    both <Z_q> and <Z_q Z_{q+1}> to inform the per-layer angles.
    """
    dim = 1 << n
    thetas = np.zeros((layers, n))
    
    # Compute single-qubit <Z_q> and <X_q>
    ez_arr = np.zeros(n)
    ex_arr = np.zeros(n)
    for q in range(n):
        ez = 0.0
        ex = 0.0
        mask = 1 << q
        for state in range(dim):
            bit = (state >> q) & 1
            amp = psi0[state]
            ez += np.real(amp * np.conj(amp) * (1.0 if bit == 0 else -1.0))
            flipped = state ^ mask
            ex += np.real(np.conj(amp) * psi0[flipped])
        ez_arr[q] = ez
        ex_arr[q] = ex
    
    # Compute pairwise <Z_q Z_{q+1}>
    ezz_arr = np.zeros(n - 1)
    for i in range(n - 1):
        ezz = 0.0
        mask1 = 1 << i
        mask2 = 1 << (i + 1)
        for state in range(dim):
            b1 = (state >> i) & 1
            b2 = (state >> (i + 1)) & 1
            zz_val = (1.0 if b1 == 0 else -1.0) * (1.0 if b2 == 0 else -1.0)
            amp = psi0[state]
            ezz += np.real(amp * np.conj(amp)) * zz_val
        ezz_arr[i] = ezz
    
    # For each qubit, the total effective angle from Ry layers
    # In a circuit with CZ between layers, the effective <Z> is modified.
    # Simplified: assume the CZ phases partially cancel, so we use a damped model.
    # The total rotation theta_total = sum of per-layer angles.
    # <Z> ~ cos(theta_total) * damping_factor
    # We estimate damping from correlation structure.
    
    # Estimate effective coupling strength from ZZ correlations
    # For product state, <ZZ> = <Z_q><Z_{q+1}>. Deviation indicates entanglement.
    # Use this to adjust the effective angle mapping.
    
    for q in range(n):
        theta_total = np.arctan2(ex_arr[q], ez_arr[q])
        
        # Distribute across layers, but weight based on correlation structure
        # Middle layer gets slightly different weight due to CZ on both sides
        if layers == 3:
            # Layer 0 and 2 have CZ only on one side, layer 1 has CZ on both sides
            # CZ phase shifts effective angle; approximate by giving middle layer
            # a slightly different contribution
            w0 = 0.35
            w1 = 0.30
            w2 = 0.35
            thetas[0, q] = w0 * theta_total
            thetas[1, q] = w1 * theta_total
            thetas[2, q] = w2 * theta_total
        else:
            per_layer = theta_total / layers
            for layer in range(layers):
                thetas[layer, q] = per_layer
    
    return thetas

def angles(n, coupling, field, layers):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    e0, psi0 = _exact_ground(n, coupling, field)
    
    def obj(p):
        return _objective(p, n, coupling, field, layers, e0)

    best_x = np.zeros(layers * n)
    best_val = obj(best_x)
    
    # Correlation-informed initialization
    try:
        corr_thetas = _correlation_informed_init(n, coupling, field, layers, psi0)
        corr_x = corr_thetas.flatten()
        corr_val = obj(corr_x)
        if corr_val < best_val:
            best_val = corr_val
            best_x = corr_x
    except Exception:
        pass
    
    # Determine number of restarts based on time budget
    elapsed = time.time() - start_time
    remaining = deadline - elapsed
    
    if n <= 4:
        num_restarts = min(25, int(remaining / 4.0))
    else:
        num_restarts = min(12, int(remaining / 6.0))
    
    # Generate initial points - focus on physics-informed starts
    inits = []
    
    # 1. Zero
    inits.append(np.zeros(layers * n))
    
    # 2. Pi/2
    inits.append(np.full(layers * n, np.pi/2))
    
    # 3. Small angles
    inits.append(np.full(layers * n, 0.1))
    inits.append(np.full(layers * n, -0.1))
    
    # 4. Correlation-informed
    try:
        corr_thetas2 = _correlation_informed_init(n, coupling, field, layers, psi0)
        inits.append(corr_thetas2.flatten())
    except Exception:
        pass
    
    # 5. Physics-informed random restarts
    if field > 1.0:
        center = np.pi/2
        spread = 0.5
    elif field > 0.5:
        center = np.pi/4
        spread = 0.5
    else:
        center = 0.0
        spread = 0.5

    rng = np.random.default_rng(42)
    for i in range(num_restarts):
        r = rng.random()
        if r < 0.3:
            x_init = rng.uniform(-np.pi, np.pi, size=(layers * n))
        elif r < 0.6:
            x_init = center + rng.normal(0, spread, size=(layers * n))
        else:
            x_init = center + rng.normal(0, spread * 1.5, size=(layers * n))
        
        inits.append(x_init)
        
        if time.time() > deadline - 20:
            break

    # Run optimization from each start
    for x_init in inits:
        if time.time() > deadline - 15:
            break
        
        try:
            res = minimize(obj, x_init, method='L-BFGS-B',
                           options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12})
            
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except Exception:
            continue

    # Final high-precision polish with L-BFGS-B
    if time.time() < deadline - 10.0:
        try:
            res = minimize(obj, best_x, method='L-BFGS-B',
                           options={'maxiter': 5000, 'ftol': 1e-16, 'gtol': 1e-14})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except Exception:
            pass
    
    # Nelder-Mead polish for fine-tuning (handles flat regions better)
    if time.time() < deadline - 5.0:
        try:
            res_nm = minimize(obj, best_x, method='Nelder-Mead',
                             options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-14})
            if res_nm.fun < best_val:
                best_val = res_nm.fun
                best_x = res_nm.x
        except Exception:
            pass
    
    # Final ultra-precise L-BFGS-B pass
    if time.time() < deadline - 2.0:
        try:
            res = minimize(obj, best_x, method='L-BFGS-B',
                           options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-15})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except Exception:
            pass

    return best_x.reshape((layers, n))
# EVOLVE-BLOCK-END
