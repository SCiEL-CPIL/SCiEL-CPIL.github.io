"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, least_squares

# Caches for Hamiltonian and ground state data
_H_cache = {}
_e0_cache = {}
_gs_cache = {}
_circuit_indices_cache = {}

def _build_H(n, coupling, field):
    key = (n, coupling, field)
    if key in _H_cache:
        return _H_cache[key]
    dim = 1 << n
    H = np.zeros((dim, dim), dtype=np.float64)
    # Z terms: -coupling * Z_i Z_{i+1}
    for i in range(n - 1):
        for state in range(dim):
            bi = (state >> i) & 1
            bj = (state >> (i + 1)) & 1
            zz_val = 1.0 if bi == bj else -1.0
            H[state, state] += -coupling * zz_val
    # X terms: -field * X_i
    for i in range(n):
        mask = 1 << i
        for state in range(dim):
            flipped = state ^ mask
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

def _exact_ground_state(n, coupling, field):
    key = (n, coupling, field)
    if key in _gs_cache:
        return _gs_cache[key]
    H = _build_H(n, coupling, field)
    eigvals, eigvecs = np.linalg.eigh(H)
    idx = np.argmin(eigvals)
    psi = eigvecs[:, idx].copy()
    # Fix global phase: make the largest magnitude component real and positive
    max_idx = np.argmax(np.abs(psi))
    phase = np.angle(psi[max_idx])
    psi = psi * np.exp(-1j * phase)
    _gs_cache[key] = psi
    return _gs_cache[key]

def _get_circuit_indices(n, layers):
    """Pre-compute index pairs for fast state evolution."""
    key = (n, layers)
    if key in _circuit_indices_cache:
        return _circuit_indices_cache[key]
    
    dim = 1 << n
    idx = np.arange(dim)
    
    # Pre-compute indices for Ry(q)
    ry_indices = []
    for q in range(n):
        mask = 1 << q
        mask_bit = (idx & mask) == 0
        idx_0 = idx[mask_bit]
        idx_1 = idx_0 + mask
        ry_indices.append((idx_0, idx_1))
        
    # Pre-compute indices for CZ(i, i+1)
    cz_indices = []
    for i in range(n - 1):
        mask1 = 1 << i
        mask2 = 1 << (i + 1)
        both_set = ((idx & mask1) != 0) & ((idx & mask2) != 0)
        cz_indices.append(both_set)
        
    _circuit_indices_cache[key] = (ry_indices, cz_indices)
    return _circuit_indices_cache[key]

def _circuit_state(n, thetas, layers):
    """Fast circuit state computation using pre-computed indices."""
    dim = 1 << n
    state = np.zeros(dim, dtype=np.complex128)
    state[0] = 1.0 + 0j
    
    ry_indices, cz_indices = _get_circuit_indices(n, layers)
    
    for layer in range(layers):
        # Apply Ry to each qubit
        for q in range(n):
            theta = thetas[layer, q]
            c = np.cos(theta / 2.0)
            s = np.sin(theta / 2.0)
            
            idx_0, idx_1 = ry_indices[q]
            
            v0 = state[idx_0].copy()
            v1 = state[idx_1].copy()
            
            state[idx_0] = c * v0 - s * v1
            state[idx_1] = s * v0 + c * v1
        
        # Apply CZ between adjacent qubits (except after last layer)
        if layer < layers - 1:
            for i in range(n - 1):
                mask_idx = cz_indices[i]
                state[mask_idx] *= -1.0
    
    return state

def _circuit_energy(n, coupling, field, thetas, layers):
    H = _build_H(n, coupling, field)
    psi = _circuit_state(n, thetas, layers)
    return np.real(np.vdot(psi, H @ psi))

def _objective_energy(params, n, coupling, field, layers, e0):
    thetas = params.reshape((layers, n))
    e_circuit = _circuit_energy(n, coupling, field, thetas, layers)
    return (e_circuit - e0) / n

def _residuals_ls(params, n, coupling, field, layers, psi_target):
    """Residuals for least-squares: match circuit state to target ground state."""
    thetas = params.reshape((layers, n))
    psi_circuit = _circuit_state(n, thetas, layers)
    
    # Complex residuals: real and imaginary parts
    diff = psi_circuit - psi_target
    residuals = np.concatenate([diff.real, diff.imag])
    return residuals

def angles(n, coupling, field, layers):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin (evaluator kills at 150s)

    e0 = _exact_ground_energy(n, coupling, field)
    psi_target = _exact_ground_state(n, coupling, field)
    
    def obj_energy(p):
        return _objective_energy(p, n, coupling, field, layers, e0)

    def residuals(p):
        return _residuals_ls(p, n, coupling, field, layers, psi_target)

    best_x = np.zeros(layers * n)
    best_val = obj_energy(best_x)
    
    # Generate initial points
    inits = []
    
    # 1. Zero (Ferromagnetic limit)
    inits.append(np.zeros(layers * n))
    
    # 2. Pi/2 (Paramagnetic limit)
    inits.append(np.full(layers * n, np.pi/2))
    
    # 3. Physics-informed mean-field estimate
    # tan(theta/2) ~ field/coupling
    ratio = field / coupling if coupling > 0 else 1.0
    theta_mf = 2.0 * np.arctan(ratio)
    inits.append(np.full(layers * n, theta_mf))
    
    # 4. Alternating pattern (Antiferromagnetic hint)
    alt = np.zeros(layers * n).reshape(layers, n)
    for l in range(layers):
        for q in range(n):
            if q % 2 == 0:
                alt[l, q] = theta_mf
            else:
                alt[l, q] = -theta_mf
    inits.append(alt.flatten())
    
    # 5. Random restarts
    num_random = 25 if n <= 4 else 15
    
    for i in range(num_random):
        if time.time() > deadline - 20:
            break
        if i % 2 == 0:
            x_init = np.random.uniform(-np.pi, np.pi, size=(layers * n))
        else:
            x_init = theta_mf + np.random.normal(0, 0.8, size=(layers * n))
        
        inits.append(x_init)
        
    # Sort inits by objective value to prioritize good starts
    init_objs = []
    for x_init in inits:
        if time.time() > deadline - 15:
            break
        try:
            v = obj_energy(x_init)
            init_objs.append((v, x_init))
        except Exception:
            continue
    
    init_objs.sort(key=lambda x: x[0])
    
    # Phase 1: L-BFGS-B on energy objective from each start
    for v_init, x_init in init_objs:
        if time.time() > deadline - 20:
            break
        
        try:
            res = minimize(obj_energy, x_init, method='L-BFGS-B',
                           options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-13,
                                   'maxls': 50})
            
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            continue
    
    # Phase 2: Least-squares refinement from best energy solution
    if time.time() < deadline - 15:
        try:
            res_ls = least_squares(residuals, best_x, method='trf',
                                   max_nfev=5000, xtol=1e-12, ftol=1e-12, gtol=1e-12)
            # Check if the LS solution improves the energy
            e_ls = obj_energy(res_ls.x)
            if e_ls < best_val:
                best_val = e_ls
                best_x = res_ls.x.copy()
        except Exception:
            pass
    
    # Phase 3: More L-BFGS-B from LS-refined point (if it improved)
    if time.time() < deadline - 10:
        try:
            res2 = minimize(obj_energy, best_x, method='L-BFGS-B',
                           options={'maxiter': 3000, 'ftol': 1e-16, 'gtol': 1e-14,
                                   'maxls': 50})
            if res2.fun < best_val:
                best_val = res2.fun
                best_x = res2.x.copy()
        except Exception:
            pass
    
    # Phase 4: Another LS polish
    if time.time() < deadline - 8:
        try:
            res_ls2 = least_squares(residuals, best_x, method='trf',
                                    max_nfev=3000, xtol=1e-13, ftol=1e-13, gtol=1e-13)
            e_ls2 = obj_energy(res_ls2.x)
            if e_ls2 < best_val:
                best_val = e_ls2
                best_x = res_ls2.x.copy()
        except Exception:
            pass
    
    # Phase 5: Nelder-Mead if time remains
    if time.time() < deadline - 5:
        try:
            res_nm = minimize(obj_energy, best_x, method='Nelder-Mead',
                              options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-12})
            if res_nm.fun < best_val:
                best_val = res_nm.fun
                best_x = res_nm.x.copy()
        except Exception:
            pass
            
    # Final L-BFGS-B polish
    if time.time() < deadline - 2.0:
        try:
            res = minimize(obj_energy, best_x, method='L-BFGS-B',
                           options={'maxiter': 5000, 'ftol': 1e-18, 'gtol': 1e-16})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass

    return best_x.reshape((layers, n))
# EVOLVE-BLOCK-END
