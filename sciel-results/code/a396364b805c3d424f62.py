import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

def _build_matrices():
    """Build A (odd bonds + half field) and B (even bonds + half field) for 6-site XXZ chain."""
    N = 6
    anis = 0.8
    field = 0.3
    dim = 2 ** N
    
    # Pauli matrices
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    I2 = np.eye(2, dtype=complex)
    
    def kron_all(mats):
        result = mats[0]
        for m in mats[1:]:
            result = np.kron(result, m)
        return result
    
    def bond_term(i, j, anis):
        terms = []
        mats = [I2] * N
        mats[i] = sx
        mats[j] = sx
        terms.append(0.5 * kron_all(mats))
        mats = [I2] * N
        mats[i] = sy
        mats[j] = sy
        terms.append(0.5 * kron_all(mats))
        mats = [I2] * N
        mats[i] = sz
        mats[j] = sz
        terms.append(anis * kron_all(mats))
        return sum(terms)
    
    A = np.zeros((dim, dim), dtype=complex)
    B = np.zeros((dim, dim), dtype=complex)
    
    # Odd bonds for A: (0,1), (2,3), (4,5)
    for i, j in [(0,1), (2,3), (4,5)]:
        A += bond_term(i, j, anis)
        
    # Even bonds for B: (1,2), (3,4)
    for i, j in [(1,2), (3,4)]:
        B += bond_term(i, j, anis)
        
    # Uniform field
    for site in range(N):
        mats = [I2] * N
        mats[site] = sz
        field_term = -field * kron_all(mats)
        A += 0.5 * field_term
        B += 0.5 * field_term
        
    return A, B

def _precompute_eigs(A, B):
    """Precompute eigendecompositions for fast exponentiation."""
    wA, vA = np.linalg.eigh(A)
    wB, vB = np.linalg.eigh(B)
    wAB, vAB = np.linalg.eigh(A + B)
    inv_vA = np.linalg.inv(vA)
    inv_vB = np.linalg.inv(vB)
    inv_vAB = np.linalg.inv(vAB)
    return wA, vA, inv_vA, wB, vB, inv_vB, wAB, vAB, inv_vAB

def _fast_exp_matrix(w, v, inv_v, x):
    """Compute exp(-i * x * H) using precomputed eigendecomp."""
    phases = np.exp(-1j * x * w)
    return v @ np.diag(phases) @ inv_v

def coefficients(stages):
    start_time = time.time()
    deadline = start_time + 140.0
    
    A, B = _build_matrices()
    times = [0.25, 0.5, 1.0]
    
    # Precompute
    wA, vA, inv_vA, wB, vB, inv_vB, wAB, vAB, inv_vAB = _precompute_eigs(A, B)
    
    def objective(params):
        if time.time() > deadline:
            return 1e10
        a = params[:stages]
        b = params[stages:]
        
        # Enforce symmetry
        a_sym = a.copy()
        b_sym = b.copy()
        for k in range(stages // 2):
            a_sym[stages - 1 - k] = a[k]
            b_sym[stages - 1 - k] = b[k]
            
        # Normalize
        sa = np.sum(a_sym)
        sb = np.sum(b_sym)
        if abs(sa) < 1e-12 or abs(sb) < 1e-12:
            return 1e10
        a_sym /= sa
        b_sym /= sb
        
        # Check bounds
        if np.any(a_sym < -50) or np.any(a_sym > 50) or np.any(b_sym < -50) or np.any(b_sym > 50):
            return 1e10
            
        errors = []
        for t in times:
            U = np.eye(2**6, dtype=complex)
            for k in range(stages):
                Ek_B = _fast_exp_matrix(wB, vB, inv_vB, b_sym[k] * t)
                Ek_A = _fast_exp_matrix(wA, vA, inv_vA, a_sym[k] * t)
                U = Ek_B @ Ek_A @ U
                
            exact = vAB @ np.diag(np.exp(-1j * t * wAB)) @ inv_vAB
            err = np.linalg.norm(U - exact, 2)
            errors.append(err)
            
        return np.mean(errors)
        
    # Initial guess: Symmetric equal weights
    x0 = np.ones(2 * stages) / stages
    
    best_x = x0
    best_f = objective(x0)
    
    # Global search: Differential Evolution
    if time.time() < deadline - 40:
        try:
            bounds = [(-50, 50)] * (2 * stages)
            # Use a smaller popsize to save time
            res_de = differential_evolution(objective, bounds, maxiter=20, popsize=10, tol=1e-4, seed=42, polish=False, workers=-1)
            if res_de.fun < best_f:
                best_x = res_de.x
                best_f = res_de.fun
        except Exception:
            pass
            
    # Local polish: Nelder-Mead
    if time.time() < deadline - 10:
        try:
            res_nm = minimize(objective, best_x, method='Nelder-Mead', 
                              options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10, 'disp': False})
            if res_nm.fun < best_f:
                best_x = res_nm.x
                best_f = res_nm.fun
        except Exception:
            pass
            
    best_a = best_x[:stages].copy()
    best_b = best_x[stages:].copy()
    
    # Symmetrize and normalize final result
    for k in range(stages // 2):
        best_a[stages - 1 - k] = best_a[k]
        best_b[stages - 1 - k] = best_b[k]
        
    sa = np.sum(best_a)
    sb = np.sum(best_b)
    if abs(sa) > 1e-12: best_a /= sa
    if abs(sb) > 1e-12: best_b /= sb
    
    # Clip to bounds
    best_a = np.clip(best_a, -50, 50)
    best_b = np.clip(best_b, -50, 50)
    
    # Final normalization after clipping
    sa = np.sum(best_a)
    sb = np.sum(best_b)
    if abs(sa) > 1e-12: best_a /= sa
    if abs(sb) > 1e-12: best_b /= sb
    
    return list(best_a), list(best_b)
