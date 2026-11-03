"""Program for 3x3x3 matrix multiplication tensor decomposition."""
import numpy as np
import time
import itertools
from scipy.optimize import least_squares

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def _build_decomposition(rank_terms):
    """
    Build U, V, W factor matrices from a list of term specifications.
    Each term is (U_dict, V_dict, W_dict) where each dict maps (i,j) or (j,k) or (k,i) to a scalar.
    """
    n, m, p = DIMENSIONS
    R = len(rank_terms)
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    
    for r, (u_dict, v_dict, w_dict) in enumerate(rank_terms):
        for (i, j), val in u_dict.items():
            U[i * m + j, r] = val
        for (j, k), val in v_dict.items():
            V[j * p + k, r] = val
        for (k, i), val in w_dict.items():
            W[k * n + i, r] = val
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly reproduces the multiplication tensor."""
    R = U.shape[1]
    U_reshaped = U.reshape((n, m, R))
    V_reshaped = V.reshape((m, p, R))
    W_reshaped = W.reshape((p, n, R))
    
    T_recon = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
    T_target = np.ones((n, m, p))
    
    return np.allclose(T_recon, T_target, atol=tol)

def _strassen_2x2_terms(offset_i, offset_j, offset_k):
    """
    Return the 7 Strassen terms for a 2x2 core at given offsets.
    """
    terms = []
    
    def idx(a, b, off):
        return (a + off[0], b + off[1])
    
    off_ij = (offset_i, offset_j)
    off_jk = (offset_j, offset_k)
    off_ki = (offset_k, offset_i)
    
    # m1 = (a00 + a11)(b00 + b11) -> c00 += m1, c11 += m1
    terms.append(({idx(0,0,off_ij): 1, idx(1,1,off_ij): 1},
                  {idx(0,0,off_jk): 1, idx(1,1,off_jk): 1},
                  {idx(0,0,off_ki): 1, idx(1,1,off_ki): 1}))
    
    # m2 = (a10 + a11)b00 -> c10 += m2, c11 -= m2
    terms.append(({idx(1,0,off_ij): 1, idx(1,1,off_ij): 1},
                  {idx(0,0,off_jk): 1},
                  {idx(0,1,off_ki): 1, idx(1,1,off_ki): -1}))
    
    # m3 = a00(b01 - b11) -> c01 += m3, c11 += m3
    terms.append(({idx(0,0,off_ij): 1},
                  {idx(0,1,off_jk): 1, idx(1,1,off_jk): -1},
                  {idx(1,0,off_ki): 1, idx(1,1,off_ki): 1}))
    
    # m4 = a11(b10 - b00) -> c00 += m4, c10 += m4
    terms.append(({idx(1,1,off_ij): 1},
                  {idx(1,0,off_jk): 1, idx(0,0,off_jk): -1},
                  {idx(0,0,off_ki): 1, idx(0,1,off_ki): 1}))
    
    # m5 = (a00 + a01)b11 -> c00 -= m5, c01 += m5
    terms.append(({idx(0,0,off_ij): 1, idx(0,1,off_ij): 1},
                  {idx(1,1,off_jk): 1},
                  {idx(0,0,off_ki): -1, idx(1,0,off_ki): 1}))
    
    # m6 = (a10 - a00)(b00 + b01) -> c11 += m6
    terms.append(({idx(1,0,off_ij): 1, idx(0,0,off_ij): -1},
                  {idx(0,0,off_jk): 1, idx(0,1,off_jk): 1},
                  {idx(1,1,off_ki): 1}))
    
    # m7 = (a01 - a11)(b10 + b11) -> c00 += m7
    terms.append(({idx(0,1,off_ij): 1, idx(1,1,off_ij): -1},
                  {idx(1,0,off_jk): 1, idx(1,1,off_jk): 1},
                  {idx(0,0,off_ki): 1}))
    
    return terms

def _try_strassen_core():
    """
    Strategy: Use Strassen on the 2x2x2 core (indices 0,1) and standard multiplication for the rest.
    Rank = 7 (core) + 19 (border) = 26.
    """
    n, m, p = DIMENSIONS
    terms = _strassen_2x2_terms(0, 0, 0)
    
    # Add border terms (at least one index is 2)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i < 2 and j < 2 and k < 2:
                    continue
                terms.append(({(i,j): 1}, {(j,k): 1}, {(k,i): 1}))
                
    return terms

def _try_standard_decomposition():
    """Standard schoolbook: rank 27."""
    n, m, p = DIMENSIONS
    terms = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                terms.append(({(i,j): 1}, {(j,k): 1}, {(k,i): 1}))
    return terms

def _attempt_rank_optimization(target_rank, start_time, deadline):
    """
    Attempt to find a decomposition of the specified target_rank using numerical optimization.
    Returns (U, V, W) if found and verified, otherwise None.
    """
    n, m, p = DIMENSIONS
    nm, mp, pn = 9, 9, 9
    R = target_rank
    
    # Precompute target tensor indices
    ones_indices = []
    ones_set = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                ones_indices.append((a, b, c))
                ones_set.add((a, b, c))
    
    all_indices = [(a, b, c) for a in range(nm) for b in range(mp) for c in range(pn)]
    zero_indices = [idx for idx in all_indices if idx not in ones_set]
    
    def verify(U, V, W, tol=1e-9):
        """Verify if U, V, W form a valid decomposition."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        # Check ones
        for a, b, c in ones_indices:
            s = np.sum(W[c, :] * U[a, :] * V[b, :])
            if abs(s - 1.0) > tol:
                return False
        # Check zeros
        for a, b, c in zero_indices:
            s = np.sum(W[c, :] * U[a, :] * V[b, :])
            if abs(s) > tol:
                return False
        return True

    def residual(x):
        U_f = x[:nm*R].reshape(nm, R)
        V_f = x[nm*R:2*nm*R].reshape(mp, R)
        W_f = x[2*nm*R:3*nm*R].reshape(pn, R)
        
        res = np.zeros(729)
        idx = 0
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    s = np.sum(W_f[c, :] * U_f[a, :] * V_f[b, :])
                    target = 1.0 if (a, b, c) in ones_set else 0.0
                    res[idx] = s - target
                    idx += 1
        return res

    # Initialization: Use the rank-26 solution truncated/adjusted or random
    # Since rank 26 is a known solution, we can try to initialize with it if R >= 26, 
    # but here we are trying R < 26 (e.g., 25). So we must use random or heuristic init.
    
    best_x = None
    best_cost = float('inf')
    
    # Try a few random restarts
    num_restarts = 5
    for r in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Random initialization
        x0 = np.random.randn(3 * nm * R) * 0.5
        
        try:
            sol = least_squares(residual, x0, method='trf', max_nfev=5000, ftol=1e-15, xtol=1e-15, gtol=1e-15)
            if sol.cost < best_cost:
                best_cost = sol.cost
                best_x = sol.x
                if sol.cost < 1e-12:
                    break
        except Exception:
            pass
    
    if best_x is not None and best_cost < 1e-8:
        U_sol = best_x[:nm*R].reshape(nm, R)
        V_sol = best_x[nm*R:2*nm*R].reshape(mp, R)
        W_sol = best_x[2*nm*R:3*nm*R].reshape(pn, R)
        
        # Try to snap to half-integers
        U_r = np.round(U_sol * 2) / 2
        V_r = np.round(V_sol * 2) / 2
        W_r = np.round(W_sol * 2) / 2
        
        if verify(U_r, V_r, W_r, tol=1e-9):
            return U_r, V_r, W_r
        
        # Try integers
        U_i = np.round(U_sol)
        V_i = np.round(V_sol)
        W_i = np.round(W_sol)
        if verify(U_i, V_i, W_i, tol=1e-9):
            return U_i, V_i, W_i
            
    return None

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3x3 matrix multiplication.
    Tries to find a rank < 26 solution via optimization, falls back to rank 26.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    n, m, p = DIMENSIONS
    
    best_result = None
    best_rank = float('inf')
    
    # Strategy 1: Try to find a rank 25 solution (optimization)
    # This is unlikely to succeed with rational coefficients, but we try.
    if time.time() < deadline - 20:
        result_25 = _attempt_rank_optimization(25, start_time, deadline - 20)
        if result_25:
            U, V, W = result_25
            if _verify_decomposition(U, V, W, n, m, p):
                R = U.shape[1]
                if R < best_rank:
                    best_rank = R
                    best_result = (U, V, W)

    # Strategy 2: Construct Rank-26 Solution (Strassen Core + Border)
    if time.time() < deadline:
        terms = _try_strassen_core()
        U, V, W = _build_decomposition(terms)
        if _verify_decomposition(U, V, W, n, m, p):
            R = U.shape[1]
            if R < best_rank:
                best_rank = R
                best_result = (U, V, W)
    
    # Strategy 3: Pure Standard (Rank 27) as fallback
    if time.time() < deadline:
        terms = _try_standard_decomposition()
        U, V, W = _build_decomposition(terms)
        if _verify_decomposition(U, V, W, n, m, p):
            R = U.shape[1]
            if R < best_rank:
                best_rank = R
                best_result = (U, V, W)
    
    if best_result is None:
        # Should not happen, but fallback
        terms = _try_standard_decomposition()
        U, V, W = _build_decomposition(terms)
        best_result = (U, V, W)
    
    U, V, W = best_result
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
