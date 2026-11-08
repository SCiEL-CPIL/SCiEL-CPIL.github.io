import time
import sys
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Precompute active and inactive indices
    # Active indices correspond to the terms in the matrix product C = A * B
    # C_{i,k} = sum_j A_{i,j} B_{j,k}
    # The tensor T has indices (i, j, k) where the term is A_{i,j} B_{j,k} C_{k,i} (using trace convention for W)
    # Actually, the task says W[k*n+i, r].
    # The multiplication tensor M_{i,j,k} is 1 if the term A_{i,j} B_{j,k} contributes to C_{i,k}.
    # The decomposition is sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r} = M_{i,j,k}
    
    active_indices = []
    active_set = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                active_indices.append((a, b, c))
                active_set.add((a, b, c))
                
    inactive_indices = []
    for a in range(nm):
        for b in range(mp):
            for c in range(pn):
                if (a, b, c) not in active_set:
                    inactive_indices.append((a, b, c))
                    
    active_a = np.array([x[0] for x in active_indices])
    active_b = np.array([x[1] for x in active_indices])
    active_c = np.array([x[2] for x in active_indices])
    
    inactive_a = np.array([x[0] for x in inactive_indices])
    inactive_b = np.array([x[1] for x in inactive_indices])
    inactive_c = np.array([x[2] for x in inactive_indices])
    
    num_active = len(active_indices)
    num_inactive = len(inactive_indices)
    
    def get_residuals(params, R):
        U = params[:R*nm].reshape(nm, R)
        V = params[R*nm:R*nm+R*mp].reshape(mp, R)
        W = params[R*nm+R*mp:].reshape(pn, R)
        
        # Vectorized computation
        # Active
        U_a = U[active_a]
        V_a = V[active_b]
        W_a = W[active_c]
        res_active = np.sum(U_a * V_a * W_a, axis=1) - 1.0
        
        # Inactive
        U_i = U[inactive_a]
        V_i = V[inactive_b]
        W_i = W[inactive_c]
        res_inactive = np.sum(U_i * V_i * W_i, axis=1)
        
        return np.concatenate([res_active, res_inactive])

    # Trivial solution
    rank_trivial = n * m * p
    U_trivial = np.zeros((nm, rank_trivial))
    V_trivial = np.zeros((mp, rank_trivial))
    W_trivial = np.zeros((pn, rank_trivial))
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_trivial[i * m + j, r] = 1
                V_trivial[j * p + k, r] = 1
                W_trivial[k * n + i, r] = 1
                r += 1

    best_result = {"U": U_trivial.tolist(), "V": V_trivial.tolist(), "W": W_trivial.tolist()}
    
    start_time = time.time()
    deadline = start_time + 110
    
    # Search for lower rank
    # Known minimal rank for 2x3x5 is 23.
    # We try 23 first. If found, return.
    # We also try 24, 25... up to 29 just in case, but 23 is the target.
    
    def verify_solution(U, V, W):
        U_a = U[active_a]
        V_a = V[active_b]
        W_a = W[active_c]
        res_active = np.sum(U_a * V_a * W_a, axis=1) - 1.0
        if np.max(np.abs(res_active)) > 1e-9:
            return False
            
        U_i = U[inactive_a]
        V_i = V[inactive_b]
        W_i = W[inactive_c]
        res_inactive = np.sum(U_i * V_i * W_i, axis=1)
        if np.max(np.abs(res_inactive)) > 1e-9:
            return False
            
        # Check for integer/half-integer entries
        U_check = np.round(U * 2)
        if not np.allclose(U * 2, U_check):
            return False
        V_check = np.round(V * 2)
        if not np.allclose(V * 2, V_check):
            return False
        W_check = np.round(W * 2)
        if not np.allclose(W * 2, W_check):
            return False
            
        return True

    for R in range(23, 30):
        if time.time() > deadline:
            break
            
        # Number of attempts increases for lower ranks
        num_attempts = 20 if R == 23 else 10
        
        for attempt in range(num_attempts):
            if time.time() > deadline:
                break
                
            seed = R * 100 + attempt
            np.random.seed(seed)
            
            # Generate initial guess
            # Strategy: Mix of random and structured
            # Structured: Block diagonal or specific patterns
            # For now, random with scaling is robust for LM
            
            scale = 1.0
            if attempt % 3 == 0:
                scale = 0.5
            elif attempt % 3 == 1:
                scale = 1.5
            else:
                scale = 1.0
                
            x0 = np.random.randn(R * (nm + mp + pn)) * scale
            
            try:
                # Use 'lm' method
                result = least_squares(get_residuals, x0, args=(R,), method='lm', max_nfev=5000, ftol=1e-12, xtol=1e-12, gtol=1e-12)
                
                if result.success or result.cost < 1e-10:
                    params = result.x
                    U = params[:R*nm].reshape(nm, R)
                    V = params[R*nm:R*nm+R*mp].reshape(mp, R)
                    W = params[R*nm+R*mp:].reshape(pn, R)
                    
                    # Snap to nearest 0.5
                    U_snap = np.round(U * 2) / 2
                    V_snap = np.round(V * 2) / 2
                    W_snap = np.round(W * 2) / 2
                    
                    if verify_solution(U_snap, V_snap, W_snap):
                        return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
                        
            except Exception:
                pass

    return best_result
# EVOLVE-BLOCK-END
