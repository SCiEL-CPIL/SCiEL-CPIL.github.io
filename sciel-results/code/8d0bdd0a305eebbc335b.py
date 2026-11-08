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
    
    # Construct the target tensor T
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    def get_residuals(params, R):
        U = params[:R*nm].reshape(nm, R)
        V = params[R*nm:R*nm+R*mp].reshape(mp, R)
        W = params[R*nm+R*mp:].reshape(pn, R)
        
        # Compute the tensor product sum_r U[a,r] V[b,r] W[c,r]
        # Using einsum for efficiency
        T_hat = np.einsum('ar,br,cr->abc', U, V, W)
        
        return (T_hat - T).flatten()

    def verify_solution(U, V, W):
        T_hat = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T_hat, T, atol=1e-9)

    # Trivial solution as fallback
    rank_trivial = n * m * p
    U_trivial = np.zeros((nm, rank_trivial))
    V_trivial = np.zeros((mp, rank_trivial))
    W_trivial = np.zeros((pn, rank_trivial))
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U_trivial[a, r] = 1
                V_trivial[b, r] = 1
                W_trivial[c, r] = 1
                r += 1

    best_result = {
        "U": U_trivial.tolist(),
        "V": V_trivial.tolist(),
        "W": W_trivial.tolist()
    }
    best_rank = rank_trivial

    start_time = time.time()
    deadline = start_time + 110

    # Search for lower rank
    # Theoretical lower bound is max(nm, mp, pn) = 15
    # Try R from 15 to 20
    for R in range(15, 21):
        if time.time() > deadline:
            break
        
        # Try multiple random initializations
        for seed in range(10):
            if time.time() > deadline:
                break
            
            np.random.seed(seed + R * 100)
            x0 = np.random.randn(R * (nm + mp + pn)) * 0.5
            
            try:
                result = least_squares(
                    get_residuals, 
                    x0, 
                    args=(R,), 
                    method='lm', 
                    max_nfev=2000,
                    xtol=1e-10,
                    ftol=1e-10
                )
                
                if result.success or result.cost < 1e-8:
                    params = result.x
                    U = params[:R*nm].reshape(nm, R)
                    V = params[R*nm:R*nm+R*mp].reshape(mp, R)
                    W = params[R*nm+R*mp:].reshape(pn, R)
                    
                    # Snap to nearest 0.5
                    U_snap = np.round(U * 2) / 2
                    V_snap = np.round(V * 2) / 2
                    W_snap = np.round(W * 2) / 2
                    
                    if verify_solution(U_snap, V_snap, W_snap):
                        return {
                            "U": U_snap.tolist(),
                            "V": V_snap.tolist(),
                            "W": W_snap.tolist()
                        }
                        
                    # If snapped solution fails, try verifying the raw solution
                    # (might be exact rational but not half-integer, though problem implies half-integer)
                    # But problem statement says "integer/half-integer entries".
                    # So we must snap.
                    
            except Exception:
                continue

    return best_result
# EVOLVE-BLOCK-END
