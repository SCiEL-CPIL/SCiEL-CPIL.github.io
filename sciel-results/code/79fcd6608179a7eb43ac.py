import numpy as np
import time
import sys
from fractions import Fraction

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    start_time = time.time()
    deadline = 120.0
    safety_margin = 5.0
    max_time = deadline - safety_margin

    # 1. Construct the Target Tensor T as a 3D array T[u, v, w]
    # T[u, v, w] = 1 if u=i*m+j, v=j*p+k, w=k*n+i for some i,j,k
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0

    # 2. Try to find a low-rank decomposition using random rank-1 tensors
    # Strategy: generate random rank-1 tensors, solve for coefficients,
    # and check if the decomposition is exact.
    
    def try_random_decomposition(rank_target, num_trials=50, seed_base=0):
        """Try to find a rank-rank_target decomposition using random rank-1 tensors."""
        rng = np.random.RandomState(seed_base)
        
        for trial in range(num_trials):
            elapsed = time.time() - start_time
            if elapsed > max_time:
                return None
            
            # Generate rank_target random rank-1 tensors
            # Each rank-1 tensor is u_r outer v_r outer w_r
            # where u_r in R^nm, v_r in R^mp, w_r in R^pn
            U_rand = rng.randn(nm, rank_target)
            V_rand = rng.randn(mp, rank_target)
            W_rand = rng.randn(pn, rank_target)
            
            # Build the linear system: sum_r c_r * (U_rand[:,r] outer V_rand[:,r] outer W_rand[:,r]) = T
            # Flatten: for each (u,v,w), the equation is sum_r c_r * U_rand[u,r]*V_rand[v,r]*W_rand[w,r] = T[u,v,w]
            # This is a linear system A @ c = T.flatten()
            # where A[w_flat, r] = U_rand[u,r] * V_rand[v,r] * W_rand[w,r]
            
            # Build A matrix: shape (nm*mp*pn, rank_target)
            # For memory efficiency, we can use the fact that the system is structured.
            # But for nm*mp*pn = 6*21*14 = 1764 and rank_target up to ~30, this is fine.
            
            # A[row, r] where row = u*mp*pn + v*pn + w
            A = np.zeros((nm * mp * pn, rank_target))
            for u in range(nm):
                for v in range(mp):
                    base_row = u * mp * pn + v * pn
                    for w in range(pn):
                        row = base_row + w
                        for r in range(rank_target):
                            A[row, r] = U_rand[u, r] * V_rand[v, r] * W_rand[w, r]
            
            T_flat = T.flatten()
            
            # Solve the linear system
            try:
                c, residuals, rank_sol, sv = np.linalg.lstsq(A, T_flat, rcond=None)
            except Exception:
                continue
            
            # Check if the solution is exact
            A_c = A @ c
            err = np.max(np.abs(A_c - T_flat))
            
            if err < 1e-10:
                # Found a valid decomposition!
                return {
                    "U": U_rand.tolist(),
                    "V": V_rand.tolist(),
                    "W": W_rand.tolist(),
                    "coeffs": c.tolist(),
                    "rank": rank_target
                }
        
        return None

    # 3. Try ranks from low to high
    # Theoretical lower bound for 2x3x7: max(nm, mp, pn) = max(6, 21, 14) = 21
    # But tensor rank can be lower. Let's try from 20 up.
    
    best_result = None
    
    # First, try to find a decomposition with rank close to the lower bound
    # Start from rank 20 and go up
    for rank_target in range(20, 43):
        elapsed = time.time() - start_time
        if elapsed > max_time - 10:  # Leave some time for finalization
            break
        
        result = try_random_decomposition(rank_target, num_trials=30, seed_base=rank_target * 100)
        if result is not None:
            best_result = result
            break
    
    # 4. If we found a decomposition, combine U, V, W with coefficients
    if best_result is not None:
        U = np.array(best_result["U"])
        V = np.array(best_result["V"])
        W = np.array(best_result["W"])
        c = np.array(best_result["coeffs"])
        rank = best_result["rank"]
        
        # Combine: U_combined[:, r] = U[:, r] * c[r], etc.
        # Actually, the decomposition is sum_r c_r * U_r outer V_r outer W_r
        # We can absorb c into any of U, V, or W.
        # Let's absorb into U: U_combined[:, r] = U[:, r] * c[r]
        U_combined = U * c[np.newaxis, :]
        V_combined = V
        W_combined = W
        
        # Verify
        P = U_combined[:, np.newaxis, np.newaxis] * V_combined[np.newaxis, :, np.newaxis] * W_combined[np.newaxis, np.newaxis, :]
        err = np.max(np.abs(P - T))
        
        if err < 1e-8:
            return {
                "U": U_combined.tolist(),
                "V": V_combined.tolist(),
                "W": W_combined.tolist()
            }
    
    # 5. Fall back to schoolbook rank n*m*p
    rank = n * m * p
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1.0
                V[j * p + k][column] = 1.0
                W[k * n + i][column] = 1.0
                column += 1
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
