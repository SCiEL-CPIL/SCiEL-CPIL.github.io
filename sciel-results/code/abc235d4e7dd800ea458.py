"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120  # 120s limit
    
    # Schoolbook decomposition as baseline (Guaranteed Valid: Integers)
    rank_schoolbook = n * m * p
    U = [[0] * rank_schoolbook for _ in range(n * m)]
    V = [[0] * rank_schoolbook for _ in range(m * p)]
    W = [[0] * rank_schoolbook for _ in range(p * n)]
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][r] = 1
                V[j * p + k][r] = 1
                W[k * n + i][r] = 1
                r += 1
    
    best_result = {"U": U, "V": V, "W": W}
    best_rank = rank_schoolbook
    
    # Try to find lower rank using randomized search with snapping to half-integers
    np.random.seed(42)
    
    # We search for ranks lower than schoolbook
    # Lower bound is at least max(n*m, m*p, p*n) = 25
    for target_rank in range(25, min(rank_schoolbook, 60)):
        if time.time() > deadline - 5:
            break
            
        for attempt in range(3):
            if time.time() > deadline - 5:
                break
                
            # Initialize U, W with random values
            # Bias towards integers? No, just random.
            U_rand = np.random.randn(n * m, target_rank)
            W_rand = np.random.randn(p * n, target_rank)
            
            # Build linear system for V
            # Equation: sum_r U[i*m+j, r] * W[k*n+i, r] * V[j*p+k, r] = 1
            num_eqs = n * m * p
            num_unknowns = m * p * target_rank
            
            if num_unknowns < num_eqs:
                continue
            
            A_mat = np.zeros((num_eqs, num_unknowns))
            b_vec = np.ones(num_eqs)
            
            eq_idx = 0
            for i in range(n):
                for k in range(p):
                    for j in range(m):
                        for r in range(target_rank):
                            col_idx = (j * p + k) * target_rank + r
                            A_mat[eq_idx, col_idx] = U_rand[i * m + j, r] * W_rand[k * n + i, r]
                        eq_idx += 1
            
            try:
                V_flat, residuals, rank_lsq, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
            except:
                continue
            
            V_rand = V_flat.reshape(m * p, target_rank)
            
            # Snap U, V, W to nearest half-integers
            # Multiply by 2, round, divide by 2
            U_snap = np.rint(U_rand * 2) / 2.0
            V_snap = np.rint(V_rand * 2) / 2.0
            W_snap = np.rint(W_rand * 2) / 2.0
            
            # Verify the snapped solution exactly
            valid = True
            # Use a tight tolerance for verification of the snapped values
            # Since they are half-integers, the sum should be exactly 1.0
            # Floating point errors might accumulate, so check with 1e-9
            for i in range(n):
                for k in range(p):
                    for j in range(m):
                        val = 0.0
                        for r in range(target_rank):
                            val += U_snap[i * m + j, r] * V_snap[j * p + k, r] * W_snap[k * n + i, r]
                        if abs(val - 1.0) > 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid and target_rank < best_rank:
                best_rank = target_rank
                best_result = {
                    "U": U_snap.tolist(),
                    "V": V_snap.tolist(),
                    "W": W_snap.tolist()
                }
    
    return best_result

# EVOLVE-BLOCK-END
