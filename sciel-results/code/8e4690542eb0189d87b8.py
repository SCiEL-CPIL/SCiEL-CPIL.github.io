"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Helper to validate a decomposition
    def is_valid_decomposition(U, V, W, rank):
        U_mat = np.array(U)
        V_mat = np.array(V)
        W_mat = np.array(W)
        
        # T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # For matrix multiplication, T[i, j, k] = 1 for all i, j, k
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_val = U_mat[i*m+j, :]
                    v_val = V_mat[j*p+k, :]
                    w_val = W_mat[k*n+i, :]
                    total = np.sum(u_val * v_val * w_val)
                    if not np.isclose(total, 1.0, atol=1e-9):
                        return False
        return True

    # 1. Establish Baseline (Schoolbook)
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb, "rank": rank_sb}
    
    start_time = time.time()
    time_limit = 135.0  # Safety margin before 150s kill
    
    def time_remaining():
        return time_limit - (time.time() - start_time)
    
    # Search for lower rank using structured random search
    # Try ranks from rank_sb - 1 down to a reasonable lower bound
    min_rank = 48  # Known lower bounds suggest ~50+ for 2x5x6
    
    found_better = False
    
    # Different search strategies to try
    densities = [0.3, 0.4, 0.5, 0.6]
    value_sets = [
        [-1, 1],
        [-1, 0, 1],
        [-2, -1, 1, 2]
    ]
    
    for r in range(rank_sb - 1, min_rank - 1, -1):
        if time_remaining() < 5.0:
            break
            
        # Determine number of attempts based on time remaining
        # More attempts for higher ranks (closer to schoolbook)
        if r > rank_sb - 5:
            attempts = 500
        elif r > rank_sb - 15:
            attempts = 300
        else:
            attempts = 100
        
        for attempt in range(attempts):
            if time_remaining() < 2.0:
                break
            
            # Choose density and value set
            density = densities[attempt % len(densities)]
            values = value_sets[(attempt // len(densities)) % len(value_sets)]
            
            # Generate random sparse U, V
            U_rand = np.zeros((n * m, r), dtype=float)
            V_rand = np.zeros((m * p, r), dtype=float)
            
            mask_u = np.random.rand(n * m, r) < density
            mask_v = np.random.rand(m * p, r) < density
            
            U_rand[mask_u] = np.random.choice(values, size=mask_u.sum())
            V_rand[mask_v] = np.random.choice(values, size=mask_v.sum())
            
            # Construct Linear System for W
            num_vars = p * n * r
            num_eqs = n * m * p
            
            A = np.zeros((num_eqs, num_vars))
            b = np.ones(num_eqs)
            
            # Fill A efficiently
            for i in range(n):
                for j in range(m):
                    u_row = U_rand[i * m + j]
                    for k in range(p):
                        eq_idx = i * m * p + j * p + k
                        v_row = V_rand[j * p + k]
                        coeffs = u_row * v_row  # shape (r,)
                        base_var_idx = (k * n + i) * r
                        A[eq_idx, base_var_idx:base_var_idx + r] = coeffs
            
            # Solve A W_vec = b
            try:
                W_vec, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
            except Exception:
                continue
                
            # Check if W_vec entries are close to integers or half-integers
            # Try both integer and half-integer snapping
            W_check_int = W_vec
            W_rounded_int = np.round(W_check_int)
            diff_int = np.abs(W_check_int - W_rounded_int)
            
            W_check_half = W_vec * 2
            W_rounded_half = np.round(W_check_half)
            diff_half = np.abs(W_check_half - W_rounded_half)
            
            # Try integer snap first
            if np.max(diff_int) < 1e-6:
                W_snap = W_rounded_int
                W_mat = W_snap.reshape(p * n, r)
                if is_valid_decomposition(U_rand.tolist(), V_rand.tolist(), W_mat.tolist(), r):
                    best_result = {"U": U_rand.tolist(), "V": V_rand.tolist(), "W": W_mat.tolist(), "rank": r}
                    found_better = True
                    break
            # Try half-integer snap
            elif np.max(diff_half) < 1e-6:
                W_snap = W_rounded_half / 2.0
                W_mat = W_snap.reshape(p * n, r)
                if is_valid_decomposition(U_rand.tolist(), V_rand.tolist(), W_mat.tolist(), r):
                    best_result = {"U": U_rand.tolist(), "V": V_rand.tolist(), "W": W_mat.tolist(), "rank": r}
                    found_better = True
                    break
                    
        if found_better:
            break
        if time_remaining() < 5.0:
            break
            
    return {"U": best_result["U"], "V": best_result["V"], "W": best_result["W"]}
# EVOLVE-BLOCK-END
