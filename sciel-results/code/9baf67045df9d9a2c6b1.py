"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Helper to validate a decomposition
    def is_valid_decomposition(U, V, W, rank):
        U_mat = np.array(U)
        V_mat = np.array(V)
        W_mat = np.array(W)
        
        # Check consistent terms: sum_c U[i,j,c] V[j,k,c] W[k,i,c] == 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_val = U_mat[i*m+j, :]
                    v_val = V_mat[j*p+k, :]
                    w_val = W_mat[k*n+i, :]
                    total = np.sum(u_val * v_val * w_val)
                    if not np.isclose(total, 1.0, atol=1e-9):
                        return False
                    
                    # Check inconsistent j terms: sum_c U[i,j',c] V[j,k,c] W[k,i,c] == 0 for j' != j
                    for j_prime in range(m):
                        if j_prime == j:
                            continue
                        u_val_p = U_mat[i*m+j_prime, :]
                        # v_val and w_val are same as above
                        total_p = np.sum(u_val_p * v_val * w_val)
                        if not np.isclose(total_p, 0.0, atol=1e-9):
                            return False
                            
                    # Check inconsistent k terms: sum_c U[i,j,c] V[j,k',c] W[k',i,c] == 0 for k' != k
                    for k_prime in range(p):
                        if k_prime == k:
                            continue
                        v_val_p = V_mat[j*p+k_prime, :]
                        w_val_p = W_mat[k_prime*n+i, :]
                        total_k = np.sum(u_val * v_val_p * w_val_p)
                        if not np.isclose(total_k, 0.0, atol=1e-9):
                            return False

                    # Check inconsistent i terms: sum_c U[i',j,c] V[j,k,c] W[k,i',c] == 0 for i' != i
                    for i_prime in range(n):
                        if i_prime == i:
                            continue
                        u_val_i = U_mat[i_prime*m+j, :]
                        w_val_i = W_mat[k*n+i_prime, :]
                        total_i = np.sum(u_val_i * v_val * w_val_i)
                        if not np.isclose(total_i, 0.0, atol=1e-9):
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
    time_limit = 140.0 # Safety margin
    
    # Search for lower rank
    # We search from a reasonable lower bound up to schoolbook - 1.
    # Lower bound is at least max(n*m, m*p, p*n) / something?
    # Actually, rank >= max(n, m, p) is trivial.
    # For 2x5x6, let's try from 10 up to 59.
    
    found_better = False
    
    # Try ranks from low to high to find the minimal one quickly if possible
    # But low ranks are harder to find.
    # Let's try a range.
    for r in range(10, rank_sb):
        if time.time() - start_time > time_limit:
            break
            
        attempts = 200 # Limit attempts per rank
        
        for _ in range(attempts):
            if time.time() - start_time > time_limit:
                break
            
            # Generate random integer matrices U, V
            # Use {-1, 0, 1}
            U_rand = np.zeros((n * m, r), dtype=int)
            V_rand = np.zeros((m * p, r), dtype=int)
            
            # Randomly fill with -1, 0, 1
            # Density 0.5
            mask_u = np.random.rand(n * m, r) < 0.5
            mask_v = np.random.rand(m * p, r) < 0.5
            
            U_rand[mask_u] = np.random.choice([-1, 1], size=mask_u.sum())
            V_rand[mask_v] = np.random.choice([-1, 1], size=mask_v.sum())
            
            # Construct Linear System for W
            # Variables: W[k*n + i, c] for k in 0..p-1, i in 0..n-1, c in 0..r-1
            # Total vars: p * n * r
            num_vars = p * n * r
            
            # Equations:
            # For each (i, j, k):
            # 1. Consistent: sum_c U[i,j,c] V[j,k,c] W[k,i,c] = 1
            # 2. Inconsistent j: sum_c U[i,j',c] V[j,k,c] W[k,i,c] = 0 for j' != j
            # 3. Inconsistent k: sum_c U[i,j,c] V[j,k',c] W[k',i,c] = 0 for k' != k
            # 4. Inconsistent i: sum_c U[i',j,c] V[j,k,c] W[k,i',c] = 0 for i' != i
            
            # Note: The inconsistent equations are coupled.
            # We can form a big matrix A and vector b.
            
            num_eqs = 0
            # Count equations:
            # For each i (n), j (m), k (p):
            # 1 consistent
            # (m-1) inconsistent j
            # (p-1) inconsistent k
            # (n-1) inconsistent i
            # Total per (i,j,k): 1 + m-1 + p-1 + n-1 = n+m+p-2
            # Total: n*m*p*(n+m+p-2)
            
            # This is large. n=2, m=5, p=6.
            # 2*5*6*(2+5+6-2) = 60 * 11 = 660 equations.
            # Variables: 12*r.
            # If r=20, vars=240.
            # Overdetermined.
            
            # We can just solve the consistent ones first?
            # No, we must satisfy all.
            
            # Let's build the matrix.
            A = np.zeros((660, num_vars))
            b = np.zeros(660)
            
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # 1. Consistent
                        # Eq: sum_c U[i,j,c] V[j,k,c] W[k,i,c] = 1
                        # Var indices for W[k,i,c]: (k*n+i)*r + c
                        base_var = (k * n + i) * r
                        coeffs = U_rand[i*m+j, :] * V_rand[j*p+k, :]
                        A[eq_idx, base_var:base_var+r] = coeffs
                        b[eq_idx] = 1
                        eq_idx += 1
                        
                        # 2. Inconsistent j
                        for j_prime in range(m):
                            if j_prime == j: continue
                            # Eq: sum_c U[i,j',c] V[j,k,c] W[k,i,c] = 0
                            coeffs_p = U_rand[i*m+j_prime, :] * V_rand[j*p+k, :]
                            A[eq_idx, base_var:base_var+r] = coeffs_p
                            b[eq_idx] = 0
                            eq_idx += 1
                            
                        # 3. Inconsistent k
                        for k_prime in range(p):
                            if k_prime == k: continue
                            # Eq: sum_c U[i,j,c] V[j,k',c] W[k',i,c] = 0
                            base_var_k = (k_prime * n + i) * r
                            coeffs_k = U_rand[i*m+j, :] * V_rand[j*p+k_prime, :]
                            A[eq_idx, base_var_k:base_var_k+r] = coeffs_k
                            b[eq_idx] = 0
                            eq_idx += 1
                            
                        # 4. Inconsistent i
                        for i_prime in range(n):
                            if i_prime == i: continue
                            # Eq: sum_c U[i',j,c] V[j,k,c] W[k,i',c] = 0
                            base_var_i = (k * n + i_prime) * r
                            coeffs_i = U_rand[i_prime*m+j, :] * V_rand[j*p+k, :]
                            A[eq_idx, base_var_i:base_var_i+r] = coeffs_i
                            b[eq_idx] = 0
                            eq_idx += 1
                            
            # Solve A W = b
            try:
                W_vec, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
            except Exception:
                continue
                
            # Check if W_vec entries are close to integers or half-integers
            W_check = W_vec * 2
            W_rounded = np.round(W_check)
            
            diff = np.abs(W_check - W_rounded)
            if np.max(diff) < 1e-6:
                W_snap = W_rounded / 2.0
                W_mat = W_snap.reshape(p * n, r)
                
                # Verify exact decomposition
                if is_valid_decomposition(U_rand.tolist(), V_rand.tolist(), W_mat.tolist(), r):
                    best_result = {"U": U_rand.tolist(), "V": V_rand.tolist(), "W": W_mat.tolist(), "rank": r}
                    found_better = True
                    break
                    
        if found_better:
            break
        if time.time() - start_time > time_limit:
            break
            
    return {"U": best_result["U"], "V": best_result["V"], "W": best_result["W"]}
# EVOLVE-BLOCK-END
