"""Matrix multiplication 2,5,6 - rank-minimal tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [2, 5, 6]
START_TIME = time.time()
DEADLINE = START_TIME + 120  # 120s wall-clock limit

def _time_remaining():
    return DEADLINE - time.time()

# EVOLVE-BLOCK-START

def construct():
    n, m, p = DIMENSIONS
    
    # Rank 54 decomposition using Strassen's algorithm for 2x2 blocks
    # A is 2x5, B is 5x6, C is 2x6
    # Partition m=5 into pairs (0,1), (2,3) and single (4).
    # Strassen blocks: 2 j-pairs * 3 k-pairs * 7 = 42
    # Naive blocks:    1 j-single * 3 k-pairs * 4 = 12
    # Total rank = 54
    
    rank = 54
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_block(i_base, j_base, k_base):
        nonlocal col
        # i_base: 0 (n=2, only one pair of rows)
        # j_base: 0 or 2 (pairs of cols in A / rows in B)
        # k_base: 0, 2, or 4 (pairs of cols in B / cols in C)
        
        # Indices: U[i*m + j], V[j*p + k], W[k*n + i]
        
        # M0: (A00+A11)(B00+B11)
        U[i_base * m + j_base][col] += 1
        U[(i_base+1) * m + (j_base+1)][col] += 1
        V[j_base * p + k_base][col] += 1
        V[(j_base+1) * p + (k_base+1)][col] += 1
        W[k_base * n + i_base][col] += 1
        W[(k_base+1) * n + (i_base+1)][col] += 1
        col += 1
        
        # M1: (A10+A11)B00
        U[(i_base+1) * m + j_base][col] += 1
        U[(i_base+1) * m + (j_base+1)][col] += 1
        V[j_base * p + k_base][col] += 1
        W[k_base * n + (i_base+1)][col] += 1
        W[(k_base+1) * n + (i_base+1)][col] -= 1
        col += 1
        
        # M2: A00(B01-B11)
        U[i_base * m + j_base][col] += 1
        V[j_base * p + (k_base+1)][col] += 1
        V[(j_base+1) * p + (k_base+1)][col] -= 1
        W[(k_base+1) * n + i_base][col] += 1
        W[(k_base+1) * n + (i_base+1)][col] += 1
        col += 1
        
        # M3: A11(B10-B00)
        U[(i_base+1) * m + (j_base+1)][col] += 1
        V[(j_base+1) * p + k_base][col] += 1
        V[j_base * p + k_base][col] -= 1
        W[k_base * n + i_base][col] += 1
        W[k_base * n + (i_base+1)][col] += 1
        col += 1
        
        # M4: (A00+A01)B11
        U[i_base * m + j_base][col] += 1
        U[i_base * m + (j_base+1)][col] += 1
        V[(j_base+1) * p + (k_base+1)][col] += 1
        W[k_base * n + i_base][col] -= 1
        W[(k_base+1) * n + i_base][col] += 1
        col += 1
        
        # M5: (A10-A00)(B00+B01)
        U[(i_base+1) * m + j_base][col] += 1
        U[i_base * m + j_base][col] -= 1
        V[j_base * p + k_base][col] += 1
        V[j_base * p + (k_base+1)][col] += 1
        W[(k_base+1) * n + (i_base+1)][col] += 1
        col += 1
        
        # M6: (A01-A11)(B10+B11)
        U[i_base * m + (j_base+1)][col] += 1
        U[(i_base+1) * m + (j_base+1)][col] -= 1
        V[(j_base+1) * p + k_base][col] += 1
        V[(j_base+1) * p + (k_base+1)][col] += 1
        W[k_base * n + i_base][col] += 1
        col += 1

    def add_naive_2112_block(i_base, j_base, k_base):
        nonlocal col
        # 2x1 * 1x2 -> 2x2, requires 4 scalar multiplications
        
        U[i_base * m + j_base][col] += 1
        V[j_base * p + k_base][col] += 1
        W[k_base * n + i_base][col] += 1
        col += 1
        
        U[i_base * m + j_base][col] += 1
        V[j_base * p + (k_base+1)][col] += 1
        W[(k_base+1) * n + i_base][col] += 1
        col += 1
        
        U[(i_base+1) * m + j_base][col] += 1
        V[j_base * p + k_base][col] += 1
        W[k_base * n + (i_base+1)][col] += 1
        col += 1
        
        U[(i_base+1) * m + j_base][col] += 1
        V[j_base * p + (k_base+1)][col] += 1
        W[(k_base+1) * n + (i_base+1)][col] += 1
        col += 1

    # Apply Strassen blocks for j-pairs (0,1) and (2,3), k-pairs (0,1), (2,3), (4,5)
    add_strassen_block(0, 0, 0)
    add_strassen_block(0, 0, 2)
    add_strassen_block(0, 0, 4)
    
    add_strassen_block(0, 2, 0)
    add_strassen_block(0, 2, 2)
    add_strassen_block(0, 2, 4)
    
    # Apply naive blocks for j=4, k-pairs (0,1), (2,3), (4,5)
    add_naive_2112_block(0, 4, 0)
    add_naive_2112_block(0, 4, 2)
    add_naive_2112_block(0, 4, 4)
    
    # Trim to actual rank used
    if col < rank:
        U = [row[:col] for row in U]
        V = [row[:col] for row in V]
        W = [row[:col] for row in W]
    
    def _verify_fast(U_list, V_list, W_list, n, m, p):
        """Fast verification using numpy."""
        U_np = np.array(U_list, dtype=float)
        V_np = np.array(V_list, dtype=float)
        W_np = np.array(W_list, dtype=float)
        r = U_np.shape[1]
        
        np.random.seed(42)
        A_test = np.random.randn(n, m)
        B_test = np.random.randn(m, p)
        C_expected = A_test @ B_test
        
        # Reshape for efficient computation
        # U: (n*m, r), V: (m*p, r), W: (p*n, r)
        # For each rank component r_idx:
        #   C[i,k] += sum over j of U[i*m+j, r_idx] * V[j*p+k, r_idx] * W[k*n+i, r_idx] * A[i,j] * B[j,k]
        
        # Vectorized: for each r_idx, compute contribution
        # U_r[i,j] = U_np[i*m+j, r_idx]
        # V_r[j,k] = V_np[j*p+k, r_idx]
        # W_r[k,i] = W_np[k*n+i, r_idx]
        
        C_computed = np.zeros((n, p))
        for r_idx in range(r):
            U_r = U_np[:, r_idx].reshape(n, m)
            V_r = V_np[:, r_idx].reshape(m, p)
            W_r = W_np[:, r_idx].reshape(p, n)
            # C[i,k] += W_r[k,i] * sum_j U_r[i,j] * A[i,j] * V_r[j,k] * B[j,k]
            # = W_r[k,i] * (U_r * A) @ (V_r * B)
            UA = U_r * A_test  # (n, m)
            VB = V_r * B_test  # (m, p)
            temp = UA @ VB  # (n, p)
            C_computed += temp * W_r.T  # W_r is (p,n), W_r.T is (n,p)
        
        return np.allclose(C_computed, C_expected, atol=1e-10)
    
    # Quick verification with random matrices (only if time permits)
    if _time_remaining() > 5:
        try:
            if not _verify_fast(U, V, W, n, m, p):
                # Fallback to schoolbook
                rank_school = n * m * p
                U_school = [[0] * rank_school for _ in range(n * m)]
                V_school = [[0] * rank_school for _ in range(m * p)]
                W_school = [[0] * rank_school for _ in range(p * n)]
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            idx = i * m * p + j * p + k
                            U_school[i*m+j][idx] = 1
                            V_school[j*p+k][idx] = 1
                            W_school[k*n+i][idx] = 1
                return {"U": U_school, "V": V_school, "W": W_school}
        except Exception:
            pass
    
    # Try to reduce rank by searching for linear dependencies
    if _time_remaining() > 30:
        try:
            best_rank = col
            best_U, best_V, best_W = U, V, W
            
            # Try multiple rank targets with different initializations
            for target_rank in range(52, 44, -1):
                if _time_remaining() < 20:
                    break
                
                # Multiple random restarts
                for restart in range(3):
                    if _time_remaining() < 15:
                        break
                    
                    np.random.seed(42 + restart * 1000)
                    U_opt = np.zeros((n*m, target_rank))
                    V_opt = np.zeros((m*p, target_rank))
                    W_opt = np.zeros((p*n, target_rank))
                    
                    # Initialize with structured patterns based on Strassen
                    if target_rank <= col:
                        # Use truncated current solution as base
                        base_U = np.array(U, dtype=float)[:, :min(target_rank, col)]
                        base_V = np.array(V, dtype=float)[:, :min(target_rank, col)]
                        base_W = np.array(W, dtype=float)[:, :min(target_rank, col)]
                        
                        if base_U.shape[1] < target_rank:
                            pad_size = target_rank - base_U.shape[1]
                            U_opt[:, :base_U.shape[1]] = base_U
                            V_opt[:, :base_V.shape[1]] = base_V
                            W_opt[:, :base_W.shape[1]] = base_W
                            U_opt[:, base_U.shape[1]:] = np.random.randn(n*m, pad_size) * 0.01
                            V_opt[:, base_V.shape[1]:] = np.random.randn(m*p, pad_size) * 0.01
                            W_opt[:, base_W.shape[1]:] = np.random.randn(p*n, pad_size) * 0.01
                        else:
                            U_opt[:, :target_rank] = base_U[:, :target_rank]
                            V_opt[:, :target_rank] = base_V[:, :target_rank]
                            W_opt[:, :target_rank] = base_W[:, :target_rank]
                    else:
                        U_opt = np.random.randn(n*m, target_rank) * 0.1
                        V_opt = np.random.randn(m*p, target_rank) * 0.1
                        W_opt = np.random.randn(p*n, target_rank) * 0.1
                    
                    # Alternating least squares with more iterations
                    for iteration in range(80):
                        if _time_remaining() < 15:
                            break
                        
                        # Update U
                        for i in range(n):
                            for j in range(m):
                                M = np.zeros((p, target_rank))
                                for k in range(p):
                                    M[k, :] = V_opt[j*p+k, :] * W_opt[k*n+i, :]
                                try:
                                    x, residuals, rank_M, sv = np.linalg.lstsq(M, np.ones(p), rcond=1e-10)
                                    U_opt[i*m+j, :] = x
                                except:
                                    pass
                        
                        # Update V
                        for j in range(m):
                            for k in range(p):
                                M = np.zeros((n, target_rank))
                                for i in range(n):
                                    M[i, :] = U_opt[i*m+j, :] * W_opt[k*n+i, :]
                                try:
                                    x, residuals, rank_M, sv = np.linalg.lstsq(M, np.ones(n), rcond=1e-10)
                                    V_opt[j*p+k, :] = x
                                except:
                                    pass
                        
                        # Update W
                        for k in range(p):
                            for i in range(n):
                                M = np.zeros((m, target_rank))
                                for j in range(m):
                                    M[j, :] = U_opt[i*m+j, :] * V_opt[j*p+k, :]
                                try:
                                    x, residuals, rank_M, sv = np.linalg.lstsq(M, np.ones(m), rcond=1e-10)
                                    W_opt[k*n+i, :] = x
                                except:
                                    pass
                        
                        if _time_remaining() < 15:
                            break
                    
                    # Verify the optimized solution
                    U_list = U_opt.tolist()
                    V_list = V_opt.tolist()
                    W_list = W_opt.tolist()
                    
                    if _verify_fast(U_list, V_list, W_list, n, m, p):
                        if target_rank < best_rank:
                            best_rank = target_rank
                            best_U, best_V, best_W = U_list, V_list, W_list
                            break  # Found valid solution for this rank, move to lower rank
        
            # Use the best solution found
            if best_rank < col:
                return {"U": best_U, "V": best_V, "W": best_W}
        except Exception:
            pass
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
