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
    
    # Start with rank 54 Strassen-based decomposition
    rank = 54
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_block(i_base, j_base, k_base):
        nonlocal col
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
    
    def _verify_error(U_list, V_list, W_list, n, m, p):
        """Compute verification error using numpy."""
        U_np = np.array(U_list, dtype=float)
        V_np = np.array(V_list, dtype=float)
        W_np = np.array(W_list, dtype=float)
        r = U_np.shape[1]
        
        np.random.seed(42)
        A_test = np.random.randn(n, m)
        B_test = np.random.randn(m, p)
        C_expected = A_test @ B_test
        
        C_computed = np.zeros((n, p))
        for r_idx in range(r):
            U_r = U_np[:, r_idx].reshape(n, m)
            V_r = V_np[:, r_idx].reshape(m, p)
            W_r = W_np[:, r_idx].reshape(p, n)
            UA = U_r * A_test
            VB = V_r * B_test
            temp = UA @ VB
            C_computed += temp * W_r.T
        
        return np.max(np.abs(C_computed - C_expected))
    
    def _verify_fast(U_list, V_list, W_list, n, m, p):
        """Fast verification using numpy."""
        return _verify_error(U_list, V_list, W_list, n, m, p) < 1e-10
    
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
    
    # Try to reduce rank using Kronecker product structure exploitation
    # Key insight: The 2x5x6 tensor can be viewed as combining smaller tensors.
    # Strategy: Use linear combinations of Kronecker products to find rank reduction.
    # 
    # The multiplication tensor T[i,j,k] = 1 for all i,j,k (since C[i,k] = sum_j A[i,j]*B[j,k])
    # We can try to express this as a sum of fewer rank-1 tensors.
    #
    # Approach: Start with the rank-54 solution and try to find linear dependencies
    # in the decomposition that allow us to reduce the rank.
    
    if _time_remaining() > 30:
        try:
            best_rank = col
            best_U, best_V, best_W = U, V, W
            
            # Convert to numpy for optimization
            U_np = np.array(U, dtype=float)
            V_np = np.array(V, dtype=float)
            W_np = np.array(W, dtype=float)
            
            # Try various target ranks using more sophisticated ALS with regularization
            for target_rank in range(53, 42, -1):
                if _time_remaining() < 20:
                    break
                    
                np.random.seed(42 + target_rank)
                U_opt = np.zeros((n*m, target_rank))
                V_opt = np.zeros((m*p, target_rank))
                W_opt = np.zeros((p*n, target_rank))
                
                # Better initialization: use SVD of the current solution
                if col >= target_rank:
                    # Try to project the existing solution
                    U_opt[:, :target_rank] = U_np[:, :target_rank]
                    V_opt[:, :target_rank] = V_np[:, :target_rank]
                    W_opt[:, :target_rank] = W_np[:, :target_rank]
                else:
                    U_opt = np.random.randn(n*m, target_rank) * 0.1
                    V_opt = np.random.randn(m*p, target_rank) * 0.1
                    W_opt = np.random.randn(p*n, target_rank) * 0.1
                
                # More iterations with better convergence criteria
                for iteration in range(80):
                    if _time_remaining() < 20:
                        break
                    
                    # Update U with better numerical handling
                    for i in range(n):
                        for j in range(m):
                            M = np.zeros((p, target_rank))
                            for k in range(p):
                                M[k, :] = V_opt[j*p+k, :] * W_opt[k*n+i, :]
                            try:
                                # Use rcond to handle ill-conditioned cases
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
                    
                    # Check convergence by measuring the error
                    if iteration % 10 == 0 and _time_remaining() > 20:
                        try:
                            err = _verify_error(U_opt.tolist(), V_opt.tolist(), W_opt.tolist(), n, m, p)
                            if err < 1e-12:
                                break
                        except:
                            pass
                
                # Final verification with high precision
                U_list = U_opt.tolist()
                V_list = V_opt.tolist()
                W_list = W_opt.tolist()
                
                if _verify_fast(U_list, V_list, W_list, n, m, p):
                    if target_rank < best_rank:
                        best_rank = target_rank
                        best_U, best_V, best_W = U_list, V_list, W_list
                        # If we found a lower rank, we can stop early
                        if target_rank <= 45:
                            break
        
            # Use the best solution found
            if best_rank < col:
                return {"U": best_U, "V": best_V, "W": best_W}
        except Exception:
            pass
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
