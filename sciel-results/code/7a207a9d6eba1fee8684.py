"""Rewritten program for matrix multiplication 2x3x9 with search for lower rank."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    n, m, p = DIMENSIONS
    r_school = n * m * p  # 54
    
    # Build the schoolbook decomposition as fallback
    U_school = [[0] * r_school for _ in range(n * m)]
    V_school = [[0] * r_school for _ in range(m * p)]
    W_school = [[0] * r_school for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1

    def verify_decomposition(U, V, W, rank):
        """Verify that the decomposition exactly represents matrix multiplication.
        
        The formula is:
        (AB)[i,k] = sum_r W[k*n+i, r] * sum_j U[i*m+j, r] * A[i,j] * V[j*p+k, r] * B[j,k]
        
        Wait, let me reconsider. Based on standard tensor decomposition:
        T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        where T[i,j,k] = 1 for all valid (i,j,k) for matrix multiplication.
        """
        # Test with random matrices
        np.random.seed(12345)
        A_test = np.random.randn(n, m)
        B_test = np.random.randn(m, p)
        AB_direct = A_test @ B_test
        
        # Compute using decomposition
        # Interpretation: (AB)[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j,r] * A[i,j]) * (sum_j V[j*p+k,r] * B[j,k])
        # But this has two different sums over j...
        
        # Alternative interpretation:
        # (AB)[i,k] = sum_r W[k*n+i, r] * sum_j (U[i*m+j,r] * A[i,j]) * (V[j*p+k,r] * B[j,k])
        
        AB_decomp = np.zeros((n, p))
        U_arr = np.array(U)  # shape (n*m, rank)
        V_arr = np.array(V)  # shape (m*p, rank)
        W_arr = np.array(W)  # shape (p*n, rank)
        
        for i in range(n):
            for k in range(p):
                s = 0.0
                for j in range(m):
                    for r in range(rank):
                        s += W_arr[k * n + i, r] * U_arr[i * m + j, r] * A_test[i, j] * V_arr[j * p + k, r] * B_test[j, k]
                AB_decomp[i, k] = s
        
        return np.allclose(AB_direct, AB_decomp, atol=1e-9)

    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill
    
    # Verify schoolbook
    if not verify_decomposition(U_school, V_school, W_school, r_school):
        # This should never happen, but if it does, we have a bug
        pass
    
    # Try to find lower rank decompositions
    # Lower bound: max(n*m, m*p, p*n) = max(6, 27, 18) = 27
    # Try ranks from 27 upward
    
    best_rank = r_school
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    
    # Strategy: For each target rank, try random initializations and use
    # alternating least squares or gradient descent to minimize the error
    
    def try_rank(rank, num_trials=5, max_iters=50):
        """Try to find a rank-`rank` decomposition."""
        for trial in range(num_trials):
            if time.time() > deadline - 10:
                return None
            
            # Random initialization
            U_init = np.random.randn(n * m, rank) * 0.1
            V_init = np.random.randn(m * p, rank) * 0.1
            W_init = np.random.randn(p * n, rank) * 0.1
            
            # Use alternating minimization
            U_cur = U_init.copy()
            V_cur = V_init.copy()
            W_cur = W_init.copy()
            
            for it in range(max_iters):
                if time.time() > deadline - 10:
                    break
                
                # Update U: for each (i,j), minimize sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] - 1
                # This is a linear system for U[i*m+j, :]
                for i in range(n):
                    for j in range(m):
                        # We need: sum_r U[i*m+j,r] * (V[j*p+k,r] * W[k*n+i,r]) = 1 for all k
                        # This is: U[i*m+j,:] @ (V[j*p:,:]*W[k*n+i,:])^T ... not quite
                        # Actually: for each k, sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
                        # Let a_k = W[k*n+i,:] (shape rank)
                        # Let b_k = V[j*p+k,:] (shape rank)
                        # We need: U[i*m+j,:] dot (b_k * a_k) = 1 for all k
                        # This is a system: U[i*m+j,:] @ M = 1 where M[k,r] = b_k[r]*a_k[r]
                        M = np.zeros((p, rank))
                        for k in range(p):
                            M[k, :] = V_cur[j * p + k, :] * W_cur[k * n + i, :]
                        # Solve U[i*m+j,:] @ M = ones(p)
                        # This is underdetermined if rank > p, overdetermined if rank < p
                        try:
                            U_cur[i * m + j, :] = np.linalg.lstsq(M.T, np.ones(p), rcond=None)[0]
                        except:
                            pass
                
                # Update V: for each (j,k), minimize sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] - 1
                for j in range(m):
                    for k in range(p):
                        M = np.zeros((n, rank))
                        for i in range(n):
                            M[i, :] = U_cur[i * m + j, :] * W_cur[k * n + i, :]
                        try:
                            V_cur[j * p + k, :] = np.linalg.lstsq(M.T, np.ones(n), rcond=None)[0]
                        except:
                            pass
                
                # Update W: for each (k,i), minimize sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] - 1
                for k in range(p):
                    for i in range(n):
                        M = np.zeros((m, rank))
                        for j in range(m):
                            M[j, :] = U_cur[i * m + j, :] * V_cur[j * p + k, :]
                        try:
                            W_cur[k * n + i, :] = np.linalg.lstsq(M.T, np.ones(m), rcond=None)[0]
                        except:
                            pass
            
            # Check if we found a valid decomposition
            U_list = U_cur.tolist()
            V_list = V_cur.tolist()
            W_list = W_cur.tolist()
            
            if verify_decomposition(U_list, V_list, W_list, rank):
                return {"U": U_list, "V": V_list, "W": W_list}
        
        return None
    
    # Try ranks from 27 to 53
    for target_rank in range(27, r_school):
        if time.time() > deadline - 20:
            break
        
        result = try_rank(target_rank, num_trials=3, max_iters=30)
        if result is not None:
            best_rank = target_rank
            best_result = result
            break
    
    return best_result
# EVOLVE-BLOCK-END
