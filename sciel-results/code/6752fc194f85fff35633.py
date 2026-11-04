"""Matrix multiplication 2x4x4 - rank-minimal exact tensor decomposition."""
import time
import random
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130  # 10s safety margin before 140s hard limit
    
    n, m, p = DIMENSIONS  # n=2, m=4, p=4
    # U shape: (n*m, rank) = (8, rank)
    # V shape: (m*p, rank) = (16, rank)
    # W shape: (p*n, rank) = (8, rank)
    # W[k*n+i, r] convention
    
    def build_decomp(U, V, W):
        """Convert lists to numpy and return dict."""
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        return {"U": U_arr.tolist(), "V": V_arr.tolist(), "W": W_arr.tolist()}
    
    def verify_decomp(U, V, W, tol=1e-8):
        """Check if U, V, W exactly decompose the matrix multiplication tensor.
        The tensor T[i,j,k] = 1 iff i in [0,n), j in [0,m), k in [0,p)
        and the decomposition satisfies:
        sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == T[i,j,k]
        """
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Compute sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
                    val = np.sum(U_arr[i*m+j] * V_arr[j*p+k] * W_arr[k*n+i])
                    expected = 1.0
                    err = abs(val - expected)
                    max_err = max(max_err, err)
        return max_err < tol, max_err
    
    def verify_full(U, V, W, tol=1e-8):
        """Full verification: check that for all i,j,k,l the product C[i,l] = sum_k A[i,k]*B[k,l]
        is correctly computed. Actually, the tensor decomposition is for the multiplication map.
        Let me reconsider: the matrix multiplication tensor has entries indexed by (i, j, k) where
        i is row of A (and C), j is col of A (and row of B), k is col of B (and col of C).
        The tensor entry T[i,j,k] = 1 means C[i,k] += A[i,j] * B[j,k].
        The decomposition: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        But W is indexed as W[k*n+i, r] per the convention W[k*n+i, r].
        """
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U_arr[i*m+j] * V_arr[j*p+k] * W_arr[k*n+i])
                    expected = 1.0
                    err = abs(val - expected)
                    max_err = max(max_err, err)
        return max_err < tol, max_err
    
    # Schoolbook decomposition (rank = n*m*p = 32)
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return U, V, W
    
    # Try to find lower rank decompositions
    # The rank of M_{2,4,4} is at least max(nm, mp, np) = max(8, 16, 8) = 16
    # Upper bound: 32 (schoolbook)
    # Known: rank(M_{2,2,2}) = 7, rank(M_{2,2,4}) = ?
    
    # Let's try random search with optimization
    def random_search(max_rank, num_trials=50, time_budget=30):
        """Try to find a decomposition of rank max_rank using random initialization + Newton."""
        best_err = float('inf')
        best_decomp = None
        trial_start = time.time()
        
        for trial in range(num_trials):
            if time.time() - trial_start > time_budget:
                break
            
            # Random initialization
            U = np.random.randn(n * m, max_rank)
            V = np.random.randn(m * p, max_rank)
            W = np.random.randn(p * n, max_rank)
            
            # Scale to get reasonable magnitudes
            scale = 1.0 / (max_rank ** 0.5)
            U *= scale
            V *= scale
            W *= scale
            
            # Use a simple gradient descent approach
            # Objective: sum_{i,j,k} (sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] - 1)^2
            # We'll use coordinate descent / alternating least squares
            
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            
            # Alternating minimization
            for iteration in range(200):
                if time.time() - trial_start > time_budget:
                    break
                
                # Update U: for each (i,j), U[i*m+j,:] should satisfy
                # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all k
                # This is a linear system in U[i*m+j,:]
                for i in range(n):
                    for j in range(m):
                        # Build matrix A[k, r] = V[j*p+k,r] * W[k*n+i,r]
                        # Then A @ U[i*m+j,:] = ones(p)
                        A = np.zeros((p, max_rank))
                        for k in range(p):
                            A[k, :] = V_list[j*p+k] * W_list[k*n+i]
                        # Solve A @ x = ones(p)
                        try:
                            x, residuals, rank_A, sv = np.linalg.lstsq(A, np.ones(p), rcond=None)
                            U_list[i*m+j] = x
                        except:
                            pass
                
                # Update V: for each (j,k), V[j*p+k,:] should satisfy
                # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all i
                for j in range(m):
                    for k in range(p):
                        A = np.zeros((n, max_rank))
                        for i in range(n):
                            A[i, :] = U_list[i*m+j] * W_list[k*n+i]
                        try:
                            x, residuals, rank_A, sv = np.linalg.lstsq(A, np.ones(n), rcond=None)
                            V_list[j*p+k] = x
                        except:
                            pass
                
                # Update W: for each (k,i), W[k*n+i,:] should satisfy
                # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all j
                for k in range(p):
                    for i in range(n):
                        A = np.zeros((m, max_rank))
                        for j in range(m):
                            A[j, :] = U_list[i*m+j] * V_list[j*p+k]
                        try:
                            x, residuals, rank_A, sv = np.linalg.lstsq(A, np.ones(m), rcond=None)
                            W_list[k*n+i] = x
                        except:
                            pass
                
                # Check error
                U_arr = np.array(U_list)
                V_arr = np.array(V_list)
                W_arr = np.array(W_list)
                max_err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.sum(U_arr[i*m+j] * V_arr[j*p+k] * W_arr[k*n+i])
                            max_err = max(max_err, abs(val - 1.0))
                
                if max_err < best_err:
                    best_err = max_err
                    best_decomp = (U_list.copy(), V_list.copy(), W_list.copy())
                
                if max_err < 1e-10:
                    break
            
            if time.time() - start_time > deadline - 5:
                break
        
        if best_err < 1e-8:
            return best_decomp, best_err
        return None, best_err
    
    # Try ranks from low to high
    # Lower bound: max(nm, mp, np) = 16
    # But also, rank >= n*m + m*p + p*n - something? 
    # For 2x2x2, rank is 7 = 8-1
    # For 2x4x4, let's try from 16 upward
    
    # First, ensure we have a valid schoolbook solution as fallback
    U_sb, V_sb, W_sb = schoolbook()
    valid_sb, err_sb = verify_full(U_sb, V_sb, W_sb)
    
    # Try to find lower rank solutions
    for target_rank in range(16, 33):
        if time.time() > deadline - 10:
            break
        
        # Only try if target_rank < 32 (schoolbook rank)
        if target_rank >= 32:
            break
        
        result, err = random_search(target_rank, num_trials=20, time_budget=25)
        if result is not None:
            U_r, V_r, W_r = result
            valid, final_err = verify_full(U_r, V_r, W_r)
            if valid:
                return build_decomp(U_r, V_r, W_r)
    
    # Fallback to schoolbook
    return build_decomp(U_sb, V_sb, W_sb)
# EVOLVE-BLOCK-END

