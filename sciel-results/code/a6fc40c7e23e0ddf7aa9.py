"""Find a rank-minimal exact tensor decomposition for multiplying a 2 by 3 matrix by a 3 by 6 matrix."""
import numpy as np
import time
import random

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 100  # Safety margin
    
    n, m, p = 2, 3, 6
    
    # Schoolbook decomposition
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school), dtype=np.float64)
    V_school = np.zeros((m * p, rank_school), dtype=np.float64)
    W_school = np.zeros((p * n, rank_school), dtype=np.float64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1.0
                V_school[j * p + k, col] = 1.0
                W_school[k * n + i, col] = 1.0
                col += 1
    
    best_rank = rank_school
    best_U = U_school.copy()
    best_V = V_school.copy()
    best_W = W_school.copy()
    
    # Try to find a better decomposition using ALS
    # We'll try various ranks below the schoolbook rank
    
    # The tensor for matrix multiplication M_{n,m,p} has entries:
    # T_{i,j,k} = 1 if the multiplication (i,j) * (j,k) -> (i,k) is valid
    # In our indexing: U is (n*m) x rank, V is (m*p) x rank, W is (p*n) x rank
    # The condition is: for all (i,j,k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # Wait, let me re-read the convention.
    # U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    # The tensor entry T[i,j,k] should equal sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    # And T[i,j,k] = 1 for all valid (i,j,k) in the multiplication.
    # Actually, for matrix multiplication C = AB, C_ik = sum_j A_ij B_jk
    # So the tensor T[i,j,k] = delta_{i,j,k} is 1 when the indices form a valid multiplication.
    # In the standard tensor decomposition for matrix multiplication:
    # T_{i,j,k} = sum_r U_{ij,r} V_{jk,r} W_{ki,r} where T_{i,j,k} = 1
    
    # Let's verify the schoolbook:
    # U[i*m+j, col] = 1, V[j*p+k, col] = 1, W[k*n+i, col] = 1
    # Product = 1*1*1 = 1 for each (i,j,k). Correct.
    
    # Now let's try to find a lower rank using ALS
    # We'll formulate this as a least squares problem
    
    def verify_decomposition(U, V, W, tol=1e-6):
        """Verify that the decomposition is exact."""
        # Check that for each (i,j,k), sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1
        errors = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(U.shape[1]):
                        val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    errors.append(abs(val - 1.0))
        return max(errors) if errors else 0.0
    
    def try_als(rank, max_iter=500, lr=0.01):
        """Try ALS to find a decomposition of the given rank."""
        # Initialize randomly
        U = np.random.randn(n * m, rank) * 0.1
        V = np.random.randn(m * p, rank) * 0.1
        W = np.random.randn(p * n, rank) * 0.1
        
        for it in range(max_iter):
            if time.time() > deadline:
                return None
            
            # Update U
            # For each (i,j), we want sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1 for all k
            # This is a system of linear equations in U[i*m+j, :]
            # Let A be a (p x rank) matrix where A[k, r] = V[j*p+k, r] * W[k*n+i, r]
            # Then we want A @ U[i*m+j, :].T = ones(p)
            for i in range(n):
                for j in range(m):
                    A = np.zeros((p, rank))
                    for k in range(p):
                        A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                    # Solve A @ x = ones(p)
                    try:
                        U[i * m + j, :] = np.linalg.lstsq(A, np.ones(p), rcond=None)[0]
                    except:
                        pass
            
            # Update V
            for j in range(m):
                for k in range(p):
                    A = np.zeros((n, rank))
                    for i in range(n):
                        A[i, :] = U[i * m + j, :] * W[k * n + i, :]
                    try:
                        V[j * p + k, :] = np.linalg.lstsq(A, np.ones(n), rcond=None)[0]
                    except:
                        pass
            
            # Update W
            for k in range(p):
                for i in range(n):
                    A = np.zeros((m, rank))
                    for j in range(m):
                        A[j, :] = U[i * m + j, :] * V[j * p + k, :]
                    try:
                        W[k * n + i, :] = np.linalg.lstsq(A, np.ones(m), rcond=None)[0]
                    except:
                        pass
            
            # Check error
            err = verify_decomposition(U, V, W, tol=1e-8)
            if err < 1e-4:
                # Round to nearest 0.5 if close
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                err_rounded = verify_decomposition(U_rounded, V_rounded, W_rounded, tol=1e-4)
                if err_rounded < 1e-4:
                    return U_rounded, V_rounded, W_rounded
                else:
                    return U, V, W
        
        return None
    
    # Try ranks from just below schoolbook down to some minimum
    # We'll try a few ranks
    for rank in range(rank_school - 1, max(15, rank_school - 20), -1):
        if time.time() > deadline:
            break
        
        # Try a few random restarts
        for restart in range(3):
            if time.time() > deadline:
                break
            
            result = try_als(rank, max_iter=100, lr=0.01)
            if result is not None:
                U_test, V_test, W_test = result
                err = verify_decomposition(U_test, V_test, W_test, tol=1e-5)
                if err < 1e-4:
                    best_rank = rank
                    best_U = U_test
                    best_V = V_test
                    best_W = W_test
                    break
        else:
            continue
        break
    
    # Convert to lists
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
