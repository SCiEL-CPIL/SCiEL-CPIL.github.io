import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 105  # Safety margin
    
    # Schoolbook decomposition as fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school
    
    # Try to find lower rank decompositions
    # For 3x3x7, known results suggest rank might be around 45-55
    # Use randomized search with verification
    
    def verify_decomposition(U, V, W, rank):
        """Verify that the decomposition exactly computes matrix multiplication"""
        # Convert to numpy for efficient checking
        U_np = np.array(U, dtype=np.float64)
        V_np = np.array(V, dtype=np.float64)
        W_np = np.array(W, dtype=np.float64)
        
        # The decomposition says: (A*B)[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # We need to check this for all i,j,k
        # But actually, the tensor decomposition is:
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # where T[i,j,k] = 1 if this corresponds to a valid multiplication term
        # For matrix mult C = A*B, C[i,k] = sum_j A[i,j]*B[j,k]
        # So the tensor has T[i,j,k] = 1 for all valid (i,j,k) in the multiplication
        
        # Actually, let me reconsider. The matrix multiplication tensor for (n,m,p) is:
        # M[i,j,k] = 1 if the (i,j) entry of A and (j,k) entry of B contribute to (i,k) of C
        # This is just 1 for all i in [0,n), j in [0,m), k in [0,p)
        
        # The decomposition: M[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        # Check a few random entries for speed, then verify all if close
        np.random.seed(42)
        sample_indices = list(product(range(n), range(m), range(p)))
        
        max_err = 0.0
        for i, j, k in sample_indices:
            expected = 1.0  # All entries of the mult tensor are 1
            actual = 0.0
            for r in range(rank):
                actual += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
            err = abs(actual - expected)
            if err > max_err:
                max_err = err
        
        return max_err < 1e-9
    
    def try_rank(rank):
        """Try to find a decomposition of given rank using randomized approach"""
        nonlocal best_result, best_rank
        if rank >= best_rank:
            return False
        if time.time() > deadline - 5:
            return False
        
        # Use random initialization and gradient-like descent
        # Start with random vectors
        np.random.seed(rank * 1000 + 42)
        
        U = np.random.randn(n * m, rank) * 0.1
        V = np.random.randn(m * p, rank) * 0.1
        W = np.random.randn(p * n, rank) * 0.1
        
        # Try several random initializations
        for init in range(10):
            if time.time() > deadline - 10:
                return False
            
            np.random.seed(init * 12345 + rank)
            U = np.random.randn(n * m, rank) * 0.5
            V = np.random.randn(m * p, rank) * 0.5
            W = np.random.randn(p * n, rank) * 0.5
            
            # Simple iterative refinement
            for iteration in range(50):
                if time.time() > deadline - 10:
                    return False
                
                # Compute error
                total_err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                            err = abs(val - 1.0)
                            total_err += err
                
                if total_err < 1e-8:
                    # Found a valid decomposition
                    U_list = [[float(U[i*m+j, r]) for r in range(rank)] for i in range(n) for j in range(m)]
                    V_list = [[float(V[j*p+k, r]) for r in range(rank)] for j in range(m) for k in range(p)]
                    W_list = [[float(W[k*n+i, r]) for r in range(rank)] for k in range(p) for i in range(n)]
                    
                    # Verify
                    if verify_decomposition(U_list, V_list, W_list, rank):
                        best_result = {"U": U_list, "V": V_list, "W": W_list}
                        best_rank = rank
                        return True
                
                # Gradient step (simplified)
                lr = 0.01 / (1 + iteration * 0.1)
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            grad_U = np.zeros(rank)
                            grad_V = np.zeros(rank)
                            grad_W = np.zeros(rank)
                            for r in range(rank):
                                val = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                                err = val - 1.0
                                grad_U[r] += err * V[j*p+k, r] * W[k*n+i, r]
                                grad_V[r] += err * U[i*m+j, r] * W[k*n+i, r]
                                grad_W[r] += err * U[i*m+j, r] * V[j*p+k, r]
                            U[i*m+j] -= lr * grad_U
                            V[j*p+k] -= lr * grad_V
                            W[k*n+i] -= lr * grad_W
        
        return False
    
    # Try ranks from lower to higher
    # Based on known results, try a range
    for rank in range(45, best_rank):
        if time.time() > deadline - 5:
            break
        if try_rank(rank):
            break
    
    return best_result
# EVOLVE-BLOCK-END
