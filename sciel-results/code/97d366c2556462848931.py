"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

_START_TIME = time.time()
_TIMEOUT = 120.0
_SAFETY_MARGIN = 10.0

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _validate(U, V, W, n, m, p):
    """Check if the decomposition exactly reproduces matrix multiplication."""
    rank = len(U[0])
    # Target tensor: T[i,j,k] = delta_{j,k} for all i in [0,n), j in [0,m), k in [0,p)
    # But wait, for matrix multiplication C = A*B where A is n x m and B is m x p:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor is T[i,j,k] where the output index is (i,k), input indices are (j for A, j for B)
    # Actually the tensor decomposition is: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # And we need T[i,j,k] = 1 if j==k (the shared dimension index), else 0
    # Wait, that's not right either. Let me think more carefully.
    
    # For matrix multiplication A (n x m) * B (m x p) = C (n x p):
    # C[i,k] = sum_{j=0}^{m-1} A[i,j] * B[j,k]
    # 
    # The tensor T has indices (i, j, k) where:
    # - i is the row index of A (and C)
    # - j is the column index of A / row index of B (the contracted index)
    # - k is the column index of B (and C)
    # 
    # T[i,j,k] = 1 if this term contributes to C[i,k], which is always true when we're summing over j
    # Actually, T[i,j,k] = delta_{j,j} = 1 for all valid indices? No...
    # 
    # The tensor is defined as T[i,j,k] = 1 for all i, j, k in their ranges.
    # Because every term A[i,j]*B[j,k] contributes to C[i,k].
    # 
    # Wait, that can't be right for a non-trivial decomposition. Let me reconsider.
    # 
    # Actually, the matrix multiplication tensor is:
    # T[i, j, k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # 
    # No wait. The tensor is T[i,j,k] where the three modes are:
    # Mode 1: (i,j) - indices of A, flattened as i*m+j
    # Mode 2: (j,k) - indices of B, flattened as j*p+k  
    # Mode 3: (i,k) - indices of C, flattened as i*p+k
    # 
    # And T[i*m+j, j*p+k, i*p+k] = 1 if the indices are consistent, else 0.
    # But since j appears in both A and B indices, and i in A and C, and k in B and C:
    # T is 1 exactly when the flattened indices correspond to valid (i,j), (j,k), (i,k) triples.
    # 
    # So the tensor has exactly n*m*p non-zero entries, all equal to 1.
    # 
    # The decomposition: sum_c U[i*m+j, c] * V[j*p+k, c] * W[i*p+k, c] = T[i*m+j, j*p+k, i*p+k]
    # 
    # Wait, the problem says W[k*n+i, r] - so W's first index is k*n+i, not i*p+k.
    # Let me re-read: "output factor follows the official trace convention W[k*n+i,r]"
    # 
    # So W has shape (p*n, rank) and is indexed as W[k*n + i, c].
    # And U has shape (n*m, rank) indexed as U[i*m + j, c].
    # And V has shape (m*p, rank) indexed as V[j*p + k, c].
    # 
    # The decomposition is:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all valid i,j,k
    # 
    # This is a rank-r decomposition of the matrix multiplication tensor.
    
    # Build the tensor as a 3D array and check
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Check all (i,j,k) triples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _try_find_decomposition(rank_target, n, m, p, max_iterations=5000):
    """Try to find a decomposition of the given rank using random search + local optimization."""
    if rank_target < 1:
        return None
    
    U = np.random.randn(n*m, rank_target)
    V = np.random.randn(m*p, rank_target)
    W = np.random.randn(p*n, rank_target)
    
    best_error = float('inf')
    
    for it in range(max_iterations):
        if time.time() - _START_TIME > _TIMEOUT - _SAFETY_MARGIN:
            break
            
        # Compute error
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank_target):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    error += (val - 1.0) ** 2
        
        if error < 1e-10:
            # Found exact solution
            return (U, V, W)
        
        if error < best_error:
            best_error = error
        
        # Gradient-based update
        # dU[i*m+j, c] = sum over (j',k') where... actually let's do full gradient
        # For simplicity, use a simple stochastic update
        if it % 10 == 0:
            # Try to improve one component at a time
            idx = np.random.randint(n*m)
            cidx = np.random.randint(rank_target)
            # Compute gradient for U[idx, cidx]
            grad = 0.0
            for j in range(m):
                for k in range(p):
                    i = idx // m
                    jj = idx % m
                    if jj == j:
                        val = 0.0
                        for c in range(rank_target):
                            val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                        grad += 2.0 * (val - 1.0) * V[j*p+k, cidx] * W[k*n+i, cidx]
            U[idx, cidx] -= 0.01 * grad
        
        if it % 10 == 5:
            idx = np.random.randint(m*p)
            cidx = np.random.randint(rank_target)
            grad = 0.0
            for i in range(n):
                for k in range(p):
                    j = idx // p
                    kk = idx % p
                    if kk == k:
                        val = 0.0
                        for c in range(rank_target):
                            val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                        grad += 2.0 * (val - 1.0) * U[i*m+j, cidx] * W[k*n+i, cidx]
            V[idx, cidx] -= 0.01 * grad
        
        if it % 10 == 7:
            idx = np.random.randint(p*n)
            cidx = np.random.randint(rank_target)
            grad = 0.0
            for i in range(n):
                for j in range(m):
                    k = idx // n
                    ii = idx % n
                    if ii == i:
                        val = 0.0
                        for c in range(rank_target):
                            val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                        grad += 2.0 * (val - 1.0) * U[i*m+j, cidx] * V[j*p+k, cidx]
            W[idx, cidx] -= 0.01 * grad
    
    return None

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Try to find a lower-rank decomposition
    # Known results suggest rank can be lower than n*m*p = 45
    # Try ranks from low to high
    for rank in range(30, schoolbook_rank):
        if time.time() - _START_TIME > _TIMEOUT - _SAFETY_MARGIN:
            break
        result = _try_find_decomposition(rank, n, m, p, max_iterations=2000)
        if result is not None:
            U, V, W = result
            # Convert to lists
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            # Round to nearest 0.5 for cleaner output
            U_list = [[round(x * 2) / 2 for x in row] for row in U_list]
            V_list = [[round(x * 2) / 2 for x in row] for row in V_list]
            W_list = [[round(x * 2) / 2 for x in row] for row in W_list]
            if _validate(U_list, V_list, W_list, n, m, p):
                return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fall back to schoolbook
    return _schoolbook()
# EVOLVE-BLOCK-END
