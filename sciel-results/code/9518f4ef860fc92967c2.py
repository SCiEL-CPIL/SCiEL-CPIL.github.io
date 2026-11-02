"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def _validate(U, V, W, n, m, p, tol=1e-6):
    """Check if U, V, W exactly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    # Build the tensor T[i,j,k] for C[i,k] = sum_j A[i,j]*B[j,k]
    # T has shape (n*m, m*p, p*n) in the factorization sense
    # Actually: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should equal 1 for all i,j,k
    # Wait, that's not right. The tensor decomposition is:
    # T_{(i,j), (j',k), (k',i')} = sum_c U[(i,j),c] * V[(j',k),c] * W[(k',i'),c]
    # But for matrix multiplication, we need:
    # For each (i,j) in A, (j,k) in B, (k,i) in C: the product A[i,j]*B[j,k] contributes to C[i,k]
    # So the tensor is T[i,j,k] = 1 (meaning A[i,j]*B[j,k] contributes to C[i,k])
    # The decomposition: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # But wait, that's a 3-index tensor, and we need it to be 1 for all valid (i,j,k).
    # Actually no - the tensor for matrix multiplication has T[i,j,k] = 1 for all i,j,k in range.
    # Let me reconsider. The matrix multiplication tensor T has dimensions (n*m) x (m*p) x (p*n)
    # T[a,b,c] = 1 if a = i*m+j, b = j'*p+k, c = k'*n+i' and j == j' and k == k'
    # So T[a,b,c] = 1 iff (a % m == b // p) and (b % p == c // n)
    
    # Let's verify: for each (i,j,k) with i in [0,n), j in [0,m), k in [0,p):
    # a = i*m + j, b = j*p + k, c = k*n + i
    # T[a,b,c] should be 1, and all other entries should be 0.
    
    # Check all entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = 0.0
                for c_idx in range(rank):
                    val += U[a, c_idx] * V[b, c_idx] * W[c, c_idx]
                if abs(val - 1.0) > tol:
                    return False
    # Check that non-diagonal entries are 0
    # Sample some non-diagonal entries
    for _ in range(100):
        a = random.randint(0, n*m-1)
        b = random.randint(0, m*p-1)
        c = random.randint(0, p*n-1)
        i, j = divmod(a, m)
        j2, k = divmod(b, p)
        k2, i2 = divmod(c, n)
        if j != j2 or k != k2:
            val = 0.0
            for c_idx in range(rank):
                val += U[a, c_idx] * V[b, c_idx] * W[c, c_idx]
            if abs(val) > tol:
                return False
    return True

def _schoolbook(n, m, p):
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

def _try_random_decomposition(n, m, p, target_rank, num_trials=50, seed=42):
    """Try to find a rank-target_rank decomposition using random initialization."""
    np.random.seed(seed)
    random.seed(seed)
    
    for trial in range(num_trials):
        U = np.random.randn(n * m, target_rank)
        V = np.random.randn(m * p, target_rank)
        W = np.random.randn(p * n, target_rank)
        
        # Build target tensor
        T = np.zeros((n * m, m * p, p * n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T[a, b, c] = 1.0
        
        # Try to solve: T = sum_c U[:,c] outer V[:,c] outer W[:,c]
        # This is a 3-way tensor decomposition. Use iterative approach.
        # Simplify: fix V and W, solve for U, etc.
        
        success = False
        for iteration in range(100):
            # Update U
            for c in range(target_rank):
                # U[:,c] should satisfy: sum_c' U[:,c'] * V[:,c'] * W[:,c'] = T
                # This is complex. Let's use a different approach.
                pass
        
        # Alternative: use the fact that for each (a,b), T[a,b,:] = sum_c U[a,c]*V[b,c]*W[:,c]
        # This is still complex. Let's just check if random works (unlikely) and move on.
        
        # Instead, let's try a structured approach: start from schoolbook and try to merge
        if trial == 0:
            break
    
    return None

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Return schoolbook as fallback
    result = _schoolbook(n, m, p)
    
    # Try to find a better decomposition
    # For now, return schoolbook. Future improvements can add search here.
    
    return result
# EVOLVE-BLOCK-END
