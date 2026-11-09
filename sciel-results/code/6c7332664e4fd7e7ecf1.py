"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 8
    deadline = time.time() + 120  # safety margin
    
    # Build the matrix multiplication tensor T[i,j,k] = 1 if the (i,k) entry of A@B
    # involves A[i,j]*B[j,k], i.e., T is the 3D tensor where T[i,j,k]=1 means
    # that entry (i,k) of the product uses A[i,j]*B[j,k].
    # For our decomposition: C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T has shape (n, m, p) = (2, 4, 8)
    
    # Schoolbook: rank = n*m*p = 64
    rank_schoolbook = n * m * p
    
    # Try to find a lower rank decomposition
    # The tensor T[i,j,k] = 1 for all i,j,k (it's the all-ones tensor in the
    # sense that every triple (i,j,k) contributes to C[i,k])
    
    # Actually, the matrix multiplication tensor is:
    # T[i, j, k] = 1 if we're computing C[i,k] = sum_j A[i,j]*B[j,k]
    # So T is a 3D array of shape (n, m, p) where T[i,j,k] = 1 for all i,j,k.
    
    # A rank-r decomposition gives:
    # U of shape (n*m, r), V of shape (m*p, r), W of shape (p*n, r)
    # such that for all i,j,k:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 (since T[i,j,k]=1)
    # Wait, no. The tensor decomposition is:
    # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]  (cyclic)
    # But the interface uses U of shape (n*m, rank), V of shape (m*p, rank), W of shape (p*n, rank)
    # with U indexed by (i*m+j), V by (j*p+k), W by (k*n+i).
    
    # The condition is: for all i in [0,n), j in [0,m), k in [0,p):
    #   sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # Let's try to find a rank-56 decomposition using optimization.
    
    def build_tensor(n, m, p):
        """Return the list of (i,j,k) triples, all with value 1."""
        triples = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    triples.append((i, j, k))
        return triples
    
    triples = build_tensor(n, m, p)
    num_triples = len(triples)  # 64
    
    # Try rank 56
    target_rank = 56
    
    # Initialize U, V, W with small random values and optimize
    np.random.seed(42)
    
    best_U = None
    best_V = None
    best_W = None
    best_error = float('inf')
    
    for attempt in range(10):
        if time.time() > deadline - 30:
            break
            
        U = np.random.randn(n * m, target_rank) * 0.1
        V = np.random.randn(m * p, target_rank) * 0.1
        W = np.random.randn(p * n, target_rank) * 0.1
        
        # Gradient descent on the loss: sum over all (i,j,k) of (sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] - 1)^2
        lr = 0.01
        for step in range(500):
            if time.time() > deadline - 30:
                break
            
            # Compute current values and gradients
            # This is expensive; let's use a simpler approach
            # For each triple (i,j,k), the value is sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
            # Gradient w.r.t. U[i*m+j,c] = V[j*p+k,c]*W[k*n+i,c]*(val-1)
            # etc.
            
            # This is too slow for 64 triples * 56 rank * 500 steps
            # Let's use a different approach: direct construction
            
        # Alternative: use a known Strassen-like construction
        break
    
    # Fallback: schoolbook decomposition
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
# EVOLVE-BLOCK-END
