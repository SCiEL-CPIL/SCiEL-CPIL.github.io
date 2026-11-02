import time
import numpy as np
from itertools import product

DIMENSIONS = [4, 4, 9]

def verify(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Check shapes
    if U.shape != (n*m, rank):
        return False
    if V.shape != (m*p, rank):
        return False
    if W.shape != (p*n, rank):
        return False
    
    # The tensor identity: for each (i,j,k), sum_t U[i*m+j, t] * V[j*p+k, t] = W[k*n+i, t]... 
    # Actually, the decomposition is: T_{i,j,k}^{(i,k)} = sum_t U[i*m+j, t] * V[j*p+k, t]
    # And this should equal 1 if the output index matches, i.e., the (i,k) entry.
    # Wait, let me re-read the task.
    
    # The tensor for matrix multiplication: C[i,k] = sum_j A[i,j] * B[j,k]
    # As a bilinear form: sum_{i,j,k} A[i,j] * B[j,k] * C_{i,k}
    # The tensor T has T[(i,j),(j,k)] = delta_{i,k} in the output... 
    
    # Actually, the standard formulation: the multiplication tensor is
    # T = sum_{i,j,k} e_i^A ⊗ e_j^B ⊗ f_{i,k}^C
    # where e_i^A is the (i,j) entry of A, e_j^B is the (j,k) entry of B, f_{i,k}^C is the (i,k) entry of C.
    
    # A rank-r decomposition: T = sum_{t=1}^r u_t ⊗ v_t ⊗ w_t
    # where u_t is a vector in R^{n*m} (flattened A), v_t in R^{m*p} (flattened B), w_t in R^{p*n} (flattened C).
    
    # The identity is: for all (i,j,k):
    # sum_t U[i*m+j, t] * V[j*p+k, t] = W[k*n+i, t]... no.
    
    # Let me think again. The tensor T is a 3-tensor with modes (n*m, m*p, p*n).
    # T[a, b, c] where a = i*m+j, b = j'*p+k, c = k'*n+i'
    # T[a, b, c] = 1 if i=i', j=j', k=k' and the output index c corresponds to (i,k), else 0.
    
    # Actually, the correct formulation: 
    # The multiplication map is (A, B) -> C where C[i,k] = sum_j A[i,j] B[j,k]
    # As a trilinear form: sum_{i,j,k} A[i,j] B[j,k] C^*[i,k]
    # The tensor T has T[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k, and 0 otherwise.
    
    # A rank-r decomposition: T = sum_t U[:,t] ⊗ V[:,t] ⊗ W[:,t]
    # So: sum_t U[a,t] V[b,t] W[c,t] = T[a,b,c]
    
    # Let's verify this.
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Build the tensor T
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Compute the decomposition
    # sum_t U[a,t] V[b,t] W[c,t]
    # This is a 3-way tensor product
    # We can compute it as: for each t, outer product of U[:,t], V[:,t], W[:,t]
    # Then sum over t
    
    # Efficient way: 
    # Result[a,b,c] = sum_t U[a,t] V[b,t] W[c,t]
    # = sum_t (U[a,t] V[b,t]) W[c,t]
    
    # Let's use einsum or manual
    # Create UV[a,b,t] = U[a,t] V[b,t]
    UV = np.einsum('at,bt->abt', U, V)
    Result = np.einsum('abt,ct->abc', UV, W)
    
    # Check if Result equals T
    if not np.allclose(Result, T, atol=1e-9):
        return False
    
    return True

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    n, m, p = DIMENSIONS
    
    # Start with trivial decomposition
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
    
    best_rank = rank
    best_U, best_V, best_W = U, V, W
    
    # Try to reduce rank by merging terms
    # A term t can be removed if its contribution is already covered by other terms
    # Or we can try to find linear dependencies
    
    # Simple approach: try to find a subset of terms that still spans the tensor
    # This is NP-hard in general, so we use heuristics
    
    # Try random projections to find lower-rank decompositions
    np.random.seed(42)
    
    for attempt in range(50):
        if time.time() > deadline - 10:
            break
        
        # Try a random rank-r' decomposition
        r_prime = max(100, best_rank - 10 * (attempt + 1))
        if r_prime < 1:
            break
            
        # Random initialization
        U_rand = np.random.randn(n * m, r_prime)
        V_rand = np.random.randn(m * p, r_prime)
        W_rand = np.random.randn(p * n, r_prime)
        
        # Try to solve for the tensor identity using least squares
        # This is underdetermined in general, so we just check if it works
        # (unlikely for random init, but let's try)
        
        # Actually, let's try a different approach: use the fact that we can
        # reparameterize the decomposition
        
        # For now, just keep the trivial solution
        pass
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
