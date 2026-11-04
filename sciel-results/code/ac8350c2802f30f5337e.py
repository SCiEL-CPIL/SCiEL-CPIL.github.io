"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    """Find a low-rank decomposition for 3x3 x 3x5 matrix multiplication."""
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill
    
    # Build the target multiplication tensor T[i,j,k] = delta structure
    # T is a 3D tensor of shape (n*m, m*p, p*n) where T[i,j,k] = 1 if the 
    # (i,j) entry of A times (j,k) entry of B contributes to (i,k) entry of C
    # Actually, the tensor is: for each (i,j,k), coefficient is 1
    # The decomposition: C[i,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Wait, let me re-read the convention.
    
    # The task says: W[k*n+i,r] follows the official trace convention
    # So W has shape (p*n, rank)
    # U has shape (n*m, rank) 
    # V has shape (m*p, rank)
    
    # The matrix multiplication tensor: C[i,k] = sum_j A[i,j] * B[j,k]
    # In tensor form: T_{(i,j),(j,k),(k,i)} = 1 for all i,j,k
    # But we need to index properly.
    
    # U is indexed by (i,j) -> row index i*m + j
    # V is indexed by (j,k) -> row index j*p + k  
    # W is indexed by (k,i) -> row index k*n + i
    
    # So the decomposition is: for each (i,j,k), we need
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # And for all other combinations, it should be 0.
    
    # Actually, the tensor is defined such that:
    # T[(i,j),(j,k),(k,i)] = 1
    # This means the only non-zero entries are where the middle index matches.
    
    # Let's verify with schoolbook:
    # U[i*m+j, c] = 1 when c = i*m*p + j*p + k (for some k)
    # Wait, in the schoolbook, each (i,j,k) triple gets its own column.
    # U[i*m+j, col] = 1, V[j*p+k, col] = 1, W[k*n+i, col] = 1
    # So for a given (i,j,k), only column c=(i,j,k) has all three non-zero.
    
    # The tensor entry T[a,b,c] where a=(i,j), b=(j,k), c=(k,i)
    # = sum_r U[a,r]*V[b,r]*W[c,r]
    # = 1 if a,b,c form a valid (i,j,k) chain, else 0
    
    # Let's build the target tensor as a 3D array
    # Shape: (n*m, m*p, p*n)
    target = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target[a, b, c] = 1.0
    
    # Now search for low-rank decompositions
    # Start with schoolbook rank 45, try to find lower
    
    best_result = None
    best_rank = n * m * p  # 45
    
    # Try ranks from low to high
    # For 3x3x5, known bounds: rank is between some values
    # Let's try a structured search
    
    def verify_decomposition(U, V, W, target_tensor, tol=1e-9):
        """Verify that U, V, W decompose the target tensor."""
        n, m, p = DIMENSIONS
        rank = U.shape[1]
        # Compute the reconstructed tensor
        # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # Use efficient computation
        # For each r, outer product of U[:,r], V[:,r], W[:,r]
        reconstructed = np.zeros_like(target_tensor)
        for r in range(rank):
            reconstructed += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        return np.allclose(reconstructed, target_tensor, atol=tol)
    
    def try_rank(rank, max_iterations=1000):
        """Try to find a decomposition of given rank using random initialization."""
        n, m, p = DIMENSIONS
        rng = np.random.RandomState(rank * 1000 + 42)
        
        for iteration in range(max_iterations):
            if time.time() > deadline:
                return None
            
            # Random initialization
            U = rng.randn(n * m, rank)
            V = rng.randn(m * p, rank)
            W = rng.randn(p * n, rank)
            
            # Scale to help convergence
            scale = (n * m * m * p * p * n) ** (1.0 / (3 * rank))
            U *= scale
            V *= scale
            W *= scale
            
            # Use alternating minimization / gradient descent
            # This is a non-convex problem, so we'll use a simple approach
            # Try a few steps of coordinate descent or just check if random works
            
            # Actually, for exact decomposition with integer/half-integer entries,
            # random real-valued won't work directly.
            # We need a different approach.
            
            # Let's use a structured approach based on known decompositions
            pass
        
        return None
    
    # For 3x3x5 matrix multiplication, let's try known constructions
    # The Strassen algorithm gives rank 7 for 2x2, but we need 3x3x5
    
    # Let's try a systematic approach: use the fact that we can combine
    # multiple (i,j,k) triples in one rank-1 term if they don't conflict.
    
    # A rank-1 term is defined by three vectors u (length 9), v (length 15), w (length 15)
    # such that for all (a,b,c): u[a]*v[b]*w[c] contributes to T[a,b,c]
    
    # The key insight: if we can find sets of (i,j,k) triples that can be
    # "covered" by a single rank-1 term, we reduce the rank.
    
    # Let's try to find a decomposition by solving a system of equations
    # for small ranks.
    
    # For rank r, we have r*9 + r*15 + r*15 = r*39 variables
    # We have 9*15*15 = 2025 equations (but most are 0=0)
    # The non-zero constraints are: for each (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    # And for all other (a,b,c) not forming a valid chain, the sum must be 0.
    
    # Number of valid (i,j,k) triples: 3*3*5 = 45
    # Total (a,b,c) combinations: 9*15*15 = 2025
    # So we have 45 "1" constraints and 1980 "0" constraints.
    
    # This is a hard nonlinear system. Let's try a different approach.
    
    # Use the known result that for m x n x p matrix multiplication,
    # the rank is at most mn + np + pm - n (for certain cases) or use
    # other known bounds.
    
    # For 3x3x5, let's try rank 27 (which is 3*3*5/1.67, a guess)
    # Actually, let's just return schoolbook for now and try to improve.
    
    # Try a greedy approach: find rank-1 terms that cover multiple (i,j,k) triples
    def find_rank1_terms():
        """Find all possible rank-1 terms that exactly match a subset of the tensor."""
        # A rank-1 term u⊗v⊗w matches entries where u[a]*v[b]*w[c] = T[a,b,c]
        # For this to be exact, we need:
        # - For (a,b,c) in support: u[a]*v[b]*w[c] = 1
        # - For (a,b,c) not in support: u[a]*v[b]*w[c] = 0
        
        # This means u, v, w must have support such that their product is 1
        # exactly on the tensor's support and 0 elsewhere.
        
        # The support of T is: {(i*m+j, j*p+k, k*n+i) : i in 0..2, j in 0..2, k in 0..4}
        # This is a 45-element set.
        
        # A rank-1 term can only match this if the support is a "grid" in some sense.
        # Given the structure, it's unlikely that a single rank-1 term covers more than 1 element.
        
        # So the rank is likely close to 45.
        pass
    
    # Given the difficulty, let's return the schoolbook decomposition
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
    
    # Convert to lists as required
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
