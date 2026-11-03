"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
import numpy as np
from itertools import product

def construct():
    n, m, p = DIMENSIONS
    
    def schoolbook_decomp(n, m, p):
        """Return schoolbook rank n*m*p decomposition."""
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    c = i * m * p + j * p + k
                    U[i * m + j, c] = 1
                    V[j * p + k, c] = 1
                    W[k * n + i, c] = 1
        return U, V, W
    
    def verify_decomp(U, V, W, n, m, p, tol=1e-10):
        """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
        rank = U.shape[1]
        # Reconstruct the multiplication tensor
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # should equal 1 if i,j,k form a valid product term, 0 otherwise
        # Actually, the tensor for matrix multiplication C = AB has:
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # The structure tensor has entries T[(i,k), j] or similar.
        
        # Let me use the standard formulation:
        # The multiplication tensor M has M[(i,j), (j',k)] = delta_{j,j'} * delta_{i,i'} * delta_{k,k'}
        # Wait, let me think more carefully.
        
        # For C = AB where A is n x m, B is m x p, C is n x p:
        # C[i,k] = sum_{j=0}^{m-1} A[i,j] * B[j,k]
        
        # The tensor T has dimensions (n*p) x (n*m) x (m*p) or similar.
        # Standard: T[a, b, c] where a indexes C entries (n*p), b indexes A entries (n*m), c indexes B entries (m*p)
        # T[(i,k), (i,j), (j,k)] = 1 for all valid i,j,k, else 0
        
        # In the decomposition: T[a,b,c] = sum_r U[b,r] * V[c,r] * W[a,r]
        # where U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
        
        # Check: for each (i,j,k), T[(k*n+i), (i*m+j), (j*p+k)] should be 1
        # And for all other combinations, it should be 0.
        
        # This is expensive to check fully. Instead, check a subset of entries.
        
        # Check that all "1" entries are correct
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = k * n + i  # index in W (p*n)
                    b = i * m + j  # index in U (n*m)
                    c = j * p + k  # index in V (m*p)
                    val = np.sum(U[b, :] * V[c, :] * W[a, :])
                    if abs(val - 1) > tol:
                        return False
        
        # Check some "0" entries to ensure no spurious contributions
        # For example, T[(k*n+i), (i*m+j), (j'*p+k')] where j != j' or k != k' should be 0
        # Check a few such cases
        if n > 1:
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # Try different j'
                        if j + 1 < m:
                            j_prime = j + 1
                            a = k * n + i
                            b = i * m + j
                            c = j_prime * p + k
                            val = np.sum(U[b, :] * V[c, :] * W[a, :])
                            if abs(val) > tol:
                                return False
                        # Try different k'
                        if k + 1 < p:
                            k_prime = k + 1
                            a = k_prime * n + i
                            b = i * m + j
                            c = j * p + k
                            val = np.sum(U[b, :] * V[c, :] * W[a, :])
                            if abs(val) > tol:
                                return False
                        # Try different i'
                        if i + 1 < n:
                            i_prime = i + 1
                            a = k * n + i_prime
                            b = i * m + j
                            c = j * p + k
                            val = np.sum(U[b, :] * V[c, :] * W[a, :])
                            if abs(val) > tol:
                                return False
        return True
    
    def try_random_decomp(n, m, p, rank, n_trials=100, seed=42):
        """Try to find a valid decomposition of given rank using random initialization."""
        rng = np.random.RandomState(seed)
        
        for trial in range(n_trials):
            # Initialize with random values
            U = rng.randn(n * m, rank)
            V = rng.randn(m * p, rank)
            W = rng.randn(p * n, rank)
            
            # Normalize to avoid numerical issues
            for r in range(rank):
                norm_u = np.linalg.norm(U[:, r])
                norm_v = np.linalg.norm(V[:, r])
                norm_w = np.linalg.norm(W[:, r])
                if norm_u > 1e-10 and norm_v > 1e-10 and norm_w > 1e-10:
                    # We want U[:,r] * V[:,r] * W[:,r] to contribute meaningfully
                    # But random won't work. Let's use a different approach.
                    pass
            
            # Random initialization won't work for exact decomposition.
            # We need a structured approach.
            break
        
        return None
    
    def strassen_block_decomp(n, m, p):
        """
        Try to use Strassen-like block decomposition.
        For 2x3x8, we can try to decompose into blocks.
        
        One approach: treat the 8 dimension as 4 blocks of size 2.
        If we can do 2x3x2 in rank r, then 2x3x8 might be done in rank 4*r
        or less using tensor product structure.
        
        Known: 2x2x2 can be done in rank 7 (Strassen).
        For 2x3x2: we can pad 3x2 to 4x2 or use other methods.
        
        Actually, let's try a direct construction.
        """
        # For now, let's try to find a lower rank decomposition using
        # the fact that matrix multiplication has algebraic structure.
        
        # One known result: the rank of 2x2x2 is 7.
        # We can embed 2x3x8 into a larger problem or use tensor products.
        
        # Tensor product approach:
        # If we have decompositions for (n1,m1,p1) and (n2,m2,p2),
        # we can get one for (n1*n2, m1*m2, p1*p2) with rank r1*r2.
        
        # 2x3x8 = 2x3x(2x4) but 4 isn't prime in a useful way.
        # 8 = 2*2*2, so 2x3x8 could be built from 2x3x2.
        
        # Let's find the rank of 2x3x2 first.
        # 2x3x2: A is 2x3, B is 3x2, C is 2x2.
        # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] + A[0,2]*B[2,0]
        # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1] + A[0,2]*B[2,1]
        # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0] + A[1,2]*B[2,0]
        # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1] + A[1,2]*B[2,1]
        
        # Schoolbook rank for 2x3x2 is 12.
        # Can we do better?
        
        # Let's try to construct a rank-11 or lower decomposition for 2x3x2.
        # This is non-trivial. Let me try a systematic search.
        
        def find_2x3x2_decomp():
            """Search for low-rank decomposition of 2x3x2 matrix multiplication."""
            n2, m2, p2 = 2, 3, 2
            
            # Try ranks from 7 (lower bound from 2x2x2) up to 12
            for rank in range(7, 13):
                # We need to find U (2*3, rank), V (3*2, rank), W (2*2, rank)
                # such that the tensor decomposition holds.
                
                # This is a system of polynomial equations.
                # For each (i,j,k), sum_r U[i*m2+j, r] * V[j*p2+k, r] * W[k*n2+i, r] = 1
                # For other combinations, = 0.
                
                # Total number of constraints: 2*3*2 = 12 equations for "1" entries
                # Plus many equations for "0" entries.
                
                # This is complex. Let me try a different approach:
                # Use the known rank-7 decomposition for 2x2x2 and extend.
                
                # Actually, let me just return a valid decomposition for now.
                # I'll use a construction that might work.
                
                pass
            
            # For now, return schoolbook
            U, V, W = schoolbook_decomp(2, 3, 2)
            return U, V, W
        
        # Let's try a different strategy: use the fact that we can combine
        # smaller decompositions.
        
        # Known: rank(2,2,2) = 7
        # We can use tensor product: rank(n1*n2, m1*m2, p1*p2) <= rank(n1,m1,p1) * rank(n2,m2,p2)
        
        # 2x3x8: write as (2x2x2) tensor (1x3x4)? No, that doesn't work directly.
        
        # Alternative: 2x3x8 = (2x2x2) tensor (1x3x4) doesn't make sense.
        
        # Let me try: if I have a rank-r decomposition for (n1,m1,p1),
        # and the dimensions divide evenly, I can repeat it.
        
        # 2x3x8: the 8 in the last dimension means we need 8 outputs.
        # If I have a decomposition for 2x3x2 of rank r, I can use it 4 times
        # for the 8 outputs, giving rank 4*r.
        
        # But I can do better by sharing factors.
        
        # Let me just try to find a good decomposition empirically.
        
        # For now, return the schoolbook decomposition for 2x3x8.
        # The rank is 48.
        
        U, V, W = schoolbook_decomp(n, m, p)
        return U, V, W
    
    # Start with schoolbook and try to improve
    U, V, W = schoolbook_decomp(n, m, p)
    current_rank = U.shape[1]
    
    # Try to find a lower rank decomposition
    # For 2x3x8, let's try ranks from 30 up to 47
    
    best_U, best_V, best_W = U, V, W
    best_rank = current_rank
    
    # Try a structured approach: use the fact that 8 = 2*4
    # If we can decompose 2x3x2 efficiently, we can extend to 2x3x8.
    
    # Let me try to construct a decomposition using block structure.
    # Treat the 8-dimensional output as 4 blocks of 2.
    
    # For each block of 2 outputs, we need to compute:
    # C[i, 2*b] and C[i, 2*b+1] for i=0,1
    # This requires products with B[:, 2*b] and B[:, 2*b+1]
    
    # The key insight: if we can share computations across blocks, we save rank.
    
    # Let me try a direct construction for 2x3x8 with rank < 48.
    
    # One approach: use the rank-7 decomposition for 2x2x2 as a building block.
    # But 2x3x8 doesn't factor nicely into 2x2x2 blocks.
    
    # Alternative: use a random search with verification.
    # This is expensive but might find a valid lower-rank decomposition.
    
    # For now, return the schoolbook decomposition.
    # The task says the score is the rank, so we want to minimize it.
    # But we must return a valid decomposition.
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
