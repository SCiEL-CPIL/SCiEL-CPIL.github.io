"""Matrix multiplication 2x4x8 rank-minimal search."""
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin
    
    n, m, p = 2, 4, 8
    
    # Build the target tensor
    T = np.zeros((n*m, m*p, p*n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1
    
    def verify(U, V, W):
        """Check if U, V, W form a valid decomposition of T."""
        # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
        recon = np.einsum('ac,bc,dc->abd', U, V, W)
        return np.array_equal(recon, T)
    
    # Schoolbook decomposition
    rank = n * m * p
    U = np.zeros((n*m, rank), dtype=np.int64)
    V = np.zeros((m*p, rank), dtype=np.int64)
    W = np.zeros((p*n, rank), dtype=np.int64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank
    
    # Try to find lower rank decompositions
    # Strategy: for each target rank, try random decompositions
    # A random decomposition: pick random vectors u, v, w and check
    # But this is very unlikely to work.
    
    # Better: use the structure. The tensor has a specific pattern.
    # Let's try: for each possible rank r, see if we can find a basis.
    
    # Alternative: use the fact that the problem is small enough for exhaustive search
    # of certain structured decompositions.
    
    # For 2x4x8, let's try the following construction:
    # We can write the multiplication as a sum of rank-1 tensors.
    # Each rank-1 tensor (u, v, w) satisfies:
    #   u[i,j] * v[j,k] * w[k,i] contributes to T[i,j,k]
    
    # Let's try to find a decomposition with rank < 64 by exploiting
    # the small size of n=2.
    
    # Idea: for each j, k, we have two terms (one for i=0, one for i=1).
    # Can we combine them? If we use u[0,j] and u[1,j] appropriately...
    
    # Actually, let's try a different approach: use bilinear forms.
    # For fixed i, the computation is <A[i,:], B[:,k]> for each k.
    # We can use the identity: <a,b> = 1/4 * sum_{s=0}^{3} chi_s(a) * chi_s(b)
    # where chi_s are characters of a group, but this requires complex numbers.
    
    # Given time constraints, let's just return the schoolbook if we can't find better.
    # But first, let's try a few specific constructions.
    
    # Construction 1: use the fact that 2x4x8 can be embedded in a larger problem
    # with known lower rank.
    
    # Actually, let's try: the rank of 2xn x nxm is at most 2*n*m - n
    # (from some results). For n=4, m=8: 2*4*8 - 4 = 60.
    
    # Let's try to construct a rank-60 decomposition.
    # This is non-trivial. Let's just do a random search.
    
    import random
    rng = random.Random(42)
    
    # Try random search for rank 63 down to 50
    for target_rank in range(63, 49, -1):
        if time.time() > deadline:
            break
        
        # Try a few random decompositions
        for attempt in range(20):
            if time.time() > deadline:
                break
            
            # Generate random integer vectors
            U_rand = np.zeros((n*m, target_rank), dtype=np.int64)
            V_rand = np.zeros((m*p, target_rank), dtype=np.int64)
            W_rand = np.zeros((p*n, target_rank), dtype=np.int64)
            
            # Fill with small random integers
            for c in range(target_rank):
                # Each term should be "sparse" to have a chance
                # Pick random non-zero positions
                nz_u = rng.sample(range(n*m), min(4, n*m))
                nz_v = rng.sample(range(m*p), min(6, m*p))
                nz_w = rng.sample(range(p*n), min(4, p*n))
                
                for a in nz_u:
                    U_rand[a, c] = rng.choice([-1, 1])
                for b in nz_v:
                    V_rand[b, c] = rng.choice([-1, 1])
                for d in nz_w:
                    W_rand[d, c] = rng.choice([-1, 1])
            
            if verify(U_rand, V_rand, W_rand):
                best_U, best_V, best_W = U_rand, V_rand, W_rand
                best_rank = target_rank
                break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
