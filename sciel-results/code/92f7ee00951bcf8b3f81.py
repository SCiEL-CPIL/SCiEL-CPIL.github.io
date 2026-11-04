"""Matrix multiplication 4x5x5 rank search."""
import time
import random
import itertools

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Build the target tensor: T[i,j,k] = 1 for the multiplication tensor
    # We need: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all (i,j,k)
    
    # First, try a structured algebraic approach:
    # Embed 4x5x5 into a larger structure.
    # Try rank values from low to high.
    
    def verify(U, V, W, r):
        """Check if decomposition is exact."""
        # U: (n*m, r), V: (m*p, r), W: (p*n, r)
        # For each (i,j,k): sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] should be 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(r):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if s != 1:
                        return False
        return True
    
    def verify_fast(U, V, W, r):
        """Faster verification using matrix multiplication structure."""
        # For fixed j, we need: for all i,k: sum_c U[i*m+j,c]*W[k*n+i,c] = V[j*p+k,c]... 
        # Actually let's just do it directly but vectorized
        for j in range(m):
            for k in range(p):
                # Need: for all i: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
                # This is a bilinear form
                for i in range(n):
                    s = 0
                    for c in range(r):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if s != 1:
                        return False
        return True
    
    # Strategy 1: Try known low-rank constructions
    # For 4x5x5, let's try to find a decomposition with rank < 100
    
    # Strategy: Use the fact that we can pad 4x5x5 to 5x5x5 or 6x6x6
    # 5x5x5 has known rank at most 49 (or lower with advanced algorithms)
    # But let's try direct search
    
    # Try ranks from a reasonable lower bound up
    # Lower bound: max(n*m, m*p, p*n) = max(20, 25, 20) = 25
    # But actually the tensor rank lower bound is max(n*m, m*p, p*n) for this type
    
    # Let's try a structured search with small integer entries
    # Use a greedy/iterative approach
    
    best_result = None
    best_rank = rank_schoolbook
    
    # Start with schoolbook as fallback
    def schoolbook():
        r = n * m * p
        U = [[0] * r for _ in range(n * m)]
        V = [[0] * r for _ in range(m * p)]
        W = [[0] * r for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return U, V, W, r
    
    U_sb, V_sb, W_sb, r_sb = schoolbook()
    best_result = (U_sb, V_sb, W_sb, r_sb)
    
    # Try to find better decompositions
    # Use a structured approach: fix some structure and search
    
    # Approach: For a given rank r, we need to find U, V, W with small integer entries
    # such that the tensor equation holds.
    
    # Let's try ranks from 25 upward, but with time limit
    # For each rank, use a randomized search with local improvement
    
    def search_rank(r, max_iters=5000):
        """Try to find a decomposition of rank r."""
        # Initialize with random small integers
        for it in range(max_iters):
            if time.time() > deadline - 5:
                return None
            
            # Random initialization with entries in {-1, 0, 1}
            U = [[random.randint(-1, 1) for _ in range(r)] for _ in range(n*m)]
            V = [[random.randint(-1, 1) for _ in range(r)] for _ in range(m*p)]
            W = [[random.randint(-1, 1) for _ in range(r)] for _ in range(p*n)]
            
            # Check if it works
            if verify(U, V, W, r):
                return U, V, W, r
            
            # Local improvement: try to fix one coordinate at a time
            # This is complex, so let's just try more random inits
            if it % 100 == 0 and time.time() > deadline - 10:
                return None
        return None
    
    # Try structured approaches first
    # For 4x5x5, try rank values that are multiples or have special structure
    
    # Try rank = n*m*p - some savings
    # Known: for square n x n x n, best known is ~n^2.5 for large n
    # For 5x5x5, rank is at most 49 (using Coppersmith-Winograd type)
    # For 4x5x5, let's try ranks from 30 to 99
    
    # Actually, let's be smarter. Use the embedding approach:
    # Pad 4x5x5 to 5x5x5 by adding a dummy row/col with zeros
    # Then use a 5x5x5 decomposition and project
    
    # For now, try direct search with increasing rank
    # Start from a reasonable estimate
    
    # Let's try a few specific ranks with more structured search
    candidate_ranks = list(range(30, min(100, r_sb)))
    # Sort by likelihood of success (lower ranks first, but skip if too low)
    
    for r in candidate_ranks:
        if time.time() > deadline - 15:
            break
        result = search_rank(r, max_iters=2000)
        if result is not None:
            best_result = result
            best_rank = r
            break
    
    # If we found something, return it
    if best_result is not None:
        U, V, W, r = best_result
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    return {"U": U_sb, "V": V_sb, "W": W_sb}
# EVOLVE-BLOCK-END
