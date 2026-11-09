"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
import numpy as np
import random
import time
from itertools import combinations

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill
    
    # Schoolbook decomposition as fallback (rank = n*m*p = 42)
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    best_rank = rank_sb
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    def is_valid(U, V, W, rank):
        """Check if the decomposition exactly reproduces the matrix multiplication tensor."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1) > 1e-9:
                        return False
        return True
    
    def try_structured_search(rank, max_iters=200):
        """
        Structured search using the fact that matrix multiplication tensors
        have a specific algebraic structure. We try to build decompositions
        where each term corresponds to a "partial product" pattern.
        """
        # For matrix multiplication, a natural structured approach:
        # Each term r can be thought of as contributing to a subset of (i,j,k) triples
        # The schoolbook has one term per (i,j,k). We try to merge terms.
        
        # Strategy: Start with schoolbook and try to find linear dependencies
        # that allow us to express some terms as combinations of others.
        
        # Alternative: Use a basis of "structured" vectors
        # For each term r, define:
        #   U_r is supported on a subset of {0..n*m-1}
        #   V_r is supported on a subset of {0..m*p-1}  
        #   W_r is supported on a subset of {0..p*n-1}
        # Such that the product U_r[i,j] * V_r[j,k] * W_r[k,i] = 1 for exactly the
        # triples (i,j,k) that term r is supposed to cover.
        
        # Let's try a different approach: use the bilinear structure.
        # For each j (the shared dimension), we can decompose the n*p submatrix.
        
        # Try random structured search with more diverse value sets
        value_sets = [
            [0, 1, -1],
            [0, 1, -1, 2, -2],
            [0, 0.5, 1, -0.5, -1, 2, -2],
            [0, 1, -1, 1.5, -1.5],
            [0, 1, -1, 0.5, -0.5, 1.5, -1.5, 2, -2],
        ]
        
        for vs_idx, vs in enumerate(value_sets):
            for _ in range(max_iters):
                if time.time() > deadline:
                    return None
                
                # Generate U, V, W with structure
                U = [[0] * rank for _ in range(n*m)]
                V = [[0] * rank for _ in range(m*p)]
                W = [[0] * rank for _ in range(p*n)]
                
                # For each rank term, assign random support patterns
                for r in range(rank):
                    # U_r: pick a random subset of (i,j) pairs to be nonzero
                    for i in range(n):
                        for j in range(m):
                            if random.random() < 0.6:
                                U[i*m+j][r] = random.choice(vs)
                    
                    # V_r: pick a random subset of (j,k) pairs
                    for j in range(m):
                        for k in range(p):
                            if random.random() < 0.6:
                                V[j*p+k][r] = random.choice(vs)
                    
                    # W_r: pick a random subset of (k,i) pairs
                    for k in range(p):
                        for i in range(n):
                            if random.random() < 0.6:
                                W[k*n+i][r] = random.choice(vs)
                
                if is_valid(U, V, W, rank):
                    return U, V, W
        
        return None
    
    # Try to find better decompositions starting from lower ranks
    # The theoretical lower bound is max(n*m, m*p, n*p) = max(6, 21, 14) = 21
    lower_bound = max(n*m, m*p, n*p)
    
    # First, try a more systematic approach for ranks near the lower bound
    # Use the structure: for each j, the n*p matrix of partial products must be rank <= 1
    # per term, so we need at least max(n*p, ...) terms... 
    
    # Try structured search for ranks from lower_bound to best_rank-1
    for rank in range(lower_bound, best_rank):
        if time.time() > deadline:
            break
        
        # Use more iterations for lower ranks (harder to find)
        max_iters = max(100, 500 - (rank - lower_bound) * 20)
        result = try_structured_search(rank, max_iters=max_iters)
        if result is not None:
            best_rank = rank
            best_U, best_V, best_W = result
            break
    
    # If we found a better rank, try to improve further
    if best_rank < rank_sb:
        for rank in range(max(lower_bound, best_rank - 5), best_rank):
            if time.time() > deadline:
                break
            result = try_structured_search(rank, max_iters=200)
            if result is not None:
                best_rank = rank
                best_U, best_V, best_W = result
                break
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
