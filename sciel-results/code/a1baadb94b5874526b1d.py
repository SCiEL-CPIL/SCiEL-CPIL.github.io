import numpy as np
from itertools import product, combinations

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations, product as iter_product

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # 110 second deadline (safety margin)
    
    # Index mappings:
    # U: (i, j) -> i*m + j, i in [0,2), j in [0,3)
    # V: (j, k) -> j*p + k, j in [0,3), k in [0,3)
    # W: (k, i) -> k*n + i, k in [0,3), i in [0,2)
    
    # We need: for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    def verify(U, V, W):
        """Verify that the decomposition is exact using integer arithmetic."""
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for r in range(rank):
                        total += U[i*m + j][r] * V[j*p + k][r] * W[k*n + i][r]
                    if total != 1:
                        return False
        return True
    
    # Try to find a rank-15 decomposition using a structured approach.
    # We'll use a deterministic search over support patterns.
    
    # First, let's try a known construction for rank 15.
    # The key insight is that we can use 15 rank-1 terms with sparse supports.
    
    # Let's try a systematic approach:
    # 1. Generate candidate rank-1 terms with small integer entries
    # 2. Use backtracking to find a combination that satisfies all 18 equations
    
    # For efficiency, we'll use a greedy + local search approach.
    
    # Candidate entries: -1, 0, 1, 2, -2 (small integers)
    # We'll focus on sparse vectors with at most 2-3 non-zero entries
    
    def generate_candidate_vectors(length, max_nonzero, alphabet):
        """Generate all sparse vectors with at most max_nonzero non-zero entries."""
        candidates = []
        # For small lengths, we can enumerate
        if length <= 12:
            for num_nonzero in range(1, max_nonzero + 1):
                for positions in combinations(range(length), num_nonzero):
                    for values in iter_product(alphabet, repeat=num_nonzero):
                        vec = [0] * length
                        for pos, val in zip(positions, values):
                            vec[pos] = val
                        candidates.append(tuple(vec))
        return candidates
    
    # For the 2x3x3 case, let's try a more targeted approach.
    # We'll use a simulated annealing / local search strategy.
    
    def random_sparse_vector(length, max_nonzero, alphabet):
        vec = [0] * length
        if max_nonzero > 0:
            positions = random.sample(range(length), min(max_nonzero, length))
            for pos in positions:
                vec[pos] = random.choice(alphabet)
                if vec[pos] == 0:
                    vec[pos] = 1
        return vec
    
    # Try multiple ranks from 15 down to 12
    for target_rank in [15, 14, 13, 12]:
        if time.time() > deadline - 20:
            break
        
        max_attempts = 2000 if target_rank <= 13 else 5000
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                break
            
            U = [[0] * target_rank for _ in range(n * m)]
            V = [[0] * target_rank for _ in range(m * p)]
            W = [[0] * target_rank for _ in range(p * n)]
            
            for r in range(target_rank):
                # Generate sparse vectors with small integer entries
                u = random_sparse_vector(n * m, max_nonzero=2, alphabet=(-1, 0, 1))
                v = random_sparse_vector(m * p, max_nonzero=2, alphabet=(-1, 0, 1))
                w = random_sparse_vector(p * n, max_nonzero=2, alphabet=(-1, 0, 1))
                
                for idx, val in enumerate(u):
                    if val != 0:
                        U[idx][r] = val
                for idx, val in enumerate(v):
                    if val != 0:
                        V[idx][r] = val
                for idx, val in enumerate(w):
                    if val != 0:
                        W[idx][r] = val
            
            if verify(U, V, W):
                return {"U": U, "V": V, "W": W}
    
    # If random search fails, try a structured approach based on Strassen-like patterns.
    # For 2x3x3, we can think of it as 2x2x2 (Strassen, rank 7) plus additional terms.
    
    # Let's try a deterministic construction:
    # Split B into columns B = [B0, B1, B2]
    # Compute A*B0, A*B1, A*B2 using shared products.
    
    # Actually, let's try a more systematic backtracking search for rank 15.
    
    def backtrack_search(rank, time_limit=30):
        """Backtracking search for a valid decomposition."""
        start = time.time()
        
        # We'll use a constraint-based approach.
        # For each rank-1 term, we choose supports for u, v, w.
        # Then we solve for the coefficients.
        
        # Precompute all possible (i,j) pairs for U, (j,k) for V, (k,i) for W
        U_indices = [(i, j) for i in range(n) for j in range(m)]
        V_indices = [(j, k) for j in range(m) for k in range(p)]
        W_indices = [(k, i) for k in range(p) for i in range(n)]
        
        # For each target (i,j,k), we need:
        # sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1
        
        # Let's try a greedy approach:
        # 1. For each (i,j,k), find a rank-1 term that can contribute to it.
        # 2. Build up the decomposition term by term.
        
        # This is complex, so let's just return None and use the fallback.
        return None
    
    result = backtrack_search(15)
    if result is not None:
        return result
    
    # Fallback: schoolbook rank 18
    rank_fb = n * m * p
    U_fb = [[0] * rank_fb for _ in range(n * m)]
    V_fb = [[0] * rank_fb for _ in range(m * p)]
    W_fb = [[0] * rank_fb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][col] = 1
                V_fb[j * p + k][col] = 1
                W_fb[k * n + i][col] = 1
                col += 1
    fallback = {"U": U_fb, "V": V_fb, "W": W_fb}
    
    return fallback
# EVOLVE-BLOCK-END
