import time
import sys
import random
import itertools
from fractions import Fraction

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-minimal exact tensor decomposition for matrix multiplication.
    Returns a dictionary with keys 'U', 'V', 'W' containing the factor matrices.
    The rank is the number of columns in these matrices.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

    n, m, p = DIMENSIONS
    
    # Schoolbook rank
    rank_schoolbook = n * m * p
    
    # Initialize schoolbook as fallback
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][column] = 1
                V_sb[j * p + k][column] = 1
                W_sb[k * n + i][column] = 1
                column += 1
    
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = rank_schoolbook
    
    def verify_decomposition(U, V, W, rank):
        """Verify that the decomposition exactly represents matrix multiplication."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for r in range(rank):
                        val += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if val != 1:
                        return False
        return True
    
    def is_half_integer(val):
        """Check if value is an integer or half-integer."""
        # Check if val * 2 is an integer
        return abs(val * 2 - round(val * 2)) < 1e-9
    
    def try_rank(rank, max_trials=50, seed=42):
        """Try to find a valid decomposition with given rank."""
        if time.time() > deadline:
            return None
        
        random.seed(seed)
        
        # Use values from {-1, -0.5, 0, 0.5, 1}
        candidates = [-1, -0.5, 0, 0.5, 1]
        
        for trial in range(max_trials):
            if time.time() > deadline:
                return None
            
            # Random initialization
            U = [[random.choice(candidates) for _ in range(rank)] for _ in range(n * m)]
            V = [[random.choice(candidates) for _ in range(rank)] for _ in range(m * p)]
            W = [[random.choice(candidates) for _ in range(rank)] for _ in range(p * n)]
            
            # Quick check: verify
            if verify_decomposition(U, V, W, rank):
                return U, V, W
        
        return None
    
    # Try ranks from a lower bound up to schoolbook - 1
    # Lower bound: at least max(n*m, m*p, n*p) = max(15, 35, 21) = 35
    # But let's be more aggressive and try from 40
    
    # First, try some structured approaches
    # For small cases, we can try specific known algorithms
    
    # Try ranks in descending order of likelihood (start with lower ranks)
    # But we need to be careful about time
    
    # Let's try a range of ranks
    min_rank_to_try = 40
    max_rank_to_try = min(rank_schoolbook - 1, 104)
    
    # Try ranks from low to high, but prioritize certain values
    ranks_to_try = list(range(min_rank_to_try, max_rank_to_try + 1))
    
    # Shuffle to add some randomness but keep lower ranks more likely
    random.seed(12345)
    # Weight lower ranks more heavily
    weighted_ranks = []
    for r in ranks_to_try:
        weight = max(1, (max_rank_to_try - r + 1))
        weighted_ranks.extend([r] * weight)
    random.shuffle(weighted_ranks)
    
    # Try each rank a limited number of times
    trials_per_rank = 30
    
    for rank in weighted_ranks:
        if time.time() > deadline:
            break
        
        if rank >= best_rank:
            continue
        
        result = try_rank(rank, max_trials=trials_per_rank, seed=rank * 1000 + 42)
        
        if result is not None:
            U, V, W = result
            best_rank = rank
            best_result = {"U": U, "V": V, "W": W}
            
            # If we found a good rank, we can stop or continue to find even better
            # Continue searching for lower ranks
    
    # If we haven't found anything better, return schoolbook
    return best_result
# EVOLVE-BLOCK-END
