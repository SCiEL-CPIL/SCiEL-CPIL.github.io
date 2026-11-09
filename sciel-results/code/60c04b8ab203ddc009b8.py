"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from math import gcd
from itertools import combinations, product

def construct():
    """Search for A, B with 3 distinct integers in [1,128] maximizing distinct pairwise-coprime sums."""
    
    best_count = 0
    best_A = None
    best_B = None
    
    # We'll search over a reasonable subset to stay within time limits
    # Focus on smaller numbers where coprime sums are more likely
    
    # Try all combinations of 3 distinct values from 1 to 128 for A and B
    # This is C(128,3)^2 which is too large, so we need to be smarter
    
    # Strategy: iterate over possible A sets, then for each, try B sets
    # But even C(128,3) = 341,920, and squaring that is too much
    
    # Better: use a heuristic search or limit the range
    
    # Let's try a more targeted approach:
    # 1. Generate candidate sets
    # 2. For each pair (A, B), compute distinct sums and check pairwise coprimality
    
    # To keep it fast, let's limit to smaller ranges first, then expand if needed
    
    # First, try with numbers up to 50
    limit = 50
    candidates = list(range(1, limit + 1))
    
    # Precompute all 3-element combinations
    comb3 = list(combinations(candidates, 3))
    
    # This is still C(50,3) = 19,600 combinations per set
    # 19,600^2 = ~384 million pairs - too many
    
    # Need a better approach. Let's use a greedy/heuristic search.
    
    # Alternative: try specific patterns that are known to work well
    
    # Pattern 1: Use numbers that are spaced out to create many distinct sums
    # Pattern 2: Use primes or numbers with few common factors
    
    # Let me try a randomized search with local improvement
    
    import random
    
    def check_pairwise_coprime(nums):
        """Check if all distinct values in nums are pairwise coprime."""
        distinct = list(set(nums))
        n = len(distinct)
        for i in range(n):
            for j in range(i + 1, n):
                if gcd(distinct[i], distinct[j]) != 1:
                    return False
        return True
    
    def get_distinct_sum_count(A, B):
        """Get count of distinct sums that are pairwise coprime."""
        sums = [a + b for a in A for b in B]
        distinct_sums = list(set(sums))
        if check_pairwise_coprime(distinct_sums):
            return len(distinct_sums)
        return 0
    
    # Start with the known good solution
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    best_count = get_distinct_sum_count(best_A, best_B)
    
    # Try random perturbations
    random.seed(42)
    for _ in range(50000):
        # Start from best and perturb one element
        if random.random() < 0.5:
            A = best_A[:]
            idx = random.randint(0, 2)
            A[idx] = random.randint(1, 128)
            B = best_B
        else:
            A = best_A
            B = best_B[:]
            idx = random.randint(0, 2)
            B[idx] = random.randint(1, 128)
        
        # Ensure distinct elements
        if len(set(A)) != 3 or len(set(B)) != 3:
            continue
        if min(A) < 1 or max(A) > 128 or min(B) < 1 or max(B) > 128:
            continue
        
        count = get_distinct_sum_count(A, B)
        if count > best_count:
            best_count = count
            best_A = A[:]
            best_B = B[:]
    
    return [best_A, best_B]
# EVOLVE-BLOCK-END
