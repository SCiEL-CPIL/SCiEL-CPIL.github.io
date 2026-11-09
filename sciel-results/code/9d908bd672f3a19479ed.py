"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from math import gcd
from itertools import combinations
import random

def construct():
    """
    Search for A, B with 3 distinct integers in [1,128] maximizing distinct pairwise-coprime sums.
    Strategy:
    1. Constrain A to be all odd and B to be all even (or vice versa) to avoid multiple even sums which violate pairwise coprimality.
    2. Use randomized search with local improvements in this restricted space.
    3. Objective: Maximize number of distinct sums that are pairwise coprime.
    """
    
    # Precompute gcd for fast lookup
    # Max sum is 128+128=256
    max_val = 256
    gcd_table = [[0] * (max_val + 1) for _ in range(max_val + 1)]
    for i in range(1, max_val + 1):
        for j in range(i, max_val + 1):
            g = gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g

    def is_pairwise_coprime(nums):
        """Check if all distinct values in nums are pairwise coprime."""
        distinct = list(set(nums))
        n = len(distinct)
        for i in range(n):
            for j in range(i + 1, n):
                if gcd_table[distinct[i]][distinct[j]] != 1:
                    return False
        return True

    def get_score(A, B):
        """
        Calculate score: number of distinct sums if they are pairwise coprime, else 0.
        """
        sums = [a + b for a in A for b in B]
        distinct_sums = list(set(sums))
        if is_pairwise_coprime(distinct_sums):
            return len(distinct_sums)
        return 0

    # Restricted search space
    odds = list(range(1, 129, 2))  # 1, 3, ..., 127
    evens = list(range(2, 129, 2))  # 2, 4, ..., 128
    
    # Initial best
    best_score = 0
    best_A = None
    best_B = None
    
    random.seed(42)
    
    # Helper to get a random valid set of 3 distinct numbers from a list
    def random_set(pool):
        return random.sample(pool, 3)
    
    # 1. Random Initialization
    # Try a few random starts
    for _ in range(100):
        A = random_set(odds)
        B = random_set(evens)
        score = get_score(A, B)
        if score > best_score:
            best_score = score
            best_A = A[:]
            best_B = B[:]
            
        # Also try swapped parity (A even, B odd) - symmetric
        A2 = random_set(evens)
        B2 = random_set(odds)
        score2 = get_score(A2, B2)
        if score2 > best_score:
            best_score = score2
            best_A = A2[:]
            best_B = B2[:]

    # 2. Local Search / Hill Climbing
    # We will try to improve the best found solution
    # We need to handle both parity cases. 
    # Let's just store the best A, B and determine their parity types dynamically or just try both.
    
    # To simplify, let's just run a local search on the specific best_A, best_B found.
    # We need to know if best_A is the "odd" set or "even" set.
    # We can just try perturbing elements and checking validity.
    # But we must maintain the parity constraint to avoid the "multiple even sums" trap.
    # If best_A is all odd, we must keep it all odd.
    
    if best_A is None:
        # Fallback if no valid solution found in random init (unlikely)
        best_A = [1, 3, 5]
        best_B = [2, 4, 6]
        best_score = get_score(best_A, best_B)
        
    # Determine parity of best_A
    # If all elements in best_A are odd, we search in odds for A and evens for B.
    # If all elements in best_A are even, we search in evens for A and odds for B.
    # Mixed parity is invalid for high scores, so we assume one of these.
    
    if all(x % 2 != 0 for x in best_A):
        pool_A = odds
        pool_B = evens
    else:
        pool_A = evens
        pool_B = odds
        
    # Local search
    for _ in range(20000):
        # Pick a random element to perturb
        if random.random() < 0.5:
            A = best_A[:]
            idx = random.randint(0, 2)
            # Replace with a random value from the pool, ensuring distinctness
            candidates = [x for x in pool_A if x != A[idx]]
            # Ensure we don't pick a value already in A (other than idx)
            others = set(A) - {A[idx]}
            candidates = [x for x in candidates if x not in others]
            if not candidates:
                continue
            A[idx] = random.choice(candidates)
            B = best_B
        else:
            A = best_A
            B = best_B[:]
            idx = random.randint(0, 2)
            candidates = [x for x in pool_B if x != B[idx]]
            others = set(B) - {B[idx]}
            candidates = [x for x in candidates if x not in others]
            if not candidates:
                continue
            B[idx] = random.choice(candidates)
            
        score = get_score(A, B)
        if score > best_score:
            best_score = score
            best_A = A[:]
            best_B = B[:]
            
            # If we reached 9, it's the maximum possible distinct sums. 
            # We can stop early if we want, but maybe there's a better configuration? 
            # 9 is the max distinct sums. So we can break.
            if best_score == 9:
                break

    return [best_A, best_B]
# EVOLVE-BLOCK-END
