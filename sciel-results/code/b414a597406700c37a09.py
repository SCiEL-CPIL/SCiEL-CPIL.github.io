import math
import time
from itertools import combinations

def is_pairwise_coprime(nums):
    n = len(nums)
    for i in range(n):
        for j in range(i + 1, n):
            if math.gcd(nums[i], nums[j]) != 1:
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    # Precompute: for each number, its prime factors
    def prime_factors(n):
        factors = set()
        d = 2
        while d * d <= n:
            if n % d == 0:
                factors.add(d)
                while n % d == 0:
                    n //= d
            d += 1
        if n > 1:
            factors.add(n)
        return factors
    
    # Precompute prime factors for all numbers 1-256 (max sum is 128+128=256)
    pf_cache = {i: prime_factors(i) for i in range(1, 257)}
    
    # A sum is valid if all distinct sums are pairwise coprime
    # Key insight: two sums s1, s2 share a common factor if they have a common prime factor
    
    # Strategy: enumerate all 3-element subsets for A and B
    # But C(128,3) = 341,376, so C(128,3)^2 is too large
    # Instead: fix A, then for each B, check validity
    # Even C(128,3) * C(128,3) is too large
    
    # Better: use the fact that we want max cardinality (up to 9)
    # For cardinality 9, all 9 sums must be distinct and pairwise coprime
    # For cardinality 8, 8 distinct sums pairwise coprime, etc.
    
    # Structural change: use a smarter search
    # 1. Try to find a 9-cardinality solution by systematic search
    # 2. If not found, fall back to best found
    
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    best_cardinality = 0
    best_sums = set()
    for a in best_A:
        for b in best_B:
            best_sums.add(a + b)
    if is_pairwise_coprime(list(best_sums)):
        best_cardinality = len(best_sums)
    
    # Try systematic search with pruning
    # Key optimization: if any two sums share a prime factor, the pair is invalid
    
    # For each candidate A (3-element subset), try all B
    # But we need to be smarter. Let's use a different approach:
    # Start with small numbers (more likely to be coprime)
    
    # Precompute: for each pair (x, y) where x,y in [2, 256], do they share a factor?
    # Actually, let's just do a focused search
    
    # Phase 1: Try to find 9-cardinality solution
    # Use the fact that pairwise coprime sums means no two sums share any prime factor
    # Small numbers have fewer prime factors, so prefer small sums
    
    # Let's try a different structural approach: 
    # Enumerate all possible A sets (sorted), and for each, find best B
    # Use early termination when time is up
    
    # Actually, let me think about this differently.
    # The maximum possible cardinality is 9 (3x3 grid, all distinct).
    # For 9 distinct sums that are pairwise coprime, we need 9 numbers in [2, 256]
    # that are pairwise coprime.
    
    # Let's precompute: what's the max size of a pairwise coprime subset of [2, 256]?
    # And can we represent them as a+b where a in A, b in B, |A|=|B|=3?
    
    # Simpler approach: just do a smarter random search with local improvement
    # But the directive says to make a structural change...
    
    # New structural approach: use simulated annealing / local search from multiple starts
    # with a different representation: represent the problem as choosing 6 numbers
    # (3 for A, 3 for B) and optimize the objective.
    
    import random
    random.seed(42)
    
    def evaluate(A, B):
        sums = set()
        for a in A:
            for b in B:
                sums.add(a + b)
        if not is_pairwise_coprime(list(sums)):
            return 0
        return len(sums)
    
    # Local search: start from a random valid config, try swapping elements
    def local_search(A, B, max_iter=10000):
        A = list(A)
        B = list(B)
        curr_score = evaluate(A, B)
        best_A = list(A)
        best_B = list(B)
        best_score = curr_score
        
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            
            # Try to improve by replacing one element in A or B
            improved = False
            for pos in range(3):
                # Try replacing A[pos]
                for new_val in range(1, 129):
                    if new_val in A:
                        continue
                    new_A = list(A)
                    new_A[pos] = new_val
                    new_score = evaluate(new_A, B)
                    if new_score > curr_score:
                        A = new_A
                        curr_score = new_score
                        improved = True
                        if curr_score > best_score:
                            best_score = curr_score
                            best_A = list(A)
                            best_B = list(B)
                        if curr_score == 9:
                            return best_A, best_B, best_score
                        break
                if improved:
                    break
                # Try replacing B[pos]
                for new_val in range(1, 129):
                    if new_val in B:
                        continue
                    new_B = list(B)
                    new_B[pos] = new_val
                    new_score = evaluate(A, new_B)
                    if new_score > curr_score:
                        B = new_B
                        curr_score = new_score
                        improved = True
                        if curr_score > best_score:
                            best_score = curr_score
                            best_A = list(A)
                            best_B = list(B)
                        if curr_score == 9:
                            return best_A, best_B, best_score
                        break
                if improved:
                    break
            
            if not improved:
                break
        
        return best_A, best_B, best_score
    
    # Multiple random starts
    for start in range(500):
        if time.time() > deadline:
            break
        
        # Random start with bias toward small numbers (more coprime-friendly)
        A = random.sample(range(1, 50), 3)
        B = random.sample(range(1, 50), 3)
        
        # Ensure distinct
        if len(set(A)) < 3 or len(set(B)) < 3:
            continue
        
        res_A, res_B, res_score = local_search(A, B, max_iter=5000)
        
        if res_score > best_cardinality:
            best_A = res_A
            best_B = res_B
            best_cardinality = res_score
            if res_score == 9:
                return [best_A, best_B]
    
    return [best_A, best_B]
