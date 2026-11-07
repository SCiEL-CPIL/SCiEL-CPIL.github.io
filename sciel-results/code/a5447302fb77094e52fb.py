import time
import random
import math
from itertools import combinations
from collections import defaultdict

def construct():
    """
    Returns a list of distinct integers in 2..40 such that all 2^k subset products are distinct.
    Maximizes k.
    """
    start_time = time.time()
    deadline = start_time + 50  # Safety margin

    # Candidate pool: 2..40
    candidates = list(range(2, 41))
    
    # Precompute prime factorization for each candidate
    # We need primes up to 40
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    
    # Represent each number as a tuple of exponents for each prime
    # e.g., 12 = 2^2 * 3^1 -> (2, 1, 0, ...)
    # This representation allows us to use vector addition for multiplication
    # and check for collisions in the exponent space.
    
    num_to_vec = {}
    for n in candidates:
        v = [0] * len(primes)
        temp = n
        for i, p in enumerate(primes):
            while temp % p == 0:
                v[i] += 1
                temp //= p
        num_to_vec[n] = tuple(v)
        
    # Helper to add two exponent vectors
    def add_vec(v1, v2):
        return tuple(a + b for a, b in zip(v1, v2))
        
    # Helper to check if a set of numbers has distinct subset products
    # We do this by checking if any two distinct subsets have the same sum of exponent vectors
    # This is equivalent to checking if the set of subset sum vectors is injective.
    # Since the number of subsets is 2^k, and k is small (max ~18), we can enumerate them.
    # However, for forward checking, we need a faster way or a way to prune.
    
    # Let's use a set of seen subset products (as tuples of exponents).
    # When adding a new number, the new subset products are:
    # 1. All old subset products
    # 2. All old subset products multiplied by the new number (i.e., adding the new vector)
    
    # We will maintain the set of current subset product vectors.
    # If we add a new vector v, the new set is S U {s + v for s in S}.
    # We need to ensure that S and {s + v for s in S} are disjoint, and that {s + v for s in S} has no internal collisions (which is guaranteed if S has no internal collisions and v is fixed).
    # So the condition is: for all s1, s2 in S, s1 + v != s2.
    # This is equivalent to: (S + v) is disjoint from S.
    
    # We can represent S as a set of tuples.
    # To check if (S + v) is disjoint from S, we can iterate through S and check if s + v is in S.
    # This takes O(|S|) time per check.
    # |S| is 2^k. For k=18, |S| = 262144. This is feasible for a small number of steps.
    
    # Backtracking search with forward checking
    def backtrack(idx, current_set, current_subset_products, remaining_candidates):
        """
        idx: not really used if we pick from remaining_candidates
        current_set: list of numbers selected so far
        current_subset_products: set of exponent tuples for all subset products of current_set
        remaining_candidates: list of numbers not yet considered
        """
        if time.time() > deadline:
            return None
            
        # If we've considered all candidates, return the current set
        if not remaining_candidates:
            return current_set[:]
            
        # Try to add the next candidate
        # To improve search, we can try candidates in a specific order.
        # A good heuristic is to try numbers with "simpler" factorizations first?
        # Or maybe numbers that are "far" from existing products?
        # Let's stick to a randomized order for diversity, or a fixed order for determinism.
        # Since we want to maximize size, and the space is small, we can try all possibilities?
        # No, 39 candidates is too many for brute force.
        
        # Heuristic: Try to add the candidate if possible. If not, skip it.
        # This is a greedy approach with backtracking.
        # To be a proper CSP solver, we should branch: include or exclude.
        # But excluding a candidate might prevent a larger set later?
        # Actually, if we can add a candidate without collision, adding it is always "safe" in the sense that it doesn't prevent adding others?
        # No, adding a candidate restricts the future candidates that can be added.
        # So we must branch: Try adding, and if that leads to a dead end, try not adding.
        
        # However, a key insight: If a set S is dissociated, and we add x to get S' which is also dissociated, 
        # does S' allow for a larger final set than S? Not necessarily.
        # But if we are looking for the MAXIMUM size, we should try to include as many as possible.
        
        # Let's use a simple recursive backtracking that tries to include the next candidate, 
        # and if that fails (leads to no solution or smaller max), we backtrack.
        # Actually, since we just want ANY valid set of maximum size, and the problem is small,
        # we can use a greedy approach that tries to add, and if it gets stuck, it backtracks.
        
        # Let's define the search as:
        # 1. Try to add candidate[0]. If valid, recurse on candidate[1:].
        # 2. If that path doesn't yield a better result than the best found, or leads to a dead end, 
        #    try skipping candidate[0] and recurse on candidate[1:].
        
        # To make this efficient, we need a good ordering of candidates.
        # Let's sort candidates by their "complexity" or just use a fixed order.
        
        c = remaining_candidates[0]
        rest = remaining_candidates[1:]
        
        # Option 1: Try to include c
        v_c = num_to_vec[c]
        
        # Check if adding c is valid
        valid = True
        # Check if any s + v_c is in current_subset_products
        # We can optimize this by checking if (current_subset_products + v_c) intersects current_subset_products
        # This is equivalent to checking if there exists s1, s2 in S such that s1 + v_c = s2.
        # i.e., s2 - s1 = v_c.
        # We can precompute the set of differences? No, S changes.
        # Just iterate.
        
        # Optimization: If we maintain S as a set, checking s + v_c in S is fast.
        for s in current_subset_products:
            new_s = add_vec(s, v_c)
            if new_s in current_subset_products:
                valid = False
                break
                
        if valid:
            # New subset products
            new_products = set(current_subset_products)
            for s in current_subset_products:
                new_products.add(add_vec(s, v_c))
                
            # Recurse
            res = backtrack(idx + 1, current_set + [c], new_products, rest)
            if res is not None:
                return res
                
        # Option 2: Skip c
        res = backtrack(idx + 1, current_set, current_subset_products, rest)
        if res is not None:
            return res
            
        return None

    # The above backtracking is a depth-first search. It might get stuck in a local optimum 
    # if the ordering is bad. 
    # To improve, we can randomize the order of candidates and run multiple restarts.
    
    best_set = []
    
    for restart in range(50):
        if time.time() > deadline:
            break
            
        # Randomize candidate order
        shuffled_candidates = candidates[:]
        random.shuffle(shuffled_candidates)
        
        # Initial state: empty set, subset products = {zero_vector}
        zero_vec = tuple(0 for _ in primes)
        initial_products = {zero_vec}
        
        res = backtrack(0, [], initial_products, shuffled_candidates)
        
        if res is not None and len(res) > len(best_set):
            best_set = res
            
        # If we found a set of size 18, that's likely optimal or near optimal.
        # Max possible? 
        # 2..40 has 39 numbers.
        # A dissociated set in [1, N] has size O(log N). 
        # For N=40, log2(40) ~ 5.3. But we are in multiplicative group.
        # The size is bounded by the dimension of the exponent space?
        # The exponent space is 12-dimensional.
        # A set of vectors in Z^12 that is dissociated (subset sums distinct) can be large.
        # Actually, if the vectors are linearly independent, they are dissociated.
        # Max linearly independent set in Z^12 is 12.
        # But they don't have to be linearly independent.
        # However, if we have 13 vectors in Z^12, they are linearly dependent.
        # Does linear dependence imply subset sum collision?
        # Not necessarily. e.g., v1 + v2 = v3 + v4 doesn't mean a subset sum collision unless coefficients are 0 or 1.
        # But for large sets, collisions are likely.
        
        # Let's see what size we get.
        
    # Final verification
    if best_set:
        # Verify distinctness
        prods = set()
        for r in range(len(best_set) + 1):
            for combo in combinations(best_set, r):
                prod = 1
                for x in combo:
                    prod *= x
                if prod in prods:
                    # Fallback if verification fails (should not happen)
                    return [2, 4, 16, 3, 9, 5, 25, 7, 11, 13, 17, 19, 23, 29, 31, 37]
                prods.add(prod)
        return [int(x) for x in best_set]
        
    # Fallback
    return [2, 4, 16, 3, 9, 5, 25, 7, 11, 13, 17, 19, 23, 29, 31, 37]

# EVOLVE-BLOCK-END
