"""Program for integer sum-product extremal sets using local search."""
import time
import random
import math

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 75.0
    best_set = None
    best_score = float('inf')
    
    def evaluate(A):
        """Compute max(|A+A|, |A*A|) for set A."""
        A = sorted(A)
        n = len(A)
        # Sumset A+A (with equal summands allowed)
        sumset = set()
        for i in range(n):
            for j in range(i, n):
                sumset.add(A[i] + A[j])
        # Product set A*A (with equal factors allowed)
        prodset = set()
        for i in range(n):
            for j in range(i, n):
                prodset.add(A[i] * A[j])
        return max(len(sumset), len(prodset))

    def fast_evaluate(A):
        """
        Approximate evaluation for speed during local search.
        We use the fact that for sorted A, the sumset size is roughly bounded by 
        the range and structure, but exact calculation is O(n^2). 
        To be safe, we still do exact O(n^2) but with early exit if possible? 
        No, n=20 is small. 210 pairs. 210*210 is fast.
        Let's just use evaluate() but optimize the set creation slightly.
        """
        A = sorted(A)
        n = len(A)
        sums = set()
        prods = set()
        for i in range(n):
            ai = A[i]
            for j in range(i, n):
                sums.add(ai + A[j])
                prods.add(ai * A[j])
        return max(len(sums), len(prods))

    # Initialize with a geometric-like sequence or random set
    # A good starting point is often a geometric progression with small ratio
    # e.g., 1, 2, 4, 8... but capped at 10000.
    # 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 (14 elements)
    # We need 20.
    
    # Strategy: Start with a set that has small sumset and small product set.
    # Geometric progressions minimize sumset size for fixed max element? 
    # Actually, for fixed max element, arithmetic progressions maximize sumset? No.
    # Small sumset: Geometric progression.
    # Small product set: Geometric progression (powers of 2).
    # So a pure geometric progression is a strong candidate.
    
    # Let's try to construct a set of 20 integers.
    # If we use powers of 2: 2^0 to 2^19 is huge (524288 > 10000).
    # So we can't use pure powers of 2 for 20 elements.
    # We need to compress the range.
    
    # Alternative: Start with a random set of 20 distinct integers <= 10000.
    # Or start with a structured set.
    
    # Let's try a set of the form:
    # Small integers: 1, 2, 3, ...
    # And some larger integers to fill the count?
    # Actually, mixing small and large creates huge products.
    # Best to keep the set "dense" in log space?
    
    # Let's use a local search starting from a geometric progression that fits in 10000.
    # Max ratio r such that 1 * r^19 <= 10000 => r^19 <= 10000 => r <= 10000^(1/19) ~ 1.26.
    # Integer ratio must be at least 2 for distinct integers if we start at 1.
    # So pure integer geometric progression with integer ratio >= 2 won't fit 20 elements.
    
    # What about non-integer ratio? The elements must be integers.
    # Let's try a set of powers of 2 up to a limit, then fill with other numbers?
    # Powers of 2: 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 (14 elements)
    # We need 6 more.
    # If we add numbers close to the existing ones, we increase sumset and product set.
    # If we add numbers far away, we increase product set significantly.
    
    # Let's try a different structure:
    # A = {a_i} where a_i are chosen to minimize max(|A+A|, |A*A|).
    # Known result: For sets of integers, |A+A| >= 2|A|-1. |A*A| >= |A|.
    # Lower bound for max is around 2n.
    # For n=20, lower bound is 39.
    
    # Let's use simulated annealing or hill climbing.
    
    # Initial set: Random
    random.seed(42)
    current_set = list(random.sample(range(1, 10001), 20))
    current_score = fast_evaluate(current_set)
    
    if current_score < best_score:
        best_score = current_score
        best_set = current_set.copy()
        
    # Also try a geometric-like start
    geo_set = []
    val = 1
    while len(geo_set) < 20 and val <= 10000:
        geo_set.append(val)
        val = int(val * 1.5) # Try ratio 1.5, round to int
        if val in geo_set:
            val += 1
        if val > 10000:
            break
    # Fill remaining with random large numbers? Or small?
    while len(geo_set) < 20:
        # Add numbers that are not in set
        new_val = random.randint(1, 10000)
        if new_val not in geo_set:
            geo_set.append(new_val)
    # Ensure distinct
    geo_set = list(set(geo_set))
    if len(geo_set) < 20:
        # Fallback to random
        geo_set = list(random.sample(range(1, 10001), 20))
        
    geo_score = fast_evaluate(geo_set)
    if geo_score < best_score:
        best_score = geo_score
        best_set = geo_set.copy()
        
    # Local Search
    steps = 0
    max_steps = 5000
    while time.time() < deadline and steps < max_steps:
        steps += 1
        # Pick a random element to change
        idx = random.randint(0, 19)
        old_val = current_set[idx]
        
        # Propose a new value
        # Strategy: Try values close to current, or random
        if random.random() < 0.5:
            # Local move
            delta = random.randint(-100, 100)
            new_val = old_val + delta
        else:
            # Random jump
            new_val = random.randint(1, 10000)
            
        if new_val < 1 or new_val > 10000:
            continue
            
        # Check distinctness
        if new_val in current_set and new_val != old_val:
            continue
            
        # Apply change
        current_set[idx] = new_val
        new_score = fast_evaluate(current_set)
        
        if new_score < current_score:
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_set = current_set.copy()
        else:
            # Revert
            current_set[idx] = old_val
            
    return best_set
# EVOLVE-BLOCK-END
