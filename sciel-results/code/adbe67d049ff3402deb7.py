"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x != 0:
                values.add(x)
        return sorted(values)
    
    def is_sum_free(subset):
        s = set(subset)
        for a in subset:
            for b in subset:
                if a + b in s:
                    return False
        return True
    
    def greedy_sum_free(numbers):
        """Greedy: pick elements that don't conflict with already picked ones."""
        picked = []
        picked_set = set()
        # Sort by absolute value descending - large numbers are less likely to be sums
        for x in sorted(numbers, key=lambda v: abs(v), reverse=True):
            # Check if adding x would violate sum-freeness
            conflict = False
            for a in picked:
                if a + x in picked_set or x + a in picked_set:
                    conflict = True
                    break
                if a + x == x:  # a = 0, but 0 not in set
                    conflict = True
                    break
            if not conflict:
                picked.append(x)
                picked_set.add(x)
        return picked
    
    def parity_based(numbers):
        """All odds or all evens are sum-free (sum of two odds is even, sum of two evens is even)."""
        odds = [x for x in numbers if x % 2 != 0]
        evens = [x for x in numbers if x % 2 == 0]
        # Both are sum-free, pick the larger
        if len(odds) >= len(evens):
            return odds
        else:
            return evens
    
    def sign_based(numbers):
        """All positives or all negatives: sum of two positives is positive, sum of two negatives is negative.
        But we need to check: if we have all positives, a+b=c where all are positive is possible.
        Actually, all positives is NOT necessarily sum-free. Let me reconsider.
        
        Better: take numbers from a range where sums fall outside the range.
        E.g., if all numbers are in [L, R] with R < L (impossible) or use modular approach.
        
        Alternative: take the largest half by absolute value - their sums tend to be larger.
        """
        # Take numbers with largest absolute values - their sums are likely outside the set
        sorted_by_abs = sorted(numbers, key=lambda v: abs(v), reverse=True)
        half = len(sorted_by_abs) // 2
        candidate = sorted_by_abs[:half]
        # Verify it's sum-free
        if is_sum_free(candidate):
            return candidate
        # Try the other half
        candidate2 = sorted_by_abs[half:]
        if is_sum_free(candidate2):
            return candidate2
        return candidate  # Return best effort
    
    def local_search(subset, all_numbers, max_iters=5000):
        """Improve subset by swapping elements to increase size while maintaining sum-freeness."""
        subset = list(subset)
        subset_set = set(subset)
        current_size = len(subset)
        
        for _ in range(max_iters):
            improved = False
            # Try adding an element
            for x in all_numbers:
                if x in subset_set:
                    continue
                # Check if adding x keeps sum-free
                valid = True
                for a in subset:
                    if a + x in subset_set or a + x == x:
                        valid = False
                        break
                if valid:
                    subset.append(x)
                    subset_set.add(x)
                    current_size += 1
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try swapping: remove one, add another
            for i, a in enumerate(subset):
                for x in all_numbers:
                    if x in subset_set:
                        continue
                    # Remove a, add x
                    new_set = subset_set - {a} | {x}
                    # Check sum-free for new_set
                    valid = True
                    for b in new_set:
                        if b + x in new_set and b != x:
                            # b + x = some element in new_set
                            if b + x in new_set:
                                valid = False
                                break
                    if valid:
                        subset[i] = x
                        subset_set = set(subset)
                        improved = True
                        break
                if improved:
                    break
            
            if not improved:
                break
        
        return subset
    
    instances = [instance(size, seed) for size, seed in ((60, 801), (120, 802), (240, 803))]
    
    results = []
    for numbers in instances:
        # Generate candidates from multiple strategies
        candidates = []
        
        # Parity-based
        candidates.append(parity_based(numbers))
        
        # Greedy by absolute value
        candidates.append(greedy_sum_free(numbers))
        
        # Sign/interval based
        candidates.append(sign_based(numbers))
        
        # All odds
        odds = [x for x in numbers if x % 2 != 0]
        candidates.append(odds)
        
        # All evens  
        evens = [x for x in numbers if x % 2 == 0]
        candidates.append(evens)
        
        # Positive numbers
        positives = [x for x in numbers if x > 0]
        if is_sum_free(positives):
            candidates.append(positives)
        
        # Negative numbers
        negatives = [x for x in numbers if x < 0]
        if is_sum_free(negatives):
            candidates.append(negatives)
        
        # Large absolute values
        sorted_abs = sorted(numbers, key=lambda v: abs(v), reverse=True)
        for frac in [0.5, 0.6, 0.7]:
            cand = sorted_abs[:int(len(sorted_abs) * frac)]
            if is_sum_free(cand):
                candidates.append(cand)
        
        # Pick the best candidate
        best = max(candidates, key=len)
        
        # Local search to improve
        best = local_search(best, numbers, max_iters=3000)
        
        # Verify
        if not is_sum_free(best):
            # Fallback to parity
            best = parity_based(numbers)
        
        results.append(best)
        
        if time.time() > deadline:
            break
    
    # If we ran out of time, fill remaining with parity-based
    while len(results) < 3:
        idx = len(results)
        numbers = instances[idx]
        results.append(parity_based(numbers))
    
    return results
# EVOLVE-BLOCK-END
