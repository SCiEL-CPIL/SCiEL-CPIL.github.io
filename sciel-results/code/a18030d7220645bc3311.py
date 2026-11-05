"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random

def construct():
    start_time = time.time()
    DEADLINE = 140  # Safety margin before 150s kill

    def instance(size, seed, limit=10000):
        rng = Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x != 0:
                values.add(x)
        return sorted(values)

    def is_sum_free(S):
        S_set = set(S)
        for a in S:
            for b in S:
                if a + b in S_set:
                    return False
        return True

    def can_add(S_set, x):
        # Check if adding x to S_set keeps it sum-free
        # Need: for all a in S_set, a + x not in S_set, x + a not in S_set (same), and 2*x not in S_set
        for a in S_set:
            if a + x in S_set:
                return False
        if 2 * x in S_set:
            return False
        return True

    def greedy_build(elements, strategy):
        """Build a sum-free subset using a greedy strategy."""
        S = []
        S_set = set()
        for x in elements:
            if can_add(S_set, x):
                S.append(x)
                S_set.add(x)
        return S

    def local_search(S, all_elements, max_iters=500):
        """Try to improve S by swapping elements."""
        S_set = set(S)
        S_list = list(S)
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            # Try adding each element not in S
            for x in all_elements:
                if x in S_set:
                    continue
                if can_add(S_set, x):
                    S_list.append(x)
                    S_set.add(x)
                    improved = True
            # Try removing one element and adding two others
            for i in range(len(S_list)):
                a = S_list[i]
                S_set.remove(a)
                for j in range(len(all_elements)):
                    b = all_elements[j]
                    if b in S_set:
                        continue
                    if can_add(S_set, b):
                        S_set.add(b)
                        # Try adding one more
                        for k in range(len(all_elements)):
                            c = all_elements[k]
                            if c in S_set or c == b:
                                continue
                            if can_add(S_set, c):
                                S_list.append(b)
                                S_list.append(c)
                                S_set.add(c)
                                improved = True
                                break
                        if not improved:
                            S_set.remove(b)
                        else:
                            break
                if not improved:
                    S_set.add(a)
                else:
                    break
        return S_list

    def best_subset(elements):
        """Find the best sum-free subset of elements."""
        # Strategy 1: Odd numbers
        odds = [x for x in elements if x % 2 != 0]
        
        # Strategy 2: Positive numbers (sums tend to be larger)
        positives = [x for x in elements if x > 0]
        
        # Strategy 3: Negative numbers
        negatives = [x for x in elements if x < 0]
        
        # Strategy 4: Numbers with |x| > threshold (large magnitude)
        large = [x for x in elements if abs(x) > 5000]
        
        # Strategy 5: Greedy from sorted by absolute value descending
        sorted_by_abs = sorted(elements, key=lambda x: -abs(x))
        
        # Strategy 6: Greedy from sorted by value descending
        sorted_desc = sorted(elements, reverse=True)
        
        # Strategy 7: Greedy from sorted by value ascending
        sorted_asc = sorted(elements)
        
        # Strategy 8: Numbers congruent to 1 mod 3
        mod1 = [x for x in elements if x % 3 == 1]
        
        # Strategy 9: Numbers congruent to 2 mod 3
        mod2 = [x for x in elements if x % 3 == 2]
        
        # Strategy 10: Even numbers with |x| > threshold
        large_even = [x for x in elements if x % 2 == 0 and abs(x) > 5000]
        
        candidates = [odds, positives, negatives, large, sorted_by_abs, sorted_desc, sorted_asc, mod1, mod2, large_even]
        
        best = []
        for cand in candidates:
            if not cand:
                continue
            # Greedy build from this candidate list
            S = greedy_build(cand, None)
            if len(S) > len(best):
                best = S
        
        # Also try greedy on full element list with different orderings
        for ordering in [elements, sorted_by_abs, sorted_desc, sorted_asc]:
            S = greedy_build(ordering, None)
            if len(S) > len(best):
                best = S
        
        # Local search to improve
        if len(best) > 0:
            best = local_search(best, elements, max_iters=100)
        
        return best

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        if time.time() - start_time > DEADLINE:
            # Return best effort so far
            break
        elements = instance(size, seed)
        subset = best_subset(elements)
        results.append(subset)
    
    # If we ran out of time, fill remaining with odd-number baseline
    while len(results) < 3:
        size, seed = ((60, 801), (120, 802), (240, 803))[len(results)]
        elements = instance(size, seed)
        subset = [x for x in elements if x % 2 != 0]
        results.append(subset)
    
    return results
# EVOLVE-BLOCK-END
