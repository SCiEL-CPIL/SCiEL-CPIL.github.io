# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations_with_replacement

def construct():
    t_start = time.time()
    DEADLINE = t_start + 75.0  # 15s safety margin before 90s kill
    
    N = 4096
    
    def check_can_add(selected, triple_sums, x):
        """Check if x can be added to selected without creating duplicate triple sums."""
        n = len(selected)
        # New triples involving x:
        # (x, x, x)
        s = 3 * x
        if s in triple_sums:
            return False
        # (x, x, y) for y in selected
        for y in selected:
            s = 2 * x + y
            if s in triple_sums:
                return False
        # (x, y, z) for y <= z in selected
        for i in range(n):
            for j in range(i, n):
                s = x + selected[i] + selected[j]
                if s in triple_sums:
                    return False
        return True
    
    def add_to_sets(selected, triple_sums, x):
        """Add x to selected and update triple_sums."""
        n = len(selected)
        # (x, x, x)
        triple_sums.add(3 * x)
        # (x, x, y) for y in selected
        for y in selected:
            triple_sums.add(2 * x + y)
        # (x, y, z) for y <= z in selected
        for i in range(n):
            for j in range(i, n):
                triple_sums.add(x + selected[i] + selected[j])
        selected.append(x)
    
    def greedy_build(order):
        """Greedy build using a given order of candidates."""
        selected = []
        triple_sums = set()
        
        for x in order:
            if time.time() > DEADLINE - 20:
                break
            if check_can_add(selected, triple_sums, x):
                add_to_sets(selected, triple_sums, x)
        
        return selected
    
    # Strategy 1: Standard greedy (ascending order)
    best = greedy_build(range(1, N + 1))
    best_len = len(best)
    
    if time.time() > DEADLINE - 40:
        return best
    
    # Strategy 2: Try different starting points / orderings
    # Use a heuristic: try skipping some small numbers to allow larger ones
    # Or try random permutations of subsets
    
    import random
    random.seed(42)
    
    # Try greedy with random perturbations of the order
    for trial in range(50):
        if time.time() > DEADLINE - 20:
            break
        
        # Create a modified order: keep ascending but with some random skips
        # Or try a different strategy: start with larger gaps
        
        # Try: ascending but skip every k-th number for small k
        # Or: try to maximize by using a "power" based selection
        
        # Let's try: greedy but with a random subset of candidates
        # Pick a random subset of size ~200 from [1, N], sort it, greedy on it
        subset_size = min(300, N)
        candidates = random.sample(range(1, N + 1), subset_size)
        candidates.sort()
        
        result = greedy_build(candidates)
        if len(result) > best_len:
            best = result
            best_len = len(result)
    
    # Strategy 3: Try a more structured approach
    # Use numbers with large gaps to avoid collisions
    # e.g., powers of 2 or similar
    
    if time.time() > DEADLINE - 20:
        return best
    
    # Try: use a base-b representation approach
    # For B_3[1] sets, we need all triple sums unique
    # A sufficient condition: if we use numbers a_i where a_{i+1} > 3 * a_i * (something)
    # Actually, for sum-free with triples, we need a_{i+1} > max possible sum of 3 from previous
    # If selected = {s_1, ..., s_k}, max triple sum = 3*s_k
    # Next element must be > 3*s_k to avoid any collision? No, that's too strong.
    # We need: for any a, b, c in selected, a+b+c != x+y+z for any other triple.
    
    # Let's try a geometric progression
    for base in [3, 4, 5, 6, 7, 8, 9, 10]:
        if time.time() > DEADLINE - 20:
            break
        candidates = []
        val = 1
        while val <= N:
            candidates.append(val)
            val *= base
        # Add some intermediate values?
        # Just greedy on this set
        result = greedy_build(candidates)
        if len(result) > best_len:
            best = result
            best_len = len(result)
    
    return best
# EVOLVE-BLOCK-END
