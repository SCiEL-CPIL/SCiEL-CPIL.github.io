"""Rewritten program: randomized greedy with incremental triple-sum collision checking."""
import time
import random
import numpy as np
from itertools import combinations_with_replacement

def _check_incremental(selected, x):
    """
    Check if adding x to selected keeps all unordered triple sums unique.
    We only need to check new triples that involve x.
    Triples involving x (unordered, with repetition allowed):
      - (x, x, x)
      - (x, x, y) for y in selected
      - (x, y, y) for y in selected
      - (x, y, z) for y, z in selected with y < z
    We must ensure none of these sums collide with each other or with existing sums.
    """
    n = len(selected)
    if n == 0:
        return True
    
    # Precompute existing triple sums as a set for O(1) lookup
    # But we can be smarter: only check new sums against existing + each other
    
    # Generate all new triple sums involving x
    new_sums = set()
    
    # (x, x, x)
    s = 3 * x
    new_sums.add(s)
    
    # (x, x, y) for each y in selected
    for y in selected:
        s = 2 * x + y
        new_sums.add(s)
    
    # (x, y, y) for each y in selected
    for y in selected:
        s = x + 2 * y
        new_sums.add(s)
    
    # (x, y, z) for y < z in selected
    for i in range(n):
        for j in range(i + 1, n):
            s = x + selected[i] + selected[j]
            new_sums.add(s)
    
    # Check for collisions among new sums
    if len(new_sums) != len(new_sums):  # This is always true for a set; need to check differently
        # Actually, we need to count how many triples produce these sums
        pass
    
    # Better approach: generate all new triple sums as a list and check for duplicates
    new_sums_list = []
    new_sums_list.append(3 * x)
    for y in selected:
        new_sums_list.append(2 * x + y)
    for y in selected:
        new_sums_list.append(x + 2 * y)
    for i in range(n):
        for j in range(i + 1, n):
            new_sums_list.append(x + selected[i] + selected[j])
    
    if len(new_sums_list) != len(set(new_sums_list)):
        return False
    
    # Now check against existing sums
    # We need to know existing sums. Let's compute them once and cache.
    # For efficiency, we'll compute existing sums each time (could be optimized with caching)
    existing_sums = set()
    for triple in combinations_with_replacement(selected, 3):
        existing_sums.add(sum(triple))
    
    for s in new_sums_list:
        if s in existing_sums:
            return False
    
    return True


def construct():
    start_time = time.time()
    time_limit = 55.0  # Leave buffer for 60s limit
    
    best_set = []
    best_size = 0
    
    def run_greedy(order):
        selected = []
        # Cache of existing sums for fast lookup
        existing_sums = set()
        
        for x in order:
            if x in selected:
                continue
            if x < 1 or x > 4096:
                continue
            
            # Check if adding x keeps triple sums unique
            # New triples involving x
            new_sums_list = []
            new_sums_list.append(3 * x)
            for y in selected:
                new_sums_list.append(2 * x + y)
            for y in selected:
                new_sums_list.append(x + 2 * y)
            n = len(selected)
            for i in range(n):
                for j in range(i + 1, n):
                    new_sums_list.append(x + selected[i] + selected[j])
            
            # Check for collisions among new sums
            if len(new_sums_list) != len(set(new_sums_list)):
                continue
            
            # Check against existing sums
            collision = False
            for s in new_sums_list:
                if s in existing_sums:
                    collision = True
                    break
            if collision:
                continue
            
            # Accept x
            selected.append(x)
            existing_sums.update(new_sums_list)
            
            if len(selected) > best_size:
                nonlocal best_set, best_size
                best_set = selected[:]
                best_size = len(selected)
        
        return selected
    
    # First, run deterministic greedy from 1 upward as baseline
    run_greedy(range(1, 4097))
    
    # Now do randomized restarts with different orderings
    # Strategy: random permutations of a subset, or random shuffles of [1, 4096]
    # To be efficient, we can try various random orderings
    
    rng = random.Random(42)
    
    # Try several random orderings
    # For each, do greedy in that order
    # We can also try biased orderings (e.g., prefer larger numbers, or spaced numbers)
    
    num_restarts = 50
    for restart in range(num_restarts):
        if time.time() - start_time > time_limit - 5:
            break
        
        # Create a random ordering
        # Option 1: full random permutation of [1, 4096]
        # Option 2: random sample from [1, 4096] with replacement of indices
        # Let's try full random permutation but only process until we've considered enough
        
        order = list(range(1, 4097))
        rng.shuffle(order)
        
        run_greedy(order)
    
    # Also try some structured orderings
    # Spaced numbers: 1, 2, 4, 8, ... (powers of 2) - these tend to work well for sum-free
    if time.time() - start_time < time_limit - 5:
        powers = [1]
        while powers[-1] * 2 <= 4096:
            powers.append(powers[-1] * 2)
        run_greedy(powers)
    
    # Try geometric progression with ratio 3
    if time.time() - start_time < time_limit - 5:
        geo = [1]
        while geo[-1] * 3 <= 4096:
            geo.append(geo[-1] * 3)
        run_greedy(geo)
    
    # Try random subsets with larger gaps
    if time.time() - start_time < time_limit - 5:
        for _ in range(20):
            if time.time() - start_time > time_limit - 5:
                break
            # Generate random set of ~100-200 numbers from [1, 4096]
            k = rng.randint(100, 300)
            subset = rng.sample(range(1, 4097), k)
            rng.shuffle(subset)
            run_greedy(subset)
    
    return best_set
