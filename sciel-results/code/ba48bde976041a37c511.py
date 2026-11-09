import time
import random
from itertools import combinations

def is_sidon(subset):
    """Check if a subset is a Sidon set."""
    k = len(subset)
    sums = set()
    for i in range(k):
        xi = subset[i]
        for j in range(i, k):
            xj = subset[j]
            s = xi + xj
            if s in sums:
                return False
            sums.add(s)
    return True

def max_sidon_size(ambient):
    """Find the size of the largest Sidon subset of the ambient set."""
    n = len(ambient)
    # Check from largest size down to 1
    for k in range(n, 0, -1):
        for comb in combinations(ambient, k):
            if is_sidon(comb):
                return k
    return 1

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    # Initial candidate: range(12)
    best_set = list(range(12))
    best_val = max_sidon_size(best_set)
    
    # If we already found the theoretical minimum (3), we can stop early
    # But let's try to confirm or find 2 (unlikely)
    
    random.seed(42) # For reproducibility in local testing, though evaluator might not care
    
    # Local search + Random restarts
    current_set = best_set[:]
    current_val = best_val
    
    while time.time() < deadline:
        # Random restart with some probability
        if random.random() < 0.1:
            # Generate a random set of 12 distinct integers in [0, 128]
            current_set = random.sample(range(129), 12)
            current_val = max_sidon_size(current_set)
            if current_val < best_val:
                best_val = current_val
                best_set = current_set[:]
                if best_val == 3:
                    # 3 is likely the minimum for N=12 in this range
                    # We can break or continue to be safe
                    # Let's break to save time, as 2 is impossible for 12 distinct ints?
                    # Actually, if every 3-subset is an AP, max size is 2.
                    # Can we have 12 ints where every 3 is an AP? No.
                    # So 3 is the target.
                    pass 
        else:
            # Local perturbation: pick 1-2 elements and replace with random values
            num_changes = random.choice([1, 2])
            new_set = current_set[:]
            indices = random.sample(range(12), num_changes)
            for idx in indices:
                # Pick a random value not in the set
                while True:
                    v = random.randint(0, 128)
                    if v not in new_set:
                        new_set[idx] = v
                        break
            
            new_val = max_sidon_size(new_set)
            if new_val < current_val:
                current_set = new_set
                current_val = new_val
                if current_val < best_val:
                    best_val = current_val
                    best_set = current_set[:]
            # Accept worse moves occasionally (Simulated Annealing style)
            elif new_val > current_val and random.random() < 0.1:
                current_set = new_set
                current_val = new_val
        
        # Early exit if we hit 3
        if best_val == 3:
            # We can try a few more iterations to ensure stability, 
            # but 3 is very good.
            # Let's just break to return quickly.
            break
            
    return best_set
