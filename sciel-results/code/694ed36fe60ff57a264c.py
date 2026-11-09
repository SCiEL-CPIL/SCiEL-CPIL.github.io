import time
import random
from itertools import combinations

def is_sidon(subset):
    """Check if subset is a Sidon set (all pair sums with repetition are unique)."""
    n = len(subset)
    sums = set()
    for i in range(n):
        for j in range(i, n):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def max_sidon_size(arr):
    """Find the largest Sidon subset size of arr."""
    n = len(arr)
    # Check from largest to smallest
    for k in range(n, 0, -1):
        for subset in combinations(arr, k):
            if is_sidon(subset):
                return k
    return 0

def construct():
    start = time.time()
    deadline = start + 85  # safety margin before 90s
    
    best_set = list(range(12))
    best_val = max_sidon_size(best_set)  # should be 4
    
    random.seed(42)
    
    # Strategy: Search for sets with max Sidon size 3.
    # Use a local search on "pair-structured" sets.
    # A pair-structured set is defined by 6 starting points s_i, containing s_i and s_i+1.
    
    def generate_pair_set():
        # Generate 6 distinct starting points such that intervals [s_i, s_i+1] don't overlap
        # s_i must be distinct and |s_i - s_j| > 1
        starts = []
        while len(starts) < 6:
            s = random.randint(0, 127)
            valid = True
            for existing in starts:
                if abs(s - existing) <= 1:
                    valid = False
                    break
            if valid:
                starts.append(s)
        # Construct set
        arr = []
        for s in starts:
            arr.append(s)
            arr.append(s + 1)
        arr.sort()
        return arr

    # Local search
    current_set = generate_pair_set()
    curr_val = max_sidon_size(current_set)
    if curr_val < best_val:
        best_val = curr_val
        best_set = current_set[:]
        
    no_improve = 0
    
    while time.time() < deadline:
        if best_val == 3:
            return best_set
            
        # Mutation: Change one start point
        # Represent set by its 6 pairs
        # Extract starts
        starts = []
        used = set()
        for i in range(0, 12, 2):
            # current_set is sorted. Pairs are (current_set[i], current_set[i+1])?
            # Not necessarily, if they overlap or are far.
            # But we generated them as pairs.
            # To be safe, let's just pick a random element and change it.
            pass
        
        # Simpler: Pick a random index in current_set, change value to a new random value.
        idx = random.randint(0, 11)
        old_val = current_set[idx]
        
        # Pick new value
        # To maintain "pair-like" structure, pick a value close to another element?
        # Or just random?
        new_val = random.randint(0, 128)
        
        if new_val == old_val or new_val in current_set:
            continue
            
        current_set[idx] = new_val
        current_set.sort()
        
        new_val_size = max_sidon_size(current_set)
        
        if new_val_size < best_val:
            best_val = new_val_size
            best_set = current_set[:]
            no_improve = 0
        else:
            no_improve += 1
            # Revert if no improvement for a while
            if no_improve > 20:
                current_set[idx] = old_val
                current_set.sort()
                # Random restart with pair structure
                current_set = generate_pair_set()
                curr_val = max_sidon_size(current_set)
                if curr_val < best_val:
                    best_val = curr_val
                    best_set = current_set[:]
                no_improve = 0

    return best_set
