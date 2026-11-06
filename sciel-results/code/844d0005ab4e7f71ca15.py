import random
import time
from itertools import combinations

def _is_sidon(subset):
    sums = set()
    for i in range(len(subset)):
        for j in range(i, len(subset)):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _max_sidon_size(A):
    n = len(A)
    # Check from largest size down to 1
    # If we find a Sidon subset of size k, that is the maximum.
    for k in range(n, 0, -1):
        for comb in combinations(A, k):
            if _is_sidon(comb):
                return k
    return 0 # Should not happen for n>=1

def construct():
    start_time = time.time()
    deadline = start_time + 50.0 # Safety margin
    
    random.seed(42)
    
    best_set = None
    best_val = 100 # Initialize high
    
    # Run multiple random restarts
    while time.time() < deadline:
        # Start with a random set
        current_set = sorted(random.sample(range(129), 12))
        current_val = _max_sidon_size(current_set)
        
        if current_val < best_val:
            best_val = current_val
            best_set = current_set[:]
            
        # If we found the theoretical minimum (likely 4 or 5 for N=12 in [0,128]? 
        # Actually, for N=12, max Sidon size is at least 4? 
        # Sidon sets in [0, 128] can be large. 
        # But we are MINIMIZING the max Sidon subset of the AMBIENT set.
        # The ambient set is 12 numbers. 
        # The max Sidon subset size is at most 12.
        # Can we make it 4? 3? 
        # If the set itself is not Sidon, the max subset is smaller.
        # We want the set to be "as non-Sidon as possible".
        # The minimum possible max Sidon size for any set of 12 integers is likely 4 or 5.
        # Let's assume we can get to 4.
        if best_val <= 4:
            return best_set

        # Local Search: Hill Climbing
        # Try to reduce current_val
        stuck_counter = 0
        while time.time() < deadline:
            # Pick a random element to change
            idx = random.randint(0, 11)
            old_val = current_set[idx]
            
            # Try a few random replacements
            improved = False
            for _ in range(20): # Try 20 random moves
                new_val_elem = random.randint(0, 128)
                if new_val_elem == old_val:
                    continue
                if new_val_elem in current_set:
                    continue
                
                # Perform swap
                current_set[idx] = new_val_elem
                current_set.sort()
                
                # Evaluate
                val = _max_sidon_size(current_set)
                
                if val < current_val:
                    current_val = val
                    if current_val < best_val:
                        best_val = current_val
                        best_set = current_set[:]
                    improved = True
                    break
                else:
                    # Revert
                    current_set[idx] = old_val
                    current_set.sort()
            
            if not improved:
                stuck_counter += 1
                if stuck_counter > 100:
                    break
            else:
                stuck_counter = 0

    return best_set
