"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START
def _is_sidon(subset):
    """Check if a subset has all unique pair sums (including diagonals)."""
    sums = set()
    n = len(subset)
    for i in range(n):
        for j in range(i, n):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _max_sidon_subset_size(arr):
    """Compute the largest subset of arr with all unordered pair sums unique (including diagonals)."""
    n = len(arr)
    # We want the largest k such that there exists a subset of size k that is Sidon.
    # Check k from n down to 1.
    for k in range(n, 0, -1):
        # Optimization: If k is large, the number of combinations is small.
        # If k is small, we might find one quickly.
        # For n=12, max combinations is C(12,6)=924.
        for subset in combinations(arr, k):
            if _is_sidon(subset):
                return k
    return 0

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin before 60s limit
    
    # Initial best: range(12)
    best_arr = list(range(12))
    best_size = _max_sidon_subset_size(best_arr)
    
    random.seed(42)
    
    # We will perform multiple restarts of a local search.
    # Local search minimizes _max_sidon_subset_size.
    
    for restart in range(200): # Max restarts, controlled by time
        if time.time() > deadline:
            break
            
        # Initialize with a random set or a structured set
        if restart % 2 == 0:
            arr = sorted(random.sample(range(129), 12))
        else:
            # Structured initialization: clusters
            # e.g., 0,1,2, 10,11,12, 20,21,22, 30,31,32
            base = random.randint(0, 100)
            arr = []
            for i in range(4):
                offset = base + i * 10
                arr.extend([offset, offset+1, offset+2])
            arr = sorted(arr)
            # Ensure distinctness and range
            # The above construction ensures distinctness if offsets are far enough
            # 10 > 2, so distinct.
            # Check range
            if max(arr) > 128:
                continue

        current_size = _max_sidon_subset_size(arr)
        
        if current_size < best_size:
            best_arr = arr
            best_size = current_size
            if best_size <= 4: # Theoretical lower bound for 12 elements? 
                # 4 is very good. 5 is likely the floor.
                break

        # Local Search
        no_improve_count = 0
        for step in range(1000): # Max steps per restart
            if time.time() > deadline:
                break
                
            # Pick a random index to modify
            i = random.randint(0, 11)
            
            # Try replacing arr[i] with a random value
            new_val = random.randint(0, 128)
            if new_val in arr:
                # Try swapping with another element instead
                j = random.randint(0, 11)
                if i == j:
                    continue
                new_arr = arr.copy()
                new_arr[i], new_arr[j] = new_arr[j], new_arr[i]
            else:
                new_arr = arr.copy()
                new_arr[i] = new_val
            
            new_arr = sorted(new_arr)
            
            # Calculate new size
            new_size = _max_sidon_subset_size(new_arr)
            
            if new_size < current_size:
                arr = new_arr
                current_size = new_size
                no_improve_count = 0
                if current_size < best_size:
                    best_arr = arr
                    best_size = current_size
                    if best_size <= 4:
                        return best_arr
            else:
                no_improve_count += 1
                if no_improve_count > 50:
                    break # Stuck, restart local search
        
    return best_arr
# EVOLVE-BLOCK-END
