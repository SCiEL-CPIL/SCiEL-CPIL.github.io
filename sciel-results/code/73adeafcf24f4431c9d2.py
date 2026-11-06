import random
import time

def _is_sidon(subset):
    sums = set()
    n = len(subset)
    for i in range(n):
        for j in range(i, n):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _max_sidon_size(A):
    # A is a list of 12 distinct integers
    # We check subsets from largest to smallest
    n = len(A)
    # Generate all subsets in decreasing order of size
    # Since n=12, 2^12 = 4096 subsets is small
    # We can just iterate through all masks
    # But it's faster to check larger sizes first
    
    # Helper to get subset from mask
    def get_subset(mask):
        return [A[i] for i in range(n) if (mask >> i) & 1]
    
    # Pre-compute masks by size? Or just iterate.
    # Let's iterate masks from 1 to 2^n - 1, but we want largest first.
    # We can just loop size from n down to 1 and check all combinations of that size.
    
    import itertools
    
    for size in range(n, 0, -1):
        for comb in itertools.combinations(range(n), size):
            subset = [A[i] for i in comb]
            if _is_sidon(subset):
                return size
    return 0

def construct():
    start_time = time.time()
    deadline = start_time + 50.0 # Safety margin for 60s limit
    
    best_set = None
    best_score = 13 # Max possible is 12
    
    # Initial random set
    A = random.sample(range(129), 12)
    current_score = _max_sidon_size(A)
    
    if current_score < best_score:
        best_score = current_score
        best_set = A[:]
    
    # Local Search / Simulated Annealing
    temp = 1.0
    cooling = 0.95
    
    while time.time() < deadline:
        # Create a neighbor
        neighbor = A[:]
        if random.random() < 0.5:
            # Swap two elements
            i, j = random.sample(range(12), 2)
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
        else:
            # Change one element to a new random value
            i = random.randint(0, 11)
            new_val = random.randint(0, 128)
            if new_val not in neighbor:
                neighbor[i] = new_val
            else:
                # Try a few more times to find a unique value
                for _ in range(10):
                    new_val = random.randint(0, 128)
                    if new_val not in neighbor:
                        neighbor[i] = new_val
                        break
        
        neighbor_score = _max_sidon_size(neighbor)
        
        # Acceptance criterion
        if neighbor_score < current_score:
            A = neighbor
            current_score = neighbor_score
            if current_score < best_score:
                best_score = current_score
                best_set = A[:]
            if best_score == 3: # 3 is likely the minimum for n=12, 2 is very hard/impossible in [0,128]?
                # We can break early if we hit 3, as 2 is extremely unlikely for 12 elements in small range
                # Actually, let's keep searching a bit more to be sure, or just return 3.
                # A set with max Sidon size 3 is a valid improvement over 4.
                pass
        else:
            # Accept with probability exp(-delta / temp) if worsening
            delta = current_score - neighbor_score # positive if worsening
            if delta > 0:
                p = pow(2.71828, -delta / max(temp, 0.01))
                if random.random() < p:
                    A = neighbor
                    current_score = neighbor_score
        
        temp *= cooling
        if temp < 0.01:
            # Restart
            temp = 1.0
            A = random.sample(range(129), 12)
            current_score = _max_sidon_size(A)
            if current_score < best_score:
                best_score = current_score
                best_set = A[:]

    return best_set
