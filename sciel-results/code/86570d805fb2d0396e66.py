import time
import random
import numpy as np

def construct():
    N = 1000
    start_time = time.time()
    deadline = start_time + 140
    
    best_size = 0
    best_set = []
    
    # Numpy arrays for fast vectorized checks
    # counts[s] = number of representations of s
    # in_set[x] = 1 if x in set
    
    def can_add_np(x, S_np, counts_np):
        # S_np is a numpy array of elements in S
        if len(S_np) == 0:
            return counts_np[2 * x] < 2
        
        sums = x + S_np
        # Check if any sum has count >= 2
        # Also check 2*x
        if counts_np[2 * x] >= 2:
            return False
        if np.any(counts_np[sums] >= 2):
            return False
        return True

    def add_element_np(x, S_np, counts_np, in_set_np):
        S_np = np.append(S_np, x)
        sums = x + S_np
        # We added x, so x+y for all y in NEW S?
        # Wait. If we append x to S_np, S_np now contains x.
        # The sums to update are x+y for all y in the OLD S, plus 2*x.
        # If we use the NEW S_np, we include x+x=2x and x+y for y in OLD S.
        # This is correct.
        counts_np[sums] += 1
        in_set_np[x] = 1
        return S_np, counts_np, in_set_np

    def greedy_build_np(order):
        S_np = np.array([], dtype=int)
        counts_np = np.zeros(2 * N + 1, dtype=np.uint8)
        in_set_np = np.zeros(N + 1, dtype=np.uint8)
        
        for x in order:
            if can_add_np(x, S_np, counts_np):
                S_np, counts_np, in_set_np = add_element_np(x, S_np, counts_np, in_set_np)
        
        return S_np.tolist(), counts_np, in_set_np

    def repair_np(S_np, counts_np, in_set_np, candidates_order):
        # S_np, counts_np, in_set_np are numpy arrays
        # candidates_order is a list of ints
        for x in candidates_order:
            if in_set_np[x]:
                continue
            if can_add_np(x, S_np, counts_np):
                S_np, counts_np, in_set_np = add_element_np(x, S_np, counts_np, in_set_np)
        return S_np, counts_np, in_set_np

    def try_order(order):
        nonlocal best_size, best_set
        S_list, counts_np, in_set_np = greedy_build_np(order)
        if len(S_list) > best_size:
            best_size = len(S_list)
            best_set = S_list

    # 1. Initial Greedy Attempts
    try_order(list(range(1, N + 1)))
    if time.time() < deadline:
        try_order(list(range(N, 0, -1)))
    
    # 2. Iterated Local Search
    rng = random.Random(42)
    
    # Convert best_set to numpy
    current_S_np = np.array(best_set, dtype=int)
    current_counts_np, current_in_set_np = greedy_build_np(best_set)[1:] # Rebuild counts? 
    # Actually, greedy_build_np returns counts. But we need counts for the *existing* best_set.
    # Let's just rebuild counts for best_set properly.
    
    def compute_counts_np(S_np):
        counts_np = np.zeros(2 * N + 1, dtype=np.uint8)
        in_set_np = np.zeros(N + 1, dtype=np.uint8)
        in_set_np[S_np] = 1
        # Compute counts
        # This is O(N^2) but we can vectorize partially?
        # For each x in S, add to counts[x+S]
        # This is still a loop over S.
        for x in S_np:
            sums = x + S_np
            counts_np[sums] += 1
        return counts_np, in_set_np

    current_counts_np, current_in_set_np = compute_counts_np(current_S_np)
    
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() < deadline - 5.0:
        K = rng.randint(1, 20)
        
        # Copy state
        S_perturbed = current_S_np.copy()
        counts_perturbed = current_counts_np.copy()
        in_set_perturbed = current_in_set_np.copy()
        
        # Remove K elements
        for _ in range(min(K, len(S_perturbed))):
            idx = rng.randrange(len(S_perturbed))
            x = S_perturbed[idx]
            
            # Remove x from S_perturbed
            S_perturbed = np.delete(S_perturbed, idx)
            in_set_perturbed[x] = 0
            
            # Update counts: subtract contributions of x
            # x contributed to 2*x and x+y for y in S_perturbed (remaining)
            # Wait, np.delete changes indices.
            # The remaining S_perturbed is the set after removal.
            # So we subtract x + y for y in S_perturbed (new) and 2*x.
            # But x was paired with all y in the OLD set.
            # Old set = New set + {x}.
            # Pairs involving x: (x,x) and (x,y) for y in New set.
            # So yes, subtract 1 from counts[2*x] and 1 from counts[x+y] for y in New set.
            
            counts_perturbed[2 * x] -= 1
            sums = x + S_perturbed
            counts_perturbed[sums] -= 1
        
        # Repair
        candidates = [x for x in range(1, N + 1) if not in_set_perturbed[x]]
        rng.shuffle(candidates)
        
        S_repaired, counts_repaired, in_set_repaired = repair_np(
            S_perturbed, counts_perturbed, in_set_perturbed, candidates
        )
        
        if len(S_repaired) > best_size:
            best_size = len(S_repaired)
            best_set = S_repaired.tolist()
            current_S_np = S_repaired
            current_counts_np = counts_repaired
            current_in_set_np = in_set_repaired
            no_improve_count = 0
        else:
            no_improve_count += 1
            
            if no_improve_count > max_no_improve:
                order = list(range(1, N + 1))
                rng.shuffle(order)
                S_greedy, counts_g, in_set_g = greedy_build_np(order)
                if len(S_greedy) > best_size:
                    best_size = len(S_greedy)
                    best_set = S_greedy
                    current_S_np = np.array(S_greedy, dtype=int)
                    current_counts_np = counts_g
                    current_in_set_np = in_set_g
                no_improve_count = 0

    return best_set
