"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill
    
    n = 128
    min_sum = 128
    max_sum = 256
    num_targets = max_sum - min_sum + 1  # 129 targets
    
    def compute_counts(A):
        """Fast computation of pair-sum counts using numpy."""
        A = np.array(A, dtype=np.int64)
        # Create indicator vector
        indicator = np.zeros(n + 1, dtype=np.int64)
        indicator[A] = 1
        # Convolution gives ordered pair counts
        conv = np.convolve(indicator, indicator, mode='full')
        # Unordered pairs: for a != b, count appears twice in conv; for a == b, once
        # conv[s] = sum of indicator[a] * indicator[s-a] for all a
        # This counts ordered pairs (a,b) with a+b=s
        # Unordered count = (conv[s] + number of a with 2a=s) / 2
        # But we need: for each unordered pair {a,b} with a<=b, count once
        # conv[s] = (# of ordered pairs) = 2*(# unordered with a!=b) + (# with a==b)
        # So unordered = (conv[s] - diag[s]) / 2 + diag[s] = (conv[s] + diag[s]) / 2
        # where diag[s] = 1 if s is even and s/2 in A, else 0
        counts = np.zeros(max_sum + 1, dtype=np.int64)
        for s in range(min_sum, max_sum + 1):
            if s < len(conv):
                ordered = conv[s]
                # Check if s/2 is in A (for the diagonal)
                if s % 2 == 0:
                    half = s // 2
                    if 0 <= half <= n and indicator[half] == 1:
                        diag = 1
                    else:
                        diag = 0
                else:
                    diag = 0
                unordered = (ordered + diag) // 2
                counts[s] = unordered
        return counts
    
    def get_cost(A):
        counts = compute_counts(A)
        target_counts = counts[min_sum:max_sum+1]
        if np.any(target_counts == 0):
            return float('inf')
        return int(np.max(target_counts))
    
    def get_cost_with_counts(A):
        counts = compute_counts(A)
        target_counts = counts[min_sum:max_sum+1]
        if np.any(target_counts == 0):
            return float('inf'), counts
        return int(np.max(target_counts)), counts
    
    best_score = float('inf')
    best_set = None
    best_set_np = None
    
    def try_set(A_list):
        nonlocal best_score, best_set, best_set_np
        A_sorted = sorted(A_list)
        c = get_cost(A_sorted)
        if c < best_score:
            best_score = c
            best_set = A_sorted
            best_set_np = np.array(A_sorted, dtype=np.int64)
    
    # Initial solution: the seed
    initial = list(range(64, 129))
    try_set(initial)
    
    # Try various structured initial solutions
    # Symmetric sets
    for offset in range(0, 30):
        lo = max(0, 64 - offset)
        hi = min(128, 64 + offset)
        candidate = list(range(lo, hi + 1))
        if len(candidate) > 0:
            try_set(candidate)
    
    # Random initialization with various sizes
    random.seed(42)
    
    for size in range(30, 80):
        if time.time() > deadline - 10:
            break
        for _ in range(20):
            if time.time() > deadline - 8:
                break
            A = random.sample(range(0, 129), size)
            try_set(A)
    
    # Local search: swap-based improvement
    if best_set is not None and time.time() < deadline - 5:
        current = best_set[:]
        current_np = best_set_np.copy()
        current_cost = best_score
        in_set = set(current)
        
        while time.time() < deadline - 2:
            improved = False
            # Try removing elements
            for i in range(len(current) - 1, -1, -1):
                if time.time() > deadline - 2:
                    break
                val = current[i]
                trial = current[:i] + current[i+1:]
                if len(trial) == 0:
                    continue
                tc = get_cost(trial)
                if tc <= current_cost:
                    current = trial
                    current_cost = tc
                    in_set.discard(val)
                    improved = True
                    break
            
            # Try adding elements
            if not improved:
                for new_val in range(129):
                    if time.time() > deadline - 2:
                        break
                    if new_val in in_set:
                        continue
                    trial = current + [new_val]
                    tc = get_cost(trial)
                    if tc < current_cost:
                        current = trial
                        current_cost = tc
                        in_set.add(new_val)
                        improved = True
                        break
            
            # Try swaps
            if not improved:
                for i in range(len(current)):
                    if time.time() > deadline - 2:
                        break
                    old_val = current[i]
                    for new_val in range(129):
                        if time.time() > deadline - 2:
                            break
                        if new_val == old_val or new_val in in_set:
                            continue
                        trial = current[:]
                        trial[i] = new_val
                        trial.sort()
                        tc = get_cost(trial)
                        if tc < current_cost:
                            current = trial
                            current_cost = tc
                            in_set.discard(old_val)
                            in_set.add(new_val)
                            improved = True
                            break
                    if improved:
                        break
            
            if not improved:
                break
        
        if current_cost < best_score:
            best_score = current_cost
            best_set = current
    
    # Second phase: more aggressive search if we have time
    if time.time() < deadline - 5 and best_score > 3:
        random.seed(123)
        for _ in range(100):
            if time.time() > deadline - 3:
                break
            # Start from best and perturb
            if best_set is None:
                break
            size = len(best_set) + random.randint(-5, 5)
            size = max(20, min(80, size))
            A = random.sample(range(0, 129), size)
            try_set(A)
            # Quick local improvement
            if best_score <= 3:
                break
    
    # Final local search
    if best_set is not None and time.time() < deadline - 1:
        current = best_set[:]
        current_cost = best_score
        in_set = set(current)
        
        while time.time() < deadline - 0.5:
            improved = False
            
            # Try removing
            for i in range(len(current) - 1, -1, -1):
                if time.time() > deadline - 0.5:
                    break
                val = current[i]
                trial = current[:i] + current[i+1:]
                if len(trial) == 0:
                    continue
                tc = get_cost(trial)
                if tc <= current_cost:
                    current = trial
                    current_cost = tc
                    in_set.discard(val)
                    improved = True
                    break
            
            # Try adding
            if not improved:
                for new_val in range(129):
                    if time.time() > deadline - 0.5:
                        break
                    if new_val in in_set:
                        continue
                    trial = current + [new_val]
                    tc = get_cost(trial)
                    if tc < current_cost:
                        current = trial
                        current_cost = tc
                        in_set.add(new_val)
                        improved = True
                        break
            
            # Try swaps
            if not improved:
                for i in range(len(current)):
                    if time.time() > deadline - 0.5:
                        break
                    old_val = current[i]
                    for new_val in range(129):
                        if time.time() > deadline - 0.5:
                            break
                        if new_val == old_val or new_val in in_set:
                            continue
                        trial = current[:]
                        trial[i] = new_val
                        trial.sort()
                        tc = get_cost(trial)
                        if tc < current_cost:
                            current = trial
                            current_cost = tc
                            in_set.discard(old_val)
                            in_set.add(new_val)
                            improved = True
                            break
                    if improved:
                        break
            
            if not improved:
                break
        
        if current_cost < best_score:
            best_score = current_cost
            best_set = current
    
    if best_set is None or len(best_set) == 0:
        return list(range(64, 129))
    
    return list(best_set)

# EVOLVE-BLOCK-END
