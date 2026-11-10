import time
import random
import numpy as np
from collections import Counter

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    n = 128
    target_lo = 128
    target_hi = 256
    target_range = list(range(target_lo, target_hi + 1))
    num_targets = len(target_range)
    
    def compute_multiplicities(A_set):
        """Compute multiplicity for each target value."""
        mult = [0] * (target_hi + 1)
        A_list = sorted(A_set)
        k = len(A_list)
        for i in range(k):
            for j in range(i, k):
                s = A_list[i] + A_list[j]
                if target_lo <= s <= target_hi:
                    mult[s] += 1
        return mult
    
    def is_valid(A_set):
        """Check if all targets are covered."""
        mult = compute_multiplicities(A_set)
        return all(mult[t] > 0 for t in target_range)
    
    def objective(A_set):
        """Return max multiplicity, or infinity if invalid."""
        mult = compute_multiplicities(A_set)
        if any(mult[t] == 0 for t in target_range):
            return float('inf')
        return max(mult[t] for t in target_range)
    
    # Initial solution: try a few random starts and pick the best
    best_A = None
    best_obj = float('inf')
    
    # Start with the seed solution
    seed_A = set(range(64, 129))
    obj = objective(seed_A)
    if obj < best_obj:
        best_obj = obj
        best_A = set(seed_A)
    
    # Try some structured initial solutions
    # Try smaller sets - we need at least enough pairs
    # k(k+1)/2 >= 129 => k >= 16
    for size in range(16, 30):
        if time.time() > deadline:
            break
        for attempt in range(20):
            if time.time() > deadline:
                break
            # Random set of given size
            A_set = set(random.sample(range(0, n + 1), size))
            obj = objective(A_set)
            if obj < best_obj:
                best_obj = obj
                best_A = set(A_set)
    
    # Local search with simulated annealing
    if best_A is None:
        best_A = set(seed_A)
        best_obj = objective(best_A)
    
    current_A = set(best_A)
    current_obj = best_obj
    best_obj_sa = best_obj
    best_A_sa = set(best_A)
    
    # Simulated annealing parameters
    T = 5.0
    cooling = 0.9995
    min_T = 0.01
    
    step_count = 0
    while time.time() < deadline and T > min_T:
        if step_count % 1000 == 0 and time.time() > deadline:
            break
        step_count += 1
        
        # Generate a neighbor
        op = random.random()
        A_list = list(current_A)
        
        if op < 0.3 and len(current_A) > 16:
            # Remove an element
            idx = random.randint(0, len(A_list) - 1)
            new_A = set(A_list[:idx] + A_list[idx+1:])
        elif op < 0.6:
            # Add an element
            candidates = [x for x in range(0, n + 1) if x not in current_A]
            if not candidates:
                continue
            x = random.choice(candidates)
            new_A = set(current_A)
            new_A.add(x)
        elif op < 0.8:
            # Swap an element
            idx = random.randint(0, len(A_list) - 1)
            old_x = A_list[idx]
            candidates = [x for x in range(0, n + 1) if x not in current_A]
            if not candidates:
                continue
            new_x = random.choice(candidates)
            new_A = set(current_A)
            new_A.discard(old_x)
            new_A.add(new_x)
        else:
            # Move an element to a random new position
            idx = random.randint(0, len(A_list) - 1)
            old_x = A_list[idx]
            candidates = [x for x in range(0, n + 1) if x not in current_A]
            if not candidates:
                continue
            new_x = random.choice(candidates)
            new_A = set(current_A)
            new_A.discard(old_x)
            new_A.add(new_x)
        
        new_obj = objective(new_A)
        
        # Acceptance criterion
        if new_obj <= current_obj:
            current_A = new_A
            current_obj = new_obj
            if new_obj < best_obj_sa:
                best_obj_sa = new_obj
                best_A_sa = set(new_A)
        else:
            # Accept with probability exp(-(new_obj - current_obj) / T)
            delta = new_obj - current_obj
            if random.random() < np.exp(-delta / T):
                current_A = new_A
                current_obj = new_obj
        
        T *= cooling
    
    return sorted(best_A_sa)
