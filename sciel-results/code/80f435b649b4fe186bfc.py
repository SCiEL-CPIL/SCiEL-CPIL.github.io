"""Ambient integer sets with small largest Sidon subset."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START

def _count_sum_collisions(ambient):
    """
    Count the number of sum collisions in the ambient set.
    A collision is a pair of distinct unordered pairs (i,j) and (k,l) with i<=j, k<=l,
    {i,j} != {k,l}, such that ambient[i]+ambient[j] == ambient[k]+ambient[l].
    This is a cheap proxy for how "non-Sidon" the set is.
    """
    n = len(ambient)
    if n < 2:
        return 0
    # Count frequency of each sum
    sum_count = {}
    for i in range(n):
        for j in range(i, n):
            s = ambient[i] + ambient[j]
            sum_count[s] = sum_count.get(s, 0) + 1
    # A collision group of size m contributes C(m,2) collision pairs
    collisions = 0
    for count in sum_count.values():
        if count >= 2:
            collisions += count * (count - 1) // 2
    return collisions

def _max_sidon_subset_size(ambient):
    """
    Exhaustively compute the largest subset of ambient with all unordered pair sums unique,
    INCLUDING diagonals.
    """
    n = len(ambient)
    if n <= 0:
        return 0
    if n == 1:
        return 1
    if n == 2:
        return 2
    
    # Try from largest possible size downward
    for k in range(n, 0, -1):
        for subset in combinations(ambient, k):
            sums = set()
            valid = True
            for i in range(k):
                for j in range(i, k):
                    s = subset[i] + subset[j]
                    if s in sums:
                        valid = False
                        break
                    sums.add(s)
                if not valid:
                    break
            if valid:
                return k
    return 1

def _initial_structures():
    """Generate diverse initial structures with high collision density."""
    structures = []
    
    # Structure 1: Pairs summing to same value (symmetric around 64)
    structures.append([1, 2, 3, 4, 5, 6, 58, 59, 60, 61, 62, 63])
    
    # Structure 2: Arithmetic progression (maximizes internal collisions)
    structures.append(list(range(12)))  # 0..11
    
    # Structure 3: Dense AP in middle
    structures.append(list(range(58, 70)))  # 58..69
    
    # Structure 4: Two tight clusters
    structures.append([0, 1, 2, 3, 4, 5, 64, 65, 66, 67, 68, 69])
    
    # Structure 5: AP with step 1
    structures.append(list(range(0, 12)))
    
    # Structure 6: Symmetric pairs around 32
    structures.append([1, 2, 3, 4, 5, 6, 26, 27, 28, 29, 30, 31])
    
    # Structure 7: Symmetric pairs around 96
    structures.append([90, 91, 92, 93, 94, 95, 97, 98, 99, 100, 101, 102])
    
    # Structure 8: Very dense small range
    structures.append(list(range(0, 12)))
    
    # Structure 9: AP step 1 starting at 50
    structures.append(list(range(50, 62)))
    
    # Structure 10: Mix - some symmetric, some clustered
    structures.append([2, 4, 6, 8, 10, 12, 56, 58, 60, 62, 64, 66])
    
    return structures

def _simulated_annealing(ambient, time_budget):
    """
    Simulated annealing to maximize sum collisions (minimize max Sidon subset size).
    """
    current = list(ambient)
    current_collisions = _count_sum_collisions(current)
    best = list(current)
    best_collisions = current_collisions
    best_sidon_size = _max_sidon_subset_size(current)
    
    # Track the best by actual Sidon size
    best_by_sidon = list(current)
    best_sidon_size = _max_sum_sidon = best_sidon_size
    
    start_time = time.time()
    deadline = start_time + time_budget
    
    # Temperature schedule
    T = 50.0
    T_min = 0.1
    cooling_rate = 0.995
    
    iteration = 0
    last_sidon_check = start_time
    sidon_check_interval = 2.0  # Check actual Sidon size every 2 seconds
    
    while time.time() < deadline:
        iteration += 1
        
        # Perturbation: pick a random element and change it
        idx = random.randint(0, 11)
        old_val = current[idx]
        # Try a random new value
        new_val = random.randint(0, 128)
        if new_val == old_val:
            continue
        
        candidate = list(current)
        candidate[idx] = new_val
        candidate.sort()
        
        # Check distinctness
        if len(set(candidate)) != 12:
            continue
        
        candidate_collisions = _count_sum_collisions(candidate)
        delta_collisions = candidate_collisions - current_collisions
        
        # Accept if better, or with probability exp(delta/T)
        if delta_collisions >= 0:
            current = candidate
            current_collisions = candidate_collisions
        else:
            import math
            if T > T_min:
                prob = math.exp(delta_collisions / T)
                if random.random() < prob:
                    current = candidate
                    current_collisions = candidate_collisions
        
        # Update best by collisions
        if current_collisions > best_collisions:
            best_collisions = current_collisions
            best = list(current)
        
        # Periodically check actual Sidon size
        if time.time() - last_sidon_check > sidon_check_interval:
            sidon_size = _max_sidon_subset_size(current)
            if sidon_size < best_sidon_size:
                best_sidon_size = sidon_size
                best_by_sidon = list(current)
            last_sidon_check = time.time()
        
        # Cool down
        T = max(T * cooling_rate, T_min)
        
        # If we've found a very good solution, tighten the search
        if best_sidon_size <= 4 and T < 5.0:
            T = max(T * 0.99, 0.01)
    
    # Final check
    final_sidon = _max_sidon_subset_size(best_by_sidon)
    if final_sidon < best_sidon_size:
        best_sidon_size = final_sidon
    
    return best_by_sidon, best_sidon_size

def _local_search(ambient, time_budget):
    """
    Local search: try swapping one element with another to reduce max Sidon subset size.
    """
    current = list(ambient)
    current_sidon = _max_sidon_subset_size(current)
    best = list(current)
    best_sidon = current_sidon
    
    start_time = time.time()
    deadline = start_time + time_budget
    
    all_values = set(range(129))
    
    while time.time() < deadline:
        improved = False
        for i in range(12):
            if time.time() > deadline:
                break
            current_set = set(current)
            for new_val in sorted(all_values - current_set):
                candidate = list(current)
                candidate[i] = new_val
                candidate.sort()
                if len(set(candidate)) != 12:
                    continue
                sidon = _max_sidon_subset_size(candidate)
                if sidon < best_sidon:
                    best = list(candidate)
                    best_sidon = sidon
                    current = list(candidate)
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    
    return best, best_sidon

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 85s deadline, 5s safety margin
    
    # Try multiple initial structures with simulated annealing
    structures = _initial_structures()
    
    best = None
    best_size = 12  # Maximum possible
    
    # First, evaluate all initial structures
    for struct in structures:
        if time.time() > deadline:
            break
        struct = struct[:12]
        if len(set(struct)) != 12:
            continue
        if any(x < 0 or x > 128 for x in struct):
            continue
        size = _max_sidon_subset_size(struct)
        if size < best_size:
            best_size = size
            best = list(struct)
    
    if best is None:
        best = list(range(12))
        best_size = _max_sidon_subset_size(best)
    
    # Run simulated annealing from the best initial structure
    remaining_time = deadline - time.time()
    if remaining_time > 5.0:
        sa_budget = min(remaining_time - 10.0, 50.0)  # Leave 10s for local search
        if sa_budget > 1.0:
            sa_result, sa_size = _simulated_annealing(best, sa_budget)
            if sa_size < best_size:
                best = sa_result
                best_size = sa_size
    
    # Also try SA from a few other promising structures
    remaining_time = deadline - time.time()
    if remaining_time > 15.0:
        for struct in structures[:4]:
            if time.time() > deadline - 10.0:
                break
            struct = struct[:12]
            if len(set(struct)) != 12:
                continue
            if any(x < 0 or x > 128 for x in struct):
                continue
            if struct == best:
                continue
            sa_budget = min(deadline - time.time() - 10.0, 15.0)
            if sa_budget > 1.0:
                sa_result, sa_size = _simulated_annealing(struct, sa_budget)
                if sa_size < best_size:
                    best = sa_result
                    best_size = sa_size
    
    # Final local search polish
    remaining_time = deadline - time.time()
    if remaining_time > 5.0:
        ls_budget = min(remaining_time - 2.0, 15.0)
        if ls_budget > 1.0:
            ls_result, ls_size = _local_search(best, ls_budget)
            if ls_size < best_size:
                best = ls_result
                best_size = ls_size
    
    # Ensure valid output
    best = sorted(set(best))
    if len(best) != 12:
        # Fallback
        best = list(range(12))
    
    return best

# EVOLVE-BLOCK-END

