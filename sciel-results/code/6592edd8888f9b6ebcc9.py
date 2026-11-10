import time
import random
import numpy as np

def _compute_counts(chosen):
    """Compute sum counts for a sorted list of chosen numbers."""
    counts = {}
    n = len(chosen)
    for i in range(n):
        a = chosen[i]
        for j in range(i, n):
            b = chosen[j]
            s = a + b
            counts[s] = counts.get(s, 0) + 1
    return counts

def _is_valid_addition(chosen, x, counts):
    """Check if adding x to chosen keeps all sum counts <= 2."""
    # New sums are x + y for y in chosen, plus 2*x
    # We need counts.get(x+y, 0) < 2 for all y in chosen, and counts.get(2*x, 0) < 2
    if counts.get(2 * x, 0) >= 2:
        return False
    for y in chosen:
        if counts.get(x + y, 0) >= 2:
            return False
    return True

def _greedy_build(order, counts_init=None, chosen_init=None):
    """Build a set greedily in the given order."""
    if chosen_init is None:
        chosen = []
    else:
        chosen = list(chosen_init)
    if counts_init is None:
        counts = {}
    else:
        counts = dict(counts_init)
    
    for x in order:
        if x in set(chosen):
            continue
        if _is_valid_addition(chosen, x, counts):
            chosen.append(x)
            # Update counts
            counts[2 * x] = counts.get(2 * x, 0) + 1
            for y in chosen[:-1]:  # y is all previously chosen
                s = x + y
                counts[s] = counts.get(s, 0) + 1
    return chosen, counts

def _random_greedy(rng, limit_time):
    """Randomized greedy with multiple restarts."""
    best_set = []
    best_counts = {}
    candidates = list(range(1, 1001))
    
    while time.time() < limit_time:
        rng.shuffle(candidates)
        chosen, counts = _greedy_build(candidates)
        if len(chosen) > len(best_set):
            best_set = chosen
            best_counts = counts
    
    return best_set, best_counts

def _local_search_improve(chosen, counts, rng, limit_time):
    """
    Local search: try to improve by removing one element and greedily adding others.
    """
    chosen = list(chosen)
    counts = dict(counts)
    n_original = len(chosen)
    
    improved = True
    while improved and time.time() < limit_time:
        improved = False
        candidates_to_remove = list(chosen)
        rng.shuffle(candidates_to_remove)
        
        for remove_x in candidates_to_remove:
            if time.time() >= limit_time:
                break
            
            # Remove x
            chosen_new = [y for y in chosen if y != remove_x]
            chosen_set = set(chosen_new)
            
            # Recompute counts
            counts_new = {}
            for i in range(len(chosen_new)):
                a = chosen_new[i]
                for j in range(i, len(chosen_new)):
                    b = chosen_new[j]
                    s = a + b
                    counts_new[s] = counts_new.get(s, 0) + 1
            
            # Try to add available elements
            available = [x for x in range(1, 1001) if x not in chosen_set]
            rng.shuffle(available)
            
            for add_x in available:
                if time.time() >= limit_time:
                    break
                if _is_valid_addition(chosen_new, add_x, counts_new):
                    chosen_new.append(add_x)
                    chosen_set.add(add_x)
                    counts_new[2 * add_x] = counts_new.get(2 * add_x, 0) + 1
                    for y in chosen_new[:-1]:
                        s = add_x + y
                        counts_new[s] = counts_new.get(s, 0) + 1
            
            if len(chosen_new) > len(chosen):
                chosen = chosen_new
                counts = counts_new
                improved = True
                break
    
    return chosen, counts

def _simulated_annealing(rng, limit_time):
    """
    Simulated annealing on the bitset representation.
    Energy = -size + lambda * violations (where violations = sum of max(0, count-2) for each sum)
    We want to minimize energy, so maximize size while keeping violations = 0.
    
    Actually, let's use: energy = violations * large_penalty - size
    We want energy to be negative and large in magnitude (i.e., -size when violations=0).
    """
    N = 1000
    x = np.zeros(N, dtype=np.int8)  # 1 if selected
    best_x = x.copy()
    best_energy = np.inf
    
    # Start with greedy
    chosen, _ = _greedy_build(range(1, 1001))
    for c in chosen:
        x[c-1] = 1
    
    def get_energy(vec):
        """Compute energy: penalty for violations - size."""
        # Get selected elements
        selected = np.where(vec == 1)[0] + 1
        size = len(selected)
        
        if size == 0:
            return 0.0
        
        # Compute sum counts using numpy convolution-like approach
        # For each pair (a, b) with a <= b, add to count of a+b
        # Use a 2D approach or just iterate
        counts = np.zeros(2001, dtype=np.int32)
        for i in range(size):
            a = selected[i]
            for j in range(i, size):
                b = selected[j]
                s = a + b
                counts[s] += 1
        
        # Violations: sum of max(0, count - 2)
        violations = np.sum(np.maximum(counts - 2, 0))
        
        # Energy: we want to minimize this
        # If violations > 0, energy should be large positive
        # If violations == 0, energy = -size (lower is better)
        if violations > 0:
            energy = violations * 100.0 - size
        else:
            energy = -float(size)
        
        return energy
    
    current_energy = get_energy(x)
    if current_energy < best_energy:
        best_energy = current_energy
        best_x = x.copy()
    
    T = 50.0
    T_min = 0.1
    cooling_rate = 0.999
    
    while time.time() < limit_time:
        # Pick a random position to flip
        i = rng.randint(0, N - 1)
        x[i] = 1 - x[i]
        
        new_energy = get_energy(x)
        delta = new_energy - current_energy
        
        if delta < 0:
            current_energy = new_energy
            if new_energy < best_energy:
                best_energy = new_energy
                best_x = x.copy()
        else:
            # Accept with probability exp(-delta / T)
            if T > T_min and rng.random() < np.exp(-delta / T):
                current_energy = new_energy
            else:
                x[i] = 1 - x[i]  # Revert
        
        T *= cooling_rate
        if T < T_min:
            T = T_min
    
    # Extract result from best_x
    result = [i + 1 for i in range(N) if best_x[i] == 1]
    
    # Verify validity
    counts = {}
    for i in range(len(result)):
        a = result[i]
        for j in range(i, len(result)):
            b = result[j]
            s = a + b
            counts[s] = counts.get(s, 0) + 1
    
    for s, c in counts.items():
        if c > 2:
            # Invalid, return empty or fallback
            return []
    
    return result

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    rng = random.Random(42)
    
    # Phase 1: Randomized greedy to get a good starting point
    best_set, best_counts = _random_greedy(rng, start_time + 40.0)
    
    # Phase 2: Local search improvement
    if time.time() < deadline - 50.0:
        best_set, best_counts = _local_search_improve(best_set, best_counts, rng, deadline - 50.0)
    
    # Phase 3: Simulated annealing if time permits
    if time.time() < deadline - 20.0:
        sa_result = _simulated_annealing(rng, deadline - 20.0)
        if len(sa_result) > len(best_set):
            best_set = sa_result
    
    return best_set
