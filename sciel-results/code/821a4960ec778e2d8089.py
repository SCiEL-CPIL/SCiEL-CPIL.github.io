"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

N = 256

# EVOLVE-BLOCK-START

def _sidon_sums(selected):
    """Return the set of all pair sums a+b for a<=b in selected."""
    sums = set()
    for i in range(len(selected)):
        for j in range(i, len(selected)):
            sums.add(selected[i] + selected[j])
    return sums

def _is_valid_sidon(selected):
    """Check if a set is a valid Sidon set (all pair sums distinct)."""
    sums = set()
    for i in range(len(selected)):
        for j in range(i, len(selected)):
            s = selected[i] + selected[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _is_maximal(selected, n=N):
    """Check if the Sidon set is maximal (cannot add any missing element)."""
    selected_set = set(selected)
    sums = _sidon_sums(selected)
    
    for x in range(1, n + 1):
        if x in selected_set:
            continue
        # Check if adding x would create a duplicate sum
        can_add = True
        for a in selected:
            s = x + a
            if s in sums:
                can_add = False
                break
        if can_add:
            s = 2 * x
            if s in sums:
                can_add = False
        if can_add:
            return False  # Not maximal
    return True

def _greedy_maximal_sidon(order, n=N):
    """Greedy construction given an order of elements."""
    selected = []
    sums = set()
    for x in order:
        if x < 1 or x > n:
            continue
        # Check if x can be added
        conflict = False
        for a in selected:
            s = x + a
            if s in sums:
                conflict = True
                break
        if not conflict:
            s = 2 * x
            if s in sums:
                conflict = True
        if not conflict:
            selected.append(x)
            # Add all new sums
            for a in selected:
                sums.add(x + a)
            sums.add(2 * x)
    return selected

def _try_removal(selected, n=N):
    """Try to remove elements while maintaining maximality."""
    selected = list(selected)
    improved = True
    while improved:
        improved = False
        for i in range(len(selected)):
            candidate = selected[:i] + selected[i+1:]
            if _is_maximal(candidate, n):
                selected = candidate
                improved = True
                break
    return selected

def _local_search(selected, n=N, max_iters=200):
    """Local search: try swapping elements to reduce size while maintaining maximality."""
    selected = list(selected)
    best = selected[:]
    best_len = len(selected)
    
    for _ in range(max_iters):
        improved = False
        for i in range(len(best)):
            # Try removing element i
            candidate = best[:i] + best[i+1:]
            if len(candidate) < best_len and _is_maximal(candidate, n):
                best = candidate
                best_len = len(best)
                improved = True
                break
        if not improved:
            # Try adding an element that allows a later removal
            selected_set = set(best)
            for x in range(1, n + 1):
                if x in selected_set:
                    continue
                candidate = best + [x]
                if _is_valid_sidon(candidate):
                    # Now try to remove something
                    for j in range(len(candidate)):
                        cand2 = candidate[:j] + candidate[j+1:]
                        if len(cand2) < best_len and _is_maximal(cand2, n):
                            best = cand2
                            best_len = len(cand2)
                            improved = True
                            break
                if improved:
                    break
        if not improved:
            break
    return best

def _simulated_annealing(n=N, max_time=30.0, seed=42):
    """
    Simulated annealing approach to find a small inclusion-maximal Sidon set.
    
    State: a set S of integers in [1,n]
    Objective: minimize |S| subject to S being a valid Sidon set AND maximal.
    
    Moves:
    1. Remove an element from S (if S remains valid Sidon)
    2. Add an element to S (if S remains valid Sidon)
    3. Swap: remove one element, add another (if S remains valid Sidon)
    
    We only accept states that are valid Sidon sets. Among valid Sidon sets,
    we prefer smaller ones. Maximality is checked at the end.
    """
    rng = random.Random(seed)
    start_time = time.time()
    
    # Initialize with a greedy maximal Sidon set
    order = list(range(1, n + 1))
    rng.shuffle(order)
    current = _greedy_maximal_sidon(order, n)
    current_sums = _sidon_sums(current)
    current_len = len(current)
    
    best = current[:]
    best_len = current_len
    
    # Temperature schedule
    T_start = 10.0
    T_end = 0.01
    cooling_rate = 0.999
    
    T = T_start
    
    # Precompute: for each x in [1,n], which existing elements would conflict
    # We'll use a helper to check addability quickly
    
    while T > T_end:
        if time.time() - start_time > max_time:
            break
        
        # Generate a random move
        move_type = rng.random()
        
        if move_type < 0.3:
            # Try to remove an element
            if len(current) > 1:
                idx = rng.randrange(len(current))
                candidate = current[:idx] + current[idx+1:]
                candidate_sums = _sidon_sums(candidate)
                # Always accept removals (they reduce size)
                current = candidate
                current_sums = candidate_sums
                current_len = len(current)
                if current_len < best_len:
                    best = current[:]
                    best_len = current_len
        elif move_type < 0.6:
            # Try to add an element
            missing = [x for x in range(1, n + 1) if x not in set(current)]
            if missing:
                x = rng.choice(missing)
                # Check if x can be added
                conflict = False
                for a in current:
                    if (x + a) in current_sums:
                        conflict = True
                        break
                if not conflict and (2 * x) in current_sums:
                    conflict = True
                if not conflict:
                    candidate = current + [x]
                    candidate_sums = set(current_sums)
                    for a in candidate:
                        candidate_sums.add(x + a)
                    candidate_sums.add(2 * x)
                    # Accept with probability based on size increase
                    delta = 1  # size increases by 1
                    if rng.random() < min(1.0, 1.0 / (T * 10)):
                        current = candidate
                        current_sums = candidate_sums
                        current_len = len(current)
        else:
            # Try a swap: remove one, add another
            if len(current) > 1:
                idx = rng.randrange(len(current))
                removed = current[idx]
                base = current[:idx] + current[idx+1:]
                base_sums = _sidon_sums(base)
                
                missing = [x for x in range(1, n + 1) if x not in set(base)]
                if missing:
                    x = rng.choice(missing)
                    # Check if x can be added to base
                    conflict = False
                    for a in base:
                        if (x + a) in base_sums:
                            conflict = True
                            break
                    if not conflict and (2 * x) in base_sums:
                        conflict = True
                    if not conflict:
                        candidate = base + [x]
                        candidate_sums = set(base_sums)
                        for a in base:
                            candidate_sums.add(x + a)
                        candidate_sums.add(2 * x)
                        current = candidate
                        current_sums = candidate_sums
                        current_len = len(current)
                        if current_len < best_len:
                            best = current[:]
                            best_len = current_len
        
        T *= cooling_rate
    
    return best, best_len

def _multi_restart_sa(n=N, total_time=30.0, n_restarts=5):
    """Run simulated annealing multiple times with different seeds."""
    start_time = time.time()
    best = None
    best_len = float('inf')
    
    time_per_restart = (total_time - 5) / n_restarts
    
    for i in range(n_restarts):
        if time.time() - start_time > total_time - 5:
            break
        
        remaining_time = min(time_per_restart, total_time - (time.time() - start_time))
        if remaining_time < 1.0:
            break
        
        seed = 42 + i * 137
        result, rlen = _simulated_annealing(n, max_time=remaining_time, seed=seed)
        
        if rlen < best_len:
            best_len = rlen
            best = result[:]
    
    return best, best_len

def construct():
    """Construct a small inclusion-maximal Sidon set in [1,256]."""
    start_time = time.time()
    deadline = start_time + 55.0  # 55 second safety margin
    
    best = None
    best_len = float('inf')
    
    # Phase 1: Greedy with multiple random orderings (fast, gives good starting point)
    random.seed(42)
    
    for attempt in range(100):
        if time.time() > deadline:
            break
        
        order = list(range(1, N + 1))
        random.shuffle(order)
        
        selected = _greedy_maximal_sidon(order, N)
        
        if len(selected) < best_len:
            best_len = len(selected)
            best = selected[:]
    
    # Phase 2: Try removal on best greedy result
    if time.time() < deadline - 20 and best is not None:
        improved = _try_removal(best, N)
        if len(improved) < best_len:
            best_len = len(improved)
            best = improved[:]
    
    # Phase 3: Simulated annealing from multiple starting points
    if time.time() < deadline - 10:
        sa_time = min(25.0, deadline - time.time() - 5)
        sa_result, sa_len = _multi_restart_sa(N, total_time=sa_time, n_restarts=5)
        if sa_len < best_len:
            best_len = sa_len
            best = sa_result[:]
    
    # Phase 4: Final local search refinement
    if time.time() < deadline - 3 and best is not None:
        improved = _try_removal(best, N)
        if len(improved) < best_len:
            best_len = len(improved)
            best = improved[:]
        
        # One more local search pass
        if time.time() < deadline - 1:
            improved = _local_search(best, N, max_iters=30)
            if len(improved) < best_len:
                best_len = len(improved)
                best = improved[:]
    
    return best if best is not None else [1, 2]

# EVOLVE-BLOCK-END
