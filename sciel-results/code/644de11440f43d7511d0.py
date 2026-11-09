"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math

N = 256

# EVOLVE-BLOCK-START

def _sidon_sums(selected):
    """Compute all pair sums (including diagonals) for a Sidon candidate."""
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

def _is_maximal_fast(selected, n=N):
    """Fast check if Sidon set is maximal using O(N*S) approach."""
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
            if 2 * x in sums:
                can_add = False
        if can_add:
            return False  # Not maximal
    return True

def _count_addable(selected, n=N):
    """Count how many elements can be added (non-maximality measure)."""
    selected_set = set(selected)
    sums = _sidon_sums(selected)
    count = 0
    for x in range(1, n + 1):
        if x in selected_set:
            continue
        can_add = True
        for a in selected:
            s = x + a
            if s in sums:
                can_add = False
                break
        if can_add:
            if 2 * x in sums:
                can_add = False
        if can_add:
            count += 1
    return count

def _greedy_valid_sidon(order, n=N, target_size=None):
    """Greedy construction of a valid Sidon set (not necessarily maximal)."""
    selected = []
    sums = set()
    for x in order:
        if x < 1 or x > n:
            continue
        if target_size is not None and len(selected) >= target_size:
            break
        conflict = False
        for a in selected:
            s = x + a
            if s in sums:
                conflict = True
                break
        if not conflict:
            if 2 * x in sums:
                conflict = True
        if not conflict:
            selected.append(x)
            for a in selected:
                sums.add(x + a)
            sums.add(2 * x)
    return selected

def _greedy_maximal_from_valid(valid_set, n=N):
    """Given a valid Sidon set, try to make it maximal by adding elements."""
    selected = list(valid_set)
    selected_set = set(selected)
    sums = _sidon_sums(selected)
    
    # Add elements to make it maximal
    changed = True
    while changed:
        changed = False
        for x in range(1, n + 1):
            if x in selected_set:
                continue
            can_add = True
            for a in selected:
                s = x + a
                if s in sums:
                    can_add = False
                    break
            if can_add:
                if 2 * x in sums:
                    can_add = False
            if can_add:
                selected.append(x)
                selected_set.add(x)
                for a in selected:
                    sums.add(x + a)
                sums.add(2 * x)
                changed = True
                break
    
    return selected

def _local_search_reduce(best, n=N, time_budget=2.0, start_time=None):
    """Local search: try removing elements in random order to minimize size while staying maximal."""
    if start_time is None:
        start_time = time.time()
    deadline = start_time + time_budget
    
    current = list(best)
    current_len = len(current)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try removals in random order
        indices = list(range(len(current)))
        random.shuffle(indices)
        for i in indices:
            if time.time() > deadline:
                break
            candidate = current[:i] + current[i+1:]
            if _is_maximal_fast(candidate, n):
                current = candidate
                current_len -= 1
                improved = True
                break
    
    return current

def _simulated_annealing(initial_set, n=N, max_iters=5000, t0=1.0, t_min=0.01, cooling=0.9995, deadline=None):
    """
    Simulated annealing to find small maximal Sidon sets.
    """
    current = list(initial_set)
    current_set = set(current)
    current_sums = _sidon_sums(current)
    current_size = len(current)
    current_violations = _count_addable(current, n)
    current_score = current_size + 10 * current_violations
    
    best = list(current)
    best_size = current_size
    best_score = current_score
    
    temperature = t0
    rng = random.Random(42)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        if temperature < t_min:
            break
        
        move_type = rng.random()
        
        new_set = None
        new_set_set = None
        new_sums = None
        new_size = None
        new_violations = None
        new_score = None
        
        if move_type < 0.4:
            # Try removing a random element
            if current_size > 1:
                idx = rng.randrange(current_size)
                new_set = current[:idx] + current[idx+1:]
                new_set_set = set(new_set)
                new_sums = _sidon_sums(new_set)
                new_size = current_size - 1
                new_violations = _count_addable(new_set, n)
                new_score = new_size + 10 * new_violations
                
        elif move_type < 0.7:
            # Try adding a random element
            candidates = [x for x in range(1, n + 1) if x not in current_set]
            if candidates:
                x = rng.choice(candidates)
                valid = True
                for a in current:
                    s = x + a
                    if s in current_sums:
                        valid = False
                        break
                if valid:
                    if 2 * x in current_sums:
                        valid = False
                if valid:
                    new_set = current + [x]
                    new_set_set = set(new_set)
                    new_sums = _sidon_sums(new_set)
                    new_size = current_size + 1
                    new_violations = _count_addable(new_set, n)
                    new_score = new_size + 10 * new_violations
                    
        else:
            # Try swap: remove one, add another
            if current_size > 0:
                candidates = [x for x in range(1, n + 1) if x not in current_set]
                if candidates:
                    idx = rng.randrange(current_size)
                    x = rng.choice(candidates)
                    temp_set = current[:idx] + current[idx+1:]
                    temp_sums = _sidon_sums(temp_set)
                    valid = True
                    for a in temp_set:
                        s = x + a
                        if s in temp_sums:
                            valid = False
                            break
                    if valid:
                        if 2 * x in temp_sums:
                            valid = False
                    if valid:
                        new_set = temp_set + [x]
                        new_set_set = set(new_set)
                        new_sums = _sidon_sums(new_set)
                        new_size = current_size
                        new_violations = _count_addable(new_set, n)
                        new_score = new_size + 10 * new_violations
        
        if new_set is None:
            continue
        
        delta = new_score - current_score
        if delta < 0:
            current = new_set
            current_set = new_set_set
            current_sums = new_sums
            current_size = new_size
            current_violations = new_violations
            current_score = new_score
            
            if current_score < best_score or (current_score == best_score and current_size < best_size):
                best = list(current)
                best_size = current_size
                best_score = current_score
        else:
            if delta == 0:
                accept = True
            else:
                accept_prob = math.exp(-delta / temperature)
                accept = rng.random() < accept_prob
            if accept:
                current = new_set
                current_set = new_set_set
                current_sums = new_sums
                current_size = new_size
                current_violations = new_violations
                current_score = new_score
        
        temperature *= cooling
    
    return best

def construct():
    """Construct a small inclusion-maximal Sidon set in [1,256]."""
    start_time = time.time()
    deadline = start_time + 25.0  # 25 second safety margin (well within 30s)
    
    best = None
    best_len = float('inf')
    
    random.seed(42)
    
    # Generate multiple initial valid Sidon sets
    initial_sets = []
    
    # Strategy 1: Greedy with various orderings
    for attempt in range(50):
        if time.time() > deadline - 15:
            break
        
        order = list(range(1, N + 1))
        random.shuffle(order)
        
        target_size = random.randint(10, 35)
        valid_set = _greedy_valid_sidon(order, N, target_size)
        
        if len(valid_set) >= 5:
            initial_sets.append(valid_set)
    
    # Strategy 2: Structured initializations
    for step in [5, 7, 9, 11, 13, 17, 19, 23]:
        if time.time() > deadline - 15:
            break
        for base in range(1, min(step + 1, 10)):
            if time.time() > deadline - 15:
                break
            valid_set = []
            sums = set()
            for x in range(base, N + 1, step):
                conflict = False
                for a in valid_set:
                    s = x + a
                    if s in sums:
                        conflict = True
                        break
                if not conflict:
                    if 2 * x in sums:
                        conflict = True
                if not conflict:
                    valid_set.append(x)
                    for a in valid_set:
                        sums.add(x + a)
                    sums.add(2 * x)
            if len(valid_set) >= 5:
                initial_sets.append(valid_set)
    
    # Process each initial set
    for init in initial_sets:
        if time.time() > deadline - 5:
            break
        
        # Make maximal first
        maximal_set = _greedy_maximal_from_valid(init, N)
        
        if len(maximal_set) < best_len:
            best_len = len(maximal_set)
            best = maximal_set[:]
        
        # Run SA to improve
        if time.time() < deadline - 3:
            sa_result = _simulated_annealing(maximal_set, N, max_iters=2000, t0=1.0, t_min=0.01, cooling=0.9997, deadline=deadline - 3)
            if len(sa_result) < best_len:
                best_len = len(sa_result)
                best = sa_result[:]
    
    # Final local search on best
    if best is not None and time.time() < deadline - 2:
        best = _local_search_reduce(best, N, time_budget=min(2.0, deadline - time.time() - 1.0), start_time=start_time)
        best_len = len(best)
    
    # Ensure we have a valid result
    if best is None or not _is_valid_sidon(best) or not _is_maximal_fast(best, N):
        # Fallback: greedy maximal
        order = list(range(1, N + 1))
        random.shuffle(order)
        best = _greedy_maximal_from_valid(_greedy_valid_sidon(order, N), N)
    
    return best

# EVOLVE-BLOCK-END
