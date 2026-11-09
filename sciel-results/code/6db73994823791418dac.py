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

def _is_maximal(selected, n=N):
    """Check if the Sidon set is maximal (cannot add any missing element)."""
    selected_set = set(selected)
    sums = _sidon_sums(selected)
    
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
            s = 2 * x
            if s in sums:
                can_add = False
        if can_add:
            return False
    return True

def _count_addable(selected, n=N):
    """Count how many missing elements can be added to the Sidon set."""
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
            s = 2 * x
            if s in sums:
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
            s = 2 * x
            if s in sums:
                conflict = True
        if not conflict:
            selected.append(x)
            for a in selected:
                sums.add(x + a)
            sums.add(2 * x)
    return selected

def _make_maximal(selected, n=N):
    """Add elements to a valid Sidon set until it becomes maximal."""
    selected = list(selected)
    selected_set = set(selected)
    sums = _sidon_sums(selected)
    
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
                s = 2 * x
                if s in sums:
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

def _try_remove_element(selected, idx, n=N):
    """
    Try removing element at idx. Returns (new_set, new_sums) if valid and maximal, else None.
    """
    if len(selected) <= 1:
        return None
    
    new_set = selected[:idx] + selected[idx+1:]
    new_set_set = set(new_set)
    
    # Check if still valid Sidon
    sums = set()
    for i in range(len(new_set)):
        for j in range(i, len(new_set)):
            s = new_set[i] + new_set[j]
            if s in sums:
                return None
            sums.add(s)
    
    # Check if still maximal: no missing element can be added
    for x in range(1, n + 1):
        if x in new_set_set:
            continue
        can_add = True
        for a in new_set:
            s = x + a
            if s in sums:
                can_add = False
                break
        if can_add:
            s = 2 * x
            if s in sums:
                can_add = False
        if can_add:
            return None  # Not maximal
    
    return (new_set, sums)

def _local_search_minimize(selected, n=N, deadline=None):
    """
    Local search: repeatedly try to remove elements while maintaining
    both Sidon validity and maximality.
    """
    selected = list(selected)
    improved = True
    while improved:
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        for i in range(len(selected)):
            result = _try_remove_element(selected, i, n)
            if result is not None:
                selected = result[0]
                improved = True
                break
    return selected

def _simulated_annealing_minimize(initial_set, n=N, deadline=None):
    """
    SA that works on maximal Sidon sets.
    Moves:
      - Remove an element (if result is still valid Sidon, then make maximal again, check size)
      - Swap: remove one, add another (keep same size, then make maximal)
    
    Objective: minimize |S|
    """
    current = list(initial_set)
    current_size = len(current)
    
    best = list(current)
    best_size = current_size
    
    temperature = 2.0
    t_min = 0.1
    cooling = 0.999
    rng = random.Random(123)
    
    iteration = 0
    max_iters = 20000
    
    while iteration < max_iters:
        if deadline is not None and time.time() > deadline:
            break
        
        iteration += 1
        move_type = rng.random()
        
        new_candidate = None
        
        if move_type < 0.5 and current_size > 1:
            # Try removing a random element
            idx = rng.randrange(current_size)
            new_set = current[:idx] + current[idx+1:]
            
            # Check validity
            valid = True
            sums = set()
            for i in range(len(new_set)):
                for j in range(i, len(new_set)):
                    s = new_set[i] + new_set[j]
                    if s in sums:
                        valid = False
                        break
                    sums.add(s)
                if not valid:
                    break
            
            if valid:
                # Make maximal
                new_set = _make_maximal(new_set, n)
                new_candidate = new_set
        
        elif move_type < 0.8:
            # Swap: remove one, add another
            if current_size > 0:
                idx = rng.randrange(current_size)
                removed = current[idx]
                candidates = [x for x in range(1, n + 1) if x not in set(current)]
                if candidates:
                    x = rng.choice(candidates)
                    temp_set = current[:idx] + current[idx+1:]
                    
                    # Check if adding x to temp_set is valid
                    valid = True
                    sums = set()
                    for i in range(len(temp_set)):
                        for j in range(i, len(temp_set)):
                            s = temp_set[i] + temp_set[j]
                            if s in sums:
                                valid = False
                                break
                            sums.add(s)
                        if not valid:
                            break
                    
                    if valid:
                        # Check x doesn't conflict
                        for a in temp_set:
                            s = x + a
                            if s in sums:
                                valid = False
                                break
                        if valid:
                            s = 2 * x
                            if s in sums:
                                valid = False
                    
                    if valid:
                        new_set = temp_set + [x]
                        new_set = _make_maximal(new_set, n)
                        new_candidate = new_set
        
        if new_candidate is None:
            temperature *= cooling
            continue
        
        new_size = len(new_candidate)
        delta = new_size - current_size
        
        if delta <= 0:
            current = new_candidate
            current_size = new_size
            if new_size < best_size:
                best = list(new_candidate)
                best_size = new_size
        else:
            # Accept with probability exp(-delta / temperature)
            if delta == 0:
                accept = True
            else:
                accept_prob = math.exp(-delta / temperature)
                accept = rng.random() < accept_prob
            if accept:
                current = new_candidate
                current_size = new_size
        
        temperature *= cooling
    
    return best

def construct():
    """Construct a small inclusion-maximal Sidon set in [1,256]."""
    start_time = time.time()
    deadline = start_time + 57.0  # 3s safety margin
    
    best = None
    best_len = float('inf')
    
    random.seed(42)
    
    # Generate initial valid Sidon sets and make them maximal
    initial_maximal_sets = []
    
    # Strategy 1: Greedy with various orderings
    for attempt in range(50):
        if time.time() > deadline - 15:
            break
        
        order = list(range(1, N + 1))
        random.shuffle(order)
        
        target_size = random.randint(10, 35)
        valid_set = _greedy_valid_sidon(order, N, target_size)
        
        if len(valid_set) >= 3:
            maximal_set = _make_maximal(valid_set, N)
            initial_maximal_sets.append(maximal_set)
    
    # Strategy 2: Structured initializations (arithmetic progressions)
    for step in [5, 7, 9, 11, 13, 17, 19, 23]:
        if time.time() > deadline - 15:
            break
        for base in range(1, min(step, 5) + 1):
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
                    s = 2 * x
                    if s in sums:
                        conflict = True
                if not conflict:
                    valid_set.append(x)
                    for a in valid_set:
                        sums.add(x + a)
                    sums.add(2 * x)
            if len(valid_set) >= 3:
                maximal_set = _make_maximal(valid_set, N)
                initial_maximal_sets.append(maximal_set)
    
    # Run SA and local search on each initial maximal set
    for init in initial_maximal_sets:
        if time.time() > deadline - 3:
            break
        
        current_size = len(init)
        
        if current_size < best_len:
            best_len = current_size
            best = init[:]
        
        # Run SA to try to minimize
        if time.time() < deadline - 2:
            sa_result = _simulated_annealing_minimize(init, N, deadline)
            if len(sa_result) < best_len:
                best_len = len(sa_result)
                best = sa_result[:]
        
        # Local search polish
        if best is not None and time.time() < deadline - 1:
            polished = _local_search_minimize(best, N, deadline)
            if len(polished) < best_len:
                best_len = len(polished)
                best = polished[:]
    
    # Final polish on the best
    if best is not None and time.time() < deadline - 0.5:
        final = _local_search_minimize(best, N, deadline)
        if len(final) < best_len:
            best = final
    
    # Ensure valid result
    if best is None or not _is_valid_sidon(best) or not _is_maximal(best, N):
        # Fallback
        order = list(range(1, N + 1))
        random.shuffle(order)
        best = _make_maximal(_greedy_valid_sidon(order, N), N)
    
    return best

# EVOLVE-BLOCK-END
