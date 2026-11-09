import time
import itertools
import random

_LCM = 232792560

def _get_sums(subset):
    """Calculate all subset sums of the scaled reciprocals."""
    scaled = [_LCM // x for x in subset]
    sums = {0}
    for v in scaled:
        sums |= {s + v for s in sums}
    return sums

def _is_dissociated(subset):
    """Check if the set is dissociated."""
    sums = _get_sums(subset)
    return len(sums) == (1 << len(subset))

def _min_sum_difference(subset):
    """
    Compute the minimum absolute difference between any two distinct subset sums.
    Returns infinity if the set is dissociated (all sums distinct).
    Returns 0 if there's a collision.
    """
    sums = sorted(_get_sums(subset))
    min_diff = float('inf')
    for i in range(1, len(sums)):
        diff = sums[i] - sums[i-1]
        if diff == 0:
            return 0
        if diff < min_diff:
            min_diff = diff
    return min_diff

def _try_add_to_sums(sums, subset):
    """
    Try to add all elements of subset to the existing sums set.
    Returns (new_sums, success). If success is False, a collision was found
    and new_sums is the original sums (unchanged).
    """
    current = set(sums)
    for x in subset:
        v = _LCM // x
        new_entries = []
        for s in current:
            ns = s + v
            if ns in current:
                return sums, False
            new_entries.append(ns)
        current |= set(new_entries)
    return current, True

def _greedy_construct():
    """Greedy construction: Add largest numbers first."""
    universe = list(range(1, 23))
    S = []
    sums = {0}
    
    candidates = sorted(universe, reverse=True)
    
    for x in candidates:
        v = _LCM // x
        is_valid = True
        for s in sums:
            if (s + v) in sums:
                is_valid = False
                break
        
        if is_valid:
            S.append(x)
            sums |= {s + v for s in sums}
    
    return S

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    universe = list(range(1, 23))
    
    # Phase 1: Greedy construction
    best_S = _greedy_construct()
    best_size = len(best_S)
    best_margin = _min_sum_difference(best_S)
    
    # Phase 2: Simulated Annealing local search
    # State: current set S, current margin (min pairwise sum difference)
    # Moves: 
    #   - Add one element (if valid, increases size)
    #   - Remove one element (decreases size, but may enable future additions)
    #   - Swap: remove one, add one (same size, different composition)
    #   - Remove one, add two (size +1)
    #   - Remove two, add one (size -1)
    #   - Remove two, add three (size +1)
    
    current_S = list(best_S)
    current_margin = best_margin
    
    # Temperature schedule
    T_start = 1000.0
    T_end = 1.0
    cooling_rate = 0.995
    
    T = T_start
    no_improve_count = 0
    max_no_improve = 5000
    
    while time.time() < deadline - 10:
        if no_improve_count > max_no_improve:
            break
        
        # Generate a random move
        in_set = set(current_S)
        not_in_set = [x for x in universe if x not in in_set]
        
        move_type = random.random()
        
        candidate_S = None
        
        if move_type < 0.2 and not_in_set:
            # Add one element
            x = random.choice(not_in_set)
            candidate_S = current_S + [x]
        elif move_type < 0.35 and current_S:
            # Remove one element
            a = random.choice(current_S)
            candidate_S = [x for x in current_S if x != a]
        elif move_type < 0.55 and current_S and not_in_set:
            # Swap one for one
            a = random.choice(current_S)
            x = random.choice(not_in_set)
            candidate_S = [z for z in current_S if z != a] + [x]
        elif move_type < 0.75 and current_S and len(not_in_set) >= 2:
            # Remove one, add two
            a = random.choice(current_S)
            x, y = random.sample(not_in_set, 2)
            candidate_S = [z for z in current_S if z != a] + [x, y]
        elif move_type < 0.9 and len(current_S) >= 2 and not_in_set:
            # Remove two, add one
            a, b = random.sample(current_S, 2)
            x = random.choice(not_in_set)
            candidate_S = [z for z in current_S if z != a and z != b] + [x]
        else:
            # Remove two, add three
            if len(current_S) >= 2 and len(not_in_set) >= 3:
                a, b = random.sample(current_S, 2)
                x, y, z = random.sample(not_in_set, 3)
                candidate_S = [w for w in current_S if w != a and w != b] + [x, y, z]
            elif current_S and not_in_set:
                # Fallback: swap
                a = random.choice(current_S)
                x = random.choice(not_in_set)
                candidate_S = [z for z in current_S if z != a] + [x]
            else:
                continue
        
        if candidate_S is None or not candidate_S:
            continue
        
        # Evaluate candidate
        candidate_margin = _min_sum_difference(candidate_S)
        
        # Acceptance criterion
        size_change = len(candidate_S) - len(current_S)
        
        if size_change > 0 and candidate_margin > 0:
            # Always accept size-increasing moves that are valid
            current_S = candidate_S
            current_margin = candidate_margin
            if len(current_S) > best_size:
                best_S = list(current_S)
                best_size = len(current_S)
                best_margin = current_margin
                no_improve_count = 0
            else:
                no_improve_count += 1
        elif size_change == 0:
            # Same size: accept if margin improves, or with probability based on temperature
            if candidate_margin > current_margin:
                current_S = candidate_S
                current_margin = candidate_margin
                no_improve_count = 0
            elif candidate_margin > 0:
                # Accept with probability exp(-delta/T) where delta is margin decrease
                delta = current_margin - candidate_margin
                if delta == 0 or random.random() < pow(2.718281828, -delta / max(T, 1e-6)):
                    current_S = candidate_S
                    current_margin = candidate_margin
            no_improve_count += 1
        else:
            # Size decreasing: accept with low probability to escape local optima
            if candidate_margin > 0:
                prob = 0.05 * pow(2.718281828, -1.0 / max(T, 1e-6))
                if random.random() < prob:
                    current_S = candidate_S
                    current_margin = candidate_margin
            no_improve_count += 1
        
        # Cool down
        T = max(T * cooling_rate, T_end)
    
    # Phase 3: Final local search to try to increase size
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        in_set = set(best_S)
        not_in_set = [x for x in universe if x not in in_set]
        
        # Try 1-for-2 moves
        for a in best_S:
            if time.time() > deadline - 5:
                break
            for x, y in itertools.combinations(not_in_set, 2):
                candidate = [z for z in best_S if z != a] + [x, y]
                if _is_dissociated(candidate):
                    best_S = candidate
                    best_size += 1
                    improved = True
                    break
            if improved:
                break
        
        # Try 2-for-3 moves
        if not improved:
            in_set = set(best_S)
            not_in_set = [x for x in universe if x not in in_set]
            for a, b in itertools.combinations(best_S, 2):
                if time.time() > deadline - 5:
                    break
                for x, y, z in itertools.combinations(not_in_set, 3):
                    candidate = [c for c in best_S if c != a and c != b] + [x, y, z]
                    if _is_dissociated(candidate):
                        best_S = candidate
                        best_size += 1
                        improved = True
                        break
                if improved:
                    break
    
    return best_S
