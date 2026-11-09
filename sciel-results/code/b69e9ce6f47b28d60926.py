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

def _count_violations(selected, n=N):
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

def _simulated_annealing(initial_set, n=N, max_iters=5000, t0=1.0, t_min=0.01, cooling=0.9995):
    """
    Simulated annealing to find small maximal Sidon sets.
    
    State: a valid Sidon set
    Moves: 
      - Remove a random element (if result is still valid Sidon)
      - Add a random element (if result is valid Sidon)
      - Swap: remove one, add another (if result is valid Sidon)
    
    Objective: minimize size, then minimize non-maximality
    Score function: size + penalty * (number of addable elements)
    """
    current = list(initial_set)
    current_set = set(current)
    current_sums = _sidon_sums(current)
    current_size = len(current)
    current_violations = _count_violations(current, n)
    current_score = current_size + 10 * current_violations
    
    best = list(current)
    best_size = current_size
    best_violations = current_violations
    best_score = current_score
    
    temperature = t0
    rng = random.Random(42)
    
    for it in range(max_iters):
        if temperature < t_min:
            break
        
        # Choose a random move
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
                removed = current[idx]
                new_set = current[:idx] + current[idx+1:]
                new_set_set = set(new_set)
                new_sums = _sidon_sums(new_set)
                new_size = current_size - 1
                # Check validity (should be valid if current is valid)
                new_violations = _count_violations(new_set, n)
                new_score = new_size + 10 * new_violations
                
        elif move_type < 0.7:
            # Try adding a random element
            candidates = [x for x in range(1, n + 1) if x not in current_set]
            if candidates:
                x = rng.choice(candidates)
                # Check if adding x keeps Sidon property
                valid = True
                for a in current:
                    s = x + a
                    if s in current_sums:
                        valid = False
                        break
                if valid:
                    s = 2 * x
                    if s in current_sums:
                        valid = False
                if valid:
                    new_set = current + [x]
                    new_set_set = set(new_set)
                    new_sums = _sidon_sums(new_set)
                    new_size = current_size + 1
                    new_violations = _count_violations(new_set, n)
                    new_score = new_size + 10 * new_violations
                    
        else:
            # Try swap: remove one, add another
            if current_size > 0:
                candidates = [x for x in range(1, n + 1) if x not in current_set]
                if candidates:
                    idx = rng.randrange(current_size)
                    removed = current[idx]
                    x = rng.choice(candidates)
                    # Check if swap keeps Sidon property
                    temp_set = current[:idx] + current[idx+1:]
                    temp_set_set = set(temp_set)
                    temp_sums = _sidon_sums(temp_set)
                    valid = True
                    for a in temp_set:
                        s = x + a
                        if s in temp_sums:
                            valid = False
                            break
                    if valid:
                        s = 2 * x
                        if s in temp_sums:
                            valid = False
                    if valid:
                        new_set = temp_set + [x]
                        new_set_set = set(new_set)
                        new_sums = _sidon_sums(new_set)
                        new_size = current_size
                        new_violations = _count_violations(new_set, n)
                        new_score = new_size + 10 * new_violations
        
        if new_set is None:
            continue
        
        # Accept or reject based on Metropolis criterion
        delta = new_score - current_score
        if delta < 0:
            # Better, always accept
            current = new_set
            current_set = new_set_set
            current_sums = new_sums
            current_size = new_size
            current_violations = new_violations
            current_score = new_score
            
            # Update best
            if current_score < best_score or (current_score == best_score and current_size < best_size):
                best = list(current)
                best_size = current_size
                best_violations = current_violations
                best_score = current_score
        else:
            # Worse, accept with probability exp(-delta/temperature)
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
        
        # Cool down
        temperature *= cooling
    
    return best

def _greedy_maximal_from_valid(valid_set, n=N):
    """Given a valid Sidon set, try to make it maximal by adding elements, then minimize."""
    selected = list(valid_set)
    selected_set = set(selected)
    sums = _sidon_sums(selected)
    
    # First, add elements to make it maximal
    changed = True
    while changed:
        changed = False
        for x in range(1, n + 1):
            if x in selected_set:
                continue
            # Check if x can be added
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

def construct():
    """Construct a small inclusion-maximal Sidon set in [1,256] using simulated annealing."""
    start_time = time.time()
    deadline = start_time + 55.0  # 55 second safety margin
    
    best = None
    best_len = float('inf')
    
    random.seed(42)
    
    # Generate multiple initial valid Sidon sets of different sizes
    initial_sets = []
    
    # Strategy 1: Greedy with various orderings to get valid Sidon sets
    for attempt in range(30):
        if time.time() > deadline - 20:
            break
        
        order = list(range(1, N + 1))
        random.shuffle(order)
        
        # Try different target sizes
        target_size = random.randint(15, 40)
        valid_set = _greedy_valid_sidon(order, N, target_size)
        
        if len(valid_set) >= 5:
            initial_sets.append(valid_set)
    
    # Also try some structured initializations
    # Arithmetic progression based
    for step in [7, 11, 13, 17]:
        if time.time() > deadline - 20:
            break
        base = random.randint(1, step)
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
        if len(valid_set) >= 5:
            initial_sets.append(valid_set)
    
    # Run simulated annealing on each initial set
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
            sa_result = _simulated_annealing(maximal_set, N, max_iters=3000, t0=1.0, t_min=0.01, cooling=0.9997)
            if len(sa_result) < best_len:
                best_len = len(sa_result)
                best = sa_result[:]
    
    # Final local search on best
    if best is not None and time.time() < deadline - 2:
        # Try removing elements one by one
        improved = True
        while improved and time.time() < deadline - 1:
            improved = False
            for i in range(len(best)):
                candidate = best[:i] + best[i+1:]
                if _is_maximal(candidate, N):
                    best = candidate
                    improved = True
                    break
    
    # Ensure we have a valid result
    if best is None or not _is_valid_sidon(best) or not _is_maximal(best, N):
        # Fallback: greedy maximal
        order = list(range(1, N + 1))
        random.shuffle(order)
        best = _greedy_maximal_from_valid(_greedy_valid_sidon(order, N), N)
    
    return best

# EVOLVE-BLOCK-END
