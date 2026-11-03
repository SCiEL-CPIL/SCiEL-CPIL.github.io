import time
import math
import random

def _score(U):
    """Compute the score for a given set U (as a sorted list)."""
    U_set = set(U)
    if 0 not in U_set:
        return 0.0
        
    n = len(U_set)
    if n == 0:
        return 0.0
    max_u = max(U_set)
    if max_u == 0:
        return 1.0
    
    # Compute U+U
    sum_set = set()
    for a in U_set:
        for b in U_set:
            sum_set.add(a + b)
            
    # Compute U-U
    diff_set = set()
    for a in U_set:
        for b in U_set:
            diff_set.add(a - b)
            
    ratio = len(diff_set) / len(sum_set)
    if ratio <= 1:
        return 1.0
        
    log_ratio = math.log(ratio)
    log_denom = math.log(2 * max_u + 1)
    
    if log_denom == 0:
        return 1.0
        
    return 1.0 + log_ratio / log_denom

def _score_fast(U_list):
    """Faster score using list-based approach for small sets."""
    if not U_list or 0 not in U_list:
        return 0.0
    n = len(U_list)
    if n <= 1:
        return 1.0
    max_u = max(U_list)
    if max_u == 0:
        return 1.0
    
    # Use set operations for speed
    U_set = set(U_list)
    sum_set = {a + b for a in U_set for b in U_set}
    diff_set = {a - b for a in U_set for b in U_set}
    
    ratio = len(diff_set) / len(sum_set)
    if ratio <= 1:
        return 1.0
    return 1.0 + math.log(ratio) / math.log(2 * max_u + 1)

def _generate_powers_of_b(b, n, include_zero=True):
    U = []
    if include_zero:
        U.append(0)
    for i in range(n):
        val = int(b ** i)
        if val not in U:
            U.append(val)
    U.sort()
    return U

def _generate_quadratic(n, scale=1):
    """Generate quadratic sequence: 0, scale*1, scale*4, scale*9, ..."""
    U = [0]
    for i in range(1, n + 1):
        val = scale * i * i
        if val not in U:
            U.append(val)
    U.sort()
    return U

def _generate_triangular(n, scale=1):
    """Generate triangular numbers: 0, scale*1, scale*3, scale*6, ..."""
    U = [0]
    for i in range(1, n + 1):
        val = scale * i * (i + 1) // 2
        if val not in U:
            U.append(val)
    U.sort()
    return U

def _generate_mixed_algebraic(n, base=2):
    """Mix powers and quadratic terms."""
    U = {0}
    for i in range(1, n + 1):
        # Alternate between power and quadratic
        if i % 2 == 1:
            U.add(base ** i)
        else:
            U.add((i // 2) ** 2)
    return sorted(U)

def _local_search(U, deadline, T0=5.0, cooling=0.995):
    """Local search with simulated annealing."""
    current_U = sorted(U)
    current_score = _score_fast(current_U)
    best_U = current_U[:]
    best_score = current_score
    
    T = T0
    step = 0
    while time.time() < deadline and T > 1e-6:
        step += 1
        candidate_U = current_U[:]
        max_val = max(candidate_U) if candidate_U else 0
        
        move_type = random.random()
        
        if move_type < 0.35:
            # Replace an element
            if len(candidate_U) > 1:
                idx = random.randint(1, len(candidate_U) - 1)
                # Try values near existing elements or in gaps
                r = random.random()
                if r < 0.3:
                    new_val = random.randint(1, max_val + 50)
                elif r < 0.6:
                    # Near an existing element
                    ref = random.choice(candidate_U[1:])
                    new_val = max(1, ref + random.randint(-max(1, ref//10), max(1, ref//10)))
                else:
                    new_val = random.randint(1, max_val * 2 + 100)
                if new_val not in candidate_U:
                    candidate_U[idx] = new_val
                    candidate_U.sort()
        elif move_type < 0.55:
            # Add an element
            if len(candidate_U) < 50:
                r = random.random()
                if r < 0.4:
                    new_val = random.randint(1, max_val + 50)
                elif r < 0.7:
                    ref = random.choice(candidate_U[1:]) if len(candidate_U) > 1 else 1
                    new_val = max(1, ref + random.randint(1, max(1, ref//5)))
                else:
                    new_val = random.randint(1, max_val * 2 + 100)
                if new_val not in candidate_U:
                    candidate_U.append(new_val)
                    candidate_U.sort()
        else:
            # Remove an element
            if len(candidate_U) > 2:
                idx = random.randint(1, len(candidate_U) - 1)
                candidate_U.pop(idx)
                candidate_U.sort()
        
        if candidate_U == current_U:
            continue
            
        candidate_score = _score_fast(candidate_U)
        
        delta = candidate_score - current_score
        if delta > 0 or (T > 1e-6 and random.random() < math.exp(delta / T)):
            current_U = candidate_U
            current_score = candidate_score
            if current_score > best_score:
                best_score = current_score
                best_U = current_U[:]
                
        T *= cooling
        
    return best_U, best_score

def _polish(U, deadline):
    """Final polish with small perturbations."""
    current_U = sorted(U)
    current_score = _score_fast(current_U)
    best_U = current_U[:]
    best_score = current_score
    
    # Try small shifts and replacements
    for _ in range(200):
        if time.time() > deadline:
            break
        candidate_U = current_U[:]
        max_val = max(candidate_U) if candidate_U else 0
        
        # Try shifting one element by a small amount
        if len(candidate_U) > 1:
            idx = random.randint(1, len(candidate_U) - 1)
            delta_shift = random.choice([-2, -1, 1, 2, 3, -3])
            new_val = candidate_U[idx] + delta_shift
            if new_val >= 1 and new_val not in candidate_U:
                candidate_U[idx] = new_val
                candidate_U.sort()
                s = _score_fast(candidate_U)
                if s > best_score:
                    best_score = s
                    best_U = candidate_U[:]
                    current_U = candidate_U
                    current_score = s
    
    return best_U, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    best_set = [0]
    best_score = 0.0
    
    # Phase 1: Generate diverse algebraic and structured seeds
    seeds = []
    
    # Powers of 2
    for n in range(6, 20):
        seeds.append(_generate_powers_of_b(2, n))
    
    # Powers of 3
    for n in range(5, 15):
        seeds.append(_generate_powers_of_b(3, n))
    
    # Powers of 4
    for n in range(4, 12):
        seeds.append(_generate_powers_of_b(4, n))
    
    # Quadratic sequences
    for n in range(8, 25):
        for scale in [1, 2, 3]:
            seeds.append(_generate_quadratic(n, scale))
    
    # Triangular numbers
    for n in range(8, 25):
        for scale in [1, 2, 3]:
            seeds.append(_generate_triangular(n, scale))
    
    # Mixed algebraic
    for n in range(10, 30):
        for base in [2, 3]:
            seeds.append(_generate_mixed_algebraic(n, base))
    
    # Random sets with structure
    for _ in range(20):
        size = random.randint(8, 25)
        max_val = random.randint(100, 2000)
        U = {0}
        # Start with some structure then add random
        for i in range(1, min(5, size)):
            U.add(i * random.randint(1, max_val // 5))
        while len(U) < size:
            U.add(random.randint(1, max_val))
        seeds.append(sorted(U))
    
    # Evaluate all seeds
    for U in seeds:
        if time.time() > deadline - 5:
            break
        if 0 not in U:
            U.append(0)
            U.sort()
        s = _score_fast(U)
        if s > best_score:
            best_score = s
            best_set = U[:]
    
    # Phase 2: Local search from best seed
    if time.time() < deadline - 15:
        sa_set, sa_score = _local_search(best_set, deadline - 5)
        if sa_score > best_score:
            best_score = sa_score
            best_set = sa_set
    
    # Phase 3: Polish
    if time.time() < deadline - 2:
        polished_set, polished_score = _polish(best_set, deadline - 1)
        if polished_score > best_score:
            best_score = polished_score
            best_set = polished_set
    
    # Ensure 0 is in the result
    if 0 not in best_set:
        best_set.append(0)
        best_set.sort()
    
    return best_set
