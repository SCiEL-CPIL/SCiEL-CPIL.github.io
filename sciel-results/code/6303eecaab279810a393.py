import time
import random
from itertools import product

def _square_nonzero_from_lists(exps, coeffs):
    """Compute nonzero count of square from parallel lists of exponents and coeffs."""
    n = len(exps)
    if n == 0:
        return 0
    sq = {}
    for i in range(n):
        ei = exps[i]
        ci = coeffs[i]
        for j in range(i, n):
            s = ei + exps[j]
            if i == j:
                sq[s] = sq.get(s, 0) + ci * ci
            else:
                sq[s] = sq.get(s, 0) + 2 * ci * coeffs[j]
    return sum(1 for v in sq.values() if v != 0)

def _square_dict_from_lists(exps, coeffs):
    """Return square as dict from parallel lists."""
    n = len(exps)
    if n == 0:
        return {}
    sq = {}
    for i in range(n):
        ei = exps[i]
        ci = coeffs[i]
        for j in range(i, n):
            s = ei + exps[j]
            if i == j:
                sq[s] = sq.get(s, 0) + ci * ci
            else:
                sq[s] = sq.get(s, 0) + 2 * ci * coeffs[j]
    return sq

def _make_list_from_dict(d, max_deg=40):
    """Convert dict to coefficient list in ascending degree."""
    if not d:
        return [0]
    max_e = max(d.keys())
    lst = [0] * (max_e + 1)
    for e, c in d.items():
        if e <= max_deg:
            lst[e] = c
    while len(lst) > 1 and lst[-1] == 0:
        lst.pop()
    return lst

def _validate_lists(exps, coeffs):
    """Check validity from parallel lists."""
    if len(exps) != 12:
        return False
    seen = set()
    for i in range(12):
        e = exps[i]
        c = coeffs[i]
        if c == 0 or c < -64 or c > 64:
            return False
        if e < 0 or e > 40:
            return False
        if e in seen:
            return False
        seen.add(e)
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Conservative deadline to avoid timeout
    
    random.seed(42)
    
    # Generate a few initial candidates
    candidates = []
    
    # Candidate 1: consecutive exponents
    exps1 = list(range(12))
    coeffs1 = [1] * 12
    candidates.append((exps1[:], coeffs1[:]))
    
    # Candidate 2: alternating signs
    exps2 = list(range(12))
    coeffs2 = [1 if i % 2 == 0 else -1 for i in range(12)]
    candidates.append((exps2[:], coeffs2[:]))
    
    # Candidate 3: sparse exponents
    exps3 = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]
    coeffs3 = [1] * 12
    candidates.append((exps3[:], coeffs3[:]))
    
    # A few random candidates
    for _ in range(20):
        exps = sorted(random.sample(range(0, 41), 12))
        coeffs = [random.choice([-2, -1, 1, 2]) for _ in range(12)]
        candidates.append((exps[:], coeffs[:]))
    
    # Evaluate initial candidates
    best_score = None
    best_exps = None
    best_coeffs = None
    
    for exps, coeffs in candidates:
        if not _validate_lists(exps, coeffs):
            continue
        score = _square_nonzero_from_lists(exps, coeffs)
        if best_score is None or score < best_score:
            best_score = score
            best_exps = exps[:]
            best_coeffs = coeffs[:]
    
    if best_exps is None:
        best_exps = list(range(12))
        best_coeffs = [1] * 12
        best_score = _square_nonzero_from_lists(best_exps, best_coeffs)
    
    # Local search with simulated annealing
    T = 2.0
    cooling = 0.9998
    
    current_exps = best_exps[:]
    current_coeffs = best_coeffs[:]
    current_score = best_score
    best_found_exps = best_exps[:]
    best_found_coeffs = best_coeffs[:]
    best_found_score = best_score
    
    no_improve = 0
    
    while time.time() < deadline - 10:  # Leave 10s for final phases
        # Choose a move
        move_type = random.random()
        new_exps = current_exps[:]
        new_coeffs = current_coeffs[:]
        
        if move_type < 0.35:
            # Change a coefficient
            idx = random.randrange(12)
            old_c = new_coeffs[idx]
            new_c = random.choice([-8, -4, -3, -2, -1, 1, 2, 3, 4, 8])
            if new_c == old_c:
                new_c = old_c + 1 if old_c < 64 else old_c - 1
            if new_c == 0:
                new_c = 1
            new_coeffs[idx] = new_c
        elif move_type < 0.55:
            # Change an exponent
            idx = random.randrange(12)
            old_e = new_exps[idx]
            new_e = random.randint(0, 40)
            if new_e in new_exps:
                continue
            new_exps[idx] = new_e
            new_exps.sort()
        elif move_type < 0.70:
            # Change sign of a coefficient
            idx = random.randrange(12)
            new_coeffs[idx] = -new_coeffs[idx]
        elif move_type < 0.85:
            # Move an exponent by small amount
            idx = random.randrange(12)
            old_e = new_exps[idx]
            shift = random.choice([-2, -1, 1, 2])
            new_e = old_e + shift
            if new_e < 0 or new_e > 40 or new_e in new_exps:
                continue
            new_exps[idx] = new_e
            new_exps.sort()
        else:
            # Swap two exponents
            i, j = random.sample(range(12), 2)
            new_exps[i], new_exps[j] = new_exps[j], new_exps[i]
            new_exps.sort()
        
        if not _validate_lists(new_exps, new_coeffs):
            continue
        
        new_score = _square_nonzero_from_lists(new_exps, new_coeffs)
        
        # Accept or reject
        if new_score <= current_score:
            current_exps = new_exps
            current_coeffs = new_coeffs
            current_score = new_score
            if new_score < best_found_score:
                best_found_exps = new_exps[:]
                best_found_coeffs = new_coeffs[:]
                best_found_score = new_score
                no_improve = 0
            else:
                no_improve += 1
        else:
            delta = new_score - current_score
            if random.random() < (1.0 / (1.0 + delta * T * 0.5)):
                current_exps = new_exps
                current_coeffs = new_coeffs
                current_score = new_score
            no_improve += 1
        
        T *= cooling
        
        if no_improve > 3000:
            current_exps = best_found_exps[:]
            current_coeffs = best_found_coeffs[:]
            current_score = best_found_score
            no_improve = 0
            T = max(T, 1.0)
    
    # Final phase: exhaustive sign search
    if time.time() < deadline - 5:
        for signs in product([1, -1], repeat=12):
            if time.time() > deadline - 3:
                break
            trial_coeffs = [abs(best_found_coeffs[i]) * signs[i] for i in range(12)]
            if not _validate_lists(best_found_exps, trial_coeffs):
                continue
            score = _square_nonzero_from_lists(best_found_exps, trial_coeffs)
            if score < best_found_score:
                best_found_coeffs = trial_coeffs[:]
                best_found_score = score
    
    # Convert to dict then list
    best_dict = {}
    for i in range(12):
        best_dict[best_found_exps[i]] = best_found_coeffs[i]
    
    result = _make_list_from_dict(best_dict, max_deg=40)
    
    if len(result) > 41:
        result = result[:41]
    while len(result) > 1 and result[-1] == 0:
        result.pop()
    
    nz = sum(1 for c in result if c != 0)
    if nz != 12:
        result = [1] * 12
    
    return result
