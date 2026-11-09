import random
import time

def _count_nonzero_square(coeffs):
    """Count nonzero terms in the square of a polynomial given as coefficient list."""
    n = len(coeffs)
    # Square has degree 2*(n-1), so length 2*n-1
    sq = [0] * (2 * n - 1)
    for i in range(n):
        if coeffs[i] == 0:
            continue
        ci = coeffs[i]
        for j in range(n):
            if coeffs[j] == 0:
                continue
            sq[i + j] += ci * coeffs[j]
    return sum(1 for c in sq if c != 0)

def _valid(coeffs):
    """Check if coeffs is valid: length<=41, nonzero final coeff, entries in [-64,64], exactly 12 nonzero."""
    if len(coeffs) > 41:
        return False
    if len(coeffs) == 0 or coeffs[-1] == 0:
        return False
    nz = sum(1 for c in coeffs if c != 0)
    if nz != 12:
        return False
    for c in coeffs:
        if c < -64 or c > 64:
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    best_score = None
    best_coeffs = None
    
    # Initial structured guess: try some known patterns
    # Pattern 1: symmetric-ish with small coefficients
    initial_candidates = [
        [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    ]
    
    for cand in initial_candidates:
        if _valid(cand):
            sc = _count_nonzero_square(cand)
            if best_score is None or sc < best_score:
                best_score = sc
                best_coeffs = cand[:]
    
    # Simulated annealing
    if best_coeffs is None:
        # Generate random valid polynomial
        while True:
            deg = random.randint(11, 40)
            coeffs = [0] * (deg + 1)
            positions = random.sample(range(deg + 1), 12)
            for pos in positions:
                coeffs[pos] = random.randint(1, 10)
                if random.random() < 0.5:
                    coeffs[pos] = -coeffs[pos]
            if coeffs[-1] != 0 and _valid(coeffs):
                break
        best_coeffs = coeffs
        best_score = _count_nonzero_square(best_coeffs)
    
    current = best_coeffs[:]
    current_score = best_score
    
    T = 5.0
    T_min = 0.01
    cooling = 0.995
    
    random.seed(42)
    
    while time.time() < deadline:
        if T <= T_min:
            T = T_min
        
        # Generate a move
        move_type = random.random()
        new = current[:]
        
        if move_type < 0.3:
            # Change a coefficient value
            nz_positions = [i for i, c in enumerate(new) if c != 0]
            pos = random.choice(nz_positions)
            old_val = new[pos]
            new[pos] = random.randint(-10, 10)
            if new[pos] == 0:
                new[pos] = 1 if random.random() < 0.5 else -1
            if not _valid(new):
                continue
        elif move_type < 0.5:
            # Change position of a term
            nz_positions = [i for i, c in enumerate(new) if c != 0]
            pos = random.choice(nz_positions)
            val = new[pos]
            new[pos] = 0
            # Try to place in new position
            max_deg = min(40, len(new) - 1)
            if len(new) <= 41:
                new_pos = random.randint(0, max_deg)
            else:
                new_pos = random.randint(0, len(new) - 1)
            if new[new_pos] != 0:
                continue
            new[new_pos] = val
            if new[-1] == 0:
                continue
            if not _valid(new):
                continue
        elif move_type < 0.7:
            # Sign flip
            nz_positions = [i for i, c in enumerate(new) if c != 0]
            pos = random.choice(nz_positions)
            new[pos] = -new[pos]
            if not _valid(new):
                continue
        elif move_type < 0.85:
            # Add/remove a term (keep exactly 12)
            nz_positions = [i for i, c in enumerate(new) if c != 0]
            if len(nz_positions) >= 2:
                pos = random.choice(nz_positions)
                val = new[pos]
                new[pos] = 0
                max_deg = min(40, len(new) - 1)
                new_pos = random.randint(0, max_deg)
                if new[new_pos] == 0:
                    new[new_pos] = val
                    if new[-1] == 0:
                        continue
                    if not _valid(new):
                        continue
                else:
                    continue
            else:
                continue
        else:
            # Perturb multiple coefficients
            nz_positions = [i for i, c in enumerate(new) if c != 0]
            for _ in range(2):
                pos = random.choice(nz_positions)
                new[pos] = random.randint(-10, 10)
                if new[pos] == 0:
                    new[pos] = 1 if random.random() < 0.5 else -1
            if not _valid(new):
                continue
        
        new_score = _count_nonzero_square(new)
        delta = new_score - current_score
        
        if delta <= 0 or random.random() < pow(2.718281828, -delta / T):
            current = new
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_coeffs = current[:]
        
        T *= cooling
    
    # Return the best found
    # Ensure it's valid
    if not _valid(best_coeffs):
        # Fallback
        return [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    
    return best_coeffs
