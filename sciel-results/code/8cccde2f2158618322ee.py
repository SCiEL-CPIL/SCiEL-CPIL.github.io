import random
import time
import math

DEADLINE = 82.0  # safety margin before 90s kill

def _square_support_count(coeffs):
    """Return number of nonzero coefficients in the exact square of coeffs."""
    n = len(coeffs)
    if n == 0:
        return 0
    # Convolution of coeffs with itself
    sq = [0] * (2 * n - 1)
    for i in range(n):
        ci = coeffs[i]
        if ci == 0:
            continue
        for j in range(n):
            cj = coeffs[j]
            if cj != 0:
                sq[i + j] += ci * cj
    # Count nonzeros
    cnt = 0
    for v in sq:
        if v != 0:
            cnt += 1
    return cnt

def _validate(coeffs):
    """Check all constraints. Return True if valid."""
    if len(coeffs) > 41:
        return False
    if len(coeffs) == 0:
        return False
    if coeffs[-1] == 0:
        return False
    nz = 0
    for c in coeffs:
        if not isinstance(c, int):
            return False
        if c < -64 or c > 64:
            return False
        if c != 0:
            nz += 1
    if nz != 12:
        return False
    return True

def _random_candidate(rng):
    """Generate a random valid 12-term polynomial."""
    # Choose 12 distinct exponents from [0, 40]
    exponents = rng.sample(range(41), 12)
    exponents.sort()
    coeffs = [0] * 41
    for e in exponents:
        coeffs[e] = rng.randint(1, 64)
        if rng.random() < 0.5:
            coeffs[e] = -coeffs[e]
    # Ensure final coefficient nonzero (exponents are in [0,40], max is in list)
    # Trim trailing zeros but keep length <= 41 and last nonzero
    # Actually we need length <= 41 and last entry nonzero
    # Find highest nonzero
    hi = 40
    while hi > 0 and coeffs[hi] == 0:
        hi -= 1
    # coeffs[hi] is nonzero by construction
    return coeffs[:hi + 1]

def _neighborhood_move(coeffs, rng):
    """Perform a random neighborhood move on a valid polynomial."""
    c = list(coeffs)
    n = len(c)
    move_type = rng.random()
    
    if move_type < 0.3:
        # Change a coefficient value
        nz_idx = [i for i in range(n) if c[i] != 0]
        i = rng.choice(nz_idx)
        new_val = rng.randint(1, 64)
        if rng.random() < 0.5:
            new_val = -new_val
        if new_val != c[i]:
            c[i] = new_val
            # Check bounds after change - already in range
            # Check that we still have 12 nonzero terms (value changed, not zero)
            return c
    elif move_type < 0.5:
        # Move an exponent (swap a term to a new position)
        nz_idx = [i for i in range(n) if c[i] != 0]
        i = rng.choice(nz_idx)
        # Choose new exponent
        new_e = rng.randint(0, 40)
        if new_e == i:
            return c
        if c[new_e] != 0:
            return c  # collision
        val = c[i]
        c[i] = 0
        c[new_e] = val
        # Trim
        hi = 40
        while hi >= 0 and (hi >= len(c) or c[hi] == 0):
            hi -= 1
        if hi < 0:
            return c  # all zero, invalid
        c = c[:hi + 1]
        # Verify 12 nonzero
        if sum(1 for x in c if x != 0) != 12:
            return list(coeffs)  # revert
        return c
    elif move_type < 0.7:
        # Swap two exponents
        nz_idx = [i for i in range(n) if c[i] != 0]
        if len(nz_idx) < 2:
            return c
        i, j = rng.sample(nz_idx, 2)
        c[i], c[j] = c[j], c[i]
        return c
    elif move_type < 0.85:
        # Add a zero term and remove another (change support size temporarily)
        nz_idx = [i for i in range(n) if c[i] != 0]
        z_idx = [i for i in range(n) if c[i] == 0]
        if not z_idx or len(nz_idx) < 2:
            return c
        # Remove one nonzero, add to a zero position
        i = rng.choice(nz_idx)
        j = rng.choice(z_idx)
        c[j] = c[i]
        c[i] = 0
        # Trim
        hi = len(c) - 1
        while hi >= 0 and c[hi] == 0:
            hi -= 1
        if hi < 0:
            return list(coeffs)
        c = c[:hi + 1]
        if sum(1 for x in c if x != 0) != 12:
            return list(coeffs)
        return c
    else:
        # Perturb: change two coefficients
        nz_idx = [i for i in range(n) if c[i] != 0]
        if len(nz_idx) < 2:
            return c
        i, j = rng.sample(nz_idx, 2)
        c[i] = rng.choice([-1, 1]) * rng.randint(1, 64)
        c[j] = rng.choice([-1, 1]) * rng.randint(1, 64)
        return c

def construct():
    start_time = time.time()
    rng = random.Random(42)
    
    # Initial candidate
    best = _random_candidate(rng)
    while not _validate(best):
        best = _random_candidate(rng)
    best_score = _square_support_count(best)
    
    # Also try some structured initial candidates
    structured_candidates = []
    
    # Arithmetic progression exponents with alternating signs
    for step in [2, 3, 4, 5]:
        if step * 11 <= 40:
            exponents = list(range(0, 12 * step, step))
            if len(exponents) == 12:
                coeffs = [0] * (exponents[-1] + 1)
                for k, e in enumerate(exponents):
                    coeffs[e] = (-1) ** k * (k + 1)
                if _validate(coeffs):
                    structured_candidates.append(coeffs)
    
    # Dense low-degree
    for start in range(0, 5):
        exponents = list(range(start, start + 12))
        if exponents[-1] <= 40:
            coeffs = [0] * (exponents[-1] + 1)
            for e in exponents:
                coeffs[e] = 1
            if _validate(coeffs):
                structured_candidates.append(coeffs)
    
    # Sparse with gaps
    for gap in [1, 2, 3]:
        exponents = [i * (gap + 1) for i in range(12)]
        if exponents[-1] <= 40:
            coeffs = [0] * (exponents[-1] + 1)
            for k, e in enumerate(exponents):
                coeffs[e] = (-1) ** k
            if _validate(coeffs):
                structured_candidates.append(coeffs)
    
    for sc in structured_candidates:
        s = _square_support_count(sc)
        if s < best_score:
            best_score = s
            best = sc
    
    # Simulated annealing
    T = 10.0
    T_min = 0.01
    cooling = 0.999
    
    current = list(best)
    current_score = best_score
    
    while time.time() < DEADLINE:
        neighbor = _neighborhood_move(current, rng)
        
        if not _validate(neighbor):
            continue
        
        neighbor_score = _square_support_count(neighbor)
        
        if neighbor_score < current_score:
            current = neighbor
            current_score = neighbor_score
            if current_score < best_score:
                best_score = current_score
                best = list(current)
        else:
            # Accept with probability exp(-delta/T)
            delta = neighbor_score - current_score
            if delta == 0:
                # Equal score, accept with some probability to explore
                if rng.random() < 0.3:
                    current = neighbor
            else:
                accept_prob = math.exp(-delta / max(T, T_min))
                if rng.random() < accept_prob:
                    current = neighbor
        
        T = max(T * cooling, T_min)
        
        # Periodic restart if stuck
        if T <= T_min * 1.01:
            # Restart with perturbation of best
            T = 5.0
            current = list(best)
            # Perturb current
            for _ in range(5):
                current = _neighborhood_move(current, rng)
                if not _validate(current):
                    current = list(best)
                    break
            current_score = _square_support_count(current)
    
    return best

# Quick self-test
if __name__ == "__main__":
    result = construct()
    print(f"Length: {len(result)}")
    print(f"Nonzero terms: {sum(1 for c in result if c != 0)}")
    print(f"Last coeff: {result[-1]}")
    print(f"Square support: {_square_support_count(result)}")
    print(f"Valid: {_validate(result)}")
