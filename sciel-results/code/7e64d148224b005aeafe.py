import time
import random
import numpy as np

_START = time.time()
_DEADLINE = 85.0  # safety margin before 90s kill

def _sq_nonzero_count(coefs):
    """Count nonzero coefficients in the exact square using numpy convolution."""
    n = len(coefs)
    if n == 0:
        return 0
    arr = np.array(coefs, dtype=np.int64)
    sq = np.convolve(arr, arr, mode='full')
    return int(np.count_nonzero(sq))

def _valid(coefs):
    if len(coefs) > 41:
        return False
    nz = 0
    for c in coefs:
        if not isinstance(c, int):
            return False
        if c < -64 or c > 64:
            return False
        if c != 0:
            nz += 1
    if nz != 12:
        return False
    if coefs[-1] == 0:
        return False
    return True

def _count_nz(coefs):
    return sum(1 for c in coefs if c != 0)

def _gen_product_factors(rng, n_factors, max_exp=40):
    """
    Generate a polynomial as product of (1 + s_i * x^{e_i}) factors.
    This creates structured cancellations in the square.
    Returns the coefficient list.
    """
    P = [0] * 41
    P[0] = 1
    current_max = 0
    
    for i in range(n_factors):
        s = rng.choice([-1, 1])
        e = rng.randint(1, max_exp)
        if current_max + e >= 41:
            e = max(1, 41 - current_max - 1)
            if e <= 0:
                break
        
        # Multiply by (1 + s * x^e)
        new_P = [0] * 41
        for d in range(41):
            if P[d] != 0:
                new_P[d] += P[d]
                if d + e < 41:
                    new_P[d + e] += s * P[d]
        P = new_P
        current_max += e
    
    return P

def _gen_structured_candidates(rng, count=200):
    """Generate structured candidates with algebraic properties."""
    candidates = []
    
    for _ in range(count):
        if time.time() > _DEADLINE:
            break
        
        r = rng.random()
        
        if r < 0.3:
            # Product of 2-4 binomial factors
            n_f = rng.randint(2, 4)
            P = _gen_product_factors(rng, n_f)
        elif r < 0.5:
            # Product of factors with some shared exponents (more cancellation)
            n_f = rng.randint(3, 5)
            exponents = rng.sample(range(1, 20), min(n_f, 10))
            P = [0] * 41
            P[0] = 1
            for e in exponents:
                s = rng.choice([-1, 1])
                new_P = [0] * 41
                for d in range(41):
                    if P[d] != 0:
                        new_P[d] += P[d]
                        if d + e < 41:
                            new_P[d + e] += s * P[d]
                P = new_P
        elif r < 0.7:
            # Arithmetic progression with alternating signs
            start = rng.randint(0, 5)
            step = rng.randint(2, 8)
            P = [0] * 41
            for i in range(12):
                idx = start + i * step
                if idx < 41:
                    P[idx] = 1 if i % 2 == 0 else -1
            # Ensure exactly 12 terms and p[40] != 0
            nz = _count_nz(P)
            if nz == 12 and P[40] != 0:
                candidates.append(P[:])
            elif nz == 12 and P[40] == 0:
                # Move last term to 40
                nz_idx = [i for i in range(41) if P[i] != 0]
                if nz_idx:
                    last = nz_idx[-1]
                    P[40] = P[last]
                    P[last] = 0
                    if _count_nz(P) == 12:
                        candidates.append(P[:])
        else:
            # Random sparse with ±1 coefficients
            P = [0] * 41
            indices = rng.sample(range(41), 12)
            for idx in indices:
                P[idx] = rng.choice([-1, 1])
            if P[40] == 0:
                nz_idx = [i for i in range(41) if P[i] != 0]
                if nz_idx:
                    k = rng.choice(nz_idx)
                    P[40] = P[k]
                    P[k] = 0
            if _count_nz(P) == 12:
                candidates.append(P[:])
    
    return candidates

def _simulated_annealing(P_init, rng, time_limit=30.0):
    """
    Simulated annealing to minimize nonzero terms in P^2.
    Moves: change coefficient value, add/remove/shift terms.
    """
    current = P_init[:]
    current_score = _sq_nonzero_count(current)
    best = current[:]
    best_score = current_score
    
    T = 2.0
    T_min = 0.05
    cooling = 0.9995
    deadline = time.time() + time_limit
    
    val_pool = [-8, -6, -4, -3, -2, -1, 1, 2, 3, 4, 6, 8]
    
    no_improve = 0
    
    while time.time() < deadline:
        no_improve += 1
        if no_improve > 3000:
            # Restart
            current = best[:]
            current_score = best_score
            T = 1.0
            no_improve = 0
            continue
        
        # Generate neighbor
        neighbor = current[:]
        move_type = rng.random()
        
        if move_type < 0.4:
            # Change coefficient value of a random nonzero term
            nz_idx = [i for i in range(41) if neighbor[i] != 0]
            if not nz_idx:
                continue
            idx = rng.choice(nz_idx)
            old_val = neighbor[idx]
            new_val = rng.choice(val_pool)
            if new_val == old_val:
                new_val = old_val + rng.choice([-1, 1])
            if abs(new_val) > 64:
                continue
            neighbor[idx] = new_val
        elif move_type < 0.6:
            # Add a new term
            zero_idx = [i for i in range(41) if neighbor[i] == 0]
            if not zero_idx:
                continue
            # Remove one existing term to keep 12
            nz_idx = [i for i in range(41) if neighbor[i] != 0]
            if len(nz_idx) < 12:
                idx_add = rng.choice(zero_idx)
                neighbor[idx_add] = rng.choice(val_pool)
            else:
                idx_remove = rng.choice(nz_idx)
                idx_add = rng.choice(zero_idx)
                if idx_add == idx_remove:
                    continue
                neighbor[idx_remove] = 0
                neighbor[idx_add] = rng.choice(val_pool)
        elif move_type < 0.8:
            # Shift a term
            nz_idx = [i for i in range(41) if neighbor[i] != 0]
            if not nz_idx:
                continue
            idx = rng.choice(nz_idx)
            shift = rng.choice([-3, -2, -1, 1, 2, 3])
            new_idx = idx + shift
            if 0 <= new_idx < 41 and neighbor[new_idx] == 0:
                neighbor[new_idx] = neighbor[idx]
                neighbor[idx] = 0
            else:
                # Just change value
                neighbor[idx] = rng.choice(val_pool)
        else:
            # Swap two terms' positions
            nz_idx = [i for i in range(41) if neighbor[i] != 0]
            if len(nz_idx) >= 2:
                i1, i2 = rng.sample(nz_idx, 2)
                neighbor[i1], neighbor[i2] = neighbor[i2], neighbor[i1]
        
        # Validate
        if not _valid(neighbor):
            continue
        
        neighbor_score = _sq_nonzero_count(neighbor)
        delta = neighbor_score - current_score
        
        if delta < 0 or rng.random() < np.exp(-delta / max(T, 1e-6)):
            current = neighbor
            current_score = neighbor_score
            if current_score < best_score:
                best_score = current_score
                best = current[:]
                no_improve = 0
        
        T = max(T_min, T * cooling)
    
    return best, best_score

def _local_search(P_init, rng, time_limit=10.0):
    """Greedy local search for quick improvement."""
    current = P_init[:]
    current_score = _sq_nonzero_count(current)
    best = current[:]
    best_score = current_score
    deadline = time.time() + time_limit
    
    val_pool = [-8, -6, -4, -3, -2, -1, 1, 2, 3, 4, 6, 8]
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for idx in range(41):
            if time.time() > deadline:
                break
            if current[idx] == 0:
                continue
            old_val = current[idx]
            for new_val in val_pool:
                if new_val == old_val:
                    continue
                current[idx] = new_val
                if not _valid(current):
                    current[idx] = old_val
                    continue
                sc = _sq_nonzero_count(current)
                if sc < best_score:
                    best_score = sc
                    best = current[:]
                    improved = True
                else:
                    current[idx] = old_val
        
        # Try support moves
        for idx in range(41):
            if time.time() > deadline:
                break
            if current[idx] == 0:
                continue
            old_val = current[idx]
            for new_idx in range(41):
                if time.time() > deadline:
                    break
                if current[new_idx] != 0:
                    continue
                if new_idx == idx:
                    continue
                current[idx] = 0
                current[new_idx] = old_val
                if _valid(current):
                    sc = _sq_nonzero_count(current)
                    if sc < best_score:
                        best_score = sc
                        best = current[:]
                        improved = True
                current[new_idx] = 0
                current[idx] = old_val
    
    return best, best_score

def construct():
    global _START, _DEADLINE
    _START = time.time()
    _DEADLINE = _START + 85.0
    rng = random.Random(42)
    
    best_score = 10**9
    best_poly = None
    
    # Phase 1: Generate structured candidates
    candidates = _gen_structured_candidates(rng, count=300)
    
    # Evaluate all candidates
    for p in candidates:
        if time.time() > _DEADLINE:
            break
        if not _valid(p):
            continue
        sc = _sq_nonzero_count(p)
        if sc < best_score:
            best_score = sc
            best_poly = p[:]
    
    # Fallback if nothing found
    if best_poly is None:
        best_poly = [0] * 41
        for i in range(12):
            best_poly[i] = 1
        best_score = _sq_nonzero_count(best_poly)
    
    # Phase 2: Local search from best
    best_poly, best_score = _local_search(best_poly, rng, time_limit=15.0)
    
    # Phase 3: Simulated annealing from best
    if time.time() < _DEADLINE - 5:
        sa_poly, sa_score = _simulated_annealing(best_poly, rng, time_limit=40.0)
        if sa_score < best_score:
            best_score = sa_score
            best_poly = sa_poly
    
    # Phase 4: Final local search
    if time.time() < _DEADLINE - 2:
        best_poly, best_score = _local_search(best_poly, rng, time_limit=10.0)
    
    return best_poly
