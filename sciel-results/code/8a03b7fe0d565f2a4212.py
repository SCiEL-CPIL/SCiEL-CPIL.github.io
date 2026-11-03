import itertools
import math
import time
import numpy as np
import random

def _score_of(U):
    """Calculate the score for a given set U using NumPy for speed."""
    if not U:
        return 0.0
    U = sorted(set(U))
    if 0 not in U:
        U = [0] + U
    max_u = max(U)
    if max_u == 0:
        return 1.0
    
    arr = np.array(U, dtype=np.int64)
    sums = np.unique(arr[:, None] + arr[None, :])
    diffs = np.unique(arr[:, None] - arr[None, :])
    
    size_sum = len(sums)
    size_diff = len(diffs)
    
    if size_sum == 0:
        return 0.0
        
    ratio = size_diff / size_sum
    if ratio <= 0:
        return 0.0
        
    denom = math.log(2 * max_u + 1)
    if denom == 0:
        return 1.0
        
    return 1.0 + math.log(ratio) / denom

def _simulated_annealing(U_init, max_iter=5000, t0=1.0, t_min=0.001, cooling=0.9995, deadline=None):
    """Simulated annealing to optimize the set U.
    
    Models lattice packing: minimize sumset size (tight Minkowski sum)
    while maximizing difference set size (sparse difference vectors).
    """
    U = sorted(set(U_init))
    if 0 not in U:
        U = [0] + U
    current_score = _score_of(U)
    best_score = current_score
    best_U = U[:]
    
    n = len(U)
    if n < 2:
        return U, best_score
    
    # Determine search range based on current max
    max_val = max(U)
    search_low = 1
    search_high = max(50, int(max_val * 1.5))
    
    for i in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        T = t0 * (cooling ** i)
        if T < t_min:
            T = t_min
        
        # Pick a random element to modify (not 0)
        idx = random.randint(1, n - 1)
        old_val = U[idx]
        
        # Generate new value: biased toward nearby values for local refinement
        if random.random() < 0.7:
            # Local move
            step = random.randint(-3, 3)
            new_val = old_val + step
        else:
            # Global move
            new_val = random.randint(search_low, search_high)
        
        # Clamp to valid range
        new_val = max(search_low, min(search_high, new_val))
        
        # Ensure uniqueness
        if new_val in U:
            continue
        
        U[idx] = new_val
        U.sort()
        
        new_score = _score_of(U)
        delta = new_score - current_score
        
        if delta > 0 or random.random() < math.exp(delta / T):
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_U = U[:]
        else:
            U[idx] = old_val
            U.sort()
    
    return best_U, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 40s margin for safety
    
    best_set = [0, 1, 4, 9, 15]
    best_score = _score_of(best_set)
    
    # Phase 1: Exhaustive search for small sets
    for N in range(2, 8):
        for M in range(1, 25):
            if time.time() > deadline:
                return best_set
            
            if M < N - 1:
                continue
                
            if N > 5:
                M_limit = 20
            else:
                M_limit = 30
                
            if M > M_limit:
                continue

            candidates = list(range(1, M + 1))
            for combo in itertools.combinations(candidates, N - 1):
                if time.time() > deadline:
                    return best_set
                    
                U = [0] + list(combo)
                s = _score_of(U)
                if s > best_score:
                    best_score = s
                    best_set = U[:]

    # Phase 2: Algebraic constructions as starting points
    if time.time() < deadline - 30:
        random.seed(42)
        
        # Quadratic construction
        for k in range(3, 10):
            if time.time() > deadline:
                break
            U_quad = sorted(set([i * i for i in range(k)]))
            if 0 not in U_quad:
                U_quad = [0] + U_quad
            s = _score_of(U_quad)
            if s > best_score:
                best_score = s
                best_set = U_quad[:]
        
        # Triangular construction
        for k in range(3, 12):
            if time.time() > deadline:
                break
            U_tri = sorted(set([i * (i + 1) // 2 for i in range(k)]))
            if 0 not in U_tri:
                U_tri = [0] + U_tri
            s = _score_of(U_tri)
            if s > best_score:
                best_score = s
                best_set = U_tri[:]
        
        # Geometric progression
        for base in [2, 3, 4]:
            for k in range(4, 9):
                if time.time() > deadline:
                    break
                U_geo = sorted(set([base ** i for i in range(k)]))
                if 0 not in U_geo:
                    U_geo = [0] + U_geo
                if max(U_geo) > 200:
                    continue
                s = _score_of(U_geo)
                if s > best_score:
                    best_score = s
                    best_set = U_geo[:]

    # Phase 3: Simulated annealing from multiple starting points
    if time.time() < deadline - 15:
        # Use best found set as primary starting point
        U_sa = best_set[:]
        U_sa, sa_score = _simulated_annealing(
            U_sa, max_iter=3000, t0=1.0, t_min=0.001, cooling=0.9998, deadline=deadline
        )
        if sa_score > best_score:
            best_score = sa_score
            best_set = U_sa[:]
        
        # Try from algebraic constructions
        for k in [5, 6, 7, 8]:
            if time.time() > deadline:
                break
            U_start = sorted(set([i * i for i in range(k)]))
            if 0 not in U_start:
                U_start = [0] + U_start
            U_sa, sa_score = _simulated_annealing(
                U_start, max_iter=2000, t0=0.5, t_min=0.001, cooling=0.9997, deadline=deadline
            )
            if sa_score > best_score:
                best_score = sa_score
                best_set = U_sa[:]

    # Phase 4: Final local search polish
    if time.time() < deadline - 5:
        U_polish = best_set[:]
        n = len(U_polish)
        if n >= 2:
            for _ in range(2000):
                if time.time() > deadline:
                    break
                idx = random.randint(1, n - 1)
                old_val = U_polish[idx]
                # Try small perturbations
                for delta in [-2, -1, 1, 2]:
                    new_val = old_val + delta
                    if new_val < 1:
                        continue
                    if new_val in U_polish:
                        continue
                    U_polish[idx] = new_val
                    U_polish.sort()
                    new_score = _score_of(U_polish)
                    if new_score > best_score:
                        best_score = new_score
                        best_set = U_polish[:]
                        break
                else:
                    # No improvement, revert
                    U_polish[idx] = old_val
                    U_polish.sort()

    return best_set
