"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools
import math
import sys

# EVOLVE-BLOCK-START

def _sumset_size(A):
    """Compute size of A+A (with repetition allowed)."""
    s = set()
    n = len(A)
    for i in range(n):
        ai = A[i]
        for j in range(i, n):
            s.add(ai + A[j])
    return len(s)

def _prodset_size(A):
    """Compute size of A*A (with repetition allowed)."""
    s = set()
    n = len(A)
    for i in range(n):
        ai = A[i]
        for j in range(i, n):
            s.add(ai * A[j])
    return len(s)

def _score(A):
    """max(|A+A|, |A*A|)"""
    return max(_sumset_size(A), _prodset_size(A))

def _is_valid(A, n=20, maximum=10000):
    if len(A) != n:
        return False
    if len(set(A)) != n:
        return False
    if any(x <= 0 or x > maximum for x in A):
        return False
    return True

def _geometric_progression(start, ratio, n=20, max_val=10000):
    """Generate a geometric-like progression."""
    A = []
    val = start
    for _ in range(n):
        if val > max_val:
            break
        A.append(int(val))
        val *= ratio
    # If we didn't get enough, fill with next values
    while len(A) < n:
        A.append(int(val))
        val *= ratio
    return A[:n]

def _try_local_search(A, time_budget, rng):
    """Local search: try swapping elements to improve score."""
    A = sorted(A)
    current_score = _score(A)
    best_A = list(A)
    best_score = current_score
    deadline = time.time() + time_budget
    
    # Precompute for speed
    n = 20
    
    while time.time() < deadline:
        # Pick a random element to perturb
        idx = rng.randint(0, n - 1)
        old_val = A[idx]
        
        # Try replacing with a nearby value or random value
        candidates = []
        # Nearby values
        for delta in range(-50, 51):
            nv = old_val + delta
            if 1 <= nv <= 10000 and nv != old_val:
                candidates.append(nv)
        # Some random values
        for _ in range(10):
            nv = rng.randint(1, 10000)
            if nv != old_val:
                candidates.append(nv)
        
        improved = False
        for nv in candidates:
            if nv in A:
                continue
            A[idx] = nv
            # Quick check: ensure distinct
            if len(set(A)) < n:
                A[idx] = old_val
                continue
            sc = _score(A)
            if sc < current_score:
                current_score = sc
                improved = True
                if sc < best_score:
                    best_score = sc
                    best_A = list(A)
                break
            A[idx] = old_val
        
        if not improved:
            # Try a pair swap
            i1 = rng.randint(0, n - 1)
            i2 = rng.randint(0, n - 1)
            if i1 == i2:
                continue
            v1, v2 = A[i1], A[i2]
            # Try swapping with new random values
            nv1 = rng.randint(1, 10000)
            nv2 = rng.randint(1, 10000)
            if nv1 == nv2 or nv1 in A or nv2 in A:
                A[i1] = nv1
                A[i2] = nv2
                if len(set(A)) < n:
                    A[i1] = v1
                    A[i2] = v2
                    continue
                sc = _score(A)
                if sc < current_score:
                    current_score = sc
                    if sc < best_score:
                        best_score = sc
                        best_A = list(A)
                else:
                    A[i1] = v1
                    A[i2] = v2
    
    return best_A, best_score

def _simulated_annealing(A, time_budget, rng):
    """Simulated annealing on the set of 20 integers."""
    A = list(A)
    n = 20
    current_score = _score(A)
    best_A = list(A)
    best_score = current_score
    deadline = time.time() + time_budget
    
    T = 50.0
    cooling = 0.995
    
    while time.time() < deadline and T > 0.1:
        # Perturbation: change one or two elements
        if rng.random() < 0.7:
            idx = rng.randint(0, n - 1)
            old_val = A[idx]
            # Biased toward nearby values
            if rng.random() < 0.5:
                delta = rng.choice([-1, -2, -3, -5, -10, -20, 1, 2, 3, 5, 10, 20])
                nv = max(1, min(10000, old_val + delta))
            else:
                nv = rng.randint(1, 10000)
            
            if nv != old_val and nv not in A:
                A[idx] = nv
                sc = _score(A)
                if sc < current_score:
                    current_score = sc
                    if sc < best_score:
                        best_score = sc
                        best_A = list(A)
                else:
                    delta_e = sc - current_score
                    if rng.random() < math.exp(-delta_e / T):
                        current_score = sc
                    else:
                        A[idx] = old_val
        else:
            # Swap two elements
            i1, i2 = rng.sample(range(n), 2)
            v1, v2 = A[i1], A[i2]
            A[i1], A[i2] = v2, v1
            sc = _score(A)
            if sc < current_score:
                current_score = sc
                if sc < best_score:
                    best_score = sc
                    best_A = list(A)
            else:
                delta_e = sc - current_score
                if rng.random() >= math.exp(-delta_e / T):
                    A[i1], A[i2] = v1, v2
        
        T *= cooling
    
    return best_A, best_score

def _generate_initial_candidates(rng):
    """Generate diverse initial candidate sets."""
    candidates = []
    
    # 1. Geometric progressions with different starts
    for start in [2, 3, 4, 5, 6, 8, 10]:
        for ratio in [1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 2.0]:
            A = []
            val = float(start)
            for _ in range(20):
                A.append(int(round(val)))
                val *= ratio
            # Ensure distinct and in range
            seen = set()
            valid = []
            for x in A:
                x = max(1, min(10000, x))
                if x not in seen:
                    seen.add(x)
                    valid.append(x)
            if len(valid) == 20:
                candidates.append(sorted(valid))
    
    # 2. Exponential with small base
    for base in [1.05, 1.1, 1.15, 1.2, 1.25, 1.3]:
        A = [int(round(2 * (base ** i))) for i in range(20)]
        seen = set()
        valid = []
        for x in A:
            x = max(1, min(10000, x))
            if x not in seen:
                seen.add(x)
                valid.append(x)
        if len(valid) == 20:
            candidates.append(sorted(valid))
    
    # 3. Powers of small integers
    for base in [2, 3]:
        A = [base ** i for i in range(20)]
        if all(x <= 10000 for x in A):
            candidates.append(sorted(A))
    
    # 4. Random sets
    for _ in range(20):
        A = sorted(rng.sample(range(1, 10001), 20))
        candidates.append(A)
    
    # 5. Dense small numbers
    for start in range(1, 50):
        A = list(range(start, start + 20))
        if A[-1] <= 10000:
            candidates.append(A)
    
    # 6. Sparse large numbers
    for _ in range(10):
        A = sorted(rng.sample(range(500, 10001), 20))
        candidates.append(A)
    
    # 7. Mixed: some small, some large
    for _ in range(10):
        small = rng.sample(range(1, 100), 10)
        large = rng.sample(range(500, 10001), 10)
        A = sorted(small + large)
        if len(set(A)) == 20:
            candidates.append(A)
    
    return candidates

def construct():
    """Construct 20 distinct positive integers <= 10000 minimizing max(|A+A|, |A*A|)."""
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s kill
    
    rng = random.Random(42)
    
    # Generate initial candidates
    candidates = _generate_initial_candidates(rng)
    
    # Evaluate all candidates
    best_A = None
    best_score = float('inf')
    
    for A in candidates:
        if not _is_valid(A):
            continue
        sc = _score(A)
        if sc < best_score:
            best_score = sc
            best_A = list(A)
    
    if best_A is None:
        return list(range(1, 21))
    
    # Run simulated annealing from the best candidate
    remaining = deadline - time.time()
    if remaining > 2:
        A, sc = _simulated_annealing(best_A, remaining * 0.6, rng)
        if sc < best_score:
            best_score = sc
            best_A = A
    
    # Run local search from the best candidate
    remaining = deadline - time.time()
    if remaining > 2:
        A, sc = _try_local_search(best_A, remaining * 0.7, rng)
        if sc < best_score:
            best_score = sc
            best_A = A
    
    # Try a few more random restarts if time permits
    remaining = deadline - time.time()
    while remaining > 5:
        A = sorted(rng.sample(range(1, 10001), 20))
        sc = _score(A)
        if sc < best_score:
            best_score = sc
            best_A = list(A)
            # Quick SA from this new best
            remaining = deadline - time.time()
            if remaining > 3:
                A2, sc2 = _simulated_annealing(best_A, min(remaining * 0.3, 5), rng)
                if sc2 < best_score:
                    best_score = sc2
                    best_A = A2
        remaining = deadline - time.time()
    
    return best_A

# EVOLVE-BLOCK-END
