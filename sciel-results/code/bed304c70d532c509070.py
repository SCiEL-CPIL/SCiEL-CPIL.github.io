"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START
def _score_fast(A):
    """Return max(|A+A|, |A*A|) for sorted list A of length 20, optimized."""
    n = len(A)
    ss = set()
    ps = set()
    for i in range(n):
        ai = A[i]
        for j in range(i, n):
            ss.add(ai + A[j])
            ps.add(ai * A[j])
    return max(len(ss), len(ps))

def _score_with_bound(A, current_best):
    """Return max(|A+A|, |A*A|) but stop early if either set exceeds current_best."""
    n = len(A)
    ss = set()
    ps = set()
    for i in range(n):
        ai = A[i]
        for j in range(i, n):
            s = ai + A[j]
            ss.add(s)
            if len(ss) >= current_best:
                return current_best
            p = ai * A[j]
            ps.add(p)
            if len(ps) >= current_best:
                return current_best
    return max(len(ss), len(ps))

def _gen_geometric_perturbed(base, exponents, multipliers, n=20, maximum=10000):
    """Generate set A = {base^e_i * k_i} where e_i are exponents, k_i are small multipliers."""
    A = set()
    for e, k in zip(exponents, multipliers):
        val = (base ** e) * k
        if 1 <= val <= maximum:
            A.add(val)
    if len(A) >= n:
        return sorted(list(A))[:n]
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin: evaluator kills at 90s
    
    n = 20
    maximum = 10000
    
    # Generate smooth numbers (5-smooth: products of 2, 3, 5)
    smooth = set()
    for a in range(1, maximum + 1):
        x = a
        while x % 2 == 0:
            x //= 2
        while x % 3 == 0:
            x //= 3
        while x % 5 == 0:
            x //= 5
        if x == 1:
            smooth.add(a)
    smooth_list = sorted(smooth)
    smooth_list = [x for x in smooth_list if x <= maximum]
    
    # Generate initial candidates using geometric progression + multiplicative perturbation
    candidates = []
    
    # Strategy 1: Pure geometric progressions with different bases
    for base in [2, 3]:
        exponents = list(range(0, 20))
        A = _gen_geometric_perturbed(base, exponents, [1]*20, n, maximum)
        if A and len(A) == n:
            candidates.append(A)
    
    # Strategy 2: Geometric with small coprime multipliers
    for base in [2, 3]:
        exponents = list(range(0, 14))
        multipliers = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27]
        A = _gen_geometric_perturbed(base, exponents, multipliers, n, maximum)
        if A and len(A) == n:
            candidates.append(A)
    
    # Strategy 3: Mix of geometric and smooth numbers
    for base in [2]:
        geo_part = [base**i for i in range(10)]
        smooth_extra = smooth_list[:15]
        A = sorted(set(geo_part + smooth_extra))
        if len(A) >= n:
            A = A[:n]
            candidates.append(A)
    
    # Strategy 4: First 20 smooth numbers
    if len(smooth_list) >= n:
        candidates.append(smooth_list[:n])
    
    # Strategy 5: Geometric with gaps
    for base in [2, 4]:
        exponents = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]
        A = _gen_geometric_perturbed(base, exponents, [1]*20, n, maximum)
        if A and len(A) == n:
            candidates.append(A)
    
    # Evaluate candidates
    best_A = None
    best_score = float('inf')
    for A in candidates:
        if _valid(A, n, maximum):
            sc = _score_fast(A)
            if sc < best_score:
                best_score = sc
                best_A = A
    
    if best_A is None:
        best_A = list(range(1, n+1))
        best_score = _score_fast(best_A)
    
    # Local search with improved candidate pool
    rng = random.Random(12345)
    A = list(best_A)
    
    # Build focused candidate pool
    candidate_pool = set()
    for x in A:
        candidate_pool.add(x)
    # Add smooth numbers (limited)
    for x in smooth_list[:500]:
        candidate_pool.add(x)
    # Add geometric-related numbers
    for base in [2, 3]:
        for e in range(0, 15):
            val = base ** e
            if val <= maximum:
                candidate_pool.add(val)
                for k in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
                    val2 = val * k
                    if val2 <= maximum:
                        candidate_pool.add(val2)
    # Add some random numbers
    for _ in range(200):
        candidate_pool.add(rng.randint(1, maximum))
    candidate_pool = sorted(candidate_pool)
    
    # Limit pool size
    if len(candidate_pool) > 1500:
        candidate_pool = candidate_pool[:750] + candidate_pool[-750:]
        candidate_pool = sorted(set(candidate_pool))
    
    # Local search with early termination
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(n):
            old_val = A[i]
            A_set = set(A)
            A_set.discard(old_val)
            for new_val in candidate_pool:
                if new_val == old_val or new_val in A_set:
                    continue
                A[i] = new_val
                sc = _score_with_bound(A, best_score)
                if sc < best_score:
                    best_score = sc
                    improved = True
                    break
                A[i] = old_val
            if improved:
                break
        
        # If no improvement, try random perturbations
        if not improved:
            for _ in range(30):
                if time.time() > deadline:
                    break
                i = rng.randint(0, n-1)
                old_val = A[i]
                new_val = rng.choice(candidate_pool)
                if new_val == old_val or new_val in A:
                    continue
                A[i] = new_val
                sc = _score_with_bound(A, best_score)
                if sc < best_score:
                    best_score = sc
                    improved = True
                    break
                A[i] = old_val
            if not improved:
                break
    
    # Final validation
    if not _valid(A, n, maximum):
        A = list(range(1, n+1))
    
    return A

def _valid(A, n=20, maximum=10000):
    if len(A) != n:
        return False
    if len(set(A)) != n:
        return False
    if any(x <= 0 or x > maximum for x in A):
        return False
    return True
# EVOLVE-BLOCK-END
