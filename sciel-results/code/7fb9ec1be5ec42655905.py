"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools

def _compute_sizes(A):
    """Compute |A+A| and |A*A| for set A."""
    A = list(set(A))
    sums = set()
    prods = set()
    for a in A:
        for b in A:
            sums.add(a + b)
            prods.add(a * b)
    return len(sums), len(prods)

def _score(A):
    """Score = max(|A+A|, |A*A|)."""
    s, p = _compute_sizes(A)
    return max(s, p)

def _generate_lattice_set(base_i, base_j, di, dj, offset):
    """Generate a set of 20 numbers from a 2D lattice of exponents.
    
    A = {2^(base_i + di*r) * 3^(base_j + dj*c) : r in 0..3, c in 0..4}
    """
    A = set()
    for r in range(4):
        for c in range(5):
            i = base_i + di * r
            j = base_j + dj * c
            if i < 0 or j < 0:
                continue
            val = (2 ** i) * (3 ** j)
            if val <= 10000:
                A.add(val)
    return A

def _local_search(A, time_limit, rng):
    """Local search: try swapping elements or small modifications."""
    A = sorted(set(A))
    if len(A) != 20:
        return A
    
    best_score = _score(A)
    best_A = A[:]
    start = time.time()
    
    # Precompute for speed
    def compute(A_list):
        A_set = set(A_list)
        if len(A_set) != 20:
            return 999999
        if any(x < 1 or x > 10000 for x in A_set):
            return 999999
        sums = set()
        prods = set()
        for a in A_list:
            for b in A_list:
                sums.add(a + b)
                prods.add(a * b)
        return max(len(sums), len(prods))
    
    best_score = compute(best_A)
    
    # Candidate values to try swapping in
    candidates = list(range(1, 10001))
    
    while time.time() - start < time_limit:
        # Try swapping one element
        idx = rng.randint(0, 19)
        old_val = best_A[idx]
        new_val = rng.choice(candidates)
        if new_val == old_val:
            continue
        
        new_A = best_A[:]
        new_A[idx] = new_val
        new_A = sorted(set(new_A))
        if len(new_A) != 20:
            continue
        if any(x < 1 or x > 10000 for x in new_A):
            continue
        
        new_score = compute(new_A)
        if new_score < best_score:
            best_score = new_score
            best_A = new_A
        elif new_score == best_score and rng.random() < 0.1:
            # Accept equal with small probability to escape plateaus
            best_A = new_A
    
    return best_A

def construct():
    rng = random.Random(42)
    deadline = time.time() + 80  # Safety margin for 90s limit
    
    best_overall = None
    best_overall_score = float('inf')
    
    # Phase 1: Generate diverse lattice-based initial sets
    initial_sets = []
    
    # Various lattice configurations
    configs = [
        # (base_i, base_j, di, dj)
        (0, 0, 1, 1),
        (0, 0, 1, 2),
        (0, 0, 2, 1),
        (0, 0, 2, 2),
        (1, 0, 1, 1),
        (0, 1, 1, 1),
        (1, 1, 1, 1),
        (0, 0, 1, 3),
        (0, 0, 3, 1),
        (1, 0, 2, 1),
        (0, 1, 2, 1),
        (1, 0, 1, 2),
        (0, 1, 1, 2),
        (2, 0, 1, 1),
        (0, 2, 1, 1),
        (1, 1, 1, 2),
        (1, 1, 2, 1),
        (0, 0, 1, 4),
        (0, 0, 4, 1),
        (2, 0, 1, 2),
    ]
    
    for bi, bj, di, dj in configs:
        if time.time() > deadline:
            break
        A = _generate_lattice_set(bi, bj, di, dj, 0)
        if len(A) == 20:
            initial_sets.append(sorted(A))
    
    # Also try some random multiplicative structures
    for _ in range(50):
        if time.time() > deadline:
            break
        # Random base factors
        bases = []
        for _ in range(20):
            # Random number with small prime factors
            v = 1
            for _ in range(rng.randint(0, 4)):
                v *= rng.choice([2, 3, 5])
            if v <= 10000 and v > 0:
                bases.append(v)
        if len(set(bases)) == 20:
            initial_sets.append(sorted(set(bases)))
    
    # Also include the simple range as fallback
    initial_sets.append(list(range(1, 21)))
    
    # Phase 2: Local search on each initial set
    time_per_set = 40.0 / max(len(initial_sets), 1)
    
    for A in initial_sets:
        if time.time() > deadline:
            break
        if len(set(A)) != 20:
            continue
        if any(x < 1 or x > 10000 for x in A):
            continue
        
        A_sorted = sorted(A)
        cur_score = _score(A_sorted)
        
        if cur_score < best_overall_score:
            best_overall_score = cur_score
            best_overall = A_sorted[:]
        
        # Do local search
        remaining = deadline - time.time()
        ts_time = min(time_per_set, remaining * 0.9)
        if ts_time > 0.5:
            improved = _local_search(A_sorted, ts_time, rng)
            imp_score = _score(improved)
            if imp_score < best_overall_score:
                best_overall_score = imp_score
                best_overall = improved[:]
    
    # Phase 3: If we have time, do more focused local search on best
    if best_overall is not None and time.time() < deadline - 2:
        remaining = deadline - time.time() - 1
        if remaining > 1:
            improved = _local_search(best_overall, remaining, rng)
            imp_score = _score(improved)
            if imp_score < best_overall_score:
                best_overall = improved
    
    if best_overall is None:
        return list(range(1, 21))
    
    return best_overall
# EVOLVE-BLOCK-END
