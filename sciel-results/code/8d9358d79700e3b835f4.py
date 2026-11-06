import time
import random
import numpy as np

MAX_N = 20
MAX_VAL = 10000
TIME_LIMIT = 85  # safety margin before 90s kill


def _objective(A):
    """Compute max(|A+A|, |A*A|) for a set A of distinct positive integers."""
    n = len(A)
    sums = set()
    prods = set()
    for i in range(n):
        ai = A[i]
        for j in range(i, n):
            aj = A[j]
            sums.add(ai + aj)
            prods.add(ai * aj)
    return max(len(sums), len(prods))


def _is_valid(A):
    if len(A) != 20:
        return False
    if len(set(A)) != 20:
        return False
    for x in A:
        if not isinstance(x, int) or x < 1 or x > 10000:
            return False
    return True


def _construct_candidates():
    """Generate candidate starting sets with strong multiplicative structure."""
    candidates = []
    
    # 1. Powers of 2 (10) + Powers of 3 (10): max multiplicative structure
    A1 = [2**i for i in range(10)]  # 1,2,4,8,...,512
    A2 = [3**i for i in range(10)]  # 1,3,9,...,19683 -> too big, cap at 10000
    A2 = [x for x in A2 if x <= 10000]
    # Need exactly 20 distinct, so combine and deduplicate
    combined = list(set(A1 + A2))
    if len(combined) >= 20:
        candidates.append(sorted(combined[:20]))
    elif len(combined) == 20:
        candidates.append(sorted(combined))
    
    # 2. Smooth numbers: products of small primes
    smooths = set()
    for i in range(14):  # 2^i up to 2^13=8192
        p2 = 1 << i
        if p2 > MAX_VAL:
            break
        for j in range(10):  # 3^j
            p3 = p2 * (3 ** j)
            if p3 > MAX_VAL:
                break
            for k in range(7):  # 5^k
                v = p3 * (5 ** k)
                if v > MAX_VAL:
                    break
                smooths.add(v)
    sorted_smooths = sorted(smooths)
    if len(sorted_smooths) >= 20:
        # Take the 20 smallest for compactness
        candidates.append(sorted_smooths[:20])
        # Also try a mix: some small, some larger
        if len(sorted_smooths) >= 40:
            mid = len(sorted_smooths) // 2
            candidates.append(sorted(sorted_smooths[:10] + sorted_smooths[mid:mid+10]))
    
    # 3. Two geometric progressions with different bases
    for b1, b2, n1, n2 in [(2, 3, 12, 8), (2, 3, 11, 9), (2, 5, 10, 10), (3, 5, 9, 11)]:
        gp1 = []
        v = 1
        for _ in range(n1):
            if v <= MAX_VAL and v not in gp1:
                gp1.append(v)
            v *= b1
        gp2 = []
        v = 1
        for _ in range(n2):
            if v <= MAX_VAL and v not in gp2 and v not in gp1:
                gp2.append(v)
            v *= b2
        if len(gp1) + len(gp2) >= 20:
            combined = sorted(gp1 + gp2)
            if len(set(combined)) == 20:
                candidates.append(combined)
    
    # 4. Arithmetic + geometric hybrid
    arith = list(range(1, 11))  # 1..10
    geom = [2**i for i in range(10, 20)]  # 1024..524288 -> cap
    geom = [x for x in geom if x <= 10000]
    combined = list(set(arith + geom))
    if len(combined) >= 20:
        candidates.append(sorted(combined[:20]))
    
    # 5. Random smooth-like sets
    rng = random.Random(42)
    for seed in [42, 123, 999]:
        r = random.Random(seed)
        A = set()
        while len(A) < 20:
            # Generate random smooth number
            p2 = r.randint(0, 10)
            p3 = r.randint(0, 7)
            p5 = r.randint(0, 5)
            v = (2 ** p2) * (3 ** p3) * (5 ** p5)
            if 1 <= v <= MAX_VAL:
                A.add(v)
        candidates.append(sorted(A))
    
    # 6. Pure random for diversity
    candidates.append(rng.sample(range(1, 10001), 20))
    
    # 7. Consecutive integers (baseline)
    candidates.append(list(range(1, 21)))
    
    return candidates


def _local_search(A, time_limit, rng):
    """Simulated annealing local search with swap and replace moves."""
    A = sorted(A)
    current_obj = _objective(A)
    best_A = list(A)
    best_obj = current_obj
    
    start_time = time.time()
    temperature = 30.0
    cooling_rate = 0.997
    
    iterations = 0
    
    while temperature > 0.1:
        if time.time() - start_time > time_limit:
            break
        
        iterations += 1
        move_type = rng.random()
        
        if move_type < 0.35:
            # Swap two elements with nearby values
            i, j = rng.sample(range(20), 2)
            old_i, old_j = A[i], A[j]
            for _ in range(20):
                if time.time() - start_time > time_limit:
                    break
                delta_i = rng.randint(-8, 8)
                delta_j = rng.randint(-8, 8)
                new_i = max(1, min(MAX_VAL, old_i + delta_i))
                new_j = max(1, min(MAX_VAL, old_j + delta_j))
                if new_i == new_j or new_i in A or new_j in A:
                    continue
                A[i], A[j] = new_i, new_j
                new_obj = _objective(A)
                A[i], A[j] = old_i, old_j
                
                delta = new_obj - current_obj
                if delta <= 0 or rng.random() < min(1.0, max(0.01, -delta / (temperature + 1e-10))):
                    A[i], A[j] = new_i, new_j
                    current_obj = new_obj
                    if new_obj < best_obj:
                        best_obj = new_obj
                        best_A = list(A)
                    break
        elif move_type < 0.70:
            # Replace one element
            idx = rng.randint(0, 19)
            old_val = A[idx]
            for _ in range(25):
                if time.time() - start_time > time_limit:
                    break
                r = rng.random()
                if r < 0.25:
                    new_val = max(1, min(MAX_VAL, old_val + rng.randint(-4, 4)))
                elif r < 0.50:
                    new_val = max(1, min(MAX_VAL, old_val + rng.randint(-20, 20)))
                elif r < 0.75:
                    # Try to move toward smooth number
                    nearest_smooth = min(range(1, MAX_VAL+1), key=lambda x: abs(x - old_val))
                    new_val = max(1, min(MAX_VAL, old_val + rng.randint(-50, 50)))
                else:
                    new_val = rng.randint(1, MAX_VAL)
                
                if new_val in A:
                    continue
                
                A[idx] = new_val
                new_obj = _objective(A)
                A[idx] = old_val
                
                delta = new_obj - current_obj
                if delta <= 0 or rng.random() < min(1.0, max(0.01, -delta / (temperature + 1e-10))):
                    A[idx] = new_val
                    current_obj = new_obj
                    if new_obj < best_obj:
                        best_obj = new_obj
                        best_A = list(A)
                    break
        else:
            # Mutate: pick two elements and shift them
            i, j = rng.sample(range(20), 2)
            old_i, old_j = A[i], A[j]
            for _ in range(15):
                if time.time() - start_time > time_limit:
                    break
                shift = rng.randint(-12, 12)
                new_i = max(1, min(MAX_VAL, old_i + shift))
                new_j = max(1, min(MAX_VAL, old_j - shift))
                if new_i == new_j or new_i in A or new_j in A:
                    continue
                A[i], A[j] = new_i, new_j
                new_obj = _objective(A)
                A[i], A[j] = old_i, old_j
                
                delta = new_obj - current_obj
                if delta <= 0 or rng.random() < min(1.0, max(0.01, -delta / (temperature + 1e-10))):
                    A[i], A[j] = new_i, new_j
                    current_obj = new_obj
                    if new_obj < best_obj:
                        best_obj = new_obj
                        best_A = list(A)
                    break
        
        temperature *= cooling_rate
    
    return best_A, best_obj


def construct():
    start_time = time.time()
    time_budget = TIME_LIMIT - 5
    
    candidates = _construct_candidates()
    
    best_A = None
    best_obj = float('inf')
    
    rng = random.Random(42)
    
    # Evaluate all candidates
    scored_candidates = []
    for A in candidates:
        if not _is_valid(A):
            continue
        obj = _objective(A)
        scored_candidates.append((obj, A))
    
    scored_candidates.sort(key=lambda x: x[0])
    
    # Run local search on top candidates with more time each
    n_top = min(6, len(scored_candidates))
    for idx, (obj, A) in enumerate(scored_candidates[:n_top]):
        elapsed = time.time() - start_time
        remaining = time_budget - elapsed
        if remaining < 2.0:
            break
        # Give more time to better candidates
        time_alloc = remaining * (0.5 if idx == 0 else 0.3)
        refined_A, refined_obj = _local_search(A, time_alloc, rng)
        if refined_obj < best_obj:
            best_obj = refined_obj
            best_A = refined_A
    
    if best_A is None:
        best_A = list(range(1, 21))
        best_obj = _objective(best_A)
    
    # Final intensive refinement on the best found
    remaining = TIME_LIMIT - 5 - (time.time() - start_time)
    if remaining > 3:
        try:
            rng2 = random.Random(123)
            refined_A, refined_obj = _local_search(best_A, remaining * 0.6, rng2)
            if refined_obj < best_obj:
                best_A = refined_A
                best_obj = refined_obj
        except Exception:
            pass
    
    # Final polish
    remaining = TIME_LIMIT - 5 - (time.time() - start_time)
    if remaining > 2:
        try:
            rng3 = random.Random(999)
            refined_A, refined_obj = _local_search(best_A, remaining * 0.8, rng3)
            if refined_obj < best_obj:
                best_A = refined_A
                best_obj = refined_obj
        except Exception:
            pass
    
    if not _is_valid(best_A):
        best_A = list(range(1, 21))
    
    return sorted(best_A)
