import time
import random
from math import log

def score_set(U):
    """Compute the score for a set U of nonnegative integers containing 0."""
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return 0.0
    
    # Compute sumset and difference set using bitsets for speed
    max_u = max(U)
    # Sumset: all sums u_i + u_j where i <= j
    # Use integer as bitset: bit k set means k is in the set
    sum_bits = 0
    diff_bits = 0
    for i in range(n):
        ui = U[i]
        # For sums: u_i + u_j for j >= i
        for j in range(i, n):
            s = ui + U[j]
            sum_bits |= (1 << s)
        # For diffs: u_j - u_i for j > i, and also negative
        for j in range(i + 1, n):
            d = U[j] - ui
            diff_bits |= (1 << d) | (1 << (max_u * 2 + d))  # store negative as offset
    diff_bits |= 1  # add 0
    
    # Count bits
    sums_count = bin(sum_bits).count('1')
    diffs_count = bin(diff_bits).count('1')
    
    ratio = diffs_count / sums_count
    if ratio <= 1:
        return -1
    D = 2 * max_u + 1
    if D <= 1:
        return -1
    return 1 + log(ratio) / log(D)

def local_search(U, deadline):
    """Perform local search to improve the set U."""
    current = sorted(set(U))
    current_score = score_set(current)
    if current_score < 0:
        return current, current_score
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        n = len(current)
        
        # Try moving individual elements with wider range
        for i in range(1, n):
            old_val = current[i]
            # Try a range of nearby values - wider range due to faster scoring
            for delta in range(-30, 31):
                if delta == 0:
                    continue
                new_val = old_val + delta
                if new_val <= 0:
                    continue
                # Ensure uniqueness
                if new_val in current:
                    continue
                new_set = [current[j] if j != i else new_val for j in range(n)]
                new_set = sorted(new_set)
                new_score = score_set(new_set)
                if new_score > current_score:
                    current = new_set
                    current_score = new_score
                    improved = True
                    break
            if improved:
                break
        
        # Weighted greedy addition: try candidates that maximize score improvement
        if time.time() < deadline and n < 25:
            max_val = max(current)
            candidates_to_try = []
            # Values near max with geometric spacing
            for new_val in range(max_val + 1, max_val + 80):
                candidates_to_try.append(new_val)
            # Gaps between existing elements
            for i in range(1, n):
                gap = current[i] - current[i-1]
                if gap > 3:
                    candidates_to_try.append(current[i-1] + 1)
                    candidates_to_try.append(current[i] - 1)
                    if gap > 5:
                        mid = (current[i-1] + current[i]) // 2
                        candidates_to_try.append(mid)
            # Geometric candidates based on current structure
            if n >= 2:
                ratio_est = current[1] / max(1, current[0]) if current[0] > 0 else 2
                base_est = max(2, int(ratio_est))
                for k in range(1, 6):
                    cand = max_val * (base_est ** k)
                    if cand > max_val and cand < max_val * 1000:
                        candidates_to_try.append(cand)
            random.shuffle(candidates_to_try)
            for new_val in candidates_to_try:
                if new_val <= 0 or new_val in current:
                    continue
                new_set = sorted(current + [new_val])
                new_score = score_set(new_set)
                if new_score > current_score:
                    current = new_set
                    current_score = new_score
                    improved = True
                    break
        
        # Try removing an element
        if time.time() < deadline and n > 4:
            for i in range(1, n):
                new_set = [current[j] for j in range(n) if j != i]
                new_score = score_set(new_set)
                if new_score > current_score:
                    current = new_set
                    current_score = new_score
                    improved = True
                    break
        
        # Try larger perturbations (kick) if stuck
        if time.time() < deadline and n >= 3:
            # Randomly perturb one element significantly
            i = random.randint(1, n - 1)
            old_val = current[i]
            for _ in range(5):
                new_val = old_val + random.randint(-50, 50)
                if new_val <= 0 or new_val in current:
                    continue
                new_set = [current[j] if j != i else new_val for j in range(n)]
                new_set = sorted(new_set)
                new_score = score_set(new_set)
                if new_score > current_score:
                    current = new_set
                    current_score = new_score
                    improved = True
                    break
    
    return current, current_score

def geometric_set(base, k):
    """Generate a geometric sequence: 0, 1, base, base^2, ..., base^(k-1)."""
    U = [0]
    val = 1
    for _ in range(k):
        U.append(val)
        val *= base
    return U

def base_expansion_set(base, digits, max_level):
    """Generate a set using base-expansion hierarchy.
    
    Elements are u_k = sum_{j=0}^k b_j * base^j where b_j in {0, 1}.
    We include all combinations up to a certain level.
    """
    U = set([0])
    # Generate all subsets of digits up to max_level
    for level in range(1, max_level + 1):
        for mask in range(1, 1 << level):
            val = 0
            for j in range(level):
                if mask & (1 << j):
                    val += digits[j] * (base ** j)
            U.add(val)
    return sorted(U)

def generate_candidates(deadline):
    """Generate diverse candidate sets."""
    candidates = []
    
    # 1. Geometric sequences with various bases
    for base in [2, 3, 4, 5, 7, 10, 13]:
        for k in range(3, 14):
            if time.time() > deadline - 20:
                return candidates
            U = geometric_set(base, k)
            candidates.append(U)
    
    # 2. Base-expansion sets
    for base in [2, 3, 4, 5]:
        for max_level in range(3, 9):
            if time.time() > deadline - 20:
                return candidates
            digits = [1] * max_level  # All digits are 1
            U = base_expansion_set(base, digits, max_level)
            if len(U) > 2:
                candidates.append(U)
    
    # 3. Powers with offsets
    for base in [2, 3, 4, 5]:
        for offset in [0, 1, 2, 3]:
            U = [0]
            val = 1 + offset
            for _ in range(12):
                U.append(val)
                val = val * base + offset
            candidates.append(U)
    
    # 4. Randomized geometric
    random.seed(12345)
    for seed in range(25):
        if time.time() > deadline - 20:
            return candidates
        random.seed(seed)
        base = random.choice([2, 3, 4, 5, 6, 7, 8, 11, 16])
        k = random.randint(4, 12)
        U = [0]
        val = random.randint(1, 5)
        for _ in range(k):
            U.append(val)
            val = val * base + random.randint(-2, 2)
        if len(U) > 2:
            candidates.append(sorted(set(U)))
    
    # 5. Sparse random sets
    random.seed(999)
    for _ in range(20):
        if time.time() > deadline - 20:
            return candidates
        n_elem = random.randint(5, 16)
        max_val = random.randint(100, 5000)
        U = [0] + sorted(random.sample(range(1, max_val), n_elem - 1))
        candidates.append(U)
    
    # 6. Structured sets with specific gap patterns
    for pattern in ['fib', 'prime', 'square']:
        if pattern == 'fib':
            U = [0]
            a, b = 1, 2
            for _ in range(12):
                U.append(a)
                a, b = b, a + b
        elif pattern == 'prime':
            # Approximate prime gaps
            U = [0, 1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
        elif pattern == 'square':
            U = [i * i for i in range(12)]
        candidates.append(U)
    
    return candidates

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    best_set = [0, 1, 4, 9, 11]
    best_score = score_set(best_set)
    
    # Generate and evaluate candidates
    candidates = generate_candidates(deadline)
    
    # Sort candidates by initial score to prioritize promising ones
    scored_candidates = []
    for U in candidates:
        if time.time() > deadline - 30:
            break
        try:
            s = score_set(U)
            if s > 0:
                scored_candidates.append((s, U))
        except Exception:
            continue
    
    scored_candidates.sort(key=lambda x: -x[0])
    
    # Local search on top candidates first
    for s, U in scored_candidates:
        if time.time() > deadline - 10:
            break
        try:
            remaining = deadline - time.time()
            if remaining < 5:
                break
            refined, score = local_search(U, deadline)
            if score > best_score:
                best_score = score
                best_set = refined
        except Exception:
            continue
    
    # Final polish: try to improve best_set further with more aggressive search
    if time.time() < deadline - 5:
        try:
            refined, score = local_search(best_set, deadline)
            if score > best_score:
                best_score = score
                best_set = refined
        except Exception:
            pass
    
    return best_set
