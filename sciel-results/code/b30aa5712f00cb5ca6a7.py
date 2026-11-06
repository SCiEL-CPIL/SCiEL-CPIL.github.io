"""Integer sum-product extremal sets - minimize max(|A+A|,|A*A|) for n=20 distinct positive integers <=10000."""
import time
import random
from itertools import combinations

_DEADLINE = 85  # seconds, safety margin before 90s kill

def _sumset_size(A):
    """Compute |A+A| with equal summands allowed (i.e., a+b for a,b in A)."""
    s = set()
    for a in A:
        for b in A:
            s.add(a + b)
    return len(s)

def _prodset_size(A):
    """Compute |A*A| with equal factors allowed (i.e., a*b for a,b in A)."""
    s = set()
    for a in A:
        for b in A:
            s.add(a * b)
    return len(s)

def _score(A):
    """Objective: max(|A+A|, |A*A|)."""
    return max(_sumset_size(A), _prodset_size(A))

def _is_valid(A):
    """Check: 20 distinct positive integers, all <= 10000."""
    if len(A) != 20:
        return False
    if len(set(A)) != 20:
        return False
    if any(x <= 0 or x > 10000 for x in A):
        return False
    return True

def _local_search(A, time_limit):
    """Hill-climbing: swap elements to reduce objective."""
    A = sorted(A)
    current_score = _score(A)
    start = time.time()
    
    # Candidate pool: numbers 1..10000
    # We'll try swapping each element with a random candidate from a broader pool
    pool = list(range(1, 10001))
    
    while time.time() - start < time_limit:
        improved = False
        random.shuffle(A)
        for i in range(20):
            if time.time() - start >= time_limit:
                break
            old_val = A[i]
            # Try several random replacements
            for _ in range(50):
                new_val = random.choice(pool)
                if new_val in A:
                    continue
                A[i] = new_val
                s = _score(A)
                if s < current_score:
                    current_score = s
                    improved = True
                    break
                A[i] = old_val
        if not improved:
            # No improvement found in this pass, try a few more passes with different shuffles
            for _ in range(3):
                if time.time() - start >= time_limit:
                    break
                random.shuffle(A)
                for i in range(20):
                    if time.time() - start >= time_limit:
                        break
                    old_val = A[i]
                    for _ in range(30):
                        new_val = random.choice(pool)
                        if new_val in A:
                            continue
                        A[i] = new_val
                        s = _score(A)
                        if s < current_score:
                            current_score = s
                            improved = True
                            break
                        A[i] = old_val
                if improved:
                    break
            if not improved:
                break
    
    return sorted(A), current_score

def _geometric_candidate(base_ratio, n=20, max_val=10000):
    """Generate a geometric progression candidate set."""
    A = []
    val = 1
    for _ in range(n):
        A.append(val)
        val = int(val * base_ratio + 0.5)
        if val > max_val:
            val = max_val
    # Ensure distinctness and <= max_val
    A = sorted(set(A))
    while len(A) < n:
        A.append(random.randint(1, max_val))
    A = sorted(set(A[:n]))
    # Fill if needed
    while len(A) < n:
        A.append(random.randint(1, max_val))
    return sorted(set(A))[:n]

def construct():
    start_time = time.time()
    random.seed(42)
    
    best_A = None
    best_score = float('inf')
    
    # Strategy 1: Start with various structured initializations
    initializations = []
    
    # 1a: Consecutive integers (baseline)
    initializations.append(list(range(1, 21)))
    
    # 1b: Powers of 2 (minimizes sumset, maximizes productset)
    initializations.append([2**i for i in range(20)])
    
    # 1c: Geometric progressions with various ratios
    for ratio in [1.1, 1.2, 1.3, 1.5, 1.7, 2.0, 2.5, 3.0]:
        A = _geometric_candidate(ratio)
        if len(A) == 20:
            initializations.append(A)
    
    # 1d: Log-spaced
    import math
    for base in [1.05, 1.1, 1.15, 1.2, 1.3]:
        A = [max(1, int(math.floor(base ** i))) for i in range(20)]
        A = sorted(set(A))
        while len(A) < 20:
            A.append(random.randint(1, 10000))
        A = sorted(set(A))[:20]
        if len(A) == 20:
            initializations.append(A)
    
    # 1e: Random small-range
    for _ in range(5):
        A = random.sample(range(1, 100), 20)
        initializations.append(sorted(A))
    
    # 1f: Random from full range
    for _ in range(5):
        A = random.sample(range(1, 10001), 20)
        initializations.append(sorted(A))
    
    # Evaluate all initializations
    for A in initializations:
        if time.time() - start_time > _DEADLINE - 5:
            break
        if not _is_valid(A):
            continue
        s = _score(A)
        if s < best_score:
            best_score = s
            best_A = list(A)
    
    # Strategy 2: Multi-start local search from best initializations
    # Use remaining time for local search
    remaining = _DEADLINE - (time.time() - start_time)
    if remaining > 2:
        # Local search from top candidates
        candidates = []
        for A in initializations:
            if _is_valid(A):
                candidates.append((A, _score(A)))
        candidates.sort(key=lambda x: x[1])
        
        # Take top 10 candidates for local search
        time_per_candidate = max(0.5, (remaining - 1) / min(10, len(candidates)))
        
        for A, s in candidates[:10]:
            if time.time() - start_time > _DEADLINE - 1:
                break
            A_ls, score_ls = _local_search(list(A), time_per_candidate)
            if score_ls < best_score:
                best_score = score_ls
                best_A = list(A_ls)
    
    # Final: ensure we have a valid result
    if best_A is None or not _is_valid(best_A):
        best_A = list(range(1, 21))
    
    return best_A
