import time
import math
import itertools

def _compute_score(U):
    """Compute the score for a given set U (list of nonnegative ints containing 0)."""
    U_set = set(U)
    n = len(U_set)
    if n == 0:
        return 0.0
    max_u = max(U_set)
    if max_u == 0:
        return 0.0
    
    # Compute sumset U+U
    sumset = set()
    for a in U_set:
        for b in U_set:
            sumset.add(a + b)
    
    # Compute difference set U-U
    diffset = set()
    for a in U_set:
        for b in U_set:
            diffset.add(a - b)
    
    ratio = len(diffset) / len(sumset)
    if ratio <= 0:
        return 0.0
    
    denom = math.log(2 * max_u + 1)
    if denom <= 0:
        return 0.0
    
    return 1.0 + math.log(ratio) / denom


def _try_geometric(b, m):
    """Try geometric progression U = {b^k : k=0,...,m}."""
    if b < 1:
        return None
    U = [b ** k for k in range(m + 1)]
    # Ensure 0 is in the set - it's b^0 = 1, not 0. Need to add 0.
    # Actually the problem says U must contain 0. So U = {0} ∪ {b^k : k=1,...,m}
    U = [0] + [b ** k for k in range(1, m + 1)]
    return U


def _brute_force_small(max_val, max_size):
    """Brute force search over small sets containing 0."""
    best_score = 0.0
    best_set = [0]
    
    # Try all subsets of {0, 1, 2, ..., max_val} of size up to max_size that contain 0
    candidates = list(range(1, max_val + 1))
    
    for size in range(2, max_size + 1):
        for combo in itertools.combinations(candidates, size - 1):
            U = [0] + list(combo)
            U.sort()
            score = _compute_score(U)
            if score > best_score:
                best_score = score
                best_set = U[:]
    
    return best_set, best_score


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    best_set = [0, 1, 3]
    best_score = _compute_score(best_set)
    
    # Phase 1: Brute force small sets
    if time.time() < deadline:
        # Try sets up to size 5 with values up to 10
        bs, bsc = _brute_force_small(10, 5)
        if bsc > best_score:
            best_set = bs
            best_score = bsc
    
    # Phase 2: Try geometric progressions
    if time.time() < deadline:
        for b in range(2, 20):
            for m in range(2, 15):
                if time.time() > deadline - 5:
                    break
                U = _try_geometric(b, m)
                if U is None:
                    continue
                score = _compute_score(U)
                if score > best_score:
                    best_score = score
                    best_set = U[:]
    
    # Phase 3: Try arithmetic progressions
    if time.time() < deadline:
        for a in range(1, 10):
            for d in range(1, 20):
                for m in range(2, 15):
                    if time.time() > deadline - 5:
                        break
                    U = [0] + [a + d * k for k in range(1, m + 1)]
                    score = _compute_score(U)
                    if score > best_score:
                        best_score = score
                        best_set = U[:]
    
    # Phase 4: Local search - try adding/removing elements from best set
    if time.time() < deadline - 10:
        for _ in range(500):
            if time.time() > deadline - 10:
                break
            current = best_set[:]
            current_score = best_score
            
            # Try adding a random element
            import random
            random.seed(42 + len(current) * 100 + int(time.time() * 1000) % 10000)
            
            # Try adding elements
            for _ in range(20):
                if time.time() > deadline - 10:
                    break
                new_elem = random.randint(1, 50)
                if new_elem not in current:
                    trial = current + [new_elem]
                    trial.sort()
                    trial_score = _compute_score(trial)
                    if trial_score > current_score:
                        current = trial
                        current_score = trial_score
            
            # Try removing elements (keep 0)
            for _ in range(10):
                if time.time() > deadline - 10:
                    break
                if len(current) > 2:
                    idx = random.randint(1, len(current) - 1)
                    trial = current[:idx] + current[idx + 1:]
                    trial_score = _compute_score(trial)
                    if trial_score > current_score:
                        current = trial
                        current_score = trial_score
            
            if current_score > best_score:
                best_score = current_score
                best_set = current[:]
    
    return best_set
