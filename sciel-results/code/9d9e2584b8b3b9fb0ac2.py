"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools
from collections import defaultdict

# EVOLVE-BLOCK-START
def _is_sidon_subset(s):
    """Check if subset s has all pair sums (including diagonals) unique."""
    n = len(s)
    sums = set()
    for i in range(n):
        for j in range(i, n):
            sm = s[i] + s[j]
            if sm in sums:
                return False
            sums.add(sm)
    return True

def _largest_sidon_subset(s):
    """Find the largest subset of s with all pair sums (including diagonals) unique."""
    s = sorted(s)
    n = len(s)
    best = 0
    for size in range(n, 0, -1):
        for combo in itertools.combinations(range(n), size):
            subset = [s[i] for i in combo]
            if _is_sidon_subset(subset):
                best = size
                return best
    return best

def _has_3ap(s):
    """Check if any 3-element subset of s forms an arithmetic progression."""
    s_set = set(s)
    for i in range(len(s)):
        for j in range(len(s)):
            if i == j:
                continue
            c = 2*s[i] - s[j]
            if c in s_set and c != s[i] and c != s[j]:
                return True
    return False

def _find_3ap_free_set():
    """Try to find a 12-element 3-AP-free set in [0,128]."""
    random.seed(12345)
    deadline = time.time() + 15
    best = None
    best_score = 13
    
    # Try random sets and check for 3-AP-free
    for attempt in range(200000):
        if time.time() > deadline:
            break
        candidate = sorted(random.sample(range(129), 12))
        if not _has_3ap(candidate):
            return candidate
    
    # Try greedy construction
    # Start with known 3-AP-free sets and extend
    # A well-known construction: numbers with only 0s and 1s in base 3
    base3_candidates = []
    for i in range(129):
        digits = []
        n = i
        while n > 0:
            digits.append(n % 3)
            n //= 3
        if all(d <= 1 for d in digits):
            base3_candidates.append(i)
    
    if len(base3_candidates) >= 12:
        # Try to find a 12-subset of base3_candidates that is 3-AP-free
        # Since base3 numbers with digits 0,1 are automatically 3-AP-free
        # (no carry in base 3 addition means no 3-AP)
        # Actually, the set of numbers with only 0/1 in base 3 IS 3-AP-free
        return sorted(base3_candidates[:12])
    
    return None

def _evaluate(s):
    """Return the largest Sidon subset size for set s."""
    return _largest_sidon_subset(s)

def _local_search(current, deadline):
    """Hill-climbing: try swapping one element with another from [0,128]."""
    current = sorted(current)
    best_score = _evaluate(current)
    best_set = list(current)
    
    all_vals = set(range(129))
    in_set = set(current)
    
    while time.time() < deadline:
        improved = False
        for i in range(12):
            for new_val in all_vals - in_set:
                if time.time() > deadline:
                    break
                candidate = list(current)
                candidate[i] = new_val
                candidate = sorted(candidate)
                score = _evaluate(candidate)
                if score < best_score:
                    best_score = score
                    best_set = candidate
                    improved = True
                    break
            if improved or time.time() > deadline:
                break
        if not improved:
            break
        current = best_set
        in_set = set(current)
    
    return best_set, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 70  # 70s to leave safe margin for evaluator overhead
    
    # First, try to find a 3-AP-free set (which gives answer 2)
    result = _find_3ap_free_set()
    if result is not None:
        return sorted(result)
    
    # Fallback: search for sets with small Sidon number
    best_set = list(range(12))
    best_score = _evaluate(best_set)
    
    random.seed(42)
    while time.time() < deadline:
        candidate = sorted(random.sample(range(129), 12))
        score = _evaluate(candidate)
        if score < best_score:
            best_score = score
            best_set = candidate
            if best_score <= 2:
                break
            # Local search from this candidate
            best_set, best_score = _local_search(best_set, deadline)
            if best_score <= 2:
                break
    
    return sorted(best_set)
# EVOLVE-BLOCK-END
