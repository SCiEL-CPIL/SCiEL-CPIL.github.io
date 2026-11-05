"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
from math import comb

def _max_sidon_size(subset):
    """Compute the largest subset of 'subset' with all pair sums (i<=j) distinct."""
    n = len(subset)
    if n <= 1:
        return n
    # Use backtracking to find max Sidon subset
    # Sort for determinism
    elems = sorted(subset)
    best = [0]
    
    def is_sidon(s):
        sums = set()
        for i in range(len(s)):
            for j in range(i, len(s)):
                val = s[i] + s[j]
                if val in sums:
                    return False
                sums.add(val)
        return True
    
    def backtrack(idx, current, current_sums):
        if idx >= n:
            best[0] = max(best[0], len(current))
            return
        # Prune: even if we take all remaining, can't beat best
        if len(current) + (n - idx) <= best[0]:
            return
        # Try not taking idx
        backtrack(idx + 1, current, current_sums)
        # Try taking idx
        elem = elems[idx]
        new_sums = set(current_sums)
        valid = True
        for j in range(len(current)):
            val = current[j] + elem
            if val in new_sums:
                valid = False
                break
            new_sums.add(val)
        if valid:
            val = elem + elem
            if val in new_sums:
                valid = False
            else:
                new_sums.add(val)
        if valid:
            backtrack(idx + 1, current + [elem], new_sums)
    
    backtrack(0, [], set())
    return best[0]

def _eval_candidate(cand):
    """Return the max Sidon subset size of the candidate."""
    return _max_sidon_size(cand)

def _valid(cand):
    """Check candidate is 12 distinct integers in [0,128]."""
    if len(cand) != 12:
        return False
    if len(set(cand)) != 12:
        return False
    return all(0 <= x <= 128 for x in cand)

def construct():
    start = time.time()
    deadline = start + 80.0  # 80s safety margin within 90s limit
    
    # Start with arithmetic progression - known to have small Sidon subsets
    # AP: {0, 12, 24, ..., 132} -> but 132 > 128, so {0, 11, 22, ..., 121}
    ap = list(range(0, 132, 12))  # [0,12,24,36,48,60,72,84,96,108,120,132] - 132 too big
    ap = list(range(0, 128, 12))  # [0,12,24,36,48,60,72,84,96,108,120] - only 11
    # Let's use step 11: [0,11,22,33,44,55,66,77,88,99,110,121] - 12 elements, max 121
    ap = list(range(0, 122, 11))  # [0,11,22,33,44,55,66,77,88,99,110,121]
    
    best_cand = ap
    best_score = _eval_candidate(ap)
    
    # Try different APs
    for step in range(10, 13):
        for offset in range(0, 13):
            cand = [offset + i * step for i in range(12)]
            if max(cand) <= 128 and min(cand) >= 0 and len(set(cand)) == 12:
                s = _eval_candidate(cand)
                if s < best_score:
                    best_score = s
                    best_cand = cand
            if time.time() > deadline:
                return best_cand
    
    # Try random perturbations of the best AP
    rng = random.Random(42)
    for iteration in range(5000):
        if time.time() > deadline:
            break
        cand = list(best_cand)
        # Randomly perturb a few elements
        for _ in range(rng.randint(1, 3)):
            idx = rng.randint(0, 11)
            cand[idx] = rng.randint(0, 128)
        if not _valid(cand):
            continue
        s = _eval_candidate(cand)
        if s < best_score:
            best_score = s
            best_cand = cand
    
    return best_cand
# EVOLVE-BLOCK-END
