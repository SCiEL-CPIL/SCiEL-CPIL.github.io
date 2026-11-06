import time
import random
import itertools

def is_sidon(subset):
    """Check if subset has all unordered pair sums (i<=j) unique."""
    s = sorted(subset)
    n = len(s)
    seen_sums = set()
    for i in range(n):
        for j in range(i, n):
            sm = s[i] + s[j]
            if sm in seen_sums:
                return False
            seen_sums.add(sm)
    return True

def max_sidon_size(S):
    """Find largest subset of S that is Sidon (all pair sums i<=j unique)."""
    S = sorted(S)
    n = len(S)
    for k in range(n, 0, -1):
        for comb in itertools.combinations(S, k):
            if is_sidon(list(comb)):
                return k
    return 0

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    # Strategy: Dense clusters create many pair-sum collisions.
    # A set like [0,1,2,3,4,5,6,7,8,9,10,11] has pair sums in [0,22].
    # For Sidon size 7, need 28 distinct sums but only 23 available -> impossible.
    # So max Sidon <= 6 for contiguous 12-set.
    # Can we do better (max Sidon = 5)?
    # For k=6 need 21 sums. If we can make the sum range < 21, then k=6 impossible.
    # But 12 distinct ints in [0,128] with min sum 0, max sum >= 11 (if contiguous from 0).
    # Actually min possible max sum for 12 distinct non-neg ints is 11 (0+11).
    # Min possible sum range: if set is {0,1,...,11}, sums range [0,22], width 23.
    # To get sum range < 21, need max pair sum - min pair sum < 21.
    # Min pair sum is 0 (if 0 in set), max pair sum is max(S)+second_max(S).
    # For 12 distinct in [0,128], min possible max pair sum: {0,1,...,11} gives 22.
    # So sum range >= 22 for any 12 distinct non-neg ints containing 0.
    # Without 0: min sum >= 1+2=3, max sum >= 11+12=23, range >= 20.
    # Hmm, tricky. Let's just search.
    
    # Start with known good candidates
    candidates = [
        list(range(12)),                    # 0-11, max Sidon likely 6
        list(range(1, 13)),                 # 1-12
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12],
        [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12],
        [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12],
        [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12],
        [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12],
        [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12],
        [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 14],
    ]
    
    # Deduplicate
    seen = set()
    unique_candidates = []
    for c in candidates:
        key = tuple(sorted(c))
        if key not in seen:
            seen.add(key)
            unique_candidates.append(sorted(c))
    
    current_best = unique_candidates[0]
    current_best_val = max_sidon_size(current_best)
    
    for c in unique_candidates[1:]:
        if time.time() > deadline:
            break
        val = max_sidon_size(c)
        if val < current_best_val:
            current_best_val = val
            current_best = c
        elif val == current_best_val:
            # Tie-break: prefer sets with more sum collisions (smaller sum range)
            pass  # Keep first found for simplicity
    
    # Systematic local search: try replacing each element with nearby values
    if time.time() < deadline - 10:
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for i in range(12):
                if time.time() > deadline - 5:
                    break
                orig = current_best[i]
                # Try replacing with values in a window around orig
                lo = max(0, orig - 5)
                hi = min(128, orig + 5)
                for new_val in range(lo, hi + 1):
                    if new_val == orig:
                        continue
                    cand = list(current_best)
                    if new_val in cand:
                        continue
                    cand[i] = new_val
                    cand = sorted(cand)
                    if len(set(cand)) != 12:
                        continue
                    val = max_sidon_size(cand)
                    if val < current_best_val:
                        current_best_val = val
                        current_best = cand
                        improved = True
                        break
    
    # Random search with hill climbing
    if time.time() < deadline - 3:
        random.seed(42)
        for _ in range(300):
            if time.time() > deadline - 3:
                break
            
            if random.random() < 0.7:
                cand = list(current_best)
            else:
                # Random dense set
                start = random.randint(0, 8)
                cand = list(range(start, start + 12))
            
            # Perturb 1-3 elements
            n_perturb = random.randint(1, 3)
            for _ in range(n_perturb):
                idx = random.randint(0, 11)
                new_val = random.randint(0, 30)
                attempts = 0
                while new_val in cand and attempts < 30:
                    new_val = random.randint(0, 30)
                    attempts += 1
                if new_val in cand:
                    continue
                cand[idx] = new_val
            
            cand = sorted(set(cand))
            if len(cand) != 12:
                continue
            
            val = max_sidon_size(cand)
            if val < current_best_val:
                current_best_val = val
                current_best = cand
    
    return current_best
