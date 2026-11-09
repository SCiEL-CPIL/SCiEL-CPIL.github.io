"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import itertools
import time
import random

# EVOLVE-BLOCK-START
def _is_sidon(subset):
    """Check if a sorted tuple has all pair sums (i<=j) unique."""
    sums = set()
    n = len(subset)
    for i in range(n):
        for j in range(i, n):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _max_sidon_size(candidates):
    """Find the largest Sidon subset size of the given 12-element set using branch and bound."""
    candidates = sorted(candidates)
    n = len(candidates)
    best = 0
    
    def _search(current, remaining, sums_so_far):
        nonlocal best
        if not remaining:
            return
        if len(current) + len(remaining) <= best:
            return
        for idx, val in enumerate(remaining):
            # Check if val can be added
            valid = True
            new_sums = set(sums_so_far)
            for v in current:
                s = v + val
                if s in new_sums:
                    valid = False
                    break
                new_sums.add(s)
            if valid:
                s = val + val
                if s in new_sums:
                    valid = False
                else:
                    new_sums.add(s)
            if valid:
                new_current = current + [val]
                new_remaining = remaining[idx+1:]
                if len(new_current) > best:
                    best = len(new_current)
                _search(new_current, new_remaining, new_sums)
    
    _search([], candidates, set())
    return best

def _count_sum_collisions(candidates):
    """Count how many pair sum collisions exist in the set."""
    candidates = sorted(candidates)
    n = len(candidates)
    sum_counts = {}
    for i in range(n):
        for j in range(i, n):
            s = candidates[i] + candidates[j]
            sum_counts[s] = sum_counts.get(s, 0) + 1
    collisions = sum(c - 1 for c in sum_counts.values() if c > 1)
    return collisions

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s kill
    
    best_set = None
    best_size = 13  # we want to minimize this
    
    # Strategy 1: Exhaustive search over arithmetic progressions
    # AP: {a, a+d, a+2d, ..., a+11d} all in [0,128]
    for d in range(1, 12):
        max_a = 128 - 11*d
        for a in range(0, max_a + 1):
            if time.time() > deadline:
                break
            candidates = [a + i*d for i in range(12)]
            ms = _max_sidon_size(candidates)
            if ms < best_size:
                best_size = ms
                best_set = candidates
                if best_size <= 4:
                    return best_set
        if time.time() > deadline:
            break
    
    # Strategy 2: Near-AP sets - perturb APs slightly
    for d in range(1, 12):
        max_a = 128 - 11*d
        for a in range(0, min(max_a + 1, 50)):
            if time.time() > deadline:
                break
            base = [a + i*d for i in range(12)]
            # Try small perturbations
            for perturb_idx in range(12):
                for delta in [-2, -1, 1, 2]:
                    if time.time() > deadline:
                        break
                    new_set = base[:]
                    new_val = new_set[perturb_idx] + delta
                    if 0 <= new_val <= 128 and new_val not in new_set:
                        new_set[perturb_idx] = new_val
                        new_set.sort()
                        ms = _max_sidon_size(new_set)
                        if ms < best_size:
                            best_size = ms
                            best_set = new_set
                            if best_size <= 4:
                                return best_set
        if time.time() > deadline:
            break
    
    # Strategy 3: Random search biased toward dense/low-value sets
    random.seed(42)
    for _ in range(2000):
        if time.time() > deadline:
            break
        # Bias toward smaller numbers to create more collisions
        if random.random() < 0.5:
            # Dense in low range
            start = random.randint(0, 40)
            candidates = random.sample(range(start, min(start + 50, 129)), 12)
        else:
            # AP-like with noise
            d = random.randint(1, 11)
            a = random.randint(0, max(0, 128 - 11*d))
            candidates = [a + i*d + random.randint(-1, 1) for i in range(12)]
            # Fix duplicates and out-of-range
            seen = set()
            valid_candidates = []
            for c in candidates:
                c = max(0, min(128, c))
                if c not in seen:
                    seen.add(c)
                    valid_candidates.append(c)
            while len(valid_candidates) < 12:
                c = random.randint(0, 128)
                if c not in seen:
                    seen.add(c)
                    valid_candidates.append(c)
            candidates = valid_candidates
        
        ms = _max_sidon_size(candidates)
        if ms < best_size:
            best_size = ms
            best_set = candidates
            if best_size <= 4:
                return best_set
    
    # Strategy 4: Local search from best found
    if best_set is not None:
        for _ in range(500):
            if time.time() > deadline:
                break
            current = best_set[:]
            # Try swapping one element
            idx = random.randint(0, 11)
            old_val = current[idx]
            new_val = random.randint(0, 128)
            if new_val == old_val or new_val in current:
                continue
            current[idx] = new_val
            current.sort()
            ms = _max_sidon_size(current)
            if ms < best_size:
                best_size = ms
                best_set = current
                if best_size <= 4:
                    return best_set
    
    if best_set is None:
        best_set = list(range(12))
    
    return best_set
# EVOLVE-BLOCK-END
