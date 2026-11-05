"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import random
import numpy as np
from collections import Counter

def _is_sidon_fast(subset):
    """Check if subset has all pair sums (i<=j) unique using numpy."""
    s = np.array(sorted(subset))
    n = len(s)
    # Compute all pair sums i<=j
    i_idx, j_idx = np.triu_indices(n)
    sums = s[i_idx] + s[j_idx]
    # Check for duplicates
    return len(np.unique(sums)) == len(sums)

def _exists_sidon_subset(s, k):
    """Check if there exists a k-subset of s that is Sidon (branch and bound)."""
    n = len(s)
    if k <= 1:
        return True
    if k > n:
        return False
    
    # Precompute sorted list
    s = sorted(s)
    
    def backtrack(start, chosen, sums_set):
        nonlocal found
        if found:
            return
        if len(chosen) == k:
            found = True
            return
        # Pruning: not enough elements left
        if n - start < k - len(chosen):
            return
        
        # Try each element from start
        for i in range(start, n):
            x = s[i]
            # Check if adding x creates duplicate sums
            conflict = False
            new_sums = []
            for y in chosen:
                sm = x + y
                if sm in sums_set:
                    conflict = True
                    break
                new_sums.append(sm)
            if not conflict:
                if x + x in sums_set:
                    conflict = True
            if not conflict:
                new_sums_set = sums_set | set(new_sums) | {x + x}
                chosen.append(x)
                backtrack(i + 1, chosen, new_sums_set)
                chosen.pop()
                if found:
                    return
    
    found = False
    backtrack(0, [], set())
    return found

def _max_sidon_size_fast(s):
    """Find max Sidon subset size using branch-and-bound."""
    s = sorted(s)
    n = len(s)
    
    # Quick check: is the whole set Sidon?
    if _is_sidon_fast(s):
        return n
    
    # Branch and bound
    best = 1
    # Try from largest k down
    for k in range(n - 1, 1, -1):
        # Use a greedy + backtrack approach
        # First try: can we find ANY k-subset that is Sidon?
        found = _exists_sidon_subset(s, k)
        if found:
            best = k
            return best
    return best

def _try_candidate(cands):
    """Return (valid, max_size) for a candidate list of 12 distinct ints in [0,128]."""
    if len(cands) != 12:
        return False, 99
    if len(set(cands)) != 12:
        return False, 99
    if any(x < 0 or x > 128 for x in cands):
        return False, 99
    return True, _max_sidon_size_fast(cands)

def _count_collisions(cands):
    """Count the number of duplicate pair sums (i<=j) in the set. Higher is better for minimization."""
    s = sorted(cands)
    n = len(s)
    sums = []
    for i in range(n):
        for j in range(i, n):
            sums.append(s[i] + s[j])
    c = Counter(sums)
    collisions = 0
    for v in c.values():
        if v > 1:
            collisions += v - 1
    return collisions

def _greedy_swap_optimize(cands, max_iters=500, deadline=None):
    """
    Local search: try swapping elements to maximize collisions (minimize Sidon size).
    Returns the best candidate found.
    """
    current = list(cands)
    current_collisions = _count_collisions(current)
    best = list(current)
    best_collisions = current_collisions
    
    rng = random.Random(12345)
    pool = list(range(129))
    
    for _ in range(max_iters):
        if deadline and time.time() > deadline:
            break
        
        # Pick a random index to replace
        idx = rng.randint(0, 11)
        old_val = current[idx]
        
        # Pick a new value from the pool (excluding current values)
        current_set = set(current)
        available = [x for x in pool if x not in current_set]
        if not available:
            break
        new_val = rng.choice(available)
        
        # Try the swap
        current[idx] = new_val
        new_collisions = _count_collisions(current)
        
        if new_collisions > current_collisions:
            current_collisions = new_collisions
            if new_collisions > best_collisions:
                best_collisions = new_collisions
                best = list(current)
        else:
            # Revert
            current[idx] = old_val
            
    return best

def _build_candidates():
    """Generate a small, high-quality set of candidates with high additive structure."""
    candidates = []
    seen = set()
    
    def add_candidate(c):
        key = tuple(sorted(c))
        if key not in seen and len(set(c)) == 12 and all(0 <= x <= 128 for x in c):
            seen.add(key)
            candidates.append(list(c))
    
    # 1. Arithmetic Progressions (known to have small Sidon subsets)
    for start in range(0, 10):
        for diff in range(1, 15):
            c = [start + i * diff for i in range(12)]
            if max(c) <= 128:
                add_candidate(c)
    
    # 2. Two-cluster APs
    for start1 in range(0, 5):
        for diff1 in range(1, 5):
            for start2 in range(50, 100):
                for diff2 in range(1, 5):
                    c1 = [start1 + i * diff1 for i in range(6)]
                    c2 = [start2 + i * diff2 for i in range(6)]
                    c = c1 + c2
                    if len(set(c)) == 12 and max(c) <= 128:
                        add_candidate(c)
    
    # 3. Random perturbations of APs
    rng = random.Random(123)
    for _ in range(200):
        base = [i * 10 for i in range(12)] # Large gaps to allow perturbation
        c = []
        for i, b in enumerate(base):
            p = rng.randint(-5, 5)
            c.append(b + p)
        # Ensure distinct and in range
        if len(set(c)) == 12 and all(0 <= x <= 128 for x in c):
            add_candidate(c)
            
    # 4. Specific "collision-heavy" constructions
    # e.g. pairs summing to same value
    for _ in range(100):
        c = set()
        while len(c) < 12:
            # Pick a sum S and split it
            S = rng.randint(0, 256)
            a = rng.randint(0, 128)
            b = S - a
            if 0 <= b <= 128:
                c.add(a)
                c.add(b)
        if len(c) == 12:
            add_candidate(list(c))
            
    # 5. Modular linear maps of structured bases (from Inspiration 1)
    base_sets = [
        [0, 1, 2, 4, 8, 16, 3, 5, 6, 10, 12, 20],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        [0, 1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 66],
        [0, 1, 2, 3, 4, 5, 16, 17, 18, 19, 20, 21],
        [0, 1, 2, 3, 4, 5, 32, 33, 34, 35, 36, 37],
        [i*i for i in range(12)],
    ]
    
    for base in base_sets:
        # Try a subset of modular maps to keep candidate list manageable
        for a in range(1, 129, 2): # Step 2 to reduce count
            for b in range(0, 129, 2):
                c = [(x * a + b) % 129 for x in base]
                if len(set(c)) == 12:
                    add_candidate(c)
                # Early exit if we have enough from this base
                if len([k for k in seen if all(any(x == y for y in c) for x in k)]) > 100:
                    break
            if len([k for k in seen if all(any(x == y for y in base) for x in k)]) > 100:
                break

    return candidates

def construct():
    start = time.time()
    deadline = start + 50  # 50s safety margin (evaluator kills at 90s)
    
    # Generate candidates
    candidates = _build_candidates()
    
    rng = random.Random(42)
    rng.shuffle(candidates)
    
    best_size = 99
    best_cand = None
    evaluated = 0
    
    # First pass: Evaluate raw candidates
    for cand in candidates:
        if time.time() > deadline:
            break
        valid, size = _try_candidate(cand)
        evaluated += 1
        if valid and size < best_size:
            best_size = size
            best_cand = cand
            if best_size <= 3:
                break
            if best_size <= 4:
                if evaluated > 200:
                    break
    
    # Second pass: Local search optimization on the best candidates found
    # We re-evaluate the top few candidates with greedy swapping to see if we can improve
    if best_cand is not None:
        # Keep the top 5 best candidates found so far for local search
        # (We didn't store them all, so we just optimize the current best)
        # Let's optimize the current best candidate
        optimized = _greedy_swap_optimize(best_cand, max_iters=1000, deadline=deadline - 5)
        valid, size = _try_candidate(optimized)
        if valid and size < best_size:
            best_size = size
            best_cand = optimized
            
        # Also try optimizing a few other random candidates from the pool
        for _ in range(10):
            if time.time() > deadline - 5:
                break
            if not candidates:
                break
            cand = rng.choice(candidates)
            optimized = _greedy_swap_optimize(cand, max_iters=500, deadline=deadline - 5)
            valid, size = _try_candidate(optimized)
            if valid and size < best_size:
                best_size = size
                best_cand = optimized
                if best_size <= 3:
                    break
    
    if best_cand is None:
        # Fallback: a simple AP
        return list(range(12))
    
    return best_cand
# EVOLVE-BLOCK-END
