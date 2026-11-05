"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import numpy as np
from collections import Counter
import random

def _count_duplicate_sums(subset):
    """Count the number of duplicate pair sums (i<=j) in subset."""
    s = sorted(subset)
    n = len(s)
    i_idx, j_idx = np.triu_indices(n)
    sums = np.array(s)[i_idx] + np.array(s)[j_idx]
    # Count duplicates: total pairs - unique pairs
    return len(sums) - len(set(sums.tolist()))

def _is_sidon_fast(subset):
    """Check if subset has all pair sums (i<=j) unique using numpy."""
    s = np.array(sorted(subset))
    n = len(s)
    i_idx, j_idx = np.triu_indices(n)
    sums = s[i_idx] + s[j_idx]
    return len(np.unique(sums)) == len(sums)

def _max_sidon_size_fast(s):
    """Find max Sidon subset size using branch-and-bound."""
    s = sorted(s)
    n = len(s)
    
    if _is_sidon_fast(s):
        return n
    
    best = 1
    for k in range(n - 1, 1, -1):
        found = _exists_sidon_subset(s, k)
        if found:
            best = k
            return best
    return best

def _exists_sidon_subset(s, k):
    """Check if there exists a k-subset of s that is Sidon (branch and bound)."""
    n = len(s)
    if k <= 1:
        return True
    if k > n:
        return False
    
    def backtrack(start, chosen, sums_set):
        nonlocal found
        if found:
            return
        if len(chosen) == k:
            found = True
            return
        if n - start < k - len(chosen):
            return
        for i in range(start, n):
            x = s[i]
            new_sums = []
            conflict = False
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

def _try_candidate(cands):
    """Return (valid, max_size) for a candidate list of 12 distinct ints in [0,128]."""
    if len(cands) != 12:
        return False, 99
    if len(set(cands)) != 12:
        return False, 99
    if any(x < 0 or x > 128 for x in cands):
        return False, 99
    return True, _max_sidon_size_fast(cands)

def _greedy_collision_maximize(cand, rng, max_iters=5000):
    """
    Greedy local search: swap elements to maximize duplicate pair sums.
    More duplicates => smaller max Sidon subset.
    """
    cand = list(cand)
    current_dups = _count_duplicate_sums(cand)
    
    for it in range(max_iters):
        improved = False
        # Try swapping each element with a random value
        for _ in range(5):
            i = rng.randint(0, 11)
            old_val = cand[i]
            new_val = rng.randint(0, 128)
            
            if new_val == old_val:
                continue
            if new_val in cand:
                continue
            
            cand[i] = new_val
            new_dups = _count_duplicate_sums(cand)
            
            if new_dups > current_dups:
                current_dups = new_dups
                improved = True
            else:
                cand[i] = old_val
        
        if not improved:
            # Try more aggressive: pick two elements and swap their values
            i, j = rng.sample(range(12), 2)
            cand[i], cand[j] = cand[j], cand[i]
            new_dups = _count_duplicate_sums(cand)
            if new_dups > current_dups:
                current_dups = new_dups
                improved = True
            else:
                cand[i], cand[j] = cand[j], cand[i]
        
        if not improved:
            break
    
    return cand, current_dups

def _build_candidates():
    """Generate a small, high-quality set of candidates with high additive structure."""
    candidates = []
    seen = set()
    
    def add_candidate(c):
        key = tuple(sorted(c))
        if key not in seen and len(set(c)) == 12 and all(0 <= x <= 128 for x in c):
            seen.add(key)
            candidates.append(c)
    
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
        base = [i * 10 for i in range(12)]
        c = []
        for i, b in enumerate(base):
            p = rng.randint(-5, 5)
            c.append(b + p)
        if len(set(c)) == 12 and all(0 <= x <= 128 for x in c):
            add_candidate(c)
            
    # 4. Collision-heavy constructions: pairs summing to same value
    for _ in range(100):
        c = set()
        while len(c) < 12:
            S = rng.randint(0, 256)
            a = rng.randint(0, 128)
            b = S - a
            if 0 <= b <= 128:
                c.add(a)
                c.add(b)
        if len(c) == 12:
            add_candidate(list(c))
    
    # 5. Structured collision patterns: multiple pairs with same sum
    for _ in range(50):
        c = set()
        # Create 3-4 pairs with the same sum
        for _sum_target in rng.sample(range(20, 200), 4):
            for _ in range(2):
                a = rng.randint(0, 128)
                b = _sum_target - a
                if 0 <= b <= 128:
                    c.add(a)
                    c.add(b)
        # Fill remaining with random values
        while len(c) < 12:
            c.add(rng.randint(0, 128))
        if len(c) == 12:
            add_candidate(list(c))
    
    # 6. Dense clusters with internal collisions
    for cluster_size in [6, 5, 4]:
        for _ in range(30):
            c = set()
            # First cluster: dense AP
            start = rng.randint(0, 20)
            diff = rng.randint(1, 3)
            for i in range(cluster_size):
                c.add(start + i * diff)
            # Second cluster: dense AP
            start2 = rng.randint(60, 100)
            diff2 = rng.randint(1, 3)
            for i in range(12 - cluster_size):
                c.add(start2 + i * diff2)
            if len(c) == 12 and all(0 <= x <= 128 for x in c):
                add_candidate(list(c))
    
    return candidates

def construct():
    start = time.time()
    deadline = start + 55  # 55s safety margin (evaluator kills at 90s)
    
    # Generate candidates
    candidates = _build_candidates()
    
    rng = random.Random(42)
    rng.shuffle(candidates)
    
    best_size = 99
    best_cand = None
    evaluated = 0
    best_dups = -1
    
    # Phase 1: Evaluate all generated candidates
    for cand in candidates:
        if time.time() > deadline:
            break
        valid, size = _try_candidate(cand)
        evaluated += 1
        if valid and size < best_size:
            best_size = size
            best_cand = cand
            if best_size <= 4:
                break
    
    # Phase 2: Probabilistic collision maximization via local search
    if time.time() < deadline - 10:
        # Start from best candidate found, or from a good seed
        seeds = [best_cand] if best_cand else [list(range(12))]
        # Also add some diverse seeds
        for _ in range(5):
            seed = [rng.randint(0, 128) for _ in range(12)]
            if len(set(seed)) == 12:
                seeds.append(seed)
        
        for seed in seeds:
            if time.time() > deadline - 5:
                break
            
            # Run multiple local search trials
            for trial in range(3):
                if time.time() > deadline - 5:
                    break
                
                # Perturb the seed
                perturbed = list(seed)
                for _ in range(rng.randint(2, 5)):
                    i = rng.randint(0, 11)
                    perturbed[i] = rng.randint(0, 128)
                
                if len(set(perturbed)) != 12 or any(x < 0 or x > 128 for x in perturbed):
                    continue
                
                # Greedy collision maximization
                improved_cand, dups = _greedy_collision_maximize(perturbed, rng, max_iters=2000)
                
                if time.time() > deadline - 3:
                    break
                
                valid, size = _try_candidate(improved_cand)
                if valid:
                    evaluated += 1
                    if size < best_size:
                        best_size = size
                        best_cand = improved_cand
                        if best_size <= 4:
                            break
                    elif size == best_size and dups > best_dups:
                        # Tie-break on duplicate count
                        best_cand = improved_cand
                        best_dups = dups
        
        if best_size <= 4:
            break
    
    # Phase 3: Final polish - try more intensive local search on best candidate
    if best_cand and time.time() < deadline - 5:
        for trial in range(5):
            if time.time() > deadline - 3:
                break
            perturbed = list(best_cand)
            for _ in range(rng.randint(1, 3)):
                i = rng.randint(0, 11)
                perturbed[i] = rng.randint(0, 128)
            
            if len(set(perturbed)) != 12 or any(x < 0 or x > 128 for x in perturbed):
                continue
            
            improved_cand, dups = _greedy_collision_maximize(perturbed, rng, max_iters=5000)
            valid, size = _try_candidate(improved_cand)
            if valid and size < best_size:
                best_size = size
                best_cand = improved_cand
                if best_size <= 4:
                    break
    
    if best_cand is None:
        return list(range(12))
    
    return best_cand
# EVOLVE-BLOCK-END
