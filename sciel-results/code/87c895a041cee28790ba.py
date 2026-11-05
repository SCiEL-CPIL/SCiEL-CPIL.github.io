"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import random
import numpy as np
from collections import Counter

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

def _max_sidon_size_exact(s):
    """Exactly compute the maximum Sidon subset size (pair sums unique including diagonals).
    Uses backtracking with pruning for n=12."""
    s = sorted(s)
    n = len(s)
    
    # Quick check: is the whole set Sidon?
    all_sums = []
    for i in range(n):
        for j in range(i, n):
            all_sums.append(s[i] + s[j])
    if len(set(all_sums)) == len(all_sums):
        return n
    
    best = [0]
    
    def backtrack(idx, chosen, used_sums, size):
        if size > best[0]:
            best[0] = size
        if idx >= n:
            return
        # Pruning: even if we take all remaining, can't beat best
        if size + (n - idx) <= best[0]:
            return
        
        # Option 1: skip element idx
        backtrack(idx + 1, chosen, used_sums, size)
        
        # Option 2: try to include element idx
        # Check all pairs (idx, j) for j in chosen and j=idx (diagonal)
        can_add = True
        new_sums = []
        for j in chosen:
            sm = s[idx] + s[j]
            if sm in used_sums:
                can_add = False
                break
            new_sums.append(sm)
        if can_add:
            # Check diagonal
            diag = s[idx] + s[idx]
            if diag in used_sums:
                can_add = False
            else:
                new_sums.append(diag)
        
        if can_add:
            new_used = used_sums | set(new_sums)
            backtrack(idx + 1, chosen + [idx], new_used, size + 1)
    
    backtrack(0, [], set(), 0)
    return best[0]

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
        base = [i * 10 for i in range(12)]
        c = []
        for i, b in enumerate(base):
            p = rng.randint(-5, 5)
            c.append(b + p)
        if len(set(c)) == 12 and all(0 <= x <= 128 for x in c):
            add_candidate(c)
            
    # 4. Specific "collision-heavy" constructions
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
            
    # 5. Three-cluster constructions: groups of 4 with shared sum patterns
    for _ in range(150):
        c = set()
        # Create 3 groups of 4, each group shares some additive structure
        for g in range(3):
            base_sum = rng.randint(20, 200)
            # Pick 4 numbers that create collisions
            a1 = rng.randint(0, 128)
            a2 = rng.randint(0, 128)
            a3 = rng.randint(0, 128)
            # a4 chosen to create a collision: a1+a3 == a2+a4 => a4 = a1+a3-a2
            a4 = a1 + a3 - a2
            for x in [a1, a2, a3, a4]:
                if 0 <= x <= 128:
                    c.add(x)
        if len(c) >= 12:
            c = list(c)[:12]
            if len(set(c)) == 12:
                add_candidate(c)
        elif len(c) > 0:
            # Fill remaining randomly
            remaining = 12 - len(c)
            while remaining > 0:
                x = rng.randint(0, 128)
                if x not in c:
                    c.add(x)
                    remaining -= 1
            if len(c) == 12:
                add_candidate(list(c))
    
    # 6. Affine transforms of structured bases (from inspiration 1)
    bases = [
        [0, 1, 2, 4, 8, 16, 3, 5, 6, 10, 12, 20],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        [0, 1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 66],
        [0, 1, 2, 3, 4, 5, 16, 17, 18, 19, 20, 21],
        [0, 1, 2, 3, 4, 5, 32, 33, 34, 35, 36, 37],
        [i*i for i in range(12)],
    ]
    
    for base in bases:
        if len(set(base)) != 12:
            continue
        # Only try a subset of affine transforms to keep candidate count manageable
        for a in range(1, 30):  # Smaller range for a
            for b in range(0, 129, 2):  # Smaller range for b
                c = [(x * a + b) % 129 for x in base]
                if len(set(c)) == 12 and all(0 <= x <= 128 for x in c):
                    add_candidate(c)
    
    return candidates

def construct():
    start = time.time()
    deadline = start + 50  # 50s safety margin
    
    # Generate candidates
    candidates = _build_candidates()
    
    if not candidates:
        return list(range(12))
        
    # First pass: score by collision count to get a rough ranking
    scored = []
    for cand in candidates:
        if time.time() > deadline:
            break
        col = _count_collisions(cand)
        scored.append((col, cand))
    
    # Sort by collision count descending (more collisions = likely smaller Sidon subset)
    scored.sort(key=lambda x: -x[0])
    
    # Second pass: exactly compute max Sidon size for top candidates
    best_sidon = 13  # worst case
    best_cand = candidates[0]
    
    # Check top candidates exactly
    for col, cand in scored[:50]:  # Check top 50 by collision count
        if time.time() > deadline:
            break
        sidon_size = _max_sidon_size_exact(cand)
        if sidon_size < best_sidon:
            best_sidon = sidon_size
            best_cand = cand
        # If we found a very small Sidon subset, we can stop early
        if best_sidon <= 4:
            break
            
    # If we still have time, do local search to try to reduce Sidon size further
    if time.time() < deadline - 10 and best_sidon > 4:
        current = list(best_cand)
        current_sidon = best_sidon
        rng = random.Random(999)
        pool = list(range(129))
        
        for _ in range(5000):
            if time.time() > deadline - 5:
                break
            idx = rng.randint(0, 11)
            old_val = current[idx]
            current_set = set(current)
            available = [x for x in pool if x not in current_set]
            if not available:
                break
            new_val = rng.choice(available)
            current[idx] = new_val
            # Quick collision check first
            new_col = _count_collisions(current)
            # Actually, just do exact check since n=12 is small
            new_sidon = _max_sidon_size_exact(current)
            if new_sidon < current_sidon:
                current_sidon = new_sidon
                best_sidon = new_sidon
                best_cand = list(current)
            else:
                current[idx] = old_val
                
    return best_cand
# EVOLVE-BLOCK-END
