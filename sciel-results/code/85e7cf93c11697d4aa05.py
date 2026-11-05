"""Ambient integer sets with small largest Sidon subset.
We want a 12-element set in [0,128] whose largest 'strong Sidon' subset
(all unordered pair sums including diagonals unique) is as small as possible."""
import random
import time
import math
from itertools import combinations
from collections import Counter


def max_sidon_size(S):
    """Exact largest k such that some k-subset of S has all unordered pair sums
    (including diagonals) unique. Returns that k."""
    S = sorted(S)
    n = len(S)
    if n <= 1:
        return n

    # Precompute pair sums
    pair_sums = {}
    for i in range(n):
        for j in range(i, n):
            pair_sums[(i, j)] = S[i] + S[j]

    # Check subsets in descending size order for early termination
    # Start from n down to 1
    for k in range(n, 0, -1):
        # Check all k-subsets
        found = False
        for combo in combinations(range(n), k):
            # Check if all pair sums in this subset are unique
            seen_sums = set()
            valid = True
            for idx_a in range(k):
                i = combo[idx_a]
                for idx_b in range(idx_a, k):
                    j = combo[idx_b]
                    s = S[i] + S[j]
                    if s in seen_sums:
                        valid = False
                        break
                    seen_sums.add(s)
                if not valid:
                    break
            if valid:
                return k
    return 0


def _valid(c):
    if len(c) != 12 or len(set(c)) != 12:
        return False
    return all(0 <= x <= 128 for x in c)


def _count_duplicate_sums(S):
    """Count the number of duplicate unordered pair sums (including diagonals).
    A 'duplicate' means a sum value that appears more than once among all pairs.
    Higher is better for our purpose (more collisions = smaller max Sidon subset)."""
    S = sorted(S)
    n = len(S)
    sum_counts = Counter()
    for i in range(n):
        for j in range(i, n):
            sum_counts[S[i] + S[j]] += 1
    # Count excess occurrences (sums that appear k times contribute k-1 to duplicates)
    duplicates = sum(v - 1 for v in sum_counts.values() if v > 1)
    return duplicates


def _collision_density(S):
    """Measure how many pairs share the same sum. Higher = more collisions."""
    S = sorted(S)
    n = len(S)
    sum_counts = Counter()
    for i in range(n):
        for j in range(i, n):
            sum_counts[S[i] + S[j]] += 1
    # Total pairs
    total_pairs = n * (n + 1) // 2
    unique_sums = len(sum_counts)
    # Density: 1 - (unique_sums / total_pairs)
    return 1.0 - unique_sums / total_pairs if total_pairs > 0 else 0.0


def construct():
    start = time.time()
    deadline = start + 85.0  # safety margin under 90s

    best_set = None
    best_val = 99
    seen = set()

    def consider(c):
        nonlocal best_set, best_val
        if time.time() > deadline:
            return None
        key = tuple(sorted(c))
        if key in seen:
            return None
        seen.add(key)
        if not _valid(c):
            return None
        v = max_sidon_size(c)
        if v < best_val:
            best_val = v
            best_set = list(c)
        return v

    rng = random.Random(42)

    # Phase 1: Try consecutive sets (known to have small Sidon subsets)
    for lo in range(0, 117):
        if time.time() > deadline:
            break
        c = list(range(lo, lo + 12))
        consider(c)

    # Phase 2: Structured sets with many sum collisions
    # Two interleaved APs with small differences
    for d in range(1, 10):
        if time.time() > deadline:
            break
        for a in range(0, 100):
            for gap in range(1, 12):
                b = a + gap
                if b + 5 * d > 128:
                    continue
                c1 = [a + i * d for i in range(6)]
                c2 = [b + i * d for i in range(6)]
                c = sorted(set(c1 + c2))
                if len(c) == 12 and _valid(c):
                    consider(c)

    # Phase 3: Sets with repeated small differences (lattice-like structures)
    # Use base + offset patterns that create many additive collisions
    for base in range(0, 80):
        if time.time() > deadline:
            break
        for d1 in range(1, 8):
            for d2 in range(1, 8):
                for off in range(0, 15):
                    # Create 6 elements from first AP, 6 from second
                    c1 = [base + i * d1 for i in range(6)]
                    c2 = [base + off + i * d2 for i in range(6)]
                    c = sorted(set(c1 + c2))
                    if len(c) == 12 and _valid(c):
                        consider(c)

    # Phase 4: Random sets in small windows with collision-focused sampling
    for trial in range(2000):
        if time.time() > deadline:
            break
        lo = rng.randint(0, 90)
        width = rng.randint(12, 22)
        hi = min(lo + width, 129)
        if hi - lo < 12:
            continue
        # Sample with bias toward creating collisions
        c = sorted(rng.sample(range(lo, hi), 12))
        consider(c)

    # Phase 5: Collision-maximization local search
    # Use actual Sidon size as primary objective
    if best_set is not None:
        cur = list(best_set)
        cur_val = best_val
        cur_dup = _count_duplicate_sums(cur)
        
        improved = True
        max_iter = 300
        for iteration in range(max_iter):
            if time.time() > deadline - 5:
                break
            if not improved:
                break
            improved = False
            
            # Try all possible single-element swaps
            best_swap = None
            best_swap_dup = cur_dup
            best_swap_val = cur_val
            
            for pos in range(12):
                old_val = cur[pos]
                # Try replacing cur[pos] with a new value
                # Focus on values close to existing elements to create collisions
                candidates = set()
                # Add values near existing elements
                for other in cur:
                    for delta in range(-4, 5):
                        nv = other + delta
                        if 0 <= nv <= 128 and nv != old_val:
                            candidates.add(nv)
                # Also try some random values
                for _ in range(8):
                    nv = rng.randint(0, 128)
                    if nv != old_val:
                        candidates.add(nv)
                
                for nv in candidates:
                    if nv in cur:
                        continue
                    new_cur = cur.copy()
                    new_cur[pos] = nv
                    new_cur.sort()
                    
                    # Quick check: count duplicates
                    new_dup = _count_duplicate_sums(new_cur)
                    
                    if new_dup > best_swap_dup:
                        best_swap_dup = new_dup
                        best_swap = (pos, nv)
                        # Also compute the actual Sidon size
                        new_val = max_sidon_size(new_cur)
                        best_swap_val = new_val
                        if new_val < cur_val:
                            improved = True
                            break
                    elif new_dup == best_swap_dup:
                        # Tie-breaker: prefer lower Sidon size
                        new_val = max_sidon_size(new_cur)
                        if new_val < best_swap_val:
                            best_swap = (pos, nv)
                            best_swap_val = new_val
                            if new_val < cur_val:
                                improved = True
                                break
                if improved and best_swap_val < cur_val:
                    break
            
            if best_swap is not None:
                pos, nv = best_swap
                cur[pos] = nv
                cur.sort()
                cur_dup = best_swap_dup
                if best_swap_val < cur_val:
                    cur_val = best_swap_val
                    improved = True
                    consider(cur)
        
        # Final consideration
        if cur_val < best_val or (cur_val == best_val and cur_dup > _count_duplicate_sums(best_set)):
            consider(cur)

    # Phase 6: If we have time, try more structured patterns with high collision density
    if time.time() < deadline - 10:
        # Sets with repeated differences - focus on small step sizes
        for base in range(0, 60):
            if time.time() > deadline - 5:
                break
            for d1 in range(1, 6):
                for d2 in range(1, 6):
                    for off in range(0, 10):
                        c1 = [base + i * d1 for i in range(6)]
                        c2 = [base + off + i * d2 for i in range(6)]
                        c = sorted(set(c1 + c2))
                        if len(c) == 12 and _valid(c):
                            consider(c)

    # Phase 7: Additional collision-maximization search from multiple starting points
    if time.time() < deadline - 15:
        # Start from several different initial sets and do collision-maximization
        start_sets = []
        # Consecutive sets
        for lo in [0, 10, 20, 30, 50, 70]:
            start_sets.append(list(range(lo, lo + 12)))
        # Random sets
        for _ in range(15):
            lo = rng.randint(0, 90)
            width = rng.randint(12, 22)
            hi = min(lo + width, 129)
            if hi - lo >= 12:
                start_sets.append(sorted(rng.sample(range(lo, hi), 12)))
        
        for ss in start_sets:
            if time.time() > deadline - 5:
                break
            if not _valid(ss):
                continue
            cur = list(ss)
            cur_dup = _count_duplicate_sums(cur)
            cur_val = max_sidon_size(cur)
            
            # Quick collision-maximization
            for _ in range(25):
                if time.time() > deadline - 5:
                    break
                best_swap = None
                best_swap_dup = cur_dup
                
                for pos in range(12):
                    old_val = cur[pos]
                    candidates = set()
                    for other in cur:
                        for delta in range(-3, 4):
                            nv = other + delta
                            if 0 <= nv <= 128 and nv != old_val:
                                candidates.add(nv)
                    
                    for nv in candidates:
                        if nv in cur:
                            continue
                        new_cur = cur.copy()
                        new_cur[pos] = nv
                        new_cur.sort()
                        new_dup = _count_duplicate_sums(new_cur)
                        if new_dup > best_swap_dup:
                            best_swap_dup = new_dup
                            best_swap = (pos, nv)
                
                if best_swap is None:
                    break
                
                pos, nv = best_swap
                cur[pos] = nv
                cur.sort()
                cur_dup = best_swap_dup
                cur_val = max_sidon_size(cur)
            
            consider(cur)

    # If nothing found, return a default
    if best_set is None:
        best_set = list(range(12))

    return best_set
