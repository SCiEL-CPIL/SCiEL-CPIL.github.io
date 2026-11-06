import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 85s safety margin for 90s limit

    n = 128

    # Precompute all 4-term progressions (a, a+d, a+2d, a+3d) with 0-based indices
    progressions = []
    for i in range(n):
        for d in range(1, n):
            j = i + d
            k = i + 2 * d
            l = i + 3 * d
            if l < n:
                progressions.append((i, j, k, l))

    m = len(progressions)

    # For each position, store indices of progressions it belongs to
    prog_at = [[] for _ in range(n)]
    for idx, (a, b, c, d) in enumerate(progressions):
        prog_at[a].append(idx)
        prog_at[b].append(idx)
        prog_at[c].append(idx)
        prog_at[d].append(idx)

    def count_violations(colors):
        """Count number of violated progressions (fewer than 3 distinct colors)."""
        v = 0
        for (a, b, c, d) in progressions:
            ca, cb, cc, cd = colors[a], colors[b], colors[c], colors[d]
            # Check if fewer than 3 distinct colors
            if ca == cb and cb == cc:
                v += 1  # all same or 3 same
            elif ca == cb and cb == cd:
                v += 1
            elif ca == cc and cc == cd:
                v += 1
            elif cb == cc and cc == cd:
                v += 1
            elif ca == cd and (ca == cb or ca == cc):
                v += 1
            elif cb == cd and (cb == ca or cb == cc):
                v += 1
            elif cc == cd and (cc == ca or cc == cb):
                v += 1
            elif ca == cb and cc == cd and ca != cc:
                # Pattern: A A B B - only 2 distinct colors
                v += 1
            elif ca == cc and cb == cd and ca != cb:
                # Pattern: A B A B - only 2 distinct colors
                v += 1
            elif ca == cd and cb == cc and ca != cb:
                # Pattern: A B B A - only 2 distinct colors
                v += 1
            # Count distinct colors more simply:
            # Actually let's just use a set for clarity and correctness
            # But that's slower. Let me use a faster check.
        return v

    def count_violations_fast(colors):
        """Faster violation count using bitwise tricks."""
        v = 0
        for (a, b, c, d) in progressions:
            ca, cb, cc, cd = colors[a], colors[b], colors[c], colors[d]
            # Count distinct values among 4
            # If all 4 same: 1 distinct
            # If 3 same, 1 different: 2 distinct
            # If 2+2: 2 distinct
            # If 2+1+1: 3 distinct
            # If 1+1+1+1: 4 distinct
            # We need >= 3 distinct, so violation if distinct <= 2
            
            # Quick check: if any pair is equal, we have at most 3 distinct
            # If two pairs are equal, we have at most 2 distinct (violation)
            # If three or more are equal, we have at most 2 distinct (violation)
            
            # Count how many are equal to ca
            n_ca = (ca == cb) + (ca == cc) + (ca == cd) + 1  # +1 for ca itself
            
            if n_ca >= 3:
                v += 1
            else:
                # Check if there are two other equal pairs
                # If cb == cc and cb != ca: then we have ca, cb, cb, cd
                #   distinct: ca, cb, cd -> 3 if cd != ca and cd != cb
                # If cb == cd and cb != ca: similar
                # If cc == cd and cc != ca: similar
                
                if cb == cc and cb != ca:
                    # ca, cb, cb, cd
                    if cd == ca or cd == cb:
                        v += 1
                elif cb == cd and cb != ca:
                    # ca, cb, cc, cb
                    if cc == ca or cc == cb:
                        v += 1
                elif cc == cd and cc != ca:
                    # ca, cb, cc, cc
                    if cb == ca or cb == cc:
                        v += 1
                else:
                    # No two of b,c,d are equal to each other
                    # Check if any equals ca
                    if cb == ca:
                        # ca, ca, cc, cd
                        if cc == cd:
                            v += 1
                    elif cc == ca:
                        # ca, cb, ca, cd
                        if cb == cd:
                            v += 1
                    elif cd == ca:
                        # ca, cb, cc, ca
                        if cb == cc:
                            v += 1
        return v

    def minimize_colors_local_search(k, max_iters=200000, seed=42):
        """Local search to find a valid k-coloring."""
        rng = random.Random(seed)
        colors = [rng.randint(0, k - 1) for _ in range(n)]
        
        # Initial violation count
        violations = count_violations_fast(colors)
        
        if violations == 0:
            return colors
        
        best_colors = colors[:]
        best_violations = violations
        
        # Track violation status for each progression
        is_violated = [False] * m
        for idx, (a, b, c, d) in enumerate(progressions):
            ca, cb, cc, cd = colors[a], colors[b], colors[c], colors[d]
            # Check if < 3 distinct
            distinct = len({ca, cb, cc, cd})
            is_violated[idx] = (distinct < 3)
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
                
            # Pick a random position
            i = rng.randint(0, n - 1)
            old_c = colors[i]
            
            # Find best new color for position i
            affected = prog_at[i]
            
            # Current violations among affected progressions
            current_v = sum(1 for idx in affected if is_violated[idx])
            
            best_new_c = old_c
            best_delta = 0
            
            for c in range(k):
                if c == old_c:
                    continue
                colors[i] = c
                
                new_v = 0
                for idx in affected:
                    a, b, cc, d = progressions[idx]
                    ca, cb, cc2, cd = colors[a], colors[b], colors[cc], colors[d]
                    if len({ca, cb, cc2, cd}) < 3:
                        new_v += 1
                        
                delta = new_v - current_v
                if delta < best_delta:
                    best_delta = delta
                    best_new_c = c
            
            colors[i] = best_new_c
            
            # Update violation tracking if color changed
            if best_new_c != old_c:
                for idx in affected:
                    a, b, cc, d = progressions[idx]
                    ca, cb, cc2, cd = colors[a], colors[b], colors[cc], colors[d]
                    new_violated = (len({ca, cb, cc2, cd}) < 3)
                    if new_violated != is_violated[idx]:
                        if new_violated:
                            violations += 1
                        else:
                            violations -= 1
                        is_violated[idx] = new_violated
                
                if violations < best_violations:
                    best_violations = violations
                    best_colors = colors[:]
                    if best_violations == 0:
                        return best_colors
                        
        return best_colors if best_violations == 0 else None

    # Try k=3 first with multiple random seeds
    for seed in range(10):
        if time.time() > deadline:
            break
        result = minimize_colors_local_search(3, max_iters=50000, seed=seed)
        if result:
            return result

    # Try k=4
    for seed in range(10):
        if time.time() > deadline:
            break
        result = minimize_colors_local_search(4, max_iters=50000, seed=seed)
        if result:
            return result

    # Try k=5
    for seed in range(5):
        if time.time() > deadline:
            break
        result = minimize_colors_local_search(5, max_iters=50000, seed=seed)
        if result:
            return result

    # Fallback: deterministic construction using modular arithmetic
    # Use color = i mod k for some k
    for k in [3, 4, 5, 6, 7, 8]:
        colors = [i % k for i in range(n)]
        if count_violations_fast(colors) == 0:
            return colors

    # Last resort: use many colors
    colors = list(range(n))
    return colors
