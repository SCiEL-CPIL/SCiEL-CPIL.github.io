"""Colorings with three colors in every four-term progression.
Uses graph-based color merging: build a conflict graph where edges connect
positions that cannot share a color (because they'd form a bad 2-color subset
of some 4-AP), then greedily assign colors to minimize distinct labels used."""
import time
import random
import numpy as np

def construct():
    start = time.time()
    deadline = start + 85.0  # safety margin before 90s limit
    n = 128

    # Precompute all non-constant 4-APs
    aps = []
    for a in range(n):
        for d in range(1, (n - a) // 3 + 1):
            i3 = a + 3 * d
            if i3 < n:
                aps.append((a, d))

    # For each position, list of (ap_idx, role)
    pos_aps = [[] for _ in range(n)]
    for idx, (a, d) in enumerate(aps):
        pos_aps[a].append((idx, 0))
        pos_aps[a + d].append((idx, 1))
        pos_aps[a + 2 * d].append((idx, 2))
        pos_aps[a + 3 * d].append((idx, 3))

    # Build conflict graph: two positions conflict if assigning them the same color
    # would cause some 4-AP to use at most 2 colors.
    # A 4-AP uses at most 2 colors iff:
    #   - All 4 same (4,0,0,0)
    #   - 3 same + 1 different (3,1,0,0)
    #   - 2+2 split (2,2,0,0)
    # So two positions i, j conflict if there exists a 4-AP containing both i and j
    # such that the other two positions in that AP are also assigned the same color as i (or j).
    #
    # Actually, the constraint is: in every 4-AP, at least 3 distinct colors appear.
    # This is violated when the 4 positions use at most 2 colors.
    # For a given 4-AP (p0, p1, p2, p3), the bad assignments are those where
    # the set of colors has size <= 2.
    #
    # For graph-based approach: we want to find the minimum number of colors such that
    # no 4-AP has all 4 positions colored with at most 2 colors.
    #
    # Alternative structural approach: Start with a valid coloring and try to merge colors.
    # Two colors can be merged if no 4-AP has positions colored with exactly those two colors
    # in a way that would create a violation when merged.
    #
    # Let's use a different strategy:
    # 1. Find a valid modular coloring with small k
    # 2. Use iterative color merging with careful checking
    # 3. Use local search to reduce colors

    def verify(colors):
        """Check if coloring is valid."""
        for (a, d) in aps:
            c0 = colors[a]
            c1 = colors[a + d]
            c2 = colors[a + 2 * d]
            c3 = colors[a + 3 * d]
            # Check if at least 3 distinct colors
            if c0 == c1 == c2 == c3:
                return False
            # Check 3-1 split
            if c0 == c1 == c2 and c3 != c0:
                return False
            if c0 == c1 == c3 and c2 != c0:
                return False
            if c0 == c2 == c3 and c1 != c0:
                return False
            if c1 == c2 == c3 and c0 != c1:
                return False
            # Check 2-2 split
            if c0 == c1 and c2 == c3 and c0 != c2:
                return False
            if c0 == c2 and c1 == c3 and c0 != c1:
                return False
            if c0 == c3 and c1 == c2 and c0 != c1:
                return False
        return True

    # Find smallest valid modular coloring
    best_colors = None
    for k in range(2, 128):
        if time.time() > deadline - 10.0:
            break
        colors = [x % k for x in range(n)]
        if verify(colors):
            best_colors = colors
            break

    if best_colors is None:
        # Fallback: use a known-good approach
        best_colors = [x % 43 for x in range(n)]
        if not verify(best_colors):
            best_colors = [x % 45 for x in range(n)]
            if not verify(best_colors):
                # Last resort: identity mapping (128 colors, always valid)
                best_colors = list(range(n))

    current_colors = list(best_colors)

    # Now try to reduce colors via merging
    # Two colors c1, c2 can be merged (replace all c2 with c1) if the result is still valid.
    # We try all pairs and apply merges that reduce the count.

    def count_colors(colors):
        return len(set(colors))

    current_k = count_colors(current_colors)

    # Iterative color merging
    if time.time() < deadline - 5.0:
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            used_colors = sorted(set(current_colors))
            num_used = len(used_colors)
            if num_used <= 3:
                break

            # Try merging each pair of colors
            for i in range(num_used):
                for j in range(i + 1, num_used):
                    if time.time() > deadline - 5.0:
                        break
                    c1, c2 = used_colors[i], used_colors[j]
                    # Try merging c2 into c1
                    new_colors = [c1 if x == c2 else x for x in current_colors]
                    if verify(new_colors):
                        current_colors = new_colors
                        improved = True
                        break
                if improved:
                    break

    # Local search: try random color reassignments to reduce color count
    if time.time() < deadline - 3.0:
        random.seed(42)
        best_count = count_colors(current_colors)
        best_state = list(current_colors)
        
        # Simulated annealing
        temp = 5.0
        for step in range(20000):
            if time.time() > deadline - 3.0:
                break
            
            pos = random.randint(0, n - 1)
            old_color = current_colors[pos]
            # Pick a new color from existing used colors or a slightly different one
            used = set(current_colors)
            if random.random() < 0.8:
                new_color = random.choice(list(used))
            else:
                new_color = random.randint(0, max(3, best_count) - 1)
            
            if new_color == old_color:
                continue
            
            current_colors[pos] = new_color
            if verify(current_colors):
                new_count = count_colors(current_colors)
                if new_count < best_count:
                    best_count = new_count
                    best_state = list(current_colors)
                    temp = max(0.5, temp * 0.95)
                else:
                    # Accept with probability based on temperature
                    if random.random() < temp / max(1, best_count):
                        pass
                    else:
                        current_colors[pos] = old_color
            else:
                current_colors[pos] = old_color
            
            temp *= 0.9995

        current_colors = best_state

    # Final attempt: try to find a coloring with fewer colors using structured patterns
    if time.time() < deadline - 2.0:
        current_k = count_colors(current_colors)
        for target_k in range(3, current_k):
            if time.time() > deadline - 2.0:
                break
            # Try modular coloring with target_k
            colors = [x % target_k for x in range(n)]
            if verify(colors) and count_colors(colors) < current_k:
                current_colors = colors
                current_k = count_colors(colors)
                break

    return current_colors
