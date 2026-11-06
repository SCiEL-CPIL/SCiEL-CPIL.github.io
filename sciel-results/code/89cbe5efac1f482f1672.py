"""Construct a coloring of [1..128] minimizing the number of distinct colors
such that every nonconstant 4-term arithmetic progression uses >=3 colors.
Uses a deterministic greedy construction plus limited repair.
"""
import time
import random

N = 128
DEADLINE = time.time() + 85.0  # safety margin before 90s kill


def all_4aps():
    """Yield (i, j, k, l) for all nonconstant 4-APs in [0..N-1]."""
    ap_list = []
    for a in range(N):
        for d in range(1, N):
            l = a + 3 * d
            if l >= N:
                break
            ap_list.append((a, a + d, a + 2 * d, l))
    return ap_list


def count_violations(colors, ap_list):
    cnt = 0
    for a, b, c, d in ap_list:
        s = {colors[a], colors[b], colors[c], colors[d]}
        if len(s) < 3:
            cnt += 1
    return cnt


def distinct_count(colors):
    return len(set(colors))


def try_greedy(num_colors, seed=0, ap_list=None):
    """Greedy: assign colors left-to-right; at each step pick color that
    creates no new violation. Returns (colors, violations) or None on failure."""
    if ap_list is None:
        ap_list = all_4aps()
    rng = random.Random(seed)
    colors = [-1] * N
    # Precompute for each position i the list of 4-APs where i appears,
    # and for each other position in that AP, which slot (0..3) i is in.
    # For speed we'll just check all APs containing i.
    # Actually let's precompute ap_by_pos: pos -> list of (ap_tuple, slot)
    ap_by_pos = [[] for _ in range(N)]
    for a, b, c, d in ap_list:
        ap_by_pos[a].append(((a, b, c, d), 0))
        ap_by_pos[b].append(((a, b, c, d), 1))
        ap_by_pos[c].append(((a, b, c, d), 2))
        ap_by_pos[d].append(((a, b, c, d), 3))

    for i in range(N):
        best_color = None
        # For each candidate color, check if assigning colors[i]=color keeps
        # all APs containing i valid (assuming all earlier positions are set).
        # An AP containing i is "active" only if all its positions are <= i (i.e.,
        # it's fully assigned) OR if i is the first unassigned position in the AP.
        # Actually since we go left-to-right, when we're at position i, all positions
        # < i are assigned, all positions > i are unassigned.
        # So an AP is fully determined only if its max position == i.
        # Otherwise, it's partially assigned and we can't fully check it yet.
        # However, we can check: among the assigned positions in the AP, do they
        # already have >=3 colors? If so, adding color[i]=c is fine.
        # If they have exactly 2 colors, we need c to be a third color.
        # If they have 1 color, any c is fine (as long as we don't accidentally
        # make it 2-color later, but that's for future steps).
        # Actually the constraint is >=3 distinct colors in the full AP.
        # When only some positions are assigned, we can check partial constraints.
        
        # For each AP containing position i:
        # Let assigned_colors = set of colors at positions in the AP that are < i (already set)
        # If i is the last position (d position), then all 4 are set, check full.
        # If i is not the last, we can only partially check.
        
        # Simplify: try each color 0..num_colors-1, check all APs where all 4 positions
        # are already determined (max pos == i, i.e., i is the 4th element).
        # For APs where i is not the last, we just ensure that among the determined
        # colors, we don't already have a contradiction (len < 3 is possible but
        # we hope future assignments fix it).
        
        # For a robust greedy, only enforce full AP constraints when i is the last element.
        # For partial APs, just assign the smallest available color.
        
        valid_colors = list(range(num_colors))
        rng.shuffle(valid_colors)
        
        found = False
        for c in valid_colors:
            ok = True
            for (ap, slot) in ap_by_pos[i]:
                a, b, cc, d = ap
                # Check if all 4 positions are assigned (positions < i are assigned,
                # position i will be c, positions > i are not)
                # The AP is fully assigned only if max(a,b,cc,d) == i
                if max(a, b, cc, d) == i:
                    # Full AP check
                    col = [0, 0, 0, 0]
                    col[slot] = c
                    for other_slot, pos in enumerate((a, b, cc, d)):
                        if pos != i:
                            col[other_slot] = colors[pos]
                    if len(set(col)) < 3:
                        ok = False
                        break
            if ok:
                colors[i] = c
                found = True
                break
        if not found:
            return None
    # Count final violations (should be 0 if greedy succeeded on full APs)
    v = count_violations(colors, ap_list)
    return colors, v


def local_search(colors, ap_list, num_colors, max_steps=5000, seed=0):
    """Simple local search: random restart with hill climbing on violations."""
    rng = random.Random(seed)
    best = list(colors)
    best_v = count_violations(best, ap_list)
    
    for step in range(max_steps):
        if time.time() > DEADLINE:
            break
        # Random single-position flip
        pos = rng.randint(0, N - 1)
        old_c = best[pos]
        new_c = rng.randint(0, num_colors - 1)
        if new_c == old_c:
            continue
        best[pos] = new_c
        v = count_violations(best, ap_list)
        if v <= best_v:
            best_v = v
        else:
            best[pos] = old_c
            # Occasionally accept worse moves (simulated annealing lite)
            if rng.random() < 0.1:
                best[pos] = new_c
                # Don't update best_v since it's worse, but keep the move
                # Actually let's revert
                best[pos] = old_c
    
    return best, best_v


def construct():
    start_time = time.time()
    ap_list = all_4aps()
    
    best_colors = None
    best_distinct = N  # worst case
    
    # Try greedy with increasing color counts
    for nc in [3, 4, 5, 6, 8]:
        if time.time() > DEADLINE - 5:
            break
        # Try multiple seeds
        for seed in range(50):
            if time.time() > DEADLINE - 2:
                break
            result = try_greedy(nc, seed=seed, ap_list=ap_list)
            if result is not None:
                colors, v = result
                if v == 0:
                    dc = distinct_count(colors)
                    if dc < best_distinct:
                        best_distinct = dc
                        best_colors = colors
                        if dc <= 3:
                            return best_colors
    
    # If greedy didn't find a valid solution, try local search from random starts
    if best_colors is None:
        for nc in [3, 4, 5]:
            if time.time() > DEADLINE - 10:
                break
            for seed in range(100):
                if time.time() > DEADLINE - 2:
                    break
                rng = random.Random(seed)
                colors = [rng.randint(0, nc - 1) for _ in range(N)]
                # Local search
                colors, v = local_search(colors, ap_list, nc, max_steps=2000, seed=seed)
                if v == 0:
                    dc = distinct_count(colors)
                    if dc < best_distinct:
                        best_distinct = dc
                        best_colors = colors
                        if dc <= 3:
                            return best_colors
    
    # Fallback: use the seed program's approach
    if best_colors is None:
        colors = []
        for x in range(1, N + 1):
            for color in range(128):
                if (all(len({color} | {colors[x - 1 - j * d] for j in (1, 2, 3)}) >= 3
                        for d in range(1, (x - 1) // 3 + 1))
                        and all(any(colors[x - 1 - j * d] != color for j in (1, 2))
                                for d in range(1, (x - 1) // 2 + 1))):
                    colors.append(color)
                    break
        best_colors = colors
    
    # Final polish: try to reduce distinct colors by merging
    if time.time() < DEADLINE - 5 and best_colors is not None:
        # Try removing a color by mapping it to another
        used = sorted(set(best_colors))
        if len(used) > 3:
            # Try to merge the least-used color into another
            from collections import Counter
            freq = Counter(best_colors)
            least = min(used, key=lambda c: freq[c])
            for target in used:
                if target == least:
                    continue
                if time.time() > DEADLINE - 3:
                    break
                new_colors = [target if c == least else c for c in best_colors]
                if count_violations(new_colors, ap_list) == 0:
                    dc = distinct_count(new_colors)
                    if dc < best_distinct:
                        best_distinct = dc
                        best_colors = new_colors
    
    return best_colors
