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

    # Density-incremental greedy construction
    # Process positions in order; for each position, try existing colors first,
    # only adding a new color when no existing color avoids creating a critical AP.
    # A "critical" AP is one that would have <= 2 distinct colors after assignment.

    # Precompute for each position which APs it participates in and its role
    # pos_aps[i] = list of (ap_idx, role) where role is 0,1,2,3

    # For incremental construction, we need to check: if we assign color c to position i,
    # does any AP containing i become "critical" (<=2 colors)?
    # An AP (p0,p1,p2,p3) is critical if the set of assigned colors has size <= 2.

    # Strategy: assign positions 0..127 in order. For each position, try colors 0,1,2,...
    # and pick the first one that doesn't make any AP containing this position critical
    # given the already-assigned positions.

    def ap_is_critical(ap_positions, assigned_colors, new_pos, new_color):
        """Check if after assigning new_color to new_pos, the AP has <=2 distinct colors.
        ap_positions: tuple of 4 positions
        assigned_colors: current color array (with -1 for unassigned)
        new_pos: the position being assigned
        new_color: the color being assigned
        Returns True if the AP would be critical (bad)."""
        colors_in_ap = []
        for p in ap_positions:
            if p == new_pos:
                colors_in_ap.append(new_color)
            elif assigned_colors[p] != -1:
                colors_in_ap.append(assigned_colors[p])
            # else: unassigned, skip for now (will be handled later)
        
        # Only check if all 4 are assigned (or the new one makes it complete)
        if len(colors_in_ap) < 4:
            return False  # Not yet complete, can't determine
        
        distinct = set(colors_in_ap)
        return len(distinct) <= 2

    # For efficiency, precompute APs per position
    # pos_aps[i] already has (ap_idx, role)
    # We need the full AP positions for each (ap_idx, role)
    # Let's build: for each position i, list of APs (as tuples of 4 positions) that contain i

    pos_ap_tuples = [[] for _ in range(n)]
    for idx, (a, d) in enumerate(aps):
        ap_tuple = (a, a + d, a + 2 * d, a + 3 * d)
        pos_ap_tuples[a].append(ap_tuple)
        pos_ap_tuples[a + d].append(ap_tuple)
        pos_ap_tuples[a + 2 * d].append(ap_tuple)
        pos_ap_tuples[a + 3 * d].append(ap_tuple)

    # Greedy construction
    assigned = [-1] * n
    max_color_used = -1

    for pos in range(n):
        if time.time() > deadline - 15.0:
            break

        # Try existing colors first (0 to max_color_used)
        found = False
        for c in range(max_color_used + 1):
            # Check if assigning c to pos makes any AP critical
            valid = True
            for ap_tuple in pos_ap_tuples[pos]:
                if ap_is_critical(ap_tuple, assigned, pos, c):
                    valid = False
                    break
            if valid:
                assigned[pos] = c
                found = True
                break

        if not found:
            # Need a new color
            max_color_used += 1
            assigned[pos] = max_color_used

    # If we didn't finish all positions, fill remaining with identity
    for pos in range(n):
        if assigned[pos] == -1:
            assigned[pos] = pos

    # Verify the greedy result
    if not verify(assigned):
        # Fallback to modular
        best_colors = None
        for k in range(2, 128):
            if time.time() > deadline - 10.0:
                break
            colors = [x % k for x in range(n)]
            if verify(colors):
                best_colors = colors
                break
        if best_colors is None:
            best_colors = list(range(n))
        current_colors = list(best_colors)
    else:
        current_colors = list(assigned)

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
        
        # Simulated annealing with incremental violation checking
        # For speed, we precompute which APs are affected by each position
        # and do a fast validity check on only those APs
        
        # Precompute: for each position, the list of AP indices it belongs to
        pos_ap_indices = [[] for _ in range(n)]
        for idx, (a, d) in enumerate(aps):
            pos_ap_indices[a].append(idx)
            pos_ap_indices[a + d].append(idx)
            pos_ap_indices[a + 2 * d].append(idx)
            pos_ap_indices[a + 3 * d].append(idx)

        def fast_verify_position(colors, pos):
            """Check if all APs containing pos are valid."""
            for ap_idx in pos_ap_indices[pos]:
                a, d = aps[ap_idx]
                c0 = colors[a]
                c1 = colors[a + d]
                c2 = colors[a + 2 * d]
                c3 = colors[a + 3 * d]
                # Check if at least 3 distinct colors
                if c0 == c1 == c2 == c3:
                    return False
                if c0 == c1 == c2 and c3 != c0:
                    return False
                if c0 == c1 == c3 and c2 != c0:
                    return False
                if c0 == c2 == c3 and c1 != c0:
                    return False
                if c1 == c2 == c3 and c0 != c1:
                    return False
                if c0 == c1 and c2 == c3 and c0 != c2:
                    return False
                if c0 == c2 and c1 == c3 and c0 != c1:
                    return False
                if c0 == c3 and c1 == c2 and c0 != c1:
                    return False
            return True

        temp = 5.0
        for step in range(50000):
            if time.time() > deadline - 3.0:
                break
            
            pos = random.randint(0, n - 1)
            old_color = current_colors[pos]
            used = set(current_colors)
            if random.random() < 0.8:
                new_color = random.choice(list(used))
            else:
                new_color = random.randint(0, max(3, best_count) - 1)
            
            if new_color == old_color:
                continue
            
            current_colors[pos] = new_color
            if fast_verify_position(current_colors, pos):
                new_count = count_colors(current_colors)
                if new_count < best_count:
                    best_count = new_count
                    best_state = list(current_colors)
                    temp = max(0.5, temp * 0.95)
                else:
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

    # Final polish: one more round of color merging after local search
    if time.time() < deadline - 1.0:
        improved = True
        while improved and time.time() < deadline - 1.0:
            improved = False
            used_colors = sorted(set(current_colors))
            num_used = len(used_colors)
            if num_used <= 3:
                break
            for i in range(num_used):
                for j in range(i + 1, num_used):
                    if time.time() > deadline - 1.0:
                        break
                    c1, c2 = used_colors[i], used_colors[j]
                    new_colors = [c1 if x == c2 else x for x in current_colors]
                    if verify(new_colors):
                        current_colors = new_colors
                        improved = True
                        break
                if improved:
                    break

    return current_colors
