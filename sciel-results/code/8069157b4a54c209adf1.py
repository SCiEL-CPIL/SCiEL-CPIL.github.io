"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 84.0  # Safety margin under 90s limit
    n = 128

    # Precompute all 4-term APs (0-indexed): (a, a+d, a+2d, a+3d)
    aps = []
    for a in range(n):
        for d in range(1, (n - a) // 3 + 1):
            i3 = a + 3 * d
            if i3 < n:
                aps.append((a, a + d, a + 2 * d, i3))

    # For each position, list of AP indices that include it
    pos_to_aps = [[] for _ in range(n)]
    for idx, ap in enumerate(aps):
        for pos in ap:
            pos_to_aps[pos].append(idx)

    def ap_violation(ap, colors):
        return len({colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]}) < 3

    def count_violations(colors):
        return sum(1 for ap in aps if ap_violation(ap, colors))

    # Greedy valid coloring: only avoid monochromatic 3-term APs.
    # This guarantees every 4-term AP uses >=3 distinct colors.
    # Assign smallest available color to minimize labels used.
    def greedy_valid():
        colors = [0] * n
        for x in range(n):
            if time.time() > deadline:
                return None
            found = False
            for c in range(128):
                valid = True
                # 3-term APs ending at x: (x-2d, x-d, x)
                max_d = x // 2
                for d in range(1, max_d + 1):
                    i2 = x - 2 * d
                    i1 = x - d
                    if colors[i2] == colors[i1] == c:
                        valid = False
                        break
                # 3-term APs where x is the middle: (x-d, x, x+d)
                max_d = (n - 1 - x) // 2
                for d in range(1, max_d + 1):
                    i1 = x - d
                    i3 = x + d
                    if colors[i1] == colors[i3] == c:
                        valid = False
                        break
                if valid:
                    colors[x] = c
                    found = True
                    break
            if not found:
                return None
        return colors

    best_colors = None
    best_num_colors = 128

    # Phase 1: Greedy to get a valid coloring
    g = greedy_valid()
    if g is not None:
        best_colors = g[:]
        best_num_colors = len(set(g))

    # Phase 2: Simulated annealing to find valid colorings with fewer colors
    for k in range(3, 12):
        if time.time() > deadline - 12:
            break

        # Random initialization with k colors
        colors = [random.randint(0, k - 1) for _ in range(n)]
        current_violations = count_violations(colors)

        if current_violations == 0:
            used = len(set(colors))
            if used < best_num_colors:
                best_colors = colors[:]
                best_num_colors = used
            break

        best_colors_sa = colors[:]
        best_violations = current_violations

        # Simulated annealing
        temp = 1.0
        cooling = 0.99995
        steps = 0
        max_steps = 300000

        while steps < max_steps and time.time() < deadline - 8:
            steps += 1
            i = random.randint(0, n - 1)
            old_color = colors[i]
            new_color = random.randint(0, k - 1)
            if new_color == old_color:
                continue

            # Compute delta in violations
            delta = 0
            for ap_idx in pos_to_aps[i]:
                ap = aps[ap_idx]
                was = ap_violation(ap, colors)
                colors[i] = new_color
                now = ap_violation(ap, colors)
                colors[i] = old_color
                if was and not now:
                    delta -= 1
                elif not was and now:
                    delta += 1

            new_violations = current_violations + delta

            if new_violations <= current_violations or \
               random.random() < 2.718281828 ** (-(new_violations - current_violations) / temp):
                colors[i] = new_color
                current_violations = new_violations
                if current_violations < best_violations:
                    best_violations = current_violations
                    best_colors_sa = colors[:]
                if current_violations == 0:
                    used = len(set(colors))
                    if used < best_num_colors:
                        best_colors = colors[:]
                        best_num_colors = used
                    break

            temp *= cooling

        if best_violations == 0:
            used = len(set(best_colors_sa))
            if used < best_num_colors:
                best_colors = best_colors_sa[:]
                best_num_colors = used
            break

    if best_colors is None:
        # Last resort fallback
        colors = [0] * n
        for x in range(n):
            for c in range(128):
                valid = True
                for ap_idx in pos_to_aps[x]:
                    ap = aps[ap_idx]
                    vals = [colors[p] if p != x else c for p in ap]
                    if len(set(vals)) < 3:
                        valid = False
                        break
                if valid:
                    colors[x] = c
                    break
        return colors

    # Phase 3: Polish - merge colors to reduce label count
    colors = best_colors[:]
    best_num = len(set(colors))
    best_polished = colors[:]

    # Iteratively try to merge colors: pick a color, try to reassign all its
    # positions to other existing colors while maintaining validity.
    polish_rounds = 200
    for rnd in range(polish_rounds):
        if time.time() > deadline - 3:
            break

        used_colors = sorted(set(colors))
        if len(used_colors) <= 2:
            break

        # Try to eliminate one color at a time
        for target_color in used_colors:
            if time.time() > deadline - 3:
                break
            if target_color not in colors:
                continue

            # Find all positions with target_color
            positions = [i for i in range(n) if colors[i] == target_color]
            if not positions:
                continue

            # Try to reassign each position to another existing color
            success = True
            new_colors = colors[:]
            for pos in positions:
                placed = False
                for c in used_colors:
                    if c == target_color:
                        continue
                    # Check validity of assigning c to pos
                    valid = True
                    for ap_idx in pos_to_aps[pos]:
                        ap = aps[ap_idx]
                        vals = [new_colors[p] if p != pos else c for p in ap]
                        if len(set(vals)) < 3:
                            valid = False
                            break
                    if valid:
                        new_colors[pos] = c
                        placed = True
                        break
                if not placed:
                    success = False
                    break

            if success:
                colors = new_colors
                best_num = len(set(colors))
                best_polished = colors[:]
                # Restart loop since used_colors changed
                break

    if best_num < best_num_colors:
        return best_polished
    return best_colors
# EVOLVE-BLOCK-END
