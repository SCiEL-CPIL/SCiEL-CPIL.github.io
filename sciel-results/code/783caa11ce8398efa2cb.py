import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Leave 5s margin for safety
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

    # Fast greedy to get an initial valid coloring (small colors first)
    def greedy_valid():
        colors = [0] * n
        for x in range(n):
            if time.time() > deadline:
                return None
            found = False
            for c in range(128):
                valid = True
                for ap_idx in pos_to_aps[x]:
                    ap = aps[ap_idx]
                    # Build the 4-tuple with x assigned c
                    vals = []
                    for p in ap:
                        if p == x:
                            vals.append(c)
                        else:
                            vals.append(colors[p])
                    if len(set(vals)) < 3:
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

    # Try increasing number of colors with simulated annealing
    for k in range(3, 20):
        if time.time() > deadline - 15:
            break

        # Random initialization
        colors = [random.randint(0, k - 1) for _ in range(n)]
        current_violations = count_violations(colors)

        if current_violations == 0:
            best_colors = colors[:]
            best_num_colors = len(set(colors))
            break

        best_colors = colors[:]
        best_violations = current_violations

        # Simulated annealing
        temp = 1.0
        cooling = 0.9999
        steps = 0
        max_steps = 200000

        while steps < max_steps and time.time() < deadline - 10:
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

            if new_violations <= current_violations or random.random() < pow(2.718281828, -(new_violations - current_violations) / temp):
                colors[i] = new_color
                current_violations = new_violations
                if current_violations < best_violations:
                    best_violations = current_violations
                    best_colors = colors[:]
                if current_violations == 0:
                    break

            temp *= cooling

        if best_violations == 0:
            best_colors = best_colors[:]
            best_num_colors = len(set(best_colors))
            break
        elif best_violations < 50:
            # Keep track but continue to try larger k
            if best_colors is not None and len(set(best_colors)) < best_num_colors:
                best_num_colors = len(set(best_colors))

    if best_colors is None:
        # Fallback: greedy
        g = greedy_valid()
        if g is not None:
            return g
        # Last resort
        colors = [0] * n
        for x in range(n):
            for c in range(128):
                valid = True
                for ap_idx in pos_to_aps[x]:
                    ap = aps[ap_idx]
                    vals = []
                    for p in ap:
                        if p == x:
                            vals.append(c)
                        else:
                            vals.append(colors[p])
                    if len(set(vals)) < 3:
                        valid = False
                        break
                if valid:
                    colors[x] = c
                    break
        return colors

    # Now polish: try to reduce the number of colors used
    # Use local search: reassign positions to merge colors
    colors = best_colors[:]
    current_num = len(set(colors))
    best_num = current_num
    best_polished = colors[:]

    for _ in range(5000):
        if time.time() > deadline - 3:
            break
        pos = random.randint(0, n - 1)
        old_color = colors[pos]
        # Try to reassign to an existing color that is not old_color
        existing = list(set(colors))
        random.shuffle(existing)
        found_improve = False
        for new_color in existing:
            if new_color == old_color:
                continue
            # Check validity
            valid = True
            for ap_idx in pos_to_aps[pos]:
                ap = aps[ap_idx]
                vals = []
                for p in ap:
                    if p == pos:
                        vals.append(new_color)
                    else:
                        vals.append(colors[p])
                if len(set(vals)) < 3:
                    valid = False
                    break
            if valid:
                # Check if this reduces the number of colors
                colors[pos] = new_color
                # Check if old_color is still used
                if old_color not in colors:
                    new_num = len(set(colors))
                    if new_num < best_num:
                        best_num = new_num
                        best_polished = colors[:]
                    found_improve = True
                colors[pos] = old_color
                break
        if not found_improve:
            continue

    if best_num < best_num_colors:
        return best_polished
    return best_colors
