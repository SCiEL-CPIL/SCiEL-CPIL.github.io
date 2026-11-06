import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # 8s margin
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

    # Precompute for each position, the list of (ap_idx, other_positions) for fast delta
    # pos_ap_data[pos] = list of (ap_idx, [p1,p2,p3]) where p's are the other 3 positions in the AP
    pos_ap_data = [[] for _ in range(n)]
    for pos in range(n):
        for ap_idx in pos_to_aps[pos]:
            ap = aps[ap_idx]
            others = [p for p in ap if p != pos]
            pos_ap_data[pos].append((ap_idx, others))

    def count_violations(colors):
        """Count violations using numpy for speed."""
        c = np.array(colors)
        total = 0
        for ap in aps:
            vals = c[list(ap)]
            if len(set(vals.tolist())) < 3:
                total += 1
        return total

    def count_violations_fast(colors_np):
        """Count violations using numpy."""
        total = 0
        for ap in aps:
            vals = colors_np[list(ap)]
            if len(set(vals.tolist())) < 3:
                total += 1
        return total

    def greedy_valid():
        """Greedy to get initial valid coloring."""
        colors = [0] * n
        for x in range(n):
            if time.time() > deadline:
                return None
            found = False
            for c in range(128):
                valid = True
                for ap_idx, others in pos_ap_data[x]:
                    vals = [colors[p] for p in others]
                    vals.append(c)
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
    for k in range(3, 15):
        if time.time() > deadline - 20:
            break

        # Multiple restarts for each k
        found_valid = False
        for restart in range(3):
            if time.time() > deadline - 20:
                break

            # Random initialization
            colors = [random.randint(0, k - 1) for _ in range(n)]
            current_violations = count_violations(colors)

            if current_violations == 0:
                best_colors = colors[:]
                best_num_colors = len(set(colors))
                found_valid = True
                break

            best_colors = colors[:]
            best_violations = current_violations

            # Simulated annealing with fast delta
            temp = 1.0
            cooling = 0.99995
            steps = 0
            max_steps = 150000

            while steps < max_steps and time.time() < deadline - 20:
                steps += 1
                i = random.randint(0, n - 1)
                old_color = colors[i]
                new_color = random.randint(0, k - 1)
                if new_color == old_color:
                    continue

                # Compute delta in violations using precomputed data
                delta = 0
                for ap_idx, others in pos_ap_data[i]:
                    # Old state
                    old_vals = [colors[p] for p in others] + [old_color]
                    old_viol = len(set(old_vals)) < 3
                    # New state
                    new_vals = [colors[p] for p in others] + [new_color]
                    new_viol = len(set(new_vals)) < 3
                    if old_viol and not new_viol:
                        delta -= 1
                    elif not old_viol and new_viol:
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
                found_valid = True
                break

        if found_valid:
            break

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
                for ap_idx, others in pos_ap_data[x]:
                    vals = [colors[p] for p in others]
                    vals.append(c)
                    if len(set(vals)) < 3:
                        valid = False
                        break
                if valid:
                    colors[x] = c
                    break
        return colors

    # Now polish: aggressively reduce the number of colors used
    colors = best_colors[:]
    best_num = len(set(colors))
    best_polished = colors[:]

    # Strategy 1: Try to merge colors one by one
    # For each color, try to reassign all positions with that color to other colors
    def try_eliminate_color(colors, target_color):
        """Try to eliminate target_color by reassigning its positions."""
        positions_with_target = [i for i in range(n) if colors[i] == target_color]
        if not positions_with_target:
            return None, False

        # Try to reassign each position to a valid different color
        new_colors = colors[:]
        for pos in positions_with_target:
            found = False
            # Try all other colors currently in use
            other_colors = [c for c in set(colors) if c != target_color]
            random.shuffle(other_colors)
            for new_c in other_colors:
                valid = True
                for ap_idx, others in pos_ap_data[pos]:
                    vals = [new_colors[p] for p in others]
                    vals.append(new_c)
                    if len(set(vals)) < 3:
                        valid = False
                        break
                if valid:
                    new_colors[pos] = new_c
                    found = True
                    break
            if not found:
                return None, False

        # Verify the entire coloring is valid
        if count_violations(new_colors) == 0:
            return new_colors, True
        return None, False

    # Iteratively try to eliminate colors
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        current_colors = set(colors)
        for target_color in list(current_colors):
            if time.time() > deadline - 5:
                break
            new_colors, success = try_eliminate_color(colors, target_color)
            if success:
                colors = new_colors
                best_num = len(set(colors))
                best_polished = colors[:]
                improved = True
                break

    # Strategy 2: Local search to reduce colors
    # Randomly pick a position and try to change its color to reduce total distinct colors
    for iteration in range(20000):
        if time.time() > deadline - 3:
            break
        pos = random.randint(0, n - 1)
        old_color = colors[pos]
        existing = list(set(colors))
        random.shuffle(existing)
        for new_color in existing:
            if new_color == old_color:
                continue
            # Check validity
            valid = True
            for ap_idx, others in pos_ap_data[pos]:
                vals = [colors[p] for p in others]
                vals.append(new_color)
                if len(set(vals)) < 3:
                    valid = False
                    break
            if valid:
                # Try the change
                colors[pos] = new_color
                if old_color not in colors:
                    new_num = len(set(colors))
                    if new_num < best_num:
                        best_num = new_num
                        best_polished = colors[:]
                else:
                    # Keep the change if it doesn't hurt, but revert for exploration
                    colors[pos] = old_color
                break

    if best_num < best_num_colors:
        return best_polished
    return best_colors
