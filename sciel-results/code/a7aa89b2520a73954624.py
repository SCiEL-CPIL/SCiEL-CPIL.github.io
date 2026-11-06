"""Combined program: randomized K-color greedy search + local-search refinement."""
import time
import random
import math


def construct():
    n = 128
    deadline = time.time() + 85.0  # Safety margin before the 90s kill

    # Precompute all nonconstant 4-term APs (0-indexed positions)
    # AP: (i, i+d, i+2d, i+3d) where d >= 1, all indices < n
    aps = []
    for d in range(1, (n - 3) // 3 + 1):
        for i in range(0, n - 3 * d):
            aps.append((i, i + d, i + 2 * d, i + 3 * d))

    # APs ending at each position (for fast incremental validity checks)
    aps_end = [[] for _ in range(n)]
    for idx, ap in enumerate(aps):
        aps_end[ap[3]].append(idx)

    # Map each position to the APs it belongs to
    pos_to_aps = [[] for _ in range(n)]
    for idx, ap in enumerate(aps):
        for pos in ap:
            pos_to_aps[pos].append(idx)

    def count_violations(colors):
        """Count number of APs with fewer than 3 distinct colors."""
        v = 0
        for ap in aps:
            c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
            if c0 == c1 == c2 == c3:
                v += 1
            elif c0 == c1 == c2 or c0 == c1 == c3 or c0 == c2 == c3 or c1 == c2 == c3:
                v += 1
            elif c0 == c1 and c2 == c3:
                v += 1
            elif c0 == c2 and c1 == c3:
                v += 1
            elif c0 == c3 and c1 == c2:
                v += 1
        return v

    def is_ap_valid(colors, ap):
        return len({colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]}) >= 3

    def try_random_greedy(K, rng, num_restarts=20000):
        """Find a valid coloring using only colors 0..K-1 via randomized greedy."""
        for _ in range(num_restarts):
            colors = [-1] * n
            ok = True
            for idx in range(n):
                # Try each of the K colors in a random order
                order = list(range(K))
                rng.shuffle(order)
                chosen = None
                for c in order:
                    good = True
                    for ap_idx in aps_end[idx]:
                        ap = aps[ap_idx]
                        if len({colors[ap[0]], colors[ap[1]], colors[ap[2]], c}) < 3:
                            good = False
                            break
                    if good:
                        chosen = c
                        break
                if chosen is None:
                    ok = False
                    break
                colors[idx] = chosen
            if ok:
                return colors
        return None

    def try_reduce_colors(colors, max_time):
        """Reduce distinct colors by remapping one color onto another."""
        colors = colors[:]
        used_colors = sorted(set(colors))
        best_colors = colors[:]
        best_count = len(used_colors)

        for color_to_remove in used_colors:
            if time.time() > max_time:
                break
            if len(set(best_colors)) < best_count:
                used_colors = sorted(set(best_colors))
                best_count = len(used_colors)
            if best_count <= 3:
                break
            positions = [i for i in range(n) if best_colors[i] == color_to_remove]
            if not positions:
                continue
            for target_color in used_colors:
                if target_color == color_to_remove:
                    continue
                old_vals = {pos: best_colors[pos] for pos in positions}
                for pos in positions:
                    best_colors[pos] = target_color
                if count_violations(best_colors) == 0:
                    best_count = len(set(best_colors))
                    break
                else:
                    for pos in positions:
                        best_colors[pos] = old_vals[pos]
        return best_colors, best_count

    def local_search_refine(colors, time_limit):
        """Simulated annealing: fix violations, then minimize color count."""
        colors = colors[:]
        current_violations = count_violations(colors)
        color_counts = [0] * 128
        for c in colors:
            color_counts[c] += 1
        current_colors_used = sum(1 for c in range(128) if color_counts[c] > 0)

        best_colors = colors[:]
        best_violations = current_violations
        best_colors_used = current_colors_used

        temp = 3.0
        cooling = 0.997
        min_temp = 0.01

        while time.time() < time_limit and temp > min_temp:
            pos = random.randint(0, n - 1)
            old_color = colors[pos]
            used = [c for c in range(128) if color_counts[c] > 0]
            candidates = used[:]
            for _ in range(2):
                new_c = random.randint(0, 127)
                if new_c not in candidates:
                    candidates.append(new_c)

            best_new_color = old_color
            best_delta_v = 0
            best_delta_c = 0
            best_score = 0

            affected_aps = pos_to_aps[pos]
            curr_invalid = 0
            for ap_idx in affected_aps:
                ap = aps[ap_idx]
                if not is_ap_valid(colors, ap):
                    curr_invalid += 1

            for new_color in candidates:
                if new_color == old_color:
                    continue
                colors[pos] = new_color
                new_invalid = 0
                for ap_idx in affected_aps:
                    ap = aps[ap_idx]
                    if not is_ap_valid(colors, ap):
                        new_invalid += 1
                delta_v = new_invalid - curr_invalid
                delta_c = 0
                if color_counts[old_color] == 1:
                    delta_c -= 1
                if color_counts[new_color] == 0:
                    delta_c += 1
                score = delta_v * 100 + delta_c
                if score < best_score:
                    best_score = score
                    best_delta_v = delta_v
                    best_delta_c = delta_c
                    best_new_color = new_color
                elif score == best_score and random.random() < 0.3:
                    best_new_color = new_color
                    best_delta_v = delta_v
                    best_delta_c = delta_c
            colors[pos] = old_color

            accept = False
            if best_delta_v < 0 or (best_delta_v == 0 and best_delta_c < 0):
                accept = True
            elif best_delta_v == 0 and best_delta_c == 0:
                accept = True
            else:
                energy = best_delta_v * 100 + best_delta_c
                if energy > 0 and temp > 0:
                    prob = math.exp(-energy / temp)
                    accept = random.random() < prob

            if accept:
                colors[pos] = best_new_color
                color_counts[old_color] -= 1
                color_counts[best_new_color] += 1
                current_violations += best_delta_v
                current_colors_used = sum(1 for c in range(128) if color_counts[c] > 0)
                if (current_violations < best_violations or
                        (current_violations == best_violations and
                         current_colors_used < best_colors_used)):
                    best_violations = current_violations
                    best_colors_used = current_colors_used
                    best_colors = colors[:]
                if current_violations == 0 and current_colors_used <= 3:
                    break
            temp *= cooling

        return best_colors, best_violations, best_colors_used

    # Step 1: Search directly for a small-palette coloring (K = 3, 4, 5, ...)
    best = None
    for K in range(3, 13):
        if time.time() > deadline - 8.0:
            break
        found = try_random_greedy(K, random)
        if found is not None:
            best = found
            break

    # Step 2: If no small-palette coloring found, build a valid one with full palette
    if best is None:
        colors = [None] * n
        for x in range(n):
            for color in range(128):
                valid = True
                for ap_idx in pos_to_aps[x]:
                    ap = aps[ap_idx]
                    cset = set()
                    for pos in ap:
                        if pos == x:
                            cset.add(color)
                        else:
                            if colors[pos] is None:
                                valid = False
                                break
                            cset.add(colors[pos])
                    if not valid:
                        break
                    if len(cset) < 3:
                        valid = False
                        break
                if valid:
                    colors[x] = color
                    break
            if colors[x] is None:
                colors[x] = x
        best = colors

    # Step 3: Reduce colors if already valid
    if count_violations(best) == 0:
        best, _ = try_reduce_colors(best, deadline - 6.0)

    # Step 4: Local search refinement if time remains
    remaining_time = deadline - time.time()
    if remaining_time > 3.0:
        best, final_violations, _ = local_search_refine(best, deadline - 1.0)
    else:
        final_violations = count_violations(best)

    # Ensure a valid solution is returned
    if final_violations > 0:
        return list(range(n))

    return best
