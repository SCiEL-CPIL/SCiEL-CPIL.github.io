# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit

    # Fallback: n=9 modular coloring (valid)
    def mod_color(n, mod=3):
        return [0 if (a + b + c) % mod == 0 else 1 for a, b, c in combinations(range(n), 3)]

    best_n = 9
    best_colors = mod_color(9, 3)

    def is_valid(n, colors):
        """Verify a coloring is valid (no red K4, no blue K5)."""
        cm = {}
        for i, t in enumerate(combinations(range(n), 3)):
            cm[t] = colors[i]
        for quad in combinations(range(n), 4):
            a, b, c, d = quad
            if cm[(a,b,c)] == 0 and cm[(a,b,d)] == 0 and cm[(a,c,d)] == 0 and cm[(b,c,d)] == 0:
                return False
        for quint in combinations(range(n), 5):
            all_blue = True
            for t in combinations(quint, 3):
                if cm[t] != 1:
                    all_blue = False
                    break
            if all_blue:
                return False
        return True

    def build_structures(target_n):
        """Precompute all structural mappings for fast local search."""
        triple_list = list(combinations(range(target_n), 3))
        num_triples = len(triple_list)
        triple_to_idx = {t: i for i, t in enumerate(triple_list)}

        all_quads = list(combinations(range(target_n), 4))
        quad_triples_idx = []
        for a, b, c, d in all_quads:
            quad_triples_idx.append([
                triple_to_idx[(a,b,c)], triple_to_idx[(a,b,d)],
                triple_to_idx[(a,c,d)], triple_to_idx[(b,c,d)]
            ])
        num_quads = len(quad_triples_idx)

        all_quints = list(combinations(range(target_n), 5))
        quint_triples_idx = []
        for q5 in all_quints:
            quint_triples_idx.append([triple_to_idx[t] for t in combinations(q5, 3)])
        num_quints = len(quint_triples_idx)

        # For each triple, list of quad indices containing it
        triple_in_quads = [[] for _ in range(num_triples)]
        for qi, qt in enumerate(quad_triples_idx):
            for tidx in qt:
                triple_in_quads[tidx].append(qi)

        # For each triple, list of quint indices containing it
        triple_in_quints = [[] for _ in range(num_triples)]
        for qi, qt in enumerate(quint_triples_idx):
            for tidx in qt:
                triple_in_quints[tidx].append(qi)

        # Bitsets for quads and quints (for fast red/blue-all checks)
        quad_bitsets = []
        for qt in quad_triples_idx:
            bs = 0
            for i in qt:
                bs |= (1 << i)
            quad_bitsets.append(bs)

        quint_bitsets = []
        for qt in quint_triples_idx:
            bs = 0
            for i in qt:
                bs |= (1 << i)
            quint_bitsets.append(bs)

        # Numpy arrays for vectorized delta computation
        quad_triples_np = np.array(quad_triples_idx, dtype=np.int64)
        quint_triples_np = np.array(quint_triples_idx, dtype=np.int64)
        # For each triple, the list of (structure_idx, position) - stored as flat arrays
        quad_positions = []  # (structure_idx, position_in_structure)
        for qi, qt in enumerate(quad_triples_idx):
            for pos, tidx in enumerate(qt):
                quad_positions.append((qi, pos))
        quint_positions = []
        for qi, qt in enumerate(quint_triples_idx):
            for pos, tidx in enumerate(qt):
                quint_positions.append((qi, pos))

        return (triple_list, num_triples, triple_to_idx, quad_triples_idx, num_quads,
                quint_triples_idx, num_quints, triple_in_quads, triple_in_quints,
                quad_bitsets, quint_bitsets, quad_triples_np, quint_triples_np)

    def compute_counts(color_arr, quad_triples_np, quint_triples_np):
        """Compute red counts per quad and blue counts per quint."""
        quad_red_counts = np.sum(color_arr[quad_triples_np] == 0, axis=1)
        quint_blue_counts = np.sum(color_arr[quint_triples_np] == 1, axis=1)
        return quad_red_counts, quint_blue_counts

    def delta_energy(tidx, old_color, color_arr, quad_red_counts, quint_blue_counts,
                     triple_in_quads, triple_in_quints):
        """Compute change in energy = 100*(violations) + 10*(near-violations) when flipping tidx."""
        delta = 0
        # Quads: red K4 is violation (4 red), near-violation (3 red)
        for qi in triple_in_quads[tidx]:
            cur = quad_red_counts[qi]
            if old_color == 0:  # was red -> becomes blue, red count decreases by 1
                new = cur - 1
                if cur == 4: delta -= 100
                elif cur == 3: delta -= 10
                if new == 4: delta += 100
                elif new == 3: delta += 10
            else:  # was blue -> becomes red, red count increases by 1
                new = cur + 1
                if cur == 4: delta -= 100
                elif cur == 3: delta -= 10
                if new == 4: delta += 100
                elif new == 3: delta += 10
        # Quints: blue K5 is violation (10 blue), near-violation (9 blue)
        for qi in triple_in_quints[tidx]:
            cur = quint_blue_counts[qi]
            if old_color == 1:  # was blue -> becomes red, blue count decreases by 1
                new = cur - 1
                if cur == 10: delta -= 100
                elif cur == 9: delta -= 10
                if new == 10: delta += 100
                elif new == 9: delta += 10
            else:  # was red -> becomes blue, blue count increases by 1
                new = cur + 1
                if cur == 10: delta -= 100
                elif cur == 9: delta -= 10
                if new == 10: delta += 100
                elif new == 9: delta += 10
        return delta

    def local_search(target_n, init_colors, structures, deadline, temp_start=4.0, temp_end=0.02, max_iters=400000):
        (triple_list, num_triples, triple_to_idx, quad_triples_idx, num_quads,
         quint_triples_idx, num_quints, triple_in_quads, triple_in_quints,
         quad_bitsets, quint_bitsets, quad_triples_np, quint_triples_np) = structures

        color_arr = np.array(init_colors, dtype=np.int8)
        red_mask = 0
        for i in range(num_triples):
            if color_arr[i] == 0:
                red_mask |= (1 << i)

        rk4 = 0
        for qbs in quad_bitsets:
            if (red_mask & qbs) == qbs:
                rk4 += 1
        bk5 = 0
        for qbs in quint_bitsets:
            if (red_mask & qbs) == 0:
                bk5 += 1
        if rk4 == 0 and bk5 == 0:
            return color_arr.tolist(), True

        quad_red_counts, quint_blue_counts = compute_counts(color_arr, quad_triples_np, quint_triples_np)

        # Build weight arrays for biased sampling: higher weight for triples in more violations
        quad_red_counts_arr = quad_red_counts
        quint_blue_counts_arr = quint_blue_counts
        # weights: sum of penalties for structures containing each triple
        weights = np.zeros(num_triples, dtype=np.float64)
        for qi in range(num_quads):
            w = 0.0
            rc = quad_red_counts_arr[qi]
            if rc == 4: w += 100.0
            elif rc == 3: w += 10.0
            elif rc == 2: w += 1.0
            if w > 0:
                for tidx in quad_triples_idx[qi]:
                    weights[tidx] += w
        for qi in range(num_quints):
            w = 0.0
            bc = quint_blue_counts_arr[qi]
            if bc == 10: w += 100.0
            elif bc == 9: w += 10.0
            elif bc == 8: w += 1.0
            if w > 0:
                for tidx in quint_triples_idx[qi]:
                    weights[tidx] += w
        # Normalize to probabilities
        wsum = weights.sum()
        if wsum > 0:
            weights /= wsum
        else:
            weights = np.ones(num_triples) / num_triples
        # Mix with uniform to avoid getting stuck
        weights = 0.7 * weights + 0.3 * (1.0 / num_triples)
        weights /= weights.sum()

        last_recompute = 0
        for it in range(max_iters):
            if time.time() > deadline - 3:
                break
            if rk4 == 0 and bk5 == 0:
                return color_arr.tolist(), True

            temp = temp_start * (temp_end / temp_start) ** (it / max_iters)

            # Choose triple: biased toward violating structures, with random fallback
            r = random.random()
            if r < 0.5:
                tidx = np.random.choice(num_triples, p=weights)
            else:
                tidx = random.randint(0, num_triples - 1)

            old_color = color_arr[tidx]
            d = delta_energy(tidx, old_color, color_arr, quad_red_counts, quint_blue_counts,
                             triple_in_quads, triple_in_quints)

            if d <= 0 or random.random() < np.exp(-d / max(temp, 1e-10)):
                new_color = 1 - old_color
                color_arr[tidx] = new_color
                if new_color == 0:
                    red_mask |= (1 << tidx)
                else:
                    red_mask &= ~(1 << tidx)

                # Update counts incrementally
                for qi in triple_in_quads[tidx]:
                    if old_color == 0:
                        quad_red_counts[qi] -= 1
                    else:
                        quad_red_counts[qi] += 1
                for qi in triple_in_quints[tidx]:
                    if old_color == 1:
                        quint_blue_counts[qi] -= 1
                    else:
                        quint_blue_counts[qi] += 1

                # Update rk4, bk5 incrementally
                # Recompute affected quads/quints
                for qi in triple_in_quads[tidx]:
                    rc = quad_red_counts[qi]
                    if rc == 4:
                        rk4 += 1
                    elif rc == 3:
                        # was 3, now 4 -> handled above; was 4 now 3 -> decrement
                        pass
                # Simpler: recompute rk4/bk5 from counts periodically
                if (it - last_recompute) > 200:
                    rk4 = int(np.sum(quad_red_counts == 4))
                    bk5 = int(np.sum(quint_blue_counts == 10))
                    last_recompute = it
                    # Refresh weights occasionally
                    if it % 2000 == 0:
                        weights = np.zeros(num_triples, dtype=np.float64)
                        for qi in range(num_quads):
                            rc = quad_red_counts[qi]
                            w = 100.0 if rc == 4 else (10.0 if rc == 3 else (1.0 if rc == 2 else 0.0))
                            if w > 0:
                                for tidx2 in quad_triples_idx[qi]:
                                    weights[tidx2] += w
                        for qi in range(num_quints):
                            bc = quint_blue_counts[qi]
                            w = 100.0 if bc == 10 else (10.0 if bc == 9 else (1.0 if bc == 8 else 0.0))
                            if w > 0:
                                for tidx2 in quint_triples_idx[qi]:
                                    weights[tidx2] += w
                        wsum = weights.sum()
                        if wsum > 0:
                            weights /= wsum
                        else:
                            weights = np.ones(num_triples) / num_triples
                        weights = 0.7 * weights + 0.3 * (1.0 / num_triples)
                        weights /= weights.sum()
                else:
                    # cheap incremental: only check flipped triple's structures
                    for qi in triple_in_quads[tidx]:
                        rc = quad_red_counts[qi]
                        if rc == 4:
                            rk4 += 1
                        elif rc == 3:
                            rk4 -= 1
                    for qi in triple_in_quints[tidx]:
                        bc = quint_blue_counts[qi]
                        if bc == 10:
                            bk5 += 1
                        elif bc == 9:
                            bk5 -= 1

        if rk4 == 0 and bk5 == 0:
            return color_arr.tolist(), True
        return color_arr.tolist(), False

    def hierarchical_init(target_n):
        """Hierarchical block construction: partition into two groups, color intra with
        mod-3, inter with a 2D vector scheme."""
        triple_list = list(combinations(range(target_n), 3))
        half = target_n // 2
        group = [0] * target_n
        for i in range(target_n):
            group[i] = 0 if i < half else 1
        # 2D vector for inter-group coloring
        colors = []
        for (a, b, c) in triple_list:
            ga, gb, gc = group[a], group[b], group[c]
            if ga == gb == gc:
                # intra-group: mod 3 on local index
                local = [a if ga == 0 else a - half]
                s = local[0] + local[1] + local[2]
                colors.append(0 if s % 3 == 0 else 1)
            else:
                # inter-group: use 2D vector scheme
                # map each vertex to a 2D vector
                def vec(i):
                    gi = group[i]
                    li = i if gi == 0 else i - half
                    # angle-based
                    ang = 2.0 * np.pi * li / max(half, 1)
                    return (np.cos(ang), np.sin(ang))
                va, vb, vc = vec(a), vec(b), vec(c)
                # dot products
                d1 = va[0]*vb[0] + va[1]*vb[1]
                d2 = vb[0]*vc[0] + vb[1]*vc[1]
                d3 = va[0]*vc[0] + va[1]*vc[1]
                score = d1 + d2 + d3
                colors.append(0 if score > 0 else 1)
        return colors

    def try_target_n(target_n, deadline):
        """Try to find a valid coloring for target_n vertices."""
        structures = build_structures(target_n)
        triple_list, num_triples = structures[0], structures[1]

        init_strategies = []
        # mod 3
        init_strategies.append([0 if (a+b+c) % 3 == 0 else 1 for (a,b,c) in triple_list])
        # mod 2
        init_strategies.append([0 if (a+b+c) % 2 == 0 else 1 for (a,b,c) in triple_list])
        # hierarchical
        init_strategies.append(hierarchical_init(target_n))
        # random
        init_strategies.append(np.random.randint(0, 2, size=num_triples).tolist())
        # random red bias
        arr = np.random.randint(0, 2, size=num_triples)
        arr[np.random.random(num_triples) < 0.4] = 0
        init_strategies.append(arr.tolist())
        # random blue bias
        arr = np.random.randint(0, 2, size=num_triples)
        arr[np.random.random(num_triples) < 0.4] = 1
        init_strategies.append(arr.tolist())
        # mod 3 + noise
        base = [0 if (a+b+c) % 3 == 0 else 1 for (a,b,c) in triple_list]
        noisy = base[:]
        for i in range(num_triples):
            if random.random() < 0.10:
                noisy[i] = 1 - noisy[i]
        init_strategies.append(noisy)

        random.shuffle(init_strategies)

        for init_colors in init_strategies:
            if time.time() > deadline - 4:
                break
            colors, success = local_search(target_n, init_colors, structures, deadline)
            if success:
                if is_valid(target_n, colors):
                    return colors
        return None

    # Try increasing n values
    for target_n in range(10, 35):
        if time.time() > deadline - 12:
            break
        result = try_target_n(target_n, deadline)
        if result is not None:
            best_n = target_n
            best_colors = result

    # Final safety verification
    if best_n > 9 and not is_valid(best_n, best_colors):
        best_n = 9
        best_colors = mod_color(9, 3)

    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END
