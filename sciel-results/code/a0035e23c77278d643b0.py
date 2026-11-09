"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit

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
        """Precompute all structures needed for fast local search."""
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

        # For each triple, list of (quad_idx, pos_in_quad)
        triple_in_quads = [[] for _ in range(num_triples)]
        for qi, qt in enumerate(quad_triples_idx):
            for pos, tidx in enumerate(qt):
                triple_in_quads[tidx].append(qi)

        # For each triple, list of (quint_idx, pos_in_quint)
        triple_in_quints = [[] for _ in range(num_triples)]
        for qi, qt in enumerate(quint_triples_idx):
            for pos, tidx in enumerate(qt):
                triple_in_quints[tidx].append(qi)

        # Bitsets for quads and quints
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

        return (triple_list, num_triples, triple_to_idx, quad_triples_idx, num_quads,
                quint_triples_idx, num_quints, triple_in_quads, triple_in_quints,
                quad_bitsets, quint_bitsets)

    def fast_count_violations(red_mask, quad_bitsets, quint_bitsets):
        """Count red K4 and blue K5 violations using bitsets."""
        rk4 = 0
        for qbs in quad_bitsets:
            if (red_mask & qbs) == qbs:
                rk4 += 1
        bk5 = 0
        for qbs in quint_bitsets:
            if (red_mask & qbs) == 0:
                bk5 += 1
        return rk4, bk5

    def local_search(target_n, init_colors, deadline):
        """Run local search to eliminate violations. Returns (colors, success)."""
        (triple_list, num_triples, triple_to_idx, quad_triples_idx, num_quads,
         quint_triples_idx, num_quints, triple_in_quads, triple_in_quints,
         quad_bitsets, quint_bitsets) = build_structures(target_n)

        color_arr = np.array(init_colors, dtype=np.int8)
        red_mask = 0
        for i in range(num_triples):
            if color_arr[i] == 0:
                red_mask |= (1 << i)

        rk4, bk5 = fast_count_violations(red_mask, quad_bitsets, quint_bitsets)
        if rk4 == 0 and bk5 == 0:
            return color_arr.tolist(), True

        # Precompute for each triple: which quads contain it (as list of quad indices)
        # and for each quad, the 4 triple indices
        # For delta computation:
        # delta_rk4: for each quad containing tidx, check if other 3 are red
        # delta_bk5: for each quint containing tidx, check if other 9 are blue

        max_iters = 200000
        temp_start = 2.0
        temp_end = 0.001

        for it in range(max_iters):
            if time.time() > deadline - 2:
                break
            if rk4 == 0 and bk5 == 0:
                return color_arr.tolist(), True

            temp = temp_start * (temp_end / temp_start) ** (it / max_iters)

            # Choose a triple to flip
            tidx = -1
            r = random.random()
            if rk4 > 0 and r < 0.5:
                # Pick from violating red K4s
                for _ in range(30):
                    qi = random.randint(0, num_quads - 1)
                    qt = quad_triples_idx[qi]
                    if (red_mask & quad_bitsets[qi]) == quad_bitsets[qi]:
                        tidx = qt[random.randint(0, 3)]
                        break
            elif bk5 > 0 and r < 0.85:
                # Pick from violating blue K5s
                for _ in range(30):
                    qi = random.randint(0, num_quints - 1)
                    qt = quint_triples_idx[qi]
                    if (red_mask & quint_bitsets[qi]) == 0:
                        tidx = qt[random.randint(0, 9)]
                        break

            if tidx == -1:
                tidx = random.randint(0, num_triples - 1)

            old_color = color_arr[tidx]

            # Compute delta_rk4
            delta_rk4 = 0
            for qi in triple_in_quads[tidx]:
                qt = quad_triples_idx[qi]
                # Check if other 3 triples in this quad are red
                others_all_red = True
                for p in range(4):
                    if qt[p] == tidx:
                        continue
                    if color_arr[qt[p]] != 0:
                        others_all_red = False
                        break
                if others_all_red:
                    if old_color == 0:
                        delta_rk4 -= 1
                    else:
                        delta_rk4 += 1

            # Compute delta_bk5
            delta_bk5 = 0
            for qi in triple_in_quints[tidx]:
                qt = quint_triples_idx[qi]
                others_all_blue = True
                for p in range(10):
                    if qt[p] == tidx:
                        continue
                    if color_arr[qt[p]] != 1:
                        others_all_blue = False
                        break
                if others_all_blue:
                    if old_color == 1:
                        delta_bk5 -= 1
                    else:
                        delta_bk5 += 1

            delta = delta_rk4 + delta_bk5

            if delta <= 0 or random.random() < np.exp(-delta / max(temp, 1e-10)):
                new_color = 1 - old_color
                color_arr[tidx] = new_color
                if new_color == 0:
                    red_mask |= (1 << tidx)
                else:
                    red_mask &= ~(1 << tidx)
                rk4 += delta_rk4
                bk5 += delta_bk5

        if rk4 == 0 and bk5 == 0:
            return color_arr.tolist(), True
        return color_arr.tolist(), False

    # Try increasing n
    for target_n in range(10, 35):
        if time.time() > deadline - 10:
            break

        (triple_list, num_triples, triple_to_idx, quad_triples_idx, num_quads,
         quint_triples_idx, num_quints, triple_in_quads, triple_in_quints,
         quad_bitsets, quint_bitsets) = build_structures(target_n)

        found = False
        max_restarts = 300

        for restart in range(max_restarts):
            if time.time() > deadline - 3:
                break

            # Various initialization strategies
            if restart == 0:
                init = [0 if (a+b+c) % 3 == 0 else 1 for (a,b,c) in triple_list]
            elif restart == 1:
                init = [0 if (a+b+c) % 4 == 0 else 1 for (a,b,c) in triple_list]
            elif restart == 2:
                init = [0 if (a+b+c) % 5 == 0 else 1 for (a,b,c) in triple_list]
            elif restart == 3:
                init = [0 if (a+b+c) % 2 == 0 else 1 for (a,b,c) in triple_list]
            elif restart == 4:
                # Random
                init = list(np.random.randint(0, 2, size=num_triples))
            elif restart == 5:
                # Random with red bias
                init = [0 if random.random() < 0.35 else 1 for _ in range(num_triples)]
            elif restart == 6:
                # Random with blue bias
                init = [0 if random.random() < 0.25 else 1 for _ in range(num_triples)]
            elif restart == 7:
                # Structured: based on vertex sum mod 3
                init = [0 if (a+b+c) % 3 == 0 else 1 for (a,b,c) in triple_list]
                # Perturb
                for i in range(num_triples):
                    if random.random() < 0.1:
                        init[i] = 1 - init[i]
            elif restart == 8:
                # Based on min vertex mod 2
                init = [0 if a % 2 == 0 else 1 for (a,b,c) in triple_list]
            elif restart == 9:
                # Based on max vertex mod 3
                init = [0 if c % 3 == 0 else 1 for (a,b,c) in triple_list]
            else:
                # Random perturbation of modular
                base = [0 if (a+b+c) % 3 == 0 else 1 for (a,b,c) in triple_list]
                init = base[:]
                for i in range(num_triples):
                    if random.random() < 0.12:
                        init[i] = 1 - init[i]

            colors, success = local_search(target_n, init, deadline)
            if success:
                found = True
                if is_valid(target_n, colors):
                    best_n = target_n
                    best_colors = colors
                    break
                else:
                    # Double-check
                    found = False

        if found:
            continue
        # If we couldn't find valid coloring for this n, stop trying larger n
        # but keep trying a few more restarts
        if target_n > 12:
            break

    # Final safety verification
    if best_n > 9 and not is_valid(best_n, best_colors):
        best_n = 9
        best_colors = mod_color(9, 3)

    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END
