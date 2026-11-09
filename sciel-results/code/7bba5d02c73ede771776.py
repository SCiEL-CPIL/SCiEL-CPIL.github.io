"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s limit

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
        num_quads = len(all_quads)
        quad_triples = np.array([
            [triple_to_idx[(a,b,c)], triple_to_idx[(a,b,d)],
             triple_to_idx[(a,c,d)], triple_to_idx[(b,c,d)]]
            for a, b, c, d in all_quads
        ], dtype=np.int32)

        all_quints = list(combinations(range(target_n), 5))
        num_quints = len(all_quints)
        quint_triples = np.array([
            [triple_to_idx[t] for t in combinations(q5, 3)]
            for q5 in all_quints
        ], dtype=np.int32)

        # For each triple, list of quad indices containing it
        triple_in_quads = [[] for _ in range(num_triples)]
        for qi in range(num_quads):
            for tidx in quad_triples[qi]:
                triple_in_quads[tidx].append(qi)

        # For each triple, list of quint indices containing it
        triple_in_quints = [[] for _ in range(num_triples)]
        for qi in range(num_quints):
            for tidx in quint_triples[qi]:
                triple_in_quints[tidx].append(qi)

        return (triple_list, num_triples, triple_to_idx, quad_triples, num_quads,
                quint_triples, num_quints, triple_in_quads, triple_in_quints)

    def count_violations(color_arr, structures):
        """Count red K4 and blue K5 violations using numpy vectorization."""
        (triple_list, num_triples, triple_to_idx, quad_triples, num_quads,
         quint_triples, num_quints, triple_in_quads, triple_in_quints) = structures

        # Red K4: quad is all red iff all 4 triples are color 0
        quad_red_counts = (color_arr[quad_triples] == 0).sum(axis=1)
        rk4 = np.sum(quad_red_counts == 4)

        # Blue K5: quint is all blue iff all 10 triples are color 1
        quint_blue_counts = (color_arr[quint_triples] == 1).sum(axis=1)
        bk5 = np.sum(quint_blue_counts == 10)

        return rk4, bk5

    def local_search(target_n, init_colors, structures, deadline):
        """Run local search to eliminate violations. Returns (colors, success)."""
        (triple_list, num_triples, triple_to_idx, quad_triples, num_quads,
         quint_triples, num_quints, triple_in_quads, triple_in_quints) = structures

        color_arr = np.array(init_colors, dtype=np.int8)

        rk4, bk5 = count_violations(color_arr, structures)
        if rk4 == 0 and bk5 == 0:
            return color_arr.tolist(), True

        max_iters = 400000
        temp_start = 2.5
        temp_end = 0.001

        for it in range(max_iters):
            if time.time() > deadline - 2:
                break
            if rk4 == 0 and bk5 == 0:
                return color_arr.tolist(), True

            temp = temp_start * (temp_end / temp_start) ** (it / max_iters)

            # Choose a triple to flip - bias toward violating structures
            tidx = -1
            r = random.random()
            if rk4 > 0 and r < 0.55:
                # Pick from violating red K4s
                quad_red_counts = (color_arr[quad_triples] == 0).sum(axis=1)
                violating = np.where(quad_red_counts == 4)[0]
                if len(violating) > 0:
                    qi = violating[random.randint(0, len(violating) - 1)]
                    tidx = quad_triples[qi][random.randint(0, 3)]
            elif bk5 > 0 and r < 0.90:
                # Pick from violating blue K5s
                quint_blue_counts = (color_arr[quint_triples] == 1).sum(axis=1)
                violating = np.where(quint_blue_counts == 10)[0]
                if len(violating) > 0:
                    qi = violating[random.randint(0, len(violating) - 1)]
                    tidx = quint_triples[qi][random.randint(0, 9)]

            if tidx == -1:
                tidx = random.randint(0, num_triples - 1)

            old_color = color_arr[tidx]

            # Compute delta_rk4
            delta_rk4 = 0
            for qi in triple_in_quads[tidx]:
                qt = quad_triples[qi]
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
                qt = quint_triples[qi]
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
                rk4 += delta_rk4
                bk5 += delta_bk5

        if rk4 == 0 and bk5 == 0:
            return color_arr.tolist(), True
        return color_arr.tolist(), False

    def try_target_n(target_n, deadline):
        """Try to find a valid coloring for target_n vertices."""
        structures = build_structures(target_n)
        triple_list, num_triples = structures[0], structures[1]

        # Multiple initialization strategies
        init_strategies = []

        # Strategy 1: mod 3
        init_strategies.append([0 if (a+b+c) % 3 == 0 else 1 for (a,b,c) in triple_list])
        # Strategy 2: mod 2
        init_strategies.append([0 if (a+b+c) % 2 == 0 else 1 for (a,b,c) in triple_list])
        # Strategy 3: random
        init_strategies.append(np.random.randint(0, 2, size=num_triples).tolist())
        # Strategy 4: random with red bias
        arr = np.random.randint(0, 2, size=num_triples)
        arr[np.random.random(num_triples) < 0.4] = 0
        init_strategies.append(arr.tolist())
        # Strategy 5: random with blue bias
        arr = np.random.randint(0, 2, size=num_triples)
        arr[np.random.random(num_triples) < 0.4] = 1
        init_strategies.append(arr.tolist())
        # Strategy 6: mod 3 with 10% noise
        base = [0 if (a+b+c) % 3 == 0 else 1 for (a,b,c) in triple_list]
        noisy = base[:]
        for i in range(num_triples):
            if random.random() < 0.10:
                noisy[i] = 1 - noisy[i]
        init_strategies.append(noisy)
        # Strategy 7: mod 3 with 20% noise
        base = [0 if (a+b+c) % 3 == 0 else 1 for (a,b,c) in triple_list]
        noisy = base[:]
        for i in range(num_triples):
            if random.random() < 0.20:
                noisy[i] = 1 - noisy[i]
        init_strategies.append(noisy)
        # Strategy 8: vertex parity XOR
        init_strategies.append([0 if (a % 2) ^ (b % 2) ^ (c % 2) == 0 else 1 for (a,b,c) in triple_list])
        # Strategy 9: sum mod 4
        init_strategies.append([0 if (a+b+c) % 4 < 2 else 1 for (a,b,c) in triple_list])
        # Strategy 10: product mod 3
        init_strategies.append([0 if (a*b*c) % 3 == 0 else 1 for (a,b,c) in triple_list])

        # Shuffle strategies
        random.shuffle(init_strategies)

        for init_colors in init_strategies:
            if time.time() > deadline - 3:
                break
            colors, success = local_search(target_n, init_colors, structures, deadline)
            if success:
                # Verify
                if is_valid(target_n, colors):
                    return colors
        return None

    # Try increasing n values
    for target_n in range(10, 35):
        if time.time() > deadline - 10:
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
