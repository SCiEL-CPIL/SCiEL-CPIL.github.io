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

    for target_n in range(10, 35):
        if time.time() > deadline - 8:
            break

        # Precompute structures
        triple_list = list(combinations(range(target_n), 3))
        num_triples = len(triple_list)
        triple_to_idx = {t: i for i, t in enumerate(triple_list)}

        # For each quad (4-set), the 4 triples it contains
        all_quads = list(combinations(range(target_n), 4))
        quad_triples_idx = []
        for a, b, c, d in all_quads:
            quad_triples_idx.append([
                triple_to_idx[(a,b,c)], triple_to_idx[(a,b,d)],
                triple_to_idx[(a,c,d)], triple_to_idx[(b,c,d)]
            ])
        num_quads = len(quad_triples_idx)

        # For each quint (5-set), the 10 triples it contains
        all_quints = list(combinations(range(target_n), 5))
        quint_triples_idx = []
        for q5 in all_quints:
            quint_triples_idx.append([triple_to_idx[t] for t in combinations(q5, 3)])
        num_quints = len(quint_triples_idx)

        # For each triple, which quads/quints contain it
        triple_in_quads = [[] for _ in range(num_triples)]
        for qi, qt in enumerate(quad_triples_idx):
            for pos, tidx in enumerate(qt):
                triple_in_quads[tidx].append((qi, pos))

        triple_in_quints = [[] for _ in range(num_triples)]
        for qi, qt in enumerate(quint_triples_idx):
            for pos, tidx in enumerate(qt):
                triple_in_quints[tidx].append((qi, pos))

        # Convert to numpy arrays for faster access
        quad_triples_np = np.array(quad_triples_idx, dtype=np.int32)
        quint_triples_np = np.array(quint_triples_idx, dtype=np.int32)

        found = False
        max_restarts = 200
        for restart in range(max_restarts):
            if time.time() > deadline - 3:
                break

            # Initialize coloring
            if restart == 0:
                color_arr = np.array([0 if (a+b+c) % 3 == 0 else 1 for (a,b,c) in triple_list], dtype=np.int8)
            elif restart == 1:
                color_arr = np.array([0 if (a+b+c) % 4 == 0 else 1 for (a,b,c) in triple_list], dtype=np.int8)
            elif restart == 2:
                color_arr = np.array([0 if (a+b+c) % 5 == 0 else 1 for (a,b,c) in triple_list], dtype=np.int8)
            elif restart == 3:
                color_arr = np.random.randint(0, 2, size=num_triples, dtype=np.int8)
            elif restart == 4:
                color_arr = np.random.randint(0, 2, size=num_triples, dtype=np.int8)
                color_arr[np.random.random(num_triples) < 0.4] = 0
            elif restart == 5:
                color_arr = np.random.randint(0, 2, size=num_triples, dtype=np.int8)
                color_arr[np.random.random(num_triples) < 0.4] = 1
            elif restart % 2 == 0:
                base = np.array([0 if (a+b+c) % 3 == 0 else 1 for (a,b,c) in triple_list], dtype=np.int8)
                flip_mask = np.random.random(num_triples) < 0.2
                color_arr = base.copy()
                color_arr[flip_mask] = 1 - color_arr[flip_mask]
            else:
                base = np.array([0 if (a+b+c) % 4 == 0 else 1 for (a,b,c) in triple_list], dtype=np.int8)
                flip_mask = np.random.random(num_triples) < 0.2
                color_arr = base.copy()
                color_arr[flip_mask] = 1 - color_arr[flip_mask]

            # Count violations using vectorized numpy operations
            quad_colors = quad_triples_np  # shape (num_quads, 4)
            quad_red_counts = np.sum(color_arr[quad_colors] == 0, axis=1)
            rk4 = int(np.sum(quad_red_counts == 4))

            quint_colors = quint_triples_np  # shape (num_quints, 10)
            quint_blue_counts = np.sum(color_arr[quint_colors] == 1, axis=1)
            bk5 = int(np.sum(quint_blue_counts == 10))

            if rk4 == 0 and bk5 == 0:
                found = True
                break

            # Local search with simulated annealing
            max_iters = 50000
            temp_start = 2.0
            temp_end = 0.01
            it = 0
            while it < max_iters:
                if time.time() > deadline - 2:
                    break
                if rk4 == 0 and bk5 == 0:
                    found = True
                    break

                temp = temp_start * (temp_end / temp_start) ** (it / max_iters)

                # Choose a triple to flip - bias toward violating structures
                tidx = -1
                r = random.random()
                if rk4 > 0 and r < 0.5:
                    # Find a violating red K4 and flip one of its triples
                    violating_quads = np.where(quad_red_counts == 4)[0]
                    if len(violating_quads) > 0:
                        qi = violating_quads[random.randint(0, len(violating_quads)-1)]
                        qt = quad_triples_np[qi]
                        tidx = int(qt[random.randint(0, 3)])
                elif bk5 > 0 and r < 0.85:
                    # Find a violating blue K5 and flip one of its triples
                    violating_quints = np.where(quint_blue_counts == 10)[0]
                    if len(violating_quints) > 0:
                        qi = violating_quints[random.randint(0, len(violating_quints)-1)]
                        qt = quint_triples_np[qi]
                        tidx = int(qt[random.randint(0, 9)])

                if tidx == -1:
                    tidx = random.randint(0, num_triples - 1)

                old_color = int(color_arr[tidx])

                # Compute delta_rk4 incrementally
                delta_rk4 = 0
                for (qi, pos) in triple_in_quads[tidx]:
                    qt = quad_triples_np[qi]
                    # Check if other 3 triples are red
                    others = [qt[p] for p in range(4) if p != pos]
                    others_red = all(int(color_arr[t]) == 0 for t in others)
                    if old_color == 0 and others_red:
                        delta_rk4 -= 1
                    elif old_color == 1 and others_red:
                        delta_rk4 += 1

                # Compute delta_bk5 incrementally
                delta_bk5 = 0
                for (qi, pos) in triple_in_quints[tidx]:
                    qt = quint_triples_np[qi]
                    # Check if other 9 triples are blue
                    others = [qt[p] for p in range(10) if p != pos]
                    others_blue = all(int(color_arr[t]) == 1 for t in others)
                    if old_color == 1 and others_blue:
                        delta_bk5 -= 1
                    elif old_color == 0 and others_blue:
                        delta_bk5 += 1

                delta_total = delta_rk4 + delta_bk5

                # Acceptance criterion
                if delta_total <= 0:
                    color_arr[tidx] = 1 - old_color
                    rk4 += delta_rk4
                    bk5 += delta_bk5
                else:
                    if random.random() < np.exp(-delta_total / max(temp, 1e-10)):
                        color_arr[tidx] = 1 - old_color
                        rk4 += delta_rk4
                        bk5 += delta_bk5

                it += 1

            if found:
                break

        if found:
            # Verify the solution
            colors_list = color_arr.tolist()
            if is_valid(target_n, colors_list):
                best_n = target_n
                best_colors = colors_list
            else:
                # Should not happen, but break if invalid
                break
        else:
            # If we failed to find a valid coloring for this n,
            # it's unlikely we can find one for larger n quickly
            if target_n >= 30:
                break

    return {'n': best_n, 'colors': best_colors}
