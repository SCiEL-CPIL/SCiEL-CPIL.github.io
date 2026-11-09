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

    for target_n in range(10, 35):
        if time.time() > deadline - 10:
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
            if time.time() > deadline - 5:
                break

            # Initialize coloring with diverse strategies
            if restart == 0:
                color_arr = np.array([0 if (a+b+c) % 3 == 0 else 1 for (a,b,c) in triple_list], dtype=np.int8)
            elif restart == 1:
                color_arr = np.array([0 if (a+b+c) % 4 == 0 else 1 for (a,b,c) in triple_list], dtype=np.int8)
            elif restart == 2:
                color_arr = np.array([0 if (a+b+c) % 5 == 0 else 1 for (a,b,c) in triple_list], dtype=np.int8)
            elif restart == 3:
                color_arr = np.array([0 if (a*b + b*c + c*a) % 3 == 0 else 1 for (a,b,c) in triple_list], dtype=np.int8)
            elif restart == 4:
                color_arr = np.array([0 if (a*b + b*c + c*a) % 4 == 0 else 1 for (a,b,c) in triple_list], dtype=np.int8)
            elif restart == 5:
                color_arr = np.array([0 if (a*b*c) % 2 == 0 else 1 for (a,b,c) in triple_list], dtype=np.int8)
            elif restart == 6:
                # Vertex-based: color by vertex parity
                color_arr = np.array([0 if (a % 2) ^ (b % 2) ^ (c % 2) == 0 else 1 for (a,b,c) in triple_list], dtype=np.int8)
            elif restart == 7:
                # Random with bias
                color_arr = np.random.randint(0, 2, size=num_triples, dtype=np.int8)
                color_arr[np.random.random(num_triples) < 0.45] = 0
            elif restart == 8:
                # Random with different bias
                color_arr = np.random.randint(0, 2, size=num_triples, dtype=np.int8)
                color_arr[np.random.random(num_triples) < 0.55] = 1
            elif restart % 2 == 0:
                base = np.array([0 if (a+b+c) % 3 == 0 else 1 for (a,b,c) in triple_list], dtype=np.int8)
                flip_mask = np.random.random(num_triples) < 0.15
                color_arr = base.copy()
                color_arr[flip_mask] = 1 - color_arr[flip_mask]
            else:
                base = np.array([0 if (a+b+c) % 4 == 0 else 1 for (a,b,c) in triple_list], dtype=np.int8)
                flip_mask = np.random.random(num_triples) < 0.15
                color_arr = base.copy()
                color_arr[flip_mask] = 1 - color_arr[flip_mask]

            # Count violations using vectorized numpy operations
            # Red K4: all 4 triples in a quad are red (color 0)
            quad_colors = color_arr[quad_triples_np]  # shape (num_quads, 4)
            quad_red_counts = np.sum(quad_colors == 0, axis=1)
            rk4 = int(np.sum(quad_red_counts == 4))

            # Blue K5: all 10 triples in a quint are blue (color 1)
            quint_colors = color_arr[quint_triples_np]  # shape (num_quints, 10)
            quint_blue_counts = np.sum(quint_colors == 1, axis=1)
            bk5 = int(np.sum(quint_blue_counts == 10))

            if rk4 == 0 and bk5 == 0:
                found = True
                break

            # Local search with simulated annealing
            max_iters = 50000
            temp_start = 3.0
            temp_end = 0.005
            it = 0
            stuck_counter = 0
            best_rk4_bk5 = rk4 + bk5
            
            while it < max_iters:
                if time.time() > deadline - 3:
                    break
                if rk4 == 0 and bk5 == 0:
                    found = True
                    break

                temp = temp_start * (temp_end / temp_start) ** (it / max_iters)

                # Choose a triple to flip - bias toward violating structures
                tidx = -1
                r = random.random()
                if rk4 > 0 and bk5 > 0:
                    if r < 0.5:
                        # Find a violating red K4 and flip one of its triples
                        violating_quads = np.where(quad_red_counts == 4)[0]
                        if len(violating_quads) > 0:
                            qi = violating_quads[random.randint(0, len(violating_quads)-1)]
                            qt = quad_triples_np[qi]
                            tidx = int(qt[random.randint(0, 3)])
                    else:
                        # Find a violating blue K5 and flip one of its triples
                        violating_quints = np.where(quint_blue_counts == 10)[0]
                        if len(violating_quints) > 0:
                            qi = violating_quints[random.randint(0, len(violating_quints)-1)]
                            qt = quint_triples_np[qi]
                            tidx = int(qt[random.randint(0, 9)])
                elif rk4 > 0:
                    # Find a violating red K4 and flip one of its triples
                    violating_quads = np.where(quad_red_counts == 4)[0]
                    if len(violating_quads) > 0:
                        qi = violating_quads[random.randint(0, len(violating_quads)-1)]
                        qt = quad_triples_np[qi]
                        tidx = int(qt[random.randint(0, 3)])
                elif bk5 > 0:
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

                delta = delta_rk4 + delta_bk5

                if delta <= 0 or random.random() < np.exp(-delta / max(temp, 1e-10)):
                    # Accept the flip
                    new_color = 1 - old_color
                    color_arr[tidx] = new_color
                    rk4 += delta_rk4
                    bk5 += delta_bk5

                    # Update violation counts for affected quads/quints
                    for (qi, pos) in triple_in_quads[tidx]:
                        qt = quad_triples_np[qi]
                        old_rc = int(quad_red_counts[qi])
                        if old_color == 0:
                            quad_red_counts[qi] = old_rc - 1
                        else:
                            quad_red_counts[qi] = old_rc + 1
                    for (qi, pos) in triple_in_quints[tidx]:
                        qt = quint_triples_np[qi]
                        old_bc = int(quint_blue_counts[qi])
                        if old_color == 1:
                            quint_blue_counts[qi] = old_bc - 1
                        else:
                            quint_blue_counts[qi] = old_bc + 1

                    if rk4 + bk5 < best_rk4_bk5:
                        best_rk4_bk5 = rk4 + bk5
                        stuck_counter = 0
                    else:
                        stuck_counter += 1
                else:
                    stuck_counter += 1

                # If stuck for too long, restart with a new random perturbation
                if stuck_counter > 5000:
                    stuck_counter = 0
                    # Perturb a random subset of triples
                    num_perturb = max(1, num_triples // 10)
                    perturb_idx = np.random.choice(num_triples, size=num_perturb, replace=False)
                    color_arr[perturb_idx] = 1 - color_arr[perturb_idx]
                    # Recompute violations
                    quad_colors = color_arr[quad_triples_np]
                    quad_red_counts = np.sum(quad_colors == 0, axis=1)
                    rk4 = int(np.sum(quad_red_counts == 4))
                    quint_colors = color_arr[quint_triples_np]
                    quint_blue_counts = np.sum(quint_colors == 1, axis=1)
                    bk5 = int(np.sum(quint_blue_counts == 10))

                it += 1

            if rk4 == 0 and bk5 == 0:
                found = True
                break

        if found:
            best_n = target_n
            best_colors = color_arr.tolist()
            if is_valid(target_n, best_colors):
                pass
            else:
                best_n = 9
                best_colors = mod_color(9, 3)

    # Final safety verification
    if best_n > 9 and not is_valid(best_n, best_colors):
        best_n = 9
        best_colors = mod_color(9, 3)

    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END
