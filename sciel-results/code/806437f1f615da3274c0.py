"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 117  # Safety margin before 120s limit

    best_n = 9
    best_colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(9), 3)]

    # Try increasing n
    for target_n in range(10, 35):
        if time.time() > deadline - 8:
            break

        # Precompute all quads and quints and their triple indices
        triple_list = list(combinations(range(target_n), 3))
        num_triples = len(triple_list)
        triple_to_idx = {t: i for i, t in enumerate(triple_list)}

        all_quads = list(combinations(range(target_n), 4))
        num_quads = len(all_quads)
        # quad_triple_idx[qi] = list of 4 triple indices
        quad_triple_idx = []
        for a, b, c, d in all_quads:
            quad_triple_idx.append([triple_to_idx[(a,b,c)], triple_to_idx[(a,b,d)],
                                     triple_to_idx[(a,c,d)], triple_to_idx[(b,c,d)]])

        all_quints = list(combinations(range(target_n), 5))
        num_quints = len(all_quints)
        # quint_triple_idx[qi] = list of 10 triple indices
        quint_triple_idx = []
        for q5 in all_quints:
            quint_triple_idx.append([triple_to_idx[t] for t in combinations(q5, 3)])

        # Precompute: for each triple index, which quads/quints contain it and at which position
        triple_quad_map = [[] for _ in range(num_triples)]
        for qi in range(num_quads):
            for pos, ti in enumerate(quad_triple_idx[qi]):
                triple_quad_map[ti].append((qi, pos))

        triple_quint_map = [[] for _ in range(num_triples)]
        for qi in range(num_quints):
            for pos, ti in enumerate(quint_triple_idx[qi]):
                triple_quint_map[ti].append((qi, pos))

        # Precompute bitmasks for fast "others all red/blue" checks
        # For each quad qi and position pos, bitmask of the other 3 triples
        quad_other_masks = [[0]*4 for _ in range(num_quads)]
        for qi in range(num_quads):
            idxs = quad_triple_idx[qi]
            for pos in range(4):
                mask = 0
                for p in range(4):
                    if p != pos:
                        mask |= (1 << idxs[p])
                quad_other_masks[qi][pos] = mask

        # For each quint qi and position pos, bitmask of the other 9 triples
        quint_other_masks = [[0]*10 for _ in range(num_quints)]
        for qi in range(num_quints):
            idxs = quint_triple_idx[qi]
            for pos in range(10):
                mask = 0
                for p in range(10):
                    if p != pos:
                        mask |= (1 << idxs[p])
                quint_other_masks[qi][pos] = mask

        found = False
        max_restarts = 400
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break

            # Diverse initialization strategies
            if restart == 0:
                # Modular init: (a+b+c) mod 3 == 0 -> red, else blue
                color_flat = np.array([0 if (a+b+c) % 3 == 0 else 1 for (a,b,c) in triple_list], dtype=np.int8)
            elif restart == 1:
                # Modular init: (a+b+c) mod 2 == 0 -> red, else blue
                color_flat = np.array([0 if (a+b+c) % 2 == 0 else 1 for (a,b,c) in triple_list], dtype=np.int8)
            elif restart == 2:
                # Random
                color_flat = np.random.randint(0, 2, size=num_triples, dtype=np.int8)
            elif restart == 3:
                # Modular init: (a+b+c) mod 4 == 0 -> red, else blue
                color_flat = np.array([0 if (a+b+c) % 4 == 0 else 1 for (a,b,c) in triple_list], dtype=np.int8)
            elif restart == 4:
                # Vertex-based: color by min vertex mod 3
                color_flat = np.array([0 if a % 3 == 0 else 1 for (a,b,c) in triple_list], dtype=np.int8)
            else:
                # Perturb modular init with varying flip rates
                flip_rate = 0.05 + 0.1 * ((restart % 5) / 5.0)
                base = np.array([0 if (a+b+c) % 3 == 0 else 1 for (a,b,c) in triple_list], dtype=np.int8)
                flip_mask = np.random.random(num_triples) < flip_rate
                color_flat = base.copy()
                color_flat[flip_mask] = 1 - color_flat[flip_mask]

            # Build red/blue bitmasks for fast violation checking
            # red_mask has bit i set if triple i is red (color 0)
            # blue_mask has bit i set if triple i is blue (color 1)
            red_mask = 0
            blue_mask = 0
            for i in range(num_triples):
                if color_flat[i] == 0:
                    red_mask |= (1 << i)
                else:
                    blue_mask |= (1 << i)

            # Count initial violations using bitmask checks
            rk4 = 0
            for qi in range(num_quads):
                idxs = quad_triple_idx[qi]
                # Check if all 4 triples are red
                qmask = 0
                for i in idxs:
                    qmask |= (1 << i)
                if (red_mask & qmask) == qmask:
                    rk4 += 1

            bk5 = 0
            for qi in range(num_quints):
                idxs = quint_triple_idx[qi]
                # Check if all 10 triples are blue
                qmask = 0
                for i in idxs:
                    qmask |= (1 << i)
                if (blue_mask & qmask) == qmask:
                    bk5 += 1

            if rk4 == 0 and bk5 == 0:
                found = True
                break

            # Simulated annealing local search with fast bitmask delta computation
            max_iters = 50000
            temp_start = 1.0
            temp_end = 0.01
            for it in range(max_iters):
                if time.time() > deadline - 3:
                    break
                if rk4 == 0 and bk5 == 0:
                    found = True
                    break

                temp = temp_start * (temp_end / temp_start) ** (it / max_iters)

                # Choose a triple to flip - prioritize violating structures
                if rk4 > 0 and random.random() < 0.7:
                    # Pick from a violating red K4
                    tidx = -1
                    for _ in range(20):
                        qi = random.randint(0, num_quads - 1)
                        idxs = quad_triple_idx[qi]
                        qmask = 0
                        for i in idxs:
                            qmask |= (1 << i)
                        if (red_mask & qmask) == qmask:
                            pos = random.randint(0, 3)
                            tidx = idxs[pos]
                            break
                    if tidx < 0:
                        tidx = random.randint(0, num_triples - 1)
                elif bk5 > 0 and random.random() < 0.7:
                    # Pick from a violating blue K5
                    tidx = -1
                    for _ in range(20):
                        qi = random.randint(0, num_quints - 1)
                        idxs = quint_triple_idx[qi]
                        qmask = 0
                        for i in idxs:
                            qmask |= (1 << i)
                        if (blue_mask & qmask) == qmask:
                            pos = random.randint(0, 9)
                            tidx = idxs[pos]
                            break
                    if tidx < 0:
                        tidx = random.randint(0, num_triples - 1)
                else:
                    tidx = random.randint(0, num_triples - 1)

                old_color = color_flat[tidx]
                new_color = 1 - old_color

                # Compute delta_rk4 using precomputed masks
                delta_rk4 = 0
                for (qi, pos) in triple_quad_map[tidx]:
                    others_mask = quad_other_masks[qi][pos]
                    if old_color == 0:
                        # Flipping red->blue: removes a red K4 if others were all red
                        if (red_mask & others_mask) == others_mask:
                            delta_rk4 -= 1
                    else:
                        # Flipping blue->red: creates a red K4 if others are all red
                        if (red_mask & others_mask) == others_mask:
                            delta_rk4 += 1

                # Compute delta_bk5 using precomputed masks
                delta_bk5 = 0
                for (qi, pos) in triple_quint_map[tidx]:
                    others_mask = quint_other_masks[qi][pos]
                    if old_color == 1:
                        # Flipping blue->red: removes a blue K5 if others were all blue
                        if (blue_mask & others_mask) == others_mask:
                            delta_bk5 -= 1
                    else:
                        # Flipping red->blue: creates a blue K5 if others are all blue
                        if (blue_mask & others_mask) == others_mask:
                            delta_bk5 += 1

                old_total = rk4 + bk5
                new_total = (rk4 + delta_rk4) + (bk5 + delta_bk5)
                delta = new_total - old_total

                if delta <= 0 or random.random() < np.exp(-delta / max(temp, 1e-10)):
                    color_flat[tidx] = new_color
                    # Update masks
                    if old_color == 0:
                        red_mask &= ~(1 << tidx)
                        blue_mask |= (1 << tidx)
                    else:
                        blue_mask &= ~(1 << tidx)
                        red_mask |= (1 << tidx)
                    rk4 += delta_rk4
                    bk5 += delta_bk5

            # Post-search fix pass: greedily flip triples in violating structures
            fix_passes = 50
            for fp in range(fix_passes):
                if time.time() > deadline - 3:
                    break
                if rk4 == 0 and bk5 == 0:
                    found = True
                    break

                made_progress = False
                # Try to fix red K4s
                if rk4 > 0:
                    for qi in range(num_quads):
                        idxs = quad_triple_idx[qi]
                        qmask = 0
                        for i in idxs:
                            qmask |= (1 << i)
                        if (red_mask & qmask) == qmask:
                            pos = random.randint(0, 3)
                            tidx = idxs[pos]
                            # Compute delta
                            delta_rk4 = 0
                            for (q2, p2) in triple_quad_map[tidx]:
                                others_mask = quad_other_masks[q2][p2]
                                if (red_mask & others_mask) == others_mask:
                                    delta_rk4 += 1  # flipping 0->1 removes a red K4
                            delta_bk5 = 0
                            for (q2, p2) in triple_quint_map[tidx]:
                                others_mask = quint_other_masks[q2][p2]
                                if (blue_mask & others_mask) == others_mask:
                                    delta_bk5 += 1  # flipping 0->1 may create a blue K5
                            if delta_rk4 + delta_bk5 <= 0:
                                color_flat[tidx] = 1
                                red_mask &= ~(1 << tidx)
                                blue_mask |= (1 << tidx)
                                rk4 += delta_rk4
                                bk5 += delta_bk5
                                made_progress = True
                                break
                # Try to fix blue K5s
                if bk5 > 0:
                    for qi in range(num_quints):
                        idxs = quint_triple_idx[qi]
                        qmask = 0
                        for i in idxs:
                            qmask |= (1 << i)
                        if (blue_mask & qmask) == qmask:
                            pos = random.randint(0, 9)
                            tidx = idxs[pos]
                            delta_rk4 = 0
                            for (q2, p2) in triple_quad_map[tidx]:
                                others_mask = quad_other_masks[q2][p2]
                                if (red_mask & others_mask) == others_mask:
                                    delta_rk4 += 1  # flipping 1->0 may create a red K4
                            delta_bk5 = 0
                            for (q2, p2) in triple_quint_map[tidx]:
                                others_mask = quint_other_masks[q2][p2]
                                if (blue_mask & others_mask) == others_mask:
                                    delta_bk5 -= 1  # flipping 1->0 removes a blue K5
                            if delta_rk4 + delta_bk5 <= 0:
                                color_flat[tidx] = 0
                                blue_mask &= ~(1 << tidx)
                                red_mask |= (1 << tidx)
                                rk4 += delta_rk4
                                bk5 += delta_bk5
                                made_progress = True
                                break
                if not made_progress:
                    break

            if rk4 == 0 and bk5 == 0:
                found = True
                break

        if found:
            best_n = target_n
            best_colors = color_flat.tolist()

    # Final verification
    if best_n > 9:
        color_map = {}
        for i, (a, b, c) in enumerate(combinations(range(best_n), 3)):
            color_map[(a, b, c)] = best_colors[i]

        valid = True
        for quad in combinations(range(best_n), 4):
            a, b, c, d = quad
            if all(color_map[t] == 0 for t in [(a,b,c),(a,b,d),(a,c,d),(b,c,d)]):
                valid = False
                break
        if valid:
            for quint in combinations(range(best_n), 5):
                if all(color_map[t] == 1 for t in combinations(quint, 3)):
                    valid = False
                    break

        if not valid:
            best_n = 9
            best_colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(9), 3)]

    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END
