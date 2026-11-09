"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115

    def time_left():
        return deadline - time.time()

    def precompute(n):
        triples = list(combinations(range(n), 3))
        num_triples = len(triples)
        idx_map = {t: i for i, t in enumerate(triples)}

        quad_list = list(combinations(range(n), 4))
        quint_list = list(combinations(range(n), 5))
        num_quads = len(quad_list)
        num_quints = len(quint_list)

        # For each quad, store the 4 triple indices
        quad_tris = np.zeros((num_quads, 4), dtype=np.int32)
        for qi, quad in enumerate(quad_list):
            for ti, tri in enumerate(combinations(quad, 3)):
                quad_tris[qi, ti] = idx_map[tri]

        # For each quint, store the 10 triple indices
        quint_tris = np.zeros((num_quints, 10), dtype=np.int32)
        for qi, quint in enumerate(quint_list):
            for ti, tri in enumerate(combinations(quint, 3)):
                quint_tris[qi, ti] = idx_map[tri]

        # Build reverse maps: for each triple, which quads/quints contain it
        # Use sparse representation: list of arrays
        triple_to_quads = [[] for _ in range(num_triples)]
        triple_to_quints = [[] for _ in range(num_triples)]

        for qi in range(num_quads):
            for ti in quad_tris[qi]:
                triple_to_quads[ti].append(qi)

        for qi in range(num_quints):
            for ti in quint_tris[qi]:
                triple_to_quints[ti].append(qi)

        triple_to_quads = [np.array(lst, dtype=np.int32) for lst in triple_to_quads]
        triple_to_quints = [np.array(lst, dtype=np.int32) for lst in triple_to_quints]

        return triples, quad_tris, quint_tris, triple_to_quads, triple_to_quints, num_triples, num_quads, num_quints

    def make_initial_colors(n, num_triples, pattern_idx=0):
        """Generate diverse initial colorings using different patterns."""
        triples = list(combinations(range(n), 3))
        if pattern_idx == 0:
            # Random
            return np.random.randint(0, 2, size=num_triples, dtype=np.int8)
        elif pattern_idx == 1:
            # Mod 3 on sum
            return np.array([0 if (a+b+c) % 3 == 0 else 1 for a,b,c in triples], dtype=np.int8)
        elif pattern_idx == 2:
            # Mod 4 on sum
            return np.array([0 if (a+b+c) % 4 < 2 else 1 for a,b,c in triples], dtype=np.int8)
        elif pattern_idx == 3:
            # Based on smallest vertex mod 2
            return np.array([0 if a % 2 == 0 else 1 for a,b,c in triples], dtype=np.int8)
        elif pattern_idx == 4:
            # Based on middle vertex mod 3
            return np.array([0 if b % 3 == 0 else 1 for a,b,c in triples], dtype=np.int8)
        elif pattern_idx == 5:
            # Random with bias
            return (np.random.random(num_triples) < 0.4).astype(np.int8)
        elif pattern_idx == 6:
            # Based on (a+b) mod 5
            return np.array([0 if (a+b) % 5 < 2 else 1 for a,b,c in triples], dtype=np.int8)
        else:
            return np.random.randint(0, 2, size=num_triples, dtype=np.int8)

    def count_violations(colors, quad_tris, quint_tris, num_quads, num_quints):
        """Count red triples per quad and blue triples per quint."""
        red_quad_count = np.zeros(num_quads, dtype=np.int32)
        blue_quint_count = np.zeros(num_quints, dtype=np.int32)

        # Vectorized: for each quad, count how many of its 4 triples are red (0)
        for qi in range(num_quads):
            red_quad_count[qi] = np.sum(colors[quad_tris[qi]] == 0)

        for qi in range(num_quints):
            blue_quint_count[qi] = np.sum(colors[quint_tris[qi]] == 1)

        return red_quad_count, blue_quint_count

    def local_search_bitparallel(n, time_limit, pattern_idx=0):
        """Bit-parallel local search for Max-SAT style optimization."""
        if n < 4:
            return None

        triples, quad_tris, quint_tris, triple_to_quads, triple_to_quints, num_triples, num_quads, num_quints = precompute(n)

        colors = make_initial_colors(n, num_triples, pattern_idx)

        red_quad_count, blue_quint_count = count_violations(colors, quad_tris, quint_tris, num_quads, num_quints)

        red_viol = set(np.where(red_quad_count == 4)[0].tolist())
        blue_viol = set(np.where(blue_quint_count == 10)[0].tolist())

        if not red_viol and not blue_viol:
            return colors.tolist()

        # Precompute for faster updates: for each triple, the list of affected quads/quints
        # and precompute the "delta" direction
        # For flipping triple t from red(0) to blue(1):
        #   - red_quad_count decreases by 1 for all quads containing t
        #   - blue_quint_count increases by 1 for all quints containing t
        # For flipping triple t from blue(1) to red(0):
        #   - red_quad_count increases by 1 for all quads containing t
        #   - blue_quint_count decreases by 1 for all quints containing t

        stall_count = 0
        max_stall = 5000

        while time_left() > time_limit:
            if not red_viol and not blue_viol:
                return colors.tolist()

            if stall_count > max_stall:
                # Restart with different pattern
                break

            # Choose which type of violation to fix
            if red_viol and (not blue_viol or random.random() < len(red_viol) / (len(red_viol) + len(blue_viol))):
                if not red_viol:
                    continue
                quad_idx = random.choice(list(red_viol))
                ts = quad_tris[quad_idx]
                # All 4 triples in this quad are red; pick one to flip to blue
                tri_idx = int(ts[random.randint(0, 3)])

                # Flip: red -> blue
                colors[tri_idx] = 1

                # Update red_quad_count for all quads containing tri_idx
                q_indices = triple_to_quads[tri_idx]
                if len(q_indices) > 0:
                    old = red_quad_count[q_indices].copy()
                    red_quad_count[q_indices] -= 1
                    # Quads that were at 4 are now at 3: remove from viol
                    fixed = q_indices[old == 4]
                    if len(fixed) > 0:
                        red_viol.difference_update(fixed.tolist())
                    # Quads that were at 3 are now at 4: add to viol
                    new_viol = q_indices[old == 3]
                    if len(new_viol) > 0:
                        red_viol.update(new_viol.tolist())

                # Update blue_quint_count for all quints containing tri_idx
                    q_indices2 = triple_to_quints[tri_idx]
                if len(q_indices2) > 0:
                    old2 = blue_quint_count[q_indices2].copy()
                    blue_quint_count[q_indices2] += 1
                    # Quints that were at 10 are now at 11: remove from viol (but 11>10 so they were already violating, now still violating)
                    # Actually, if old was 10, new is 11: still violating (count >= 10 means violation? No, exactly 10)
                    # Wait: violation is count == 10. If old=10, new=11: no longer exactly 10, so remove
                    fixed2 = q_indices2[old2 == 10]
                    if len(fixed2) > 0:
                        blue_viol.difference_update(fixed2.tolist())
                    # Quints that were at 9 are now at 10: add to viol
                    new_viol2 = q_indices2[old2 == 9]
                    if len(new_viol2) > 0:
                        blue_viol.update(new_viol2.tolist())

                # Check if this flip helped (reduced total violations)
                # We don't track total explicitly, but we can check if the chosen quad was fixed
                # and no new violations were created in that quad
                # For simplicity, just continue; stall detection handles bad moves

            else:
                if not blue_viol:
                    continue
                quint_idx = random.choice(list(blue_viol))
                ts = quint_tris[quint_idx]
                # All 10 triples in this quint are blue; pick one to flip to red
                tri_idx = int(ts[random.randint(0, 9)])

                # Flip: blue -> red
                colors[tri_idx] = 0

                # Update red_quad_count for all quads containing tri_idx
                q_indices = triple_to_quads[tri_idx]
                if len(q_indices) > 0:
                    old = red_quad_count[q_indices].copy()
                    red_quad_count[q_indices] += 1
                    # Quads that were at 4 are now at 5: remove from viol (no longer exactly 4)
                    fixed = q_indices[old == 4]
                    if len(fixed) > 0:
                        red_viol.difference_update(fixed.tolist())
                    # Quads that were at 3 are now at 4: add to viol
                    new_viol = q_indices[old == 3]
                    if len(new_viol) > 0:
                        red_viol.update(new_viol.tolist())

                # Update blue_quint_count for all quints containing tri_idx
                q_indices2 = triple_to_quints[tri_idx]
                if len(q_indices2) > 0:
                    old2 = blue_quint_count[q_indices2].copy()
                    blue_quint_count[q_indices2] -= 1
                    # Quints that were at 10 are now at 9: remove from viol
                    fixed2 = q_indices2[old2 == 10]
                    if len(fixed2) > 0:
                        blue_viol.difference_update(fixed2.tolist())
                    # Quints that were at 9 are now at 8: no change to viol set
                    # Quints that were at 11 are now at 10: add to viol
                    new_viol2 = q_indices2[old2 == 11]
                    if len(new_viol2) > 0:
                        blue_viol.update(new_viol2.tolist())

            # Simple stall detection: if violations not decreasing, count up
            # This is a heuristic; in practice we just keep going
            stall_count += 1
            if stall_count % 10000 == 0 and time_left() > time_limit:
                # Periodic check: if stuck, try a random batch of flips
                if len(red_viol) + len(blue_viol) > 0:
                    # Try 50 random flips
                    for _ in range(50):
                        if not red_viol and not blue_viol:
                            return colors.tolist()
                        if red_viol:
                            quad_idx = random.choice(list(red_viol))
                            ts = quad_tris[quad_idx]
                            tri_idx = int(ts[random.randint(0, 3)])
                            colors[tri_idx] = 1
                            q_indices = triple_to_quads[tri_idx]
                            if len(q_indices) > 0:
                                old = red_quad_count[q_indices].copy()
                                red_quad_count[q_indices] -= 1
                                fixed = q_indices[old == 4]
                                if len(fixed) > 0:
                                    red_viol.difference_update(fixed.tolist())
                                new_viol = q_indices[old == 3]
                                if len(new_viol) > 0:
                                    red_viol.update(new_viol.tolist())
                            q_indices2 = triple_to_quints[tri_idx]
                            if len(q_indices2) > 0:
                                old2 = blue_quint_count[q_indices2].copy()
                                blue_quint_count[q_indices2] += 1
                                fixed2 = q_indices2[old2 == 10]
                                if len(fixed2) > 0:
                                    blue_viol.difference_update(fixed2.tolist())
                                new_viol2 = q_indices2[old2 == 9]
                                if len(new_viol2) > 0:
                                    blue_viol.update(new_viol2.tolist())
                        elif blue_viol:
                            quint_idx = random.choice(list(blue_viol))
                            ts = quint_tris[quint_idx]
                            tri_idx = int(ts[random.randint(0, 9)])
                            colors[tri_idx] = 0
                            q_indices = triple_to_quads[tri_idx]
                            if len(q_indices) > 0:
                                old = red_quad_count[q_indices].copy()
                                red_quad_count[q_indices] += 1
                                fixed = q_indices[old == 4]
                                if len(fixed) > 0:
                                    red_viol.difference_update(fixed.tolist())
                                new_viol = q_indices[old == 3]
                                if len(new_viol) > 0:
                                    red_viol.update(new_viol.tolist())
                            q_indices2 = triple_to_quints[tri_idx]
                            if len(q_indices2) > 0:
                                old2 = blue_quint_count[q_indices2].copy()
                                blue_quint_count[q_indices2] -= 1
                                fixed2 = q_indices2[old2 == 10]
                                if len(fixed2) > 0:
                                    blue_viol.difference_update(fixed2.tolist())
                                new_viol2 = q_indices2[old2 == 11]
                                if len(new_viol2) > 0:
                                    blue_viol.update(new_viol2.tolist())

        return None

    # Fallback coloring for small n
    def fallback_colors(n):
        triples = list(combinations(range(n), 3))
        return [0 if (a+b+c) % 3 == 0 else 1 for a,b,c in triples]

    best_n = 9
    best_colors = fallback_colors(9)

    # Try n from 34 down to 10 with multiple restarts
    targets = [34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10]

    time_budgets = {
        34: 35,
        33: 15,
        32: 8,
        31: 5,
        30: 4,
        29: 3,
        28: 3,
        27: 2,
        26: 2,
        25: 2,
        24: 2,
        23: 2,
        22: 2,
        21: 2,
        20: 2,
        19: 2,
        18: 2,
        17: 2,
        16: 2,
        15: 2,
        14: 2,
        13: 2,
        12: 2,
        11: 2,
        10: 2
    }

    for n in targets:
        if time_left() < 5:
            break

        total_budget = time_budgets.get(n, 2)
        if time_left() < total_budget:
            total_budget = max(1, int(time_left()) - 3)
        if total_budget <= 0:
            break

        # Multiple restarts with different patterns
        num_restarts = max(1, total_budget // 3)
        per_restart = total_budget / num_restarts

        for r in range(num_restarts):
            if time_left() < 3:
                break
            pattern_idx = (r + n) % 7
            remaining = time_left() - 3
            t_limit = min(per_restart, remaining)
            if t_limit < 0.5:
                break

            result = local_search_bitparallel(n, t_limit, pattern_idx)
            if result is not None:
                best_n = n
                best_colors = result
                break

        if best_n == n:
            break

    return {'n': best_n, 'colors': best_colors}

# EVOLVE-BLOCK-END
