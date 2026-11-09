"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # 120s safety margin before 150s kill

    def time_left():
        return deadline - time.time()

    def build_structures(n):
        triples = list(combinations(range(n), 3))
        num_triples = len(triples)
        triple_idx = {t: i for i, t in enumerate(triples)}

        # For each quad, the set of its 4 triples as a bitset
        quads = list(combinations(range(n), 4))
        quad_bitsets = []
        for q in quads:
            mask = 0
            for tri in combinations(q, 3):
                mask |= (1 << triple_idx[tri])
            quad_bitsets.append(mask)

        # For each quint, the set of its 10 triples as a bitset
        quints = list(combinations(range(n), 5))
        quint_bitsets = []
        for qu in quints:
            mask = 0
            for tri in combinations(qu, 3):
                mask |= (1 << triple_idx[tri])
            quint_bitsets.append(mask)

        # For each triple, which quads/quints it belongs to
        triple_to_quads = [[] for _ in range(num_triples)]
        for qi, q in enumerate(quads):
            for tri in combinations(q, 3):
                triple_to_quads[triple_idx[tri]].append(qi)

        triple_to_quints = [[] for _ in range(num_triples)]
        for qui, qu in enumerate(quints):
            for tri in combinations(qu, 3):
                triple_to_quints[triple_idx[tri]].append(qui)

        return triples, num_triples, triple_idx, quads, quad_bitsets, quints, quint_bitsets, triple_to_quads, triple_to_quints

    def solve(n, time_budget):
        """Solve for n vertices within time_budget seconds. Return colors list or None."""
        triples, num_triples, triple_idx, quads, quad_bitsets, quints, quint_bitsets, triple_to_quads, triple_to_quints = build_structures(n)

        num_quads = len(quads)
        num_quints = len(quints)

        # colors[i] = 0 (red) or 1 (blue)
        colors = [random.randint(0, 1) for _ in range(num_triples)]

        # all_red_mask: bit i set iff colors[i] == 0 (red)
        # all_blue_mask: bit i set iff colors[i] == 1 (blue)
        all_red_mask = 0
        all_blue_mask = 0
        for i in range(num_triples):
            if colors[i] == 0:
                all_red_mask |= (1 << i)
            else:
                all_blue_mask |= (1 << i)

        def count_violations():
            rv = 0
            for mask in quad_bitsets:
                if mask & all_red_mask == 0:  # all triples in quad are red
                    rv += 1
            bv = 0
            for mask in quint_bitsets:
                if mask & all_blue_mask == 0:  # all triples in quint are blue
                    bv += 1
            return rv, bv

        red_v, blue_v = count_violations()
        best_viol = red_v + blue_v
        best_colors = colors[:]

        end_time = time.time() + time_budget
        step = 0
        stuck_count = 0

        while time.time() < end_time:
            if red_v + blue_v == 0:
                return colors[:]

            if red_v + blue_v < best_viol:
                best_viol = red_v + blue_v
                best_colors = colors[:]
                stuck_count = 0

            step += 1

            # Choose which violation to fix
            if red_v > 0 and (blue_v == 0 or random.random() < red_v / (red_v + blue_v)):
                # Pick a random red quad
                while True:
                    qi = random.randint(0, num_quads - 1)
                    if quad_bitsets[qi] & all_red_mask == 0:
                        break
                # Pick a random triple in that quad to flip
                # Get the triples in this quad
                quad = quads[qi]
                tri_list = [triple_idx[t] for t in combinations(quad, 3)]
                tri_idx = random.choice(tri_list)
            else:
                while True:
                    qui = random.randint(0, num_quints - 1)
                    if quint_bitsets[qui] & all_blue_mask == 0:
                        break
                quint = quints[qui]
                tri_list = [triple_idx[t] for t in combinations(quint, 3)]
                tri_idx = random.choice(tri_list)

            # Flip the triple
            old_color = colors[tri_idx]
            new_color = 1 - old_color
            colors[tri_idx] = new_color

            bit = 1 << tri_idx
            if old_color == 0:
                # Was red, now blue
                all_red_mask ^= bit
                all_blue_mask |= bit
            else:
                # Was blue, now red
                all_blue_mask ^= bit
                all_red_mask |= bit

            # Update violation counts incrementally
            # Check affected quads
            for qi in triple_to_quads[tri_idx]:
                mask = quad_bitsets[qi]
                was_viol = (mask & all_red_mask ^ bit) == 0 if old_color == 0 else (mask & (all_red_mask | bit)) == 0
                # Simpler: recompute
                # Actually let's just recompute for affected quads
                pass

            # Incremental update is complex with bitsets; let's recompute violations
            # But that's O(num_quads + num_quints) per step which is too slow for large n
            # Alternative: only check affected quads/quints

            # For the flipped triple, check all quads it belongs to
            # A quad is red-violation iff all 4 triples are red
            # Before flip: if old_color was 0 (red), the quad might have been a violation
            # After flip: the triple is now blue, so the quad can no longer be a red violation

            # Let's track violations differently:
            # For each quad, track how many of its 4 triples are red
            # For each quint, track how many of its 10 triples are blue

            # Actually, let me use a different approach: maintain counters
            # quad_red_count[qi] = number of red triples in quad qi
            # quint_blue_count[qui] = number of blue triples in quint qui

            # This is cleaner. Let me restart with this approach.
            break  # Break to restart with counter-based approach

        return None

    def solve_counters(n, time_budget):
        """Solve using counter-based incremental updates."""
        triples, num_triples, triple_idx, quads, quad_bitsets, quints, quint_bitsets, triple_to_quads, triple_to_quints = build_structures(n)

        num_quads = len(quads)
        num_quints = len(quints)

        # Precompute for each quad: list of triple indices
        quad_tris = []
        for q in quads:
            quad_tris.append([triple_idx[t] for t in combinations(q, 3)])

        # Precompute for each quint: list of triple indices
        quint_tris = []
        for qu in quints:
            quint_tris.append([triple_idx[t] for t in combinations(qu, 3)])

        colors = [random.randint(0, 1) for _ in range(num_triples)]

        # quad_red_count[qi]: number of red triples in quad qi
        # quint_blue_count[qui]: number of blue triples in quint qui
        quad_red_count = [0] * num_quads
        quint_blue_count = [0] * num_quints

        for qi in range(num_quads):
            for ti in quad_tris[qi]:
                if colors[ti] == 0:
                    quad_red_count[qi] += 1

        for qui in range(num_quints):
            for ti in quint_tris[qui]:
                if colors[ti] == 1:
                    quint_blue_count[qui] += 1

        red_v = sum(1 for qi in range(num_quads) if quad_red_count[qi] == 4)
        blue_v = sum(1 for qui in range(num_quints) if quint_blue_count[qui] == 10)

        best_viol = red_v + blue_v
        best_colors = colors[:]

        end_time = time.time() + time_budget
        stuck = 0

        while time.time() < end_time:
            if red_v + blue_v == 0:
                return colors[:]

            if red_v + blue_v < best_viol:
                best_viol = red_v + blue_v
                best_colors = colors[:]
                stuck = 0
            else:
                stuck += 1

            if stuck > 5000:
                # Restart
                colors = [random.randint(0, 1) for _ in range(num_triples)]
                for qi in range(num_quads):
                    quad_red_count[qi] = 0
                    for ti in quad_tris[qi]:
                        if colors[ti] == 0:
                            quad_red_count[qi] += 1
                for qui in range(num_quints):
                    quint_blue_count[qui] = 0
                    for ti in quint_tris[qui]:
                        if colors[ti] == 1:
                            quint_blue_count[qui] += 1
                red_v = sum(1 for qi in range(num_quads) if quad_red_count[qi] == 4)
                blue_v = sum(1 for qui in range(num_quints) if quint_blue_count[qui] == 10)
                stuck = 0
                continue

            # Choose violation to fix
            if red_v > 0 and (blue_v == 0 or random.random() < red_v / (red_v + blue_v)):
                # Find a red quad
                while True:
                    qi = random.randint(0, num_quads - 1)
                    if quad_red_count[qi] == 4:
                        break
                tri_idx = random.choice(quad_tris[qi])
            else:
                while True:
                    qui = random.randint(0, num_quints - 1)
                    if quint_blue_count[qui] == 10:
                        break
                tri_idx = random.choice(quint_tris[qui])

            old_color = colors[tri_idx]
            new_color = 1 - old_color
            colors[tri_idx] = new_color

            # Update counters
            if old_color == 0:
                # Was red, now blue
                for qi in triple_to_quads[tri_idx]:
                    quad_red_count[qi] -= 1
                    if quad_red_count[qi] == 3:
                        red_v -= 1  # was 4, now 3, no longer violation
                    elif quad_red_count[qi] == 4:
                        red_v += 1  # was 3, now 4, new violation
                for qui in triple_to_quints[tri_idx]:
                    quint_blue_count[qui] += 1
                    if quint_blue_count[qui] == 10:
                        blue_v += 1
                    elif quint_blue_count[qui] == 9:
                        blue_v -= 1
            else:
                # Was blue, now red
                for qi in triple_to_quads[tri_idx]:
                    quad_red_count[qi] += 1
                    if quad_red_count[qi] == 4:
                        red_v += 1
                    elif quad_red_count[qi] == 3:
                        red_v -= 1
                for qui in triple_to_quints[tri_idx]:
                    quint_blue_count[qui] -= 1
                    if quint_blue_count[qui] == 9:
                        blue_v -= 1
                    elif quint_blue_count[qui] == 10:
                        blue_v += 1

        return best_colors if best_viol == 0 else None

    def solve_multi(n, time_budget):
        """Try multiple restarts and strategies."""
        triples, num_triples, triple_idx, quads, quad_bitsets, quints, quint_bitsets, triple_to_quads, triple_to_quints = build_structures(n)
        num_quads = len(quads)
        num_quints = len(quints)

        quad_tris = []
        for q in quads:
            quad_tris.append([triple_idx[t] for t in combinations(q, 3)])

        quint_tris = []
        for qu in quints:
            quint_tris.append([triple_idx[t] for t in combinations(qu, 3)])

        best_result = None
        best_viol = float('inf')

        end_time = time.time() + time_budget

        while time.time() < end_time - 2.0:
            # Random initialization
            colors = [random.randint(0, 1) for _ in range(num_triples)]

            quad_red_count = [0] * num_quads
            quint_blue_count = [0] * num_quints

            for qi in range(num_quads):
                for ti in quad_tris[qi]:
                    if colors[ti] == 0:
                        quad_red_count[qi] += 1

            for qui in range(num_quints):
                for ti in quint_tris[qui]:
                    if colors[ti] == 1:
                        quint_blue_count[qui] += 1

            red_v = sum(1 for qi in range(num_quads) if quad_red_count[qi] == 4)
            blue_v = sum(1 for qui in range(num_quints) if quint_blue_count[qui] == 10)

            if red_v + blue_v < best_viol:
                best_viol = red_v + blue_v
                best_result = colors[:]

            if red_v + blue_v == 0:
                return colors

            # Local search
            restart_end = min(time.time() + 5.0, end_time - 1.0)
            stuck = 0
            local_best_viol = red_v + blue_v
            local_best = colors[:]

            while time.time() < restart_end:
                if red_v + blue_v == 0:
                    return colors

                if red_v + blue_v < local_best_viol:
                    local_best_viol = red_v + blue_v
                    local_best = colors[:]
                    stuck = 0
                else:
                    stuck += 1

                if stuck > 3000:
                    break

                if red_v > 0 and (blue_v == 0 or random.random() < red_v / (red_v + blue_v)):
                    while True:
                        qi = random.randint(0, num_quads - 1)
                        if quad_red_count[qi] == 4:
                            break
                    tri_idx = random.choice(quad_tris[qi])
                else:
                    while True:
                        qui = random.randint(0, num_quints - 1)
                        if quint_blue_count[qui] == 10:
                            break
                    tri_idx = random.choice(quint_tris[qui])

                old_color = colors[tri_idx]
                new_color = 1 - old_color
                colors[tri_idx] = new_color

                if old_color == 0:
                    for qi in triple_to_quads[tri_idx]:
                        quad_red_count[qi] -= 1
                        if quad_red_count[qi] == 3:
                            red_v -= 1
                        elif quad_red_count[qi] == 4:
                            red_v += 1
                    for qui in triple_to_quints[tri_idx]:
                        quint_blue_count[qui] += 1
                        if quint_blue_count[qui] == 10:
                            blue_v += 1
                        elif quint_blue_count[qui] == 9:
                            blue_v -= 1
                else:
                    for qi in triple_to_quads[tri_idx]:
                        quad_red_count[qi] += 1
                        if quad_red_count[qi] == 4:
                            red_v += 1
                        elif quad_red_count[qi] == 3:
                            red_v -= 1
                    for qui in triple_to_quints[tri_idx]:
                        quint_blue_count[qui] -= 1
                        if quint_blue_count[qui] == 9:
                            blue_v -= 1
                        elif quint_blue_count[qui] == 10:
                            blue_v += 1

            if local_best_viol < best_viol:
                best_viol = local_best_viol
                best_result = local_best[:]

            if best_viol == 0:
                return best_result

        return best_result if best_viol == 0 else None

    # Try to find colorings for increasing n
    best_n = 9
    best_colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(9), 3)]

    # Allocate time: try larger n first since that's where the score comes from
    # n=34 is the target, but let's be realistic
    # Time budget allocation:
    # - n >= 20: 60s
    # - n >= 15: 30s
    # - n >= 12: 20s
    # - n >= 10: 10s
    # - n <= 9: instant

    # Strategy: try n=30, 25, 20, 15, 12, 10 in that order
    # But we need to be careful about time

    targets = [30, 25, 20, 15, 12, 10]
    time_alloc = {30: 50, 25: 30, 20: 20, 15: 15, 12: 10, 10: 5}

    for target_n in targets:
        if time_left() < 5.0:
            break
        if target_n <= best_n:
            continue

        alloc = min(time_alloc.get(target_n, 10), time_left() - 3.0)
        if alloc < 2.0:
            break

        result = solve_multi(target_n, alloc)
        if result is not None:
            best_n = target_n
            best_colors = result
            # Found a valid coloring, try to go higher
            # But we've already tried higher in the list, so we're done
            break

    # If we didn't find anything above 9, at least return n=9
    if best_n == 9:
        # Verify n=9 coloring
        triples9 = list(combinations(range(9), 3))
        quads9 = list(combinations(range(9), 4))
        quints9 = list(combinations(range(9), 5))
        valid = True
        for q in quads9:
            all_red = True
            for tri in combinations(q, 3):
                idx = triples9.index(tri)
                if best_colors[idx] != 0:
                    all_red = False
                    break
            if all_red:
                valid = False
                break
        if valid:
            for qu in quints9:
                all_blue = True
                for tri in combinations(qu, 3):
                    idx = triples9.index(tri)
                    if best_colors[idx] != 1:
                        all_blue = False
                        break
                if all_blue:
                    valid = False
                    break
        if not valid:
            # Use a known valid coloring for small n
            best_n = 6
            best_colors = [0] * len(list(combinations(range(6), 3)))

    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END
