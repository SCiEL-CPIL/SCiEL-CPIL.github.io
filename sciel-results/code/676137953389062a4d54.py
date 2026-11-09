"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Leave 5s margin for overhead/return

    def time_left():
        return deadline - time.time()

    def get_structures(n):
        triples = list(combinations(range(n), 3))
        num_triples = len(triples)
        idx_map = {t: i for i, t in enumerate(triples)}

        quad_list = list(combinations(range(n), 4))
        quint_list = list(combinations(range(n), 5))

        # Map each quad/quint to its constituent triple indices
        # We need reverse mapping: triple index -> list of quad indices, list of quint indices
        triple_to_quads = [[] for _ in range(num_triples)]
        triple_to_quints = [[] for _ in range(num_triples)]
        
        quad_tris = []
        for quad in quad_list:
            ts = [idx_map[tri] for tri in combinations(quad, 3)]
            quad_tris.append(ts)
            for ti in ts:
                triple_to_quads[ti].append(len(quad_list) - 1) # Current index

        quint_tris = []
        for quint in quint_list:
            ts = [idx_map[tri] for tri in combinations(quint, 3)]
            quint_tris.append(ts)
            for ti in ts:
                triple_to_quints[ti].append(len(quint_list) - 1) # Current index

        # Convert to numpy arrays for faster access
        triple_to_quads_np = [np.array(lst, dtype=np.int32) for lst in triple_to_quads]
        triple_to_quints_np = [np.array(lst, dtype=np.int32) for lst in triple_to_quints]

        return triples, quad_tris, quint_tris, triple_to_quads_np, triple_to_quints_np

    def solve(n, time_limit):
        if n < 4:
            return None

        triples, quad_tris, quint_tris, triple_to_quads, triple_to_quints = get_structures(n)
        num_triples = len(triples)
        num_quads = len(quad_tris)
        num_quints = len(quint_tris)

        # Initialize random coloring
        # 0: Red, 1: Blue
        colors = np.random.randint(0, 2, size=num_triples, dtype=np.int8)

        # Count red triples in each quad (violation if count == 4)
        # Count blue triples in each quint (violation if count == 10)
        red_quad_count = np.zeros(num_quads, dtype=np.int32)
        blue_quint_count = np.zeros(num_quints, dtype=np.int32)

        # Efficiently initialize counts
        # Instead of looping over quads, we can use the fact that each triple is in multiple quads.
        # But looping over quads is O(num_quads * 4) which is ~ 46k * 4 = 180k ops. Fast enough.
        for qi, ts in enumerate(quad_tris):
            red_quad_count[qi] = np.sum(colors[ts] == 0)

        for qi, ts in enumerate(quint_tris):
            blue_quint_count[qi] = np.sum(colors[ts] == 1)

        # Track violations using sets for O(1) removal
        red_viol_set = set(np.where(red_quad_count == 4)[0])
        blue_viol_set = set(np.where(blue_quint_count == 10)[0])

        # If no violations, we are done
        if not red_viol_set and not blue_viol_set:
            return colors.tolist()

        # Local search loop
        while time_left() > time_limit:
            if not red_viol_set and not blue_viol_set:
                return colors.tolist()

            # Decide which violation to fix
            # Probability proportional to count, but ensure we try both if both exist
            if red_viol_set and (not blue_viol_set or random.random() < len(red_viol_set) / (len(red_viol_set) + len(blue_viol_set))):
                # Fix a red violation (quad with 4 red triples)
                # Pick a random violating quad
                quad_idx = random.choice(list(red_viol_set))
                ts = quad_tris[quad_idx]
                # Pick a random triple in this quad to flip (0 -> 1)
                # All triples in a violating quad are red (0), so flipping any makes it blue (1)
                tri_idx = ts[random.randint(0, len(ts) - 1)]

                # Flip color
                colors[tri_idx] = 1

                # Update counts for affected quads and quints
                # This triple was Red (0), now Blue (1)

                # Update red_quad_count for all quads containing tri_idx
                quad_indices = triple_to_quads[tri_idx]
                if len(quad_indices) > 0:
                    old_cnts = red_quad_count[quad_indices].copy()
                    red_quad_count[quad_indices] -= 1
                    new_cnts = red_quad_count[quad_indices]
                    
                    # Update violation set
                    removed = quad_indices[old_cnts == 4]
                    if len(removed) > 0:
                        red_viol_set.difference_update(removed.tolist())
                    added = quad_indices[old_cnts == 3]
                    if len(added) > 0:
                        red_viol_set.update(added.tolist())

                # Update blue_quint_count for all quints containing tri_idx
                quint_indices = triple_to_quints[tri_idx]
                if len(quint_indices) > 0:
                    old_cnts = blue_quint_count[quint_indices].copy()
                    blue_quint_count[quint_indices] += 1
                    new_cnts = blue_quint_count[quint_indices]
                    
                    # Update violation set
                    removed = quint_indices[old_cnts == 10]
                    if len(removed) > 0:
                        blue_viol_set.difference_update(removed.tolist())
                    added = quint_indices[old_cnts == 9]
                    if len(added) > 0:
                        blue_viol_set.update(added.tolist())

            else:
                # Fix a blue violation (quint with 10 blue triples)
                if not blue_viol_set:
                    continue # Should not happen if red_viol_set is also empty, but safety check
                # Pick a random violating quint
                quint_idx = random.choice(list(blue_viol_set))
                ts = quint_tris[quint_idx]
                # Pick a random triple in this quint to flip (1 -> 0)
                # All triples in a violating quint are blue (1), so flipping any makes it red (0)
                tri_idx = ts[random.randint(0, len(ts) - 1)]

                # Flip color
                colors[tri_idx] = 0

                # Update counts for affected quads and quints
                # This triple was Blue (1), now Red (0)

                # Update red_quad_count for all quads containing tri_idx
                quad_indices = triple_to_quads[tri_idx]
                if len(quad_indices) > 0:
                    old_cnts = red_quad_count[quad_indices].copy()
                    red_quad_count[quad_indices] += 1
                    new_cnts = red_quad_count[quad_indices]
                    
                    # Update violation set
                    removed = quad_indices[old_cnts == 4]
                    if len(removed) > 0:
                        red_viol_set.difference_update(removed.tolist())
                    added = quad_indices[old_cnts == 3]
                    if len(added) > 0:
                        red_viol_set.update(added.tolist())

                # Update blue_quint_count for all quints containing tri_idx
                quint_indices = triple_to_quints[tri_idx]
                if len(quint_indices) > 0:
                    old_cnts = blue_quint_count[quint_indices].copy()
                    blue_quint_count[quint_indices] -= 1
                    new_cnts = blue_quint_count[quint_indices]
                    
                    # Update violation set
                    removed = quint_indices[old_cnts == 10]
                    if len(removed) > 0:
                        blue_viol_set.difference_update(removed.tolist())
                    added = quint_indices[old_cnts == 9]
                    if len(added) > 0:
                        blue_viol_set.update(added.tolist())

        return None

    # Fallback: known valid small coloring
    # n=9 construction from inspiration 1
    best_n = 9
    best_colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(9), 3)]

    # Strategy: Try n from 34 down to 10.
    # If we find a valid coloring for n, return it immediately (since we are descending, this is the max found so far).
    # We must manage time carefully.
    
    targets = [34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10]
    
    # Assign time budgets. We have ~115s total.
    # Give more time to higher n.
    time_budgets = {
        34: 40,
        33: 20,
        32: 10,
        31: 5,
        30: 3,
        29: 2,
        28: 2,
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
        if time_left() < 5: # Safety margin
            break
        
        t_limit = time_budgets.get(n, 2)
        
        # Cap t_limit by remaining time
        if time_left() < t_limit:
            t_limit = time_left() - 2 # Leave 2s margin
            
        if t_limit <= 0:
            break
            
        result = solve(n, t_limit)
        
        if result is not None:
            best_n = n
            best_colors = result
            break

    return {'n': best_n, 'colors': best_colors}

# EVOLVE-BLOCK-END
