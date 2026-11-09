"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
from itertools import combinations
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Leave 5s margin for overhead/return

    def time_left():
        return deadline - time.time()

    def get_structures(n):
        triples = list(combinations(range(n), 3))
        num_triples = len(triples)
        idx_map = {t: i for i, t in enumerate(triples)}

        quad_list = list(combinations(range(n), 4))
        quint_list = list(combinations(range(n), 5))

        triple_to_quads = [[] for _ in range(num_triples)]
        triple_to_quints = [[] for _ in range(num_triples)]
        
        quad_tris = []
        for qi, quad in enumerate(quad_list):
            ts = [idx_map[tri] for tri in combinations(quad, 3)]
            quad_tris.append(ts)
            for ti in ts:
                triple_to_quads[ti].append(qi)

        quint_tris = []
        for qi, quint in enumerate(quint_list):
            ts = [idx_map[tri] for tri in combinations(quint, 3)]
            quint_tris.append(ts)
            for ti in ts:
                triple_to_quints[ti].append(qi)

        return triples, quad_tris, quint_tris, triple_to_quads, triple_to_quints

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

        for qi, ts in enumerate(quad_tris):
            red_quad_count[qi] = np.sum(colors[ts] == 0)

        for qi, ts in enumerate(quint_tris):
            blue_quint_count[qi] = np.sum(colors[ts] == 1)

        # Track violations using sets for O(1) removal
        red_viol_set = set(np.where(red_quad_count == 4)[0].tolist())
        blue_viol_set = set(np.where(blue_quint_count == 10)[0].tolist())

        if not red_viol_set and not blue_viol_set:
            return colors.tolist()

        # Simulated Annealing parameters
        T = 1.0
        cooling = 0.9995
        min_T = 0.01
        best_score = len(red_viol_set) + len(blue_viol_set)
        best_colors = colors.copy()
        
        end_time = time.time() + time_limit
        
        while time.time() < end_time:
            if not red_viol_set and not blue_viol_set:
                return colors.tolist()

            # Decide which violation to fix
            # Bias towards fixing the type with more violations, but ensure exploration
            if red_viol_set and (not blue_viol_set or random.random() < len(red_viol_set) / (len(red_viol_set) + len(blue_viol_set))):
                # Fix a red violation
                if not red_viol_set: continue
                quad_idx = random.choice(list(red_viol_set))
                ts = quad_tris[quad_idx]
                tri_idx = ts[random.randint(0, len(ts) - 1)]
                old_color = colors[tri_idx]
                new_color = 1 - old_color
            else:
                if not blue_viol_set: continue
                quint_idx = random.choice(list(blue_viol_set))
                ts = quint_tris[quint_idx]
                tri_idx = ts[random.randint(0, len(ts) - 1)]
                old_color = colors[tri_idx]
                new_color = 1 - old_color

            # Calculate delta violations
            delta_red = 0
            delta_blue = 0
            
            # Update red_quad_count for all quads containing tri_idx
            quad_indices = triple_to_quads[tri_idx]
            if len(quad_indices) > 0:
                old_cnts = red_quad_count[quad_indices].copy()
                if old_color == 0: # Red -> Blue
                    red_quad_count[quad_indices] -= 1
                    # Removed violation: old==4, new==3
                    delta_red -= np.sum(old_cnts == 4)
                    # Added violation: old==3, new==4 (impossible here since we decreased)
                    # Wait, if old==3, new==2. No new violation.
                    # If old==4, new==3. Removed violation.
                    # If old==3, new==2. No change in violation status (3 is not 4).
                    # Actually, we only care about count==4.
                    # Decreasing count:
                    # 4->3: violation removed (-1)
                    # 3->2: no change
                    # 2->1: no change
                    # 1->0: no change
                    # So delta_red -= count of (old==4)
                else: # Blue -> Red
                    red_quad_count[quad_indices] += 1
                    # Increasing count:
                    # 3->4: violation added (+1)
                    # 4->5: impossible (max 4)
                    # So delta_red += count of (old==3)
                    delta_red += np.sum(old_cnts == 3)
                
            # Update blue_quint_count for all quints containing tri_idx
            quint_indices = triple_to_quints[tri_idx]
            if len(quint_indices) > 0:
                old_cnts = blue_quint_count[quint_indices].copy()
                if old_color == 1: # Blue -> Red
                    blue_quint_count[quint_indices] -= 1
                    # Decreasing count:
                    # 10->9: violation removed (-1)
                    # So delta_blue -= count of (old==10)
                    delta_blue -= np.sum(old_cnts == 10)
                else: # Red -> Blue
                    blue_quint_count[quint_indices] += 1
                    # Increasing count:
                    # 9->10: violation added (+1)
                    # So delta_blue += count of (old==9)
                    delta_blue += np.sum(old_cnts == 9)

            delta_E = delta_red + delta_blue
            
            # Simulated Annealing acceptance
            accept = False
            if delta_E < 0:
                accept = True
            else:
                if T > min_T:
                    prob = math.exp(-delta_E / T)
                    if random.random() < prob:
                        accept = True

            if accept:
                colors[tri_idx] = new_color
                
                # Update violation sets
                # We need to update the sets based on the specific quads/quints affected
                # This is tricky because we only have the delta, not which specific ones changed.
                # For efficiency and correctness, let's just recalculate the sets occasionally
                # or update them precisely.
                
                # Precise update is complex. Let's use a hybrid:
                # If we accept, we update the counts. 
                # The sets `red_viol_set` and `blue_viol_set` might be stale.
                # Instead of maintaining sets perfectly, let's just check validity periodically 
                # or use the counts directly to pick violations.
                
                # A simpler approach for SA: 
                # Don't maintain sets for picking. Just pick a random triple involved in ANY violation?
                # No, we picked from a violation at the start of the loop.
                # So the sets used for picking MUST be accurate.
                
                # Let's update the sets accurately.
                # For red_quad_count:
                if len(quad_indices) > 0:
                    new_cnts = red_quad_count[quad_indices]
                    # Removed: old==4
                    removed = quad_indices[old_cnts == 4]
                    if len(removed) > 0:
                        red_viol_set.difference_update(removed.tolist())
                    # Added: old==3 (for Blue->Red)
                    if old_color == 1: # Blue -> Red
                        added = quad_indices[old_cnts == 3]
                        if len(added) > 0:
                            red_viol_set.update(added.tolist())
                    # Note: For Red->Blue, no new violations are added (count decreases).
                    
                # For blue_quint_count:
                if len(quint_indices) > 0:
                    new_cnts = blue_quint_count[quint_indices]
                    # Removed: old==10
                    removed = quint_indices[old_cnts == 10]
                    if len(removed) > 0:
                        blue_viol_set.difference_update(removed.tolist())
                    # Added: old==9 (for Red->Blue)
                    if old_color == 0: # Red -> Blue
                        added = quint_indices[old_cnts == 9]
                        if len(added) > 0:
                            blue_viol_set.update(added.tolist())

                # Update best
                current_score = len(red_viol_set) + len(blue_viol_set)
                if current_score < best_score:
                    best_score = current_score
                    best_colors = colors.copy()
                    if best_score == 0:
                        return colors.tolist()
            else:
                # Rollback counts
                if len(quad_indices) > 0:
                    if old_color == 0:
                        red_quad_count[quad_indices] += 1
                    else:
                        red_quad_count[quad_indices] -= 1
                if len(quint_indices) > 0:
                    if old_color == 1:
                        blue_quint_count[quint_indices] += 1
                    else:
                        blue_quint_count[quint_indices] -= 1

            T *= cooling

        # Return best found if not solved
        if best_score == 0:
            return best_colors.tolist()
        return None

    # Fallback: known valid small coloring
    # n=9 construction
    best_n = 9
    best_colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(9), 3)]

    # Strategy: Try n from 34 down to 10.
    targets = [34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10]
    
    # Assign time budgets. We have ~115s total.
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
