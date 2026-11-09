"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin for overhead/return

    def time_left():
        return deadline - time.time()

    def get_structures(n):
        triples = list(combinations(range(n), 3))
        num_triples = len(triples)
        idx_map = {t: i for i, t in enumerate(triples)}

        quad_list = list(combinations(range(n), 4))
        quint_list = list(combinations(range(n), 5))

        # Map each quad/quint to its constituent triple indices
        quad_tris = []
        for quad in quad_list:
            ts = [idx_map[tri] for tri in combinations(quad, 3)]
            quad_tris.append(ts)

        quint_tris = []
        for quint in quint_list:
            ts = [idx_map[tri] for tri in combinations(quint, 3)]
            quint_tris.append(ts)

        # Reverse mapping: triple index -> list of quad indices, list of quint indices
        triple_to_quads = [[] for _ in range(num_triples)]
        for qi, ts in enumerate(quad_tris):
            for ti in ts:
                triple_to_quads[ti].append(qi)

        triple_to_quints = [[] for _ in range(num_triples)]
        for qi, ts in enumerate(quint_tris):
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
                for q_i in triple_to_quads[tri_idx]:
                    old_cnt = red_quad_count[q_i]
                    red_quad_count[q_i] = old_cnt - 1
                    if old_cnt == 4 and red_quad_count[q_i] != 4:
                        red_viol_set.discard(q_i)
                    elif old_cnt != 4 and red_quad_count[q_i] == 4:
                        red_viol_set.add(q_i)

                # Update blue_quint_count for all quints containing tri_idx
                for q_i in triple_to_quints[tri_idx]:
                    old_cnt = blue_quint_count[q_i]
                    blue_quint_count[q_i] = old_cnt + 1
                    if old_cnt == 10 and blue_quint_count[q_i] != 10:
                        blue_viol_set.discard(q_i)
                    elif old_cnt != 10 and blue_quint_count[q_i] == 10:
                        blue_viol_set.add(q_i)

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
                for q_i in triple_to_quads[tri_idx]:
                    old_cnt = red_quad_count[q_i]
                    red_quad_count[q_i] = old_cnt + 1
                    if old_cnt == 4 and red_quad_count[q_i] != 4:
                        red_viol_set.discard(q_i)
                    elif old_cnt != 4 and red_quad_count[q_i] == 4:
                        red_viol_set.add(q_i)

                # Update blue_quint_count for all quints containing tri_idx
                for q_i in triple_to_quints[tri_idx]:
                    old_cnt = blue_quint_count[q_i]
                    blue_quint_count[q_i] = old_cnt - 1
                    if old_cnt == 10 and blue_quint_count[q_i] != 10:
                        blue_viol_set.discard(q_i)
                    elif old_cnt != 10 and blue_quint_count[q_i] == 10:
                        blue_viol_set.add(q_i)

        return None

    # Fallback: known valid small coloring
    # n=9 construction from inspiration 1
    best_n = 9
    best_colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(9), 3)]

    # Try to find larger valid colorings
    # We start from a reasonable n and go up.
    # The problem states max n is 34.
    # We allocate time to try higher n first if possible, or start from a moderate n and increase.
    # Given the difficulty, starting from n=20 or so might be safer to ensure we get *some* improvement over 9.
    # However, the current program starts from 10 and goes up.
    # Let's try a strategy: try n=34, if fails quickly, try n=30, etc.
    # Or just iterate 10..34 with time budget.
    # Since local search is fast, we can try many n values.

    # Let's try to solve for n=34 first with a generous time slice.
    # If that fails, try 33, 32, etc.
    # But we must ensure we return *something* valid.
    # The loop below will update best_n and best_colors if a solution is found.

    # Strategy: Try n from 34 down to 10? Or 10 up to 34?
    # Going up is safer to guarantee monotonic improvement if we find 10, then 11, etc.
    # But finding 34 is much harder.
    # Let's try a few high values first with a time limit, then fall back to lower ones if time permits.
    
    # Actually, the current program (score 1.11) iterates 10 to 35.
    # The inspiration 2 program tries 34 first.
    # Let's combine: Try a few high targets (34, 33, 32) with limited time, then if not solved,
    # iterate from a lower bound upwards to ensure we get the best possible n within remaining time.
    
    # Time allocation:
    # 1. Try n=34 for 30s.
    # 2. Try n=33 for 20s.
    # 3. Try n=32 for 10s.
    # 4. If any succeeded, we have a high n. If not, we have n=9.
    # 5. Then, if time remains, try to improve from the current best_n+1 upwards.
    
    # This is complex. Let's stick to a simpler robust approach:
    # Iterate n from 10 to 34. For each n, give it a small time slice.
    # If we find a solution for n, update best. Continue to n+1.
    # This guarantees we find the smallest n that works, and keeps trying for larger ones.
    # To make it more effective for high n, we can give more time to higher n if we have time to spare?
    # No, better to give a fixed small budget per n to ensure we scan the range.
    # If we find a solution for n=20, we keep it. Then try n=21.
    
    # Let's refine the time budget per n.
    # Total time ~ 140s.
    # Range 10..34 is 25 values.
    # 140 / 25 = 5.6s per n.
    # This might be too short for high n.
    
    # Alternative:
    # 1. Try n=34 with 60s.
    # 2. If fail, try n=33 with 30s.
    # 3. If fail, try n=32 with 15s.
    # 4. If fail, try n=31 with 10s.
    # 5. If fail, try n=30 with 5s.
    # 6. If still no improvement over 9, fall back to scanning 10..29 with remaining time.
    
    # This prioritizes the "prize" of high n.
    
    targets = [34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10]
    
    # Assign time budgets
    time_budgets = {
        34: 60,
        33: 30,
        32: 15,
        31: 10,
        30: 5,
    }
    # Default budget for others
    default_budget = 3
    
    for n in targets:
        if time_left() < 5: # Safety margin
            break
        
        if n in time_budgets:
            t_limit = time_budgets[n]
        else:
            t_limit = default_budget
            
        # Cap t_limit by remaining time
        if time_left() < t_limit:
            t_limit = time_left() - 2 # Leave 2s margin
            
        if t_limit <= 0:
            break
            
        result = solve(n, t_limit)
        
        if result is not None:
            best_n = n
            best_colors = result
            # If we found a solution, we can break if we found the max possible?
            # No, we want to maximize n. So we continue to try larger n?
            # Wait, my targets list is descending.
            # If I find 34, I should stop? No, 34 is the max known.
            # If I find 33, I should try 34? No, I already tried 34 first.
            # So if I iterate descending, and I find a solution for n, I should stop?
            # No, because maybe 34 failed but 33 worked. Is there a 34 solution?
            # The solver is stochastic. Maybe it failed to find 34 but a 34 solution exists.
            # However, re-running 34 is expensive.
            # Given the time constraints, if we find a high n, it's a good bet.
            # But the task says "Maximize n".
            # If I find n=33, and I have time, should I try 34 again?
            # Probably not worth it if 34 already failed for 60s.
            # So, if I find a solution for n, I can break?
            # Only if I am confident no higher n exists or is findable.
            # Since I try in descending order, if I find 33, I have already tried 34.
            # So I can break?
            # Yes, because 34 was tried and failed (or timed out).
            # So breaking is logical.
            
            break

    return {'n': best_n, 'colors': best_colors}

# EVOLVE-BLOCK-END
