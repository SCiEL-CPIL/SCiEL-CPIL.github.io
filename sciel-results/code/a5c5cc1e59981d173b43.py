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

        # For each triple, list of quad indices containing it
        triple_in_quads = [[] for _ in range(num_triples)]
        for qi, qt in enumerate(quad_triples_idx):
            for tidx in qt:
                triple_in_quads[tidx].append(qi)

        # For each triple, list of quint indices containing it
        triple_in_quints = [[] for _ in range(num_triples)]
        for qi, qt in enumerate(quint_triples_idx):
            for tidx in qt:
                triple_in_quints[tidx].append(qi)

        return (triple_list, num_triples, triple_to_idx, quad_triples_idx, num_quads,
                quint_triples_idx, num_quints, triple_in_quads, triple_in_quints)

    def simulated_annealing(target_n, init_colors, deadline):
        """Run Simulated Annealing to eliminate violations. Returns (colors, success)."""
        (triple_list, num_triples, triple_to_idx, quad_triples_idx, num_quads,
         quint_triples_idx, num_quints, triple_in_quads, triple_in_quints) = build_structures(target_n)

        color_arr = np.array(init_colors, dtype=np.int8)
        
        # Initialize counts
        red_count_in_quad = np.zeros(num_quads, dtype=np.int32)
        for qi, qt in enumerate(quad_triples_idx):
            for tidx in qt:
                if color_arr[tidx] == 0:
                    red_count_in_quad[qi] += 1
        
        blue_count_in_quint = np.zeros(num_quints, dtype=np.int32)
        for qi, qt in enumerate(quint_triples_idx):
            for tidx in qt:
                if color_arr[tidx] == 1:
                    blue_count_in_quint[qi] += 1

        # Initial violation counts
        rk4 = np.sum(red_count_in_quad == 4)
        bk5 = np.sum(blue_count_in_quint == 10)
        total_violations = rk4 + bk5

        if total_violations == 0:
            return color_arr.tolist(), True

        # Simulated Annealing parameters
        # We want to minimize total_violations
        start_temp = 1.0
        end_temp = 0.001
        max_iters = 200000  # High iteration count, but we break early if valid
        
        # Precompute lists of triples for faster random access? No, random index is fine.
        
        for it in range(max_iters):
            if time.time() > deadline - 1:
                break
            if total_violations == 0:
                return color_arr.tolist(), True
            
            # Cooling schedule
            if it < max_iters // 2:
                temp = start_temp
            else:
                progress = (it - max_iters // 2) / (max_iters // 2)
                temp = start_temp * (end_temp / start_temp) ** progress

            # Choose a triple to flip
            # Strategy: 50% random, 50% from a random violation
            r = random.random()
            tidx = -1
            
            if r < 0.5 and rk4 > 0:
                # Pick a random violating quad and flip one of its triples
                # To avoid O(N) scan for violating quads every time, we can maintain a list?
                # For now, random sampling of quads is okay if num_quads isn't huge.
                # n=34 -> num_quads = C(34,4) = 46376. Sampling 10 times is fast.
                found = False
                for _ in range(20):
                    qi = random.randint(0, num_quads - 1)
                    if red_count_in_quad[qi] == 4:
                        qt = quad_triples_idx[qi]
                        tidx = qt[random.randint(0, 3)]
                        found = True
                        break
                if not found:
                    tidx = random.randint(0, num_triples - 1)
            elif r < 0.8 and bk5 > 0:
                # Pick a random violating quint and flip one of its triples
                found = False
                for _ in range(20):
                    qi = random.randint(0, num_quints - 1)
                    if blue_count_in_quint[qi] == 10:
                        qt = quint_triples_idx[qi]
                        tidx = qt[random.randint(0, 9)]
                        found = True
                        break
                if not found:
                    tidx = random.randint(0, num_triples - 1)
            else:
                tidx = random.randint(0, num_triples - 1)

            old_color = color_arr[tidx]
            new_color = 1 - old_color

            # Compute delta in violations
            delta_rk4 = 0
            delta_bk5 = 0

            # Update red counts for quads containing tidx
            for qi in triple_in_quads[tidx]:
                if old_color == 0:
                    # Red -> Blue
                    old_rc = red_count_in_quad[qi]
                    new_rc = old_rc - 1
                    if old_rc == 4:
                        delta_rk4 -= 1
                    if new_rc == 4:
                        delta_rk4 += 1
                    red_count_in_quad[qi] = new_rc
                else:
                    # Blue -> Red
                    old_rc = red_count_in_quad[qi]
                    new_rc = old_rc + 1
                    if old_rc == 4:
                        delta_rk4 -= 1
                    if new_rc == 4:
                        delta_rk4 += 1
                    red_count_in_quad[qi] = new_rc

            # Update blue counts for quints containing tidx
            for qi in triple_in_quints[tidx]:
                if old_color == 1:
                    # Blue -> Red
                    old_bc = blue_count_in_quint[qi]
                    new_bc = old_bc - 1
                    if old_bc == 10:
                        delta_bk5 -= 1
                    if new_bc == 10:
                        delta_bk5 += 1
                    blue_count_in_quint[qi] = new_bc
                else:
                    # Red -> Blue
                    old_bc = blue_count_in_quint[qi]
                    new_bc = old_bc + 1
                    if old_bc == 10:
                        delta_bk5 -= 1
                    if new_bc == 10:
                        delta_bk5 += 1
                    blue_count_in_quint[qi] = new_bc

            delta = delta_rk4 + delta_bk5

            # Acceptance probability
            if delta <= 0:
                accept = True
            else:
                # Metropolis criterion
                # We want to minimize violations, so "energy" is +delta
                # P = exp(-delta / T)
                if temp > 1e-9:
                    accept = random.random() < np.exp(-delta / temp)
                else:
                    accept = False

            if accept:
                color_arr[tidx] = new_color
                rk4 += delta_rk4
                bk5 += delta_bk5
                total_violations += delta
            else:
                # Revert counts
                for qi in triple_in_quads[tidx]:
                    if old_color == 0:
                        red_count_in_quad[qi] += 1
                    else:
                        red_count_in_quad[qi] -= 1
                for qi in triple_in_quints[tidx]:
                    if old_color == 1:
                        blue_count_in_quint[qi] += 1
                    else:
                        blue_count_in_quint[qi] -= 1

        if total_violations == 0:
            return color_arr.tolist(), True
        return color_arr.tolist(), False

    # Try decreasing n from 34 down to 10
    # This prioritizes finding the largest n possible
    for target_n in range(34, 9, -1):
        if time.time() > deadline - 5:
            break

        (triple_list, num_triples, triple_to_idx, quad_triples_idx, num_quads,
         quint_triples_idx, num_quints, triple_in_quads, triple_in_quints) = build_structures(target_n)

        found = False
        # Allocate more restarts for larger n if time permits, but keep it balanced
        # n=34 is hard, so give it a good chance
        max_restarts = 50 if target_n >= 30 else 100

        for restart in range(max_restarts):
            if time.time() > deadline - 2:
                break

            # Initialization strategies
            # 1. Random
            # 2. Modular
            # 3. Mixed
            if restart % 3 == 0:
                init = list(np.random.randint(0, 2, size=num_triples))
            elif restart % 3 == 1:
                init = [0 if (a+b+c) % 3 == 0 else 1 for (a,b,c) in triple_list]
            else:
                # Start with modular, then random flip 10%
                init = [0 if (a+b+c) % 3 == 0 else 1 for (a,b,c) in triple_list]
                for i in range(num_triples):
                    if random.random() < 0.1:
                        init[i] = 1 - init[i]

            colors, success = simulated_annealing(target_n, init, deadline)
            if success:
                if is_valid(target_n, colors):
                    best_n = target_n
                    best_colors = colors
                    found = True
                    break

        if found:
            break # Found valid coloring for this n, which is the max we tried. Since we go down, this is the best.

    # Final safety verification
    if best_n > 9 and not is_valid(best_n, best_colors):
        best_n = 9
        best_colors = mod_color(9, 3)

    return {'n': best_n, 'colors': best_colors}
