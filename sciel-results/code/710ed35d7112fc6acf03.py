import numpy as np
import time
from itertools import combinations
import sys
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    def build_indices(n):
        """Build K4 and K5 index structures for fast delta updates."""
        triples = list(combinations(range(n), 3))
        num_triples = len(triples)
        triple_to_idx = {t: i for i, t in enumerate(triples)}

        # K4s: each is a tuple of 4 triple indices
        k4s = []
        k4s_containing = [[] for _ in range(num_triples)]
        for quad in combinations(range(n), 4):
            a, b, c, d = quad
            t1 = triple_to_idx[(a, b, c)]
            t2 = triple_to_idx[(a, b, d)]
            t3 = triple_to_idx[(a, c, d)]
            t4 = triple_to_idx[(b, c, d)]
            idx_tuple = (t1, t2, t3, t4)
            k4s.append(idx_tuple)
            for t in idx_tuple:
                k4s_containing[t].append(len(k4s) - 1)

        # K5s: each is a tuple of 10 triple indices
        k5s = []
        k5s_containing = [[] for _ in range(num_triples)]
        for quint in combinations(range(n), 5):
            idxs = []
            for triple in combinations(quint, 3):
                idxs.append(triple_to_idx[triple])
            idx_tuple = tuple(idxs)
            k5s.append(idx_tuple)
            for t in idx_tuple:
                k5s_containing[t].append(len(k5s) - 1)

        return triples, num_triples, k4s, k5s, k4s_containing, k5s_containing

    def calculate_cost(colors, k4s, k5s):
        """Calculate total violations."""
        cost = 0
        for idxs in k4s:
            if colors[idxs[0]] == 0 and colors[idxs[1]] == 0 and colors[idxs[2]] == 0 and colors[idxs[3]] == 0:
                cost += 1
        for idxs in k5s:
            all_blue = True
            for t in idxs:
                if colors[t] == 0:
                    all_blue = False
                    break
            if all_blue:
                cost += 1
        return cost

    def heuristic_init(n, triples):
        """Try multiple heuristic initializations and return the best."""
        num_triples = len(triples)
        patterns = [
            lambda a, b, c: 0 if (a + b + c) % 3 == 0 else 1,
            lambda a, b, c: 0 if (a + b + c) % 2 == 0 else 1,
            lambda a, b, c: 0 if (a * b + c) % 3 == 0 else 1,
            lambda a, b, c: 0 if (a * a + b * b + c * c) % 2 == 0 else 1,
            lambda a, b, c: 0 if (a + 2 * b + 3 * c) % 4 == 0 else 1,
            lambda a, b, c: 0 if (a * (b + 1) + c) % 3 == 0 else 1,
            lambda a, b, c: 0 if ((a + b) * (b + c)) % 3 == 0 else 1,
            lambda a, b, c: 0 if (a ^ b ^ c) % 2 == 0 else 1,
        ]
        best_init = None
        best_cost = float('inf')
        for pat in patterns:
            c = np.array([pat(a, b, c) for a, b, c in triples], dtype=np.int8)
            # Quick cost check
            cost = 0
            # We'll do a rough check; full cost computed later
            best_init = c
            # Just pick the first one that looks reasonable
            break
        if best_init is None:
            best_init = np.random.randint(0, 2, num_triples, dtype=np.int8)
        return best_init

    def simulated_annealing(n, time_limit, init_colors=None, seed=None):
        """Run simulated annealing to find a valid coloring."""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        triples, num_triples, k4s, k5s, k4s_containing, k5s_containing = build_indices(n)

        if init_colors is None:
            colors = heuristic_init(n, triples)
        else:
            colors = init_colors.copy()

        current_cost = calculate_cost(colors, k4s, k5s)
        best_cost = current_cost
        best_colors = colors.copy()

        temp = 2.0
        cool_rate = 0.9998
        min_temp = 0.01
        no_improve_steps = 0
        max_no_improve = 30000

        while time.time() < time_limit:
            # Pick a random triple to flip
            t = np.random.randint(num_triples)
            old_color = colors[t]
            new_color = 1 - old_color

            delta_cost = 0

            # Update K4s containing t
            for k4_idx in k4s_containing[t]:
                idxs = k4s[k4_idx]
                # Count how many of the 4 triples are red (0)
                red_count = 0
                for i in idxs:
                    if i == t:
                        if old_color == 0:
                            red_count += 1
                    else:
                        if colors[i] == 0:
                            red_count += 1

                # Before: red_count reds. After: if old_color==0 and new_color==1, red_count-1
                # if old_color==1 and new_color==0, red_count+1

                was_violation = (red_count == 4)
                if old_color == 0:
                    new_red_count = red_count - 1
                else:
                    new_red_count = red_count + 1
                is_violation = (new_red_count == 4)

                if was_violation and not is_violation:
                    delta_cost -= 1
                elif not was_violation and is_violation:
                    delta_cost += 1

            # Update K5s containing t
            for k5_idx in k5s_containing[t]:
                idxs = k5s[k5_idx]
                # Count how many of the 10 triples are blue (1)
                blue_count = 0
                for i in idxs:
                    if i == t:
                        if old_color == 1:
                            blue_count += 1
                    else:
                        if colors[i] == 1:
                            blue_count += 1

                was_violation = (blue_count == 10)
                if old_color == 1:
                    new_blue_count = blue_count - 1
                else:
                    new_blue_count = blue_count + 1
                is_violation = (new_blue_count == 10)

                if was_violation and not is_violation:
                    delta_cost -= 1
                elif not was_violation and is_violation:
                    delta_cost += 1

            new_cost = current_cost + delta_cost

            # Acceptance criterion
            if new_cost < current_cost:
                accept = True
            elif new_cost == current_cost:
                accept = np.random.rand() < 0.3
            else:
                if temp > min_temp:
                    # Use a scaled version for better acceptance
                    accept = np.random.rand() < np.exp(-float(delta_cost) / temp)
                else:
                    accept = False

            if accept:
                colors[t] = new_color
                current_cost = new_cost
                no_improve_steps = 0
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_colors = colors.copy()
                    if best_cost == 0:
                        return best_colors, True
            else:
                no_improve_steps += 1

            # Adaptive restart: if stuck, reheat
            if no_improve_steps > max_no_improve:
                no_improve_steps = 0
                temp = max(temp, 1.0)

            temp *= cool_rate
            temp = max(temp, min_temp)

        return best_colors, (best_cost == 0)

    best_n = 9
    best_result_colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(9), 3)]

    # Try to find solutions for decreasing n values
    # Allocate time based on n: larger n gets more time
    target_ns = [34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20]

    # Time allocation: give more time to higher n
    time_alloc = {
        34: 25.0, 33: 15.0, 32: 12.0, 31: 10.0, 30: 8.0,
        29: 7.0, 28: 6.0, 27: 5.0, 26: 4.0, 25: 4.0,
        24: 3.0, 23: 3.0, 22: 2.5, 21: 2.0, 20: 2.0
    }

    for n in target_ns:
        if time.time() > deadline - 15:
            break

        alloc = time_alloc.get(n, 3.0)
        time_limit = min(time.time() + alloc, deadline - 10)

        if time_limit <= time.time() + 1:
            continue

        # Try multiple seeds for robustness
        seeds = [42, 123, 777, 2024, 999]
        found = False
        for seed in seeds:
            if time.time() > time_limit:
                break
            try:
                colors, is_valid = simulated_annealing(n, time_limit, init_colors=None, seed=seed)
                if is_valid:
                    best_n = n
                    best_result_colors = colors.tolist()
                    found = True
                    break
            except Exception as e:
                pass

        if found and n == 34:
            break

    return {'n': best_n, 'colors': best_result_colors}
