import numpy as np
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    def get_triple_indices(n):
        return list(combinations(range(n), 3))

    def build_k4_k5(n):
        triples = get_triple_indices(n)
        num_triples = len(triples)
        triple_to_idx = {t: i for i, t in enumerate(triples)}

        k4s = []
        for quad in combinations(range(n), 4):
            a, b, c, d = quad
            k4s.append((
                triple_to_idx[(a, b, c)],
                triple_to_idx[(a, b, d)],
                triple_to_idx[(a, c, d)],
                triple_to_idx[(b, c, d)]
            ))

        k5s = []
        for quint in combinations(range(n), 5):
            idxs = []
            for triple in combinations(quint, 3):
                idxs.append(triple_to_idx[triple])
            k5s.append(tuple(idxs))

        triple_to_k4s = [[] for _ in range(num_triples)]
        triple_to_k5s = [[] for _ in range(num_triples)]

        for i, idxs in enumerate(k4s):
            for t in idxs:
                triple_to_k4s[t].append(i)

        for i, idxs in enumerate(k5s):
            for t in idxs:
                triple_to_k5s[t].append(i)

        # Pre-compute numpy arrays for faster operations
        k4s_arr = np.array(k4s, dtype=np.int32) if k4s else np.empty((0, 4), dtype=np.int32)
        k5s_arr = np.array(k5s, dtype=np.int32) if k5s else np.empty((0, 10), dtype=np.int32)
        
        # Pre-compute red/blue counts for each K4/K5
        k4_red_counts = np.zeros(len(k4s), dtype=np.int32)
        k5_blue_counts = np.zeros(len(k5s), dtype=np.int32)

        return triples, num_triples, k4s, k5s, triple_to_k4s, triple_to_k5s, k4s_arr, k5s_arr

    def count_violations(colors, k4s, k5s):
        red_count = 0
        for idxs in k4s:
            if all(colors[i] == 0 for i in idxs):
                red_count += 1
        blue_count = 0
        for idxs in k5s:
            if all(colors[i] == 1 for i in idxs):
                blue_count += 1
        return red_count + blue_count

    def delta_cost(colors, t, old_color, new_color, k4s, k5s, triple_to_k4s, triple_to_k5s):
        delta = 0
        for k4_idx in triple_to_k4s[t]:
            idxs = k4s[k4_idx]
            red_count = sum(1 for i in idxs if colors[i] == 0)
            if old_color == 0:
                if red_count == 4:
                    delta -= 1
            else:
                if red_count == 3:
                    delta += 1
        for k5_idx in triple_to_k5s[t]:
            idxs = k5s[k5_idx]
            blue_count = sum(1 for i in idxs if colors[i] == 1)
            if old_color == 1:
                if blue_count == 10:
                    delta -= 1
            else:
                if blue_count == 9:
                    delta += 1
        return delta

    def structured_init(n, variant=0, seed=0):
        triples = get_triple_indices(n)
        colors = []
        
        if variant == 0:
            for a, b, c in triples:
                colors.append(0 if (a + b + c) % 3 == 0 else 1)
        elif variant == 1:
            for a, b, c in triples:
                colors.append(0 if (a + b + 2*c) % 3 == 0 else 1)
        elif variant == 2:
            for a, b, c in triples:
                colors.append((a*b + b*c + c*a) % 2)
        elif variant == 3:
            for a, b, c in triples:
                colors.append((a + b + c) % 2)
        elif variant == 4:
            for a, b, c in triples:
                colors.append((a*b*c) % 2)
        elif variant == 5:
            for a, b, c in triples:
                colors.append(0 if (a + b) % 3 == 0 else 1)
        elif variant == 6:
            for a, b, c in triples:
                colors.append(0 if (b + c) % 3 == 0 else 1)
        elif variant == 7:
            for a, b, c in triples:
                colors.append(0 if (a + c) % 3 == 0 else 1)
        elif variant == 8:
            np.random.seed(12345 + seed)
            colors = np.random.randint(0, 2, len(triples))
            return np.array(colors, dtype=np.int8)
        elif variant == 9:
            np.random.seed(42 + seed)
            pair_vals = {}
            for i in range(n):
                for j in range(i+1, n):
                    pair_vals[(i,j)] = np.random.randint(0, 3)
            for a, b, c in triples:
                s = (pair_vals[(a,b)] + pair_vals[(b,c)] + pair_vals[(a,c)]) % 3
                colors.append(0 if s == 0 else 1)
        elif variant == 10:
            np.random.seed(99 + seed)
            pair_vals = {}
            for i in range(n):
                for j in range(i+1, n):
                    pair_vals[(i,j)] = np.random.randint(0, 3)
            for a, b, c in triples:
                s = (pair_vals[(a,b)] + 2*pair_vals[(b,c)] + pair_vals[(a,c)]) % 3
                colors.append(0 if s == 0 else 1)
        elif variant == 11:
            # Quadratic form mod 2
            for a, b, c in triples:
                colors.append((a*b + b*c) % 2)
        elif variant == 12:
            # Based on max element
            for a, b, c in triples:
                colors.append(0 if c % 2 == 0 else 1)
        elif variant == 13:
            # Based on sum of pairs
            for a, b, c in triples:
                colors.append(0 if (a+b+c) % 2 == 0 and (a*b + b*c + c*a) % 2 == 0 else 1)
        elif variant == 14:
            # Random with bias
            np.random.seed(777 + seed)
            colors = np.random.randint(0, 2, len(triples))
            return np.array(colors, dtype=np.int8)
        else:
            for a, b, c in triples:
                colors.append(0 if (a + c) % 3 == 0 else 1)
        
        return np.array(colors, dtype=np.int8)

    def repair_search(n, time_limit, seed=0, init_variant=0):
        triples, num_triples, k4s, k5s, triple_to_k4s, triple_to_k5s, k4s_arr, k5s_arr = build_k4_k5(n)
        np.random.seed(seed)

        colors = structured_init(n, init_variant, seed)
        current_cost = count_violations(colors, k4s, k5s)
        best_cost = current_cost
        best_colors = colors.copy()

        if current_cost == 0:
            return colors, True

        violated_k4s = set()
        violated_k5s = set()
        for i, idxs in enumerate(k4s):
            if all(colors[j] == 0 for j in idxs):
                violated_k4s.add(i)
        for i, idxs in enumerate(k5s):
            if all(colors[j] == 1 for j in idxs):
                violated_k5s.add(i)

        temp = 1.0
        cool_rate = 0.9999
        min_temp = 0.001

        step = 0
        no_improve = 0
        max_no_improve = 8000

        while time.time() < time_limit:
            step += 1

            # Strategy: prioritize fixing violations
            if (violated_k4s or violated_k5s) and np.random.rand() < 0.85:
                if violated_k4s:
                    if np.random.rand() < 0.65 or not violated_k5s:
                        k4_idx = np.random.choice(list(violated_k4s))
                        idxs = k4s[k4_idx]
                        t = idxs[np.random.randint(4)]
                    else:
                        k5_idx = np.random.choice(list(violated_k5s))
                        idxs = k5s[k5_idx]
                        t = idxs[np.random.randint(10)]
                else:
                    k5_idx = np.random.choice(list(violated_k5s))
                    idxs = k5s[k5_idx]
                    t = idxs[np.random.randint(10)]
            else:
                t = np.random.randint(num_triples)

            old_color = colors[t]
            new_color = 1 - old_color

            dc = delta_cost(colors, t, old_color, new_color, k4s, k5s, triple_to_k4s, triple_to_k5s)
            new_cost = current_cost + dc

            if dc <= 0:
                accept = True
            else:
                if temp > min_temp:
                    accept = np.random.rand() < np.exp(-dc / temp)
                else:
                    accept = False

            if accept:
                colors[t] = new_color
                current_cost = new_cost

                for k4_i in triple_to_k4s[t]:
                    idxs = k4s[k4_i]
                    if all(colors[j] == 0 for j in idxs):
                        violated_k4s.add(k4_i)
                    else:
                        violated_k4s.discard(k4_i)

                for k5_i in triple_to_k5s[t]:
                    idxs = k5s[k5_i]
                    if all(colors[j] == 1 for j in idxs):
                        violated_k5s.add(k5_i)
                    else:
                        violated_k5s.discard(k5_i)

                if current_cost < best_cost:
                    best_cost = current_cost
                    best_colors = colors.copy()
                    no_improve = 0
                    if best_cost == 0:
                        return best_colors, True
                else:
                    no_improve += 1
            else:
                no_improve += 1

            temp *= cool_rate

            if no_improve > max_no_improve and temp > min_temp:
                temp = max(0.3, temp * 1.3)
                no_improve = 0

            if no_improve > max_no_improve * 5:
                colors = best_colors.copy()
                current_cost = best_cost
                temp = max(0.5, temp)
                no_improve = 0
                violated_k4s = set()
                violated_k5s = set()
                for i, idxs in enumerate(k4s):
                    if all(colors[j] == 0 for j in idxs):
                        violated_k4s.add(i)
                for i, idxs in enumerate(k5s):
                    if all(colors[j] == 1 for j in idxs):
                        violated_k5s.add(i)

        return best_colors, (best_cost == 0)

    def verify(n, colors):
        triples = get_triple_indices(n)
        triple_to_idx = {t: i for i, t in enumerate(triples)}

        for quad in combinations(range(n), 4):
            a, b, c, d = quad
            if (colors[triple_to_idx[(a, b, c)]] == 0 and
                colors[triple_to_idx[(a, b, d)]] == 0 and
                colors[triple_to_idx[(a, c, d)]] == 0 and
                colors[triple_to_idx[(b, c, d)]] == 0):
                return False

        for quint in combinations(range(n), 5):
            all_blue = True
            for triple in combinations(quint, 3):
                if colors[triple_to_idx[triple]] != 1:
                    all_blue = False
                    break
            if all_blue:
                return False

        return True

    final_n = 0
    final_colors = []

    # Focus on highest n values first with more time
    target_ns = [35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9]

    for n in target_ns:
        if time.time() > deadline - 5:
            break

        if n <= final_n:
            continue

        remaining = deadline - time.time()
        if remaining < 3:
            break

        # Allocate more time to higher n values
        if n == 35:
            time_budget = min(60.0, remaining * 0.8)
        elif n == 34:
            time_budget = min(50.0, remaining * 0.7)
        elif n >= 30:
            time_budget = min(25.0, remaining * 0.5)
        elif n >= 25:
            time_budget = min(15.0, remaining * 0.4)
        elif n >= 20:
            time_budget = min(10.0, remaining * 0.3)
        else:
            time_budget = min(5.0, remaining * 0.2)

        if time_budget < 1.0:
            break

        time_limit = time.time() + time_budget

        found = False
        num_seeds = 40 if n >= 35 else 35 if n >= 34 else 25

        for seed in range(num_seeds):
            if time.time() > time_limit:
                break

            variant = seed % 15

            try:
                colors, is_valid = repair_search(n, time_limit, seed=seed, init_variant=variant)

                if is_valid:
                    if verify(n, colors.tolist()):
                        final_n = n
                        final_colors = colors.tolist()
                        found = True
                        break
            except Exception:
                continue

        if found:
            break

    if final_n == 0 or not final_colors:
        n = 9
        triples = get_triple_indices(n)
        colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in triples]

        if verify(n, colors):
            final_n = n
            final_colors = colors
        else:
            np.random.seed(42)
            for _ in range(1000):
                colors = [np.random.randint(0, 2) for _ in range(len(triples))]
                if verify(n, colors):
                    final_n = n
                    final_colors = colors
                    break
            else:
                final_n = 1
                final_colors = [0]

    return {'n': final_n, 'colors': final_colors}
