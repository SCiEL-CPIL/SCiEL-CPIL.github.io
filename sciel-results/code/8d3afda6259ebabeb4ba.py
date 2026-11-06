import time
import random
import math
from itertools import combinations
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin before 120s limit

    def try_find_valid(n, time_budget):
        """Try to find a valid coloring for n vertices within time_budget seconds."""
        triples = list(combinations(range(n), 3))
        num_triples = len(triples)
        triple_index = {t: i for i, t in enumerate(triples)}

        # Precompute K4 and K5 structures
        k4s = list(combinations(range(n), 4))
        k5s = list(combinations(range(n), 5))
        num_k4 = len(k4s)
        num_k5 = len(k5s)

        # For each K4, the 4 triple indices
        k4_triples = []
        for quad in k4s:
            a, b, c, d = quad
            k4_triples.append([
                triple_index[(a, b, c)],
                triple_index[(a, b, d)],
                triple_index[(a, c, d)],
                triple_index[(b, c, d)]
            ])

        # For each K5, the 10 triple indices
        k5_triples = []
        for quint in k5s:
            vs = list(quint)
            idxs = []
            for i in range(5):
                for j in range(i+1, 5):
                    for k in range(j+1, 5):
                        idxs.append(triple_index[(vs[i], vs[j], vs[k])])
            k5_triples.append(idxs)

        # Build reverse maps: triple -> list of K4 indices, triple -> list of K5 indices
        triple_to_k4s = [[] for _ in range(num_triples)]
        for k4_i, idxs in enumerate(k4_triples):
            for tri_idx in idxs:
                triple_to_k4s[tri_idx].append(k4_i)

        triple_to_k5s = [[] for _ in range(num_triples)]
        for k5_i, idxs in enumerate(k5_triples):
            for tri_idx in idxs:
                triple_to_k5s[tri_idx].append(k5_i)

        def count_violations(colors):
            """Count number of violating K4s and K5s."""
            red_k4 = np.zeros(num_k4, dtype=np.int8)
            blue_k5 = np.zeros(num_k5, dtype=np.int8)
            for t in range(num_triples):
                if colors[t] == 0:
                    for k4_i in triple_to_k4s[t]:
                        red_k4[k4_i] += 1
                else:
                    for k5_i in triple_to_k5s[t]:
                        blue_k5[k5_i] += 1
            return int(np.sum(red_k4 == 4)) + int(np.sum(blue_k5 == 10))

        def local_search(colors, max_iters=200000):
            """Greedy local search: flip triples involved in violations to reduce them."""
            colors = list(colors)
            for it in range(max_iters):
                if time.time() > deadline:
                    return None
                # Find violations
                red_k4 = np.zeros(num_k4, dtype=np.int8)
                blue_k5 = np.zeros(num_k5, dtype=np.int8)
                for t in range(num_triples):
                    if colors[t] == 0:
                        for k4_i in triple_to_k4s[t]:
                            red_k4[k4_i] += 1
                    else:
                        for k5_i in triple_to_k5s[t]:
                            blue_k5[k5_i] += 1
                viol_k4 = np.where(red_k4 == 4)[0]
                viol_k5 = np.where(blue_k5 == 10)[0]
                if len(viol_k4) == 0 and len(viol_k5) == 0:
                    return colors
                # Pick a random violation and flip one of its triples
                if len(viol_k4) > 0 and (len(viol_k5) == 0 or random.random() < 0.5):
                    k4_i = random.choice(viol_k4.tolist())
                    t = random.choice(k4_triples[k4_i])
                else:
                    k5_i = random.choice(viol_k5.tolist())
                    t = random.choice(k5_triples[k5_i])
                colors[t] = 1 - colors[t]
            return None

        # Try multiple restarts with different biases
        biases = [0.5, 0.45, 0.55, 0.4, 0.6, 0.35, 0.65, 0.5]
        random.shuffle(biases)

        for restart in range(30):
            if time.time() > deadline:
                break
            p_red = biases[restart % len(biases)]
            colors = [0 if random.random() < p_red else 1 for _ in range(num_triples)]

            # Quick check
            v = count_violations(colors)
            if v == 0:
                return colors

            # Local search
            result = local_search(colors)
            if result is not None:
                return result

        return None

    # Try n=34 first
    result = try_find_valid(34, 110)
    if result is not None:
        return {'n': 34, 'colors': result}

    # Fallback: try smaller n
    for n in [30, 25, 20, 15, 10, 9]:
        if time.time() > deadline - 5:
            break
        result = try_find_valid(n, min(10, deadline - time.time()))
        if result is not None:
            return {'n': n, 'colors': result}

    # Last resort: known valid small coloring
    n = 9
    return {'n': n, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]}
