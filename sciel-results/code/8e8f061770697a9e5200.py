import random
import time
from itertools import combinations

def construct():
    deadline = time.time() + 140
    best_result = None

    # Try from 34 downward
    for n in range(34, 18, -1):
        if time.time() > deadline - 10:
            break

        triples = list(combinations(range(n), 3))
        num_triples = len(triples)
        triple_index = {t: i for i, t in enumerate(triples)}

        # Build K4 and K5 structures
        k4s = []
        k4_map = [[] for _ in range(num_triples)]
        for quad in combinations(range(n), 4):
            edges = [triple_index[tri] for tri in combinations(quad, 3)]
            k4_idx = len(k4s)
            k4s.append(edges)
            for e in edges:
                k4_map[e].append(k4_idx)

        k5s = []
        k5_map = [[] for _ in range(num_triples)]
        for quint in combinations(range(n), 5):
            edges = [triple_index[tri] for tri in combinations(quint, 3)]
            k5_idx = len(k5s)
            k5s.append(edges)
            for e in edges:
                k5_map[e].append(k5_idx)

        num_k4 = len(k4s)
        num_k5 = len(k5s)

        time_budget = 8 if n <= 30 else 5

        def count_violations(colors):
            """Count total violations (red K4s + blue K5s)."""
            violations = 0
            for k4 in k4s:
                if all(colors[e] == 0 for e in k4):
                    violations += 1
            for k5 in k5s:
                if all(colors[e] == 1 for e in k5):
                    violations += 1
            return violations

        def local_search(max_time):
            """Simulated annealing with fast violation counting."""
            start = time.time()
            best_colors = None
            best_viol = float('inf')

            while time.time() - start < max_time:
                # Random initialization with slight bias
                if random.random() < 0.3:
                    # Bias toward avoiding red K4s
                    colors = [random.randint(0, 1) for _ in range(num_triples)]
                    for e in range(num_triples):
                        if colors[e] == 0 and random.random() < 0.3:
                            colors[e] = 1
                else:
                    colors = [random.randint(0, 1) for _ in range(num_triples)]

                # Compute initial violations incrementally
                red_k4_counts = [0] * num_k4
                blue_k5_counts = [0] * num_k5
                for i, edges in enumerate(k4s):
                    for e in edges:
                        if colors[e] == 0:
                            red_k4_counts[i] += 1
                for i, edges in enumerate(k5s):
                    for e in edges:
                        if colors[e] == 1:
                            blue_k5_counts[i] += 1

                violating_k4s = [i for i in range(num_k4) if red_k4_counts[i] == 4]
                violating_k5s = [i for i in range(num_k5) if blue_k5_counts[i] == 10]
                total_viol = len(violating_k4s) + len(violating_k5s)

                if total_viol == 0:
                    return colors

                if total_viol < best_viol:
                    best_viol = total_viol
                    best_colors = colors[:]

                # Simulated annealing
                temp = 10.0
                cooling = 0.995
                stuck = 0
                last_viol = total_viol

                while time.time() - start < max_time and total_viol > 0:
                    # Pick a random triple to flip
                    e = random.randint(0, num_triples - 1)
                    old_val = colors[e]
                    new_val = 1 - old_val

                    # Compute delta in violations
                    delta = 0
                    for k4_i in k4_map[e]:
                        old_c = red_k4_counts[k4_i]
                        new_c = old_c - 1 if old_val == 0 else old_c + 1
                        if old_c == 4 and new_c != 4:
                            delta -= 1
                        elif old_c != 4 and new_c == 4:
                            delta += 1
                        red_k4_counts[k4_i] = new_c
                    for k5_i in k5_map[e]:
                        old_c = blue_k5_counts[k5_i]
                        new_c = old_c - 1 if old_val == 1 else old_c + 1
                        if old_c == 10 and new_c != 10:
                            delta -= 1
                        elif old_c != 10 and new_c == 10:
                            delta += 1
                        blue_k5_counts[k5_i] = new_c

                    # Accept or reject
                    if delta <= 0 or random.random() < (2.718281828 ** (-delta / max(temp, 1e-6))):
                        colors[e] = new_val
                        total_viol += delta

                        if total_viol < best_viol:
                            best_viol = total_viol
                            best_colors = colors[:]

                        if total_viol == 0:
                            return colors

                        if total_viol >= last_viol:
                            stuck += 1
                        else:
                            stuck = 0
                            last_viol = total_viol

                        if stuck > 2000:
                            break

                    else:
                        # Revert
                        for k4_i in k4_map[e]:
                            old_c = red_k4_counts[k4_i]
                            new_c = old_c + 1 if old_val == 0 else old_c - 1
                            red_k4_counts[k4_i] = new_c
                        for k5_i in k5_map[e]:
                            old_c = blue_k5_counts[k5_i]
                            new_c = old_c + 1 if old_val == 1 else old_c - 1
                            blue_k5_counts[k5_i] = new_c

                    temp *= cooling

                if best_viol == 0:
                    return best_colors

            return best_colors if best_colors is not None else None

        # Multiple restarts
        result = local_search(time_budget)
        if result is not None:
            # Verify
            if count_violations(result) == 0:
                best_result = {'n': n, 'colors': result}
                return best_result

        # If we get here, try a few more restarts for this n
        for _ in range(3):
            if time.time() > deadline - 10:
                break
            result = local_search(time_budget)
            if result is not None and count_violations(result) == 0:
                best_result = {'n': n, 'colors': result}
                return best_result

    # Fallback: small valid colorings
    if best_result is not None:
        return best_result

    # Trivial valid colorings for small n
    # n=1: one vertex, no triples
    return {'n': 1, 'colors': [0]}
