import numpy as np
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # 15s margin before 150s kill

    def get_k4_k5_indices(n):
        """Precompute all K4 and K5 clique triple-index tuples."""
        triples = list(combinations(range(n), 3))
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

        return triples, num_triples, k4s, k5s

    def build_reverse_indices(num_triples, k4s, k5s):
        """Build reverse index: triple_idx -> list of clique indices containing it."""
        triple_to_k4s = [[] for _ in range(num_triples)]
        triple_to_k5s = [[] for _ in range(num_triples)]

        for i, idxs in enumerate(k4s):
            for t in idxs:
                triple_to_k4s[t].append(i)

        for i, idxs in enumerate(k5s):
            for t in idxs:
                triple_to_k5s[t].append(i)

        return triple_to_k4s, triple_to_k5s

    def compute_initial_cost(colors, k4s, k5s):
        """Compute number of red K4s and blue K5s."""
        red_count = 0
        for idxs in k4s:
            if colors[idxs[0]] == 0 and colors[idxs[1]] == 0 and colors[idxs[2]] == 0 and colors[idxs[3]] == 0:
                red_count += 1
        blue_count = 0
        for idxs in k5s:
            if all(colors[i] == 1 for i in idxs):
                blue_count += 1
        return red_count + blue_count

    def local_search(n, time_limit, seed=0):
        """Simulated annealing with fast delta computation."""
        triples, num_triples, k4s, k5s = get_k4_k5_indices(n)
        triple_to_k4s, triple_to_k5s = build_reverse_indices(num_triples, k4s, k5s)

        np.random.seed(seed)

        # Multiple initialization strategies
        init_strategies = [
            lambda a, b, c: (a + b + c) % 3 == 0,
            lambda a, b, c: (a * 2 + b * 3 + c * 5) % 7 < 3,
            lambda a, b, c: (a ^ b ^ c) % 2 == 0,
            lambda a, b, c: (a + 2*b + 3*c) % 5 < 2,
            lambda a, b, c: np.random.rand() < 0.5,
            lambda a, b, c: (a * b + b * c + c * a) % 11 < 5,
            lambda a, b, c: (a * 7 + b * 13 + c * 17) % 19 < 8,
            lambda a, b, c: (a + b) % 4 < 2 and (b + c) % 4 < 2,
        ]

        best_colors = None
        best_cost = float('inf')

        for strategy_idx, init_func in enumerate(init_strategies):
            if time.time() > time_limit:
                break

            # Initialize colors
            colors = np.array([1 if init_func(a, b, c) else 0 for a, b, c in triples], dtype=np.int8)

            # Compute initial cost
            current_cost = compute_initial_cost(colors, k4s, k5s)

            if current_cost == 0:
                return colors, True

            if current_cost < best_cost:
                best_cost = current_cost
                best_colors = colors.copy()

            # Simulated annealing with adaptive temperature
            temp = 3.0
            cool_rate = 0.999
            min_temp = 0.05
            no_improve = 0
            max_no_improve = 5000

            while time.time() < time_limit:
                # Pick random triple
                t = np.random.randint(num_triples)
                old_color = colors[t]
                new_color = 1 - old_color

                # Calculate delta cost efficiently
                delta_cost = 0

                # Update K4s containing t
                for k4_idx in triple_to_k4s[t]:
                    idxs = k4s[k4_idx]
                    red_before = sum(1 for i in idxs if colors[i] == 0)
                    if old_color == 0:
                        red_after = red_before - 1
                    else:
                        red_after = red_before + 1

                    if red_before == 4 and red_after < 4:
                        delta_cost -= 1
                    elif red_before < 4 and red_after == 4:
                        delta_cost += 1

                # Update K5s containing t
                for k5_idx in triple_to_k5s[t]:
                    idxs = k5s[k5_idx]
                    blue_before = sum(1 for i in idxs if colors[i] == 1)
                    if old_color == 1:
                        blue_after = blue_before - 1
                    else:
                        blue_after = blue_before + 1

                    if blue_before == 10 and blue_after < 10:
                        delta_cost -= 1
                    elif blue_before < 10 and blue_after == 10:
                        delta_cost += 1

                new_cost = current_cost + delta_cost

                # Acceptance criterion
                if new_cost < current_cost:
                    accept = True
                    no_improve = 0
                elif new_cost == current_cost:
                    accept = np.random.rand() < 0.3
                    no_improve += 1
                else:
                    if temp > min_temp:
                        accept = np.random.rand() < np.exp(-new_cost / temp)
                    else:
                        accept = False
                    no_improve += 1

                if accept:
                    colors[t] = new_color
                    current_cost = new_cost
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_colors = colors.copy()
                        no_improve = 0
                        if best_cost == 0:
                            return best_colors, True

                temp *= cool_rate

                # Restart if stuck
                if no_improve > max_no_improve and temp > min_temp:
                    temp = max(0.5, temp * 1.5)
                    no_improve = 0

        return best_colors, (best_cost == 0)

    def verify_coloring(n, colors):
        """Verify that the coloring has no red K4 and no blue K5."""
        triples = list(combinations(range(n), 3))
        triple_to_idx = {t: i for i, t in enumerate(triples)}

        # Check no red K4
        for quad in combinations(range(n), 4):
            a, b, c, d = quad
            if (colors[triple_to_idx[(a, b, c)]] == 0 and
                colors[triple_to_idx[(a, b, d)]] == 0 and
                colors[triple_to_idx[(a, c, d)]] == 0 and
                colors[triple_to_idx[(b, c, d)]] == 0):
                return False

        # Check no blue K5
        for quint in combinations(range(n), 5):
            all_blue = True
            for triple in combinations(quint, 3):
                if colors[triple_to_idx[triple]] != 1:
                    all_blue = False
                    break
            if all_blue:
                return False

        return True

    # Start with a known small valid coloring
    final_n = 9
    triples_9 = list(combinations(range(9), 3))
    final_colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in triples_9]

    # Try to find valid colorings for larger n
    # Focus more time on larger n values where improvement is more valuable
    target_ns = list(range(10, 35))

    for n in target_ns:
        if time.time() > deadline - 3:
            break

        # Allocate time based on n: larger n gets more time
        if n <= 15:
            time_budget = min(2.0, (deadline - time.time()) * 0.25)
        elif n <= 25:
            time_budget = min(4.0, (deadline - time.time()) * 0.3)
        else:
            time_budget = min(6.0, (deadline - time.time()) * 0.35)

        if time_budget < 0.5:
            break

        time_limit = time.time() + time_budget

        try:
            # Try multiple seeds
            for seed in range(5):
                if time.time() > time_limit:
                    break

                colors, is_valid = local_search(n, time_limit, seed=seed)

                if is_valid:
                    if verify_coloring(n, colors):
                        final_n = n
                        final_colors = colors.tolist()
                        break
                    else:
                        pass
        except Exception as e:
            continue

    # Final verification
    if not verify_coloring(final_n, final_colors):
        final_n = 9
        triples_9 = list(combinations(range(9), 3))
        final_colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in triples_9]

    return {'n': final_n, 'colors': final_colors}
