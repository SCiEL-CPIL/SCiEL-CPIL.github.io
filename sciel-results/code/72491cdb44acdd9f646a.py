"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill

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
            if all(colors[i] == 0 for i in idxs):
                red_count += 1
        blue_count = 0
        for idxs in k5s:
            if all(colors[i] == 1 for i in idxs):
                blue_count += 1
        return red_count + blue_count

    def finite_field_init(n, p, mod_val):
        """Initialize coloring using a modular arithmetic pattern."""
        triples = list(combinations(range(n), 3))
        colors = np.zeros(len(triples), dtype=np.int8)
        for i, (a, b, c) in enumerate(triples):
            # Use a combination of modular patterns
            val = (a * p + b * (p + 1) + c * (p + 2)) % mod_val
            colors[i] = 0 if val < mod_val // 2 else 1
        return colors

    def xor_init(n, shift):
        """Initialize coloring using XOR-based pattern."""
        triples = list(combinations(range(n), 3))
        colors = np.zeros(len(triples), dtype=np.int8)
        for i, (a, b, c) in enumerate(triples):
            val = (a ^ (b + shift) ^ (c + 2*shift)) % 2
            colors[i] = val
        return colors

    def quadratic_init(n, a_coeff, b_coeff, c_coeff, mod_val):
        """Initialize using quadratic form."""
        triples = list(combinations(range(n), 3))
        colors = np.zeros(len(triples), dtype=np.int8)
        for i, (a, b, c) in enumerate(triples):
            val = (a_coeff * a * a + b_coeff * b * b + c_coeff * c * c +
                   a * b + b * c + c * a) % mod_val
            colors[i] = 0 if val < mod_val // 2 else 1
        return colors

    def local_search_conflict_priority(n, time_limit, seed=0):
        """Local search with conflict-priority: flip triples in most-violated cliques first."""
        triples, num_triples, k4s, k5s = get_k4_k5_indices(n)
        triple_to_k4s, triple_to_k5s = build_reverse_indices(num_triples, k4s, k5s)

        np.random.seed(seed)

        # Multiple initialization strategies
        init_strategies = [
            lambda: finite_field_init(n, 3, 7),
            lambda: finite_field_init(n, 5, 11),
            lambda: finite_field_init(n, 2, 5),
            lambda: xor_init(n, 1),
            lambda: xor_init(n, 3),
            lambda: quadratic_init(n, 1, 2, 3, 7),
            lambda: quadratic_init(n, 2, 3, 5, 11),
            lambda: np.random.randint(0, 2, num_triples, dtype=np.int8),
            lambda: np.random.randint(0, 2, num_triples, dtype=np.int8),
            lambda: np.random.randint(0, 2, num_triples, dtype=np.int8),
        ]

        best_colors = None
        best_cost = float('inf')

        for strategy_idx, init_func in enumerate(init_strategies):
            if time.time() > time_limit:
                break

            colors = init_func()
            current_cost = compute_initial_cost(colors, k4s, k5s)

            if current_cost == 0:
                return colors, True

            if current_cost < best_cost:
                best_cost = current_cost
                best_colors = colors.copy()

            # Conflict-priority local search
            # Track violation counts per clique
            k4_red_counts = np.zeros(len(k4s), dtype=np.int32)
            k5_blue_counts = np.zeros(len(k5s), dtype=np.int32)

            for i, idxs in enumerate(k4s):
                k4_red_counts[i] = np.sum(colors[list(idxs)] == 0)
            for i, idxs in enumerate(k5s):
                k5_blue_counts[i] = np.sum(colors[list(idxs)] == 1)

            # Priority queue: (negative violation_count, clique_type, clique_idx, triple_idx)
            # We'll use a simpler approach: periodically rebuild priority based on violations

            temp = 2.0
            cool_rate = 0.9995
            min_temp = 0.1
            no_improve = 0
            max_no_improve = 3000
            steps = 0

            while time.time() < time_limit:
                steps += 1

                # Every 500 steps, rebuild a priority list of triples to try
                if steps % 500 == 1 or steps == 1:
                    # Find triples that are in violated cliques
                    candidate_triples = set()

                    # Add triples from red K4 violations
                    for k4_idx in range(len(k4s)):
                        if k4_red_counts[k4_idx] == 4:
                            for t in k4s[k4_idx]:
                                candidate_triples.add(t)

                    # Add triples from blue K5 violations
                    for k5_idx in range(len(k5s)):
                        if k5_blue_counts[k5_idx] == 10:
                            for t in k5s[k5_idx]:
                                candidate_triples.add(t)

                    if not candidate_triples:
                        # No violations, try random
                        candidate_triples = set(np.random.randint(0, num_triples, min(100, num_triples)))

                t = np.random.choice(list(candidate_triples)) if candidate_triples else np.random.randint(num_triples)
                old_color = colors[t]
                new_color = 1 - old_color

                # Calculate delta cost efficiently
                delta_cost = 0

                # Update K4s containing t
                for k4_idx in triple_to_k4s[t]:
                    idxs = k4s[k4_idx]
                    red_before = k4_red_counts[k4_idx]
                    if old_color == 0:
                        red_after = red_before - 1
                    else:
                        red_after = red_before + 1

                    if red_before == 4 and red_after < 4:
                        delta_cost -= 1
                    elif red_before < 4 and red_after == 4:
                        delta_cost += 1

                    k4_red_counts[k4_idx] = red_after

                # Update K5s containing t
                for k5_idx in triple_to_k5s[t]:
                    idxs = k5s[k5_idx]
                    blue_before = k5_blue_counts[k5_idx]
                    if old_color == 1:
                        blue_after = blue_before - 1
                    else:
                        blue_after = blue_before + 1

                    if blue_before == 10 and blue_after < 10:
                        delta_cost -= 1
                    elif blue_before < 10 and blue_after == 10:
                        delta_cost += 1

                    k5_blue_counts[k5_idx] = blue_after

                new_cost = current_cost + delta_cost

                # Acceptance criterion
                if new_cost < current_cost:
                    accept = True
                    no_improve = 0
                elif new_cost == current_cost:
                    accept = np.random.rand() < 0.2
                    no_improve += 1
                else:
                    if temp > min_temp:
                        accept = np.random.rand() < np.exp(-delta_cost / temp)
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
                else:
                    # Revert the counts
                    for k4_idx in triple_to_k4s[t]:
                        if old_color == 0:
                            k4_red_counts[k4_idx] += 1
                        else:
                            k4_red_counts[k4_idx] -= 1
                    for k5_idx in triple_to_k5s[t]:
                        if old_color == 1:
                            k5_blue_counts[k5_idx] += 1
                        else:
                            k5_blue_counts[k5_idx] -= 1

                temp *= cool_rate

                # Restart if stuck
                if no_improve > max_no_improve and temp > min_temp:
                    temp = max(1.0, temp * 2.0)
                    no_improve = 0

        return best_colors, (best_cost == 0)

    def verify_coloring(n, colors):
        """Verify that the coloring has no red K4 and no blue K5."""
        triples = list(combinations(range(n), 3))
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

    # Start with a known small valid coloring
    final_n = 9
    triples_9 = list(combinations(range(9), 3))
    final_colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in triples_9]

    # Try to find valid colorings for larger n
    target_ns = list(range(10, 35))

    for n in target_ns:
        if time.time() > deadline - 5:
            break

        # Allocate time based on n
        if n <= 15:
            time_budget = min(3.0, (deadline - time.time()) * 0.3)
        elif n <= 25:
            time_budget = min(5.0, (deadline - time.time()) * 0.3)
        else:
            time_budget = min(8.0, (deadline - time.time()) * 0.35)

        if time_budget < 1.0:
            break

        colors, is_valid = local_search_conflict_priority(n, time.time() + time_budget, seed=n)

        if is_valid and verify_coloring(n, colors):
            final_n = n
            final_colors = list(colors)

    return {'n': final_n, 'colors': final_colors}
# EVOLVE-BLOCK-END
