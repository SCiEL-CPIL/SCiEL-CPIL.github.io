import time
import random
import itertools
from collections import Counter

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill

    n = 14
    m = 35

    all_pairs = list(itertools.combinations(range(n), 2))
    num_pairs = len(all_pairs)  # 91

    def calc_score(vals, edges):
        """Compute max(distinct_sums, distinct_prods) for given values and edges."""
        sums = set()
        prods = set()
        for i, j in edges:
            sums.add(vals[i] + vals[j])
            prods.add(vals[i] * vals[j])
        return max(len(sums), len(prods))

    def greedy_edges_lookahead(vals, lookahead=2):
        """Greedy edge selection with lookahead: pick edges that minimize future cost."""
        pair_data = []
        for i, j in all_pairs:
            s = vals[i] + vals[j]
            p = vals[i] * vals[j]
            pair_data.append((s, p, i, j))

        # Precompute frequency of each sum and product across all pairs
        sum_freq = Counter()
        prod_freq = Counter()
        for s, p, i, j in pair_data:
            sum_freq[s] += 1
            prod_freq[p] += 1

        # Group pairs by (sum, prod) to find collision clusters
        collision_map = {}
        for idx, (s, p, i, j) in enumerate(pair_data):
            key = (s, p)
            if key not in collision_map:
                collision_map[key] = []
            collision_map[key].append(idx)

        selected = []
        seen_sums = set()
        seen_prods = set()
        available = list(range(num_pairs))
        available_set = set(available)

        for step in range(m):
            best_idx = -1
            best_cost = float('inf')
            best_tiebreak = -1

            for idx in available:
                s, p, i, j = pair_data[idx]
                new_sums = len(seen_sums) + (1 if s not in seen_sums else 0)
                new_prods = len(seen_prods) + (1 if p not in seen_prods else 0)
                cost = max(new_sums, new_prods)

                # Tiebreak: prefer edges whose sum/prod have high collision potential
                # (more other edges share that sum or product)
                tiebreak = sum_freq[s] + prod_freq[p]

                # Lookahead: check how many remaining edges would benefit
                if cost < best_cost or (cost == best_cost and tiebreak > best_tiebreak):
                    best_cost = cost
                    best_idx = idx
                    best_tiebreak = tiebreak

            if best_idx == -1:
                break

            s, p, i, j = pair_data[best_idx]
            selected.append([i, j])
            seen_sums.add(s)
            seen_prods.add(p)
            available.remove(best_idx)
            available_set.discard(best_idx)

        return selected, calc_score(vals, selected)

    def local_search_edges(vals, edges, max_rounds=50):
        """Iterative local search: swap one edge at a time to reduce score."""
        current_edges = [e[:] for e in edges]
        current_score = calc_score(vals, current_edges)

        if current_score <= 1:
            return current_edges, current_score

        selected_set = set(frozenset(e) for e in current_edges)
        candidate_edges = [list(p) for p in all_pairs if frozenset(p) not in selected_set]

        for rnd in range(max_rounds):
            if time.time() > deadline - 3:
                break

            improved = False
            random.shuffle(candidate_edges)

            for new_edge in candidate_edges:
                if time.time() > deadline - 3:
                    break

                new_key = frozenset(new_edge)
                if new_key in selected_set:
                    continue

                for k in range(m):
                    old_key = frozenset(current_edges[k])
                    if old_key == new_key:
                        continue

                    # Quick check: what sums/prods would change?
                    old_s = vals[current_edges[k][0]] + vals[current_edges[k][1]]
                    old_p = vals[current_edges[k][0]] * vals[current_edges[k][1]]
                    new_s = vals[new_edge[0]] + vals[new_edge[1]]
                    new_p = vals[new_edge[0]] * vals[new_edge[1]]

                    # If both sum and prod are already present, score won't increase
                    # If both are new, score might increase by up to 2
                    # Only do full evaluation if potentially beneficial

                    # Build temp
                    temp_edges = current_edges[:]
                    temp_edges[k] = list(new_edge)

                    # Check duplicates
                    temp_set = set(frozenset(e) for e in temp_edges)
                    if len(temp_set) != m:
                        continue

                    new_score = calc_score(vals, temp_edges)
                    if new_score < current_score:
                        current_edges = temp_edges
                        current_score = new_score
                        selected_set = temp_set
                        candidate_edges = [list(p) for p in all_pairs if frozenset(p) not in selected_set]
                        improved = True
                        break

                if improved:
                    break

            if not improved:
                break

        return current_edges, current_score

    def two_opt_edges(vals, edges):
        """Try swapping two edges at once for more aggressive improvement."""
        current_edges = [e[:] for e in edges]
        current_score = calc_score(vals, current_edges)

        if current_score <= 2:
            return current_edges, current_score

        selected_set = set(frozenset(e) for e in current_edges)
        candidate_edges = [list(p) for p in all_pairs if frozenset(p) not in selected_set]

        for _ in range(30):
            if time.time() > deadline - 3:
                break

            improved = False
            random.shuffle(candidate_edges)

            for new_edge in candidate_edges[:20]:  # limit candidates per round
                if time.time() > deadline - 3:
                    break

                new_key = frozenset(new_edge)
                if new_key in selected_set:
                    continue

                for k in range(m):
                    old_key = frozenset(current_edges[k])
                    if old_key == new_key:
                        continue

                    temp_edges = current_edges[:]
                    temp_edges[k] = list(new_edge)

                    temp_set = set(frozenset(e) for e in temp_edges)
                    if len(temp_set) != m:
                        continue

                    new_score = calc_score(vals, temp_edges)
                    if new_score < current_score:
                        current_edges = temp_edges
                        current_score = new_score
                        selected_set = temp_set
                        candidate_edges = [list(p) for p in all_pairs if frozenset(p) not in selected_set]
                        improved = True
                        break

                if improved:
                    break

            if not improved:
                break

        return current_edges, current_score

    # Generate diverse value sets designed for high collision
    value_sets = []

    # 1. Small consecutive: 1-14
    value_sets.append(list(range(1, 15)))

    # 2. Powers of 2 (great for product collisions)
    value_sets.append([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192])

    # 3. Near powers of 2 with small perturbations
    value_sets.append([1, 2, 4, 8, 16, 33, 64, 129, 256, 513, 1024, 2049, 4096, 8193])

    # 4. Multiplicative sequence: a*b for small a,b
    value_sets.append([1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 36, 48, 72])

    # 5. Small primes
    value_sets.append([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43])

    # 6. Mixed: some small, some geometric
    value_sets.append([1, 2, 3, 6, 12, 24, 48, 96, 192, 384, 768, 1536, 3072, 6144])

    # 7. All small: 1-7, 9, 11, 13, 15, 17, 19, 21
    value_sets.append([1, 2, 3, 4, 5, 6, 7, 9, 11, 13, 15, 17, 19, 21])

    # 8. Densely packed small values
    value_sets.append([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])

    # 9. Random small values
    for _ in range(3):
        random.seed(random.randint(0, 100000))
        v = random.sample(range(1, 40), n)
        v.sort()
        value_sets.append(v)

    # 10. Another structured set: multiples
    value_sets.append([2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28])

    best_score = float('inf')
    best_values = None
    best_edges = None

    for vals in value_sets:
        if time.time() > deadline - 5:
            break

        # Greedy + local search
        edges, score = greedy_edges_lookahead(vals)

        if score < best_score:
            best_score = score
            best_values = vals[:]
            best_edges = [e[:] for e in edges]

        # Local search refinement
        if time.time() < deadline - 5:
            edges2, score2 = local_search_edges(vals, edges)
            if score2 < score:
                score = score2
                edges = edges2

        # Two-opt refinement
        if time.time() < deadline - 3:
            edges3, score3 = two_opt_edges(vals, edges)
            if score3 < score:
                score = score3
                edges = edges3

        if score < best_score:
            best_score = score
            best_values = vals[:]
            best_edges = [e[:] for e in edges]

        # Try perturbing this value set
        if time.time() < deadline - 10 and score <= best_score:
            current_vals = vals[:]
            current_edges = edges[:]
            current_score = score

            for _ in range(30):
                if time.time() > deadline - 5:
                    break

                idx = random.randint(0, n - 1)
                old_val = current_vals[idx]
                # Try nearby values
                candidates = [old_val + d for d in [-3, -2, -1, 1, 2, 3] if 1 <= old_val + d <= 10000]
                candidates += [random.randint(1, 60) for _ in range(2)]

                for new_val in candidates:
                    if new_val == old_val or new_val in current_vals:
                        continue

                    temp_vals = current_vals[:]
                    temp_vals[idx] = new_val
                    temp_vals.sort()

                    te, ts = greedy_edges_lookahead(temp_vals)
                    if time.time() < deadline - 3:
                        te, ts = local_search_edges(temp_vals, te)

                    if ts < current_score:
                        current_vals = temp_vals
                        current_edges = te
                        current_score = ts
                        if current_score < best_score:
                            best_score = current_score
                            best_values = current_vals[:]
                            best_edges = [e[:] for e in current_edges]
                        break

    if best_values is None:
        best_values = list(range(1, n + 1))
        best_edges, _ = greedy_edges_lookahead(best_values)

    # Final type conversion
    best_values = [int(v) for v in best_values]
    best_edges = [[int(i), int(j)] for i, j in best_edges]

    return {'values': best_values, 'edges': best_edges}
