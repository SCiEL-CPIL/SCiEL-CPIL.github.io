import time
import random
from itertools import combinations
from collections import Counter
import math

DEADLINE = 87.0  # Safety margin for 90s limit


def _evaluate_edge_set(values, edges):
    """Compute the raw value: max(distinct sums, distinct products)."""
    sums = set()
    prods = set()
    for i, j in edges:
        sums.add(values[i] + values[j])
        prods.add(values[i] * values[j])
    return max(len(sums), len(prods))


def _optimize_edges(values, m, deadline):
    """
    Optimize the set of m edges for the given values.
    Uses Simulated Annealing with distinct-count tracking and aggressive polish.
    """
    n = len(values)
    all_edges = list(combinations(range(n), 2))
    num_edges = len(all_edges)

    # Precompute sums and products for all edges
    edge_sums = [values[i] + values[j] for i, j in all_edges]
    edge_prods = [values[i] * values[j] for i, j in all_edges]

    # Precompute unique sums and products for faster distinct counting
    all_sum_vals = set(edge_sums)
    all_prod_vals = set(edge_prods)

    def _score_from_counts(sum_counts, prod_counts):
        distinct_sums = sum(1 for v in sum_counts.values() if v > 0)
        distinct_prods = sum(1 for v in prod_counts.values() if v > 0)
        return max(distinct_sums, distinct_prods)

    best_score = float('inf')
    best_edges_set = None

    rng = random.Random(42)

    # Number of restarts depends on time available
    restart_count = 0
    while time.time() < deadline:
        restart_count += 1

        if restart_count == 1:
            # Greedy initialization: add edges that minimize cost increase
            current_edges_set = set()
            sum_counts = Counter()
            prod_counts = Counter()
            remaining = list(range(num_edges))
            rng.shuffle(remaining)

            for _ in range(m):
                best_idx = -1
                best_cost = float('inf')
                for idx in remaining:
                    s = edge_sums[idx]
                    p = edge_prods[idx]

                    old_s = sum_counts[s]
                    old_p = prod_counts[p]

                    sum_counts[s] = old_s + 1
                    prod_counts[p] = old_p + 1

                    cost = _score_from_counts(sum_counts, prod_counts)

                    # Revert
                    if old_s == 0:
                        del sum_counts[s]
                    else:
                        sum_counts[s] = old_s
                    if old_p == 0:
                        del prod_counts[p]
                    else:
                        prod_counts[p] = old_p

                    if cost < best_cost:
                        best_cost = cost
                        best_idx = idx

                if best_idx != -1:
                    s = edge_sums[best_idx]
                    p = edge_prods[best_idx]
                    sum_counts[s] += 1
                    prod_counts[p] += 1
                    current_edges_set.add(best_idx)
                    remaining.remove(best_idx)

        elif restart_count == 2:
            # Random initialization
            current_edges_set = set(rng.sample(range(num_edges), m))
            sum_counts = Counter()
            prod_counts = Counter()
            for idx in current_edges_set:
                sum_counts[edge_sums[idx]] += 1
                prod_counts[edge_prods[idx]] += 1

        else:
            # Perturbation of best found so far
            if best_edges_set is None:
                continue
            current_edges_set = set(best_edges_set)
            sum_counts = Counter()
            prod_counts = Counter()
            for idx in current_edges_set:
                sum_counts[edge_sums[idx]] += 1
                prod_counts[edge_prods[idx]] += 1

            # Randomly swap k edges
            k = 2 + (restart_count % 4)
            for _ in range(k):
                if not current_edges_set:
                    continue
                in_list = list(current_edges_set)
                out_list = [i for i in range(num_edges) if i not in current_edges_set]
                if not out_list:
                    continue
                idx_in = rng.choice(in_list)
                idx_out = rng.choice(out_list)

                s_in = edge_sums[idx_in]
                p_in = edge_prods[idx_in]
                s_out = edge_sums[idx_out]
                p_out = edge_prods[idx_out]

                # Remove idx_in
                sum_counts[s_in] -= 1
                if sum_counts[s_in] == 0:
                    del sum_counts[s_in]
                prod_counts[p_in] -= 1
                if prod_counts[p_in] == 0:
                    del prod_counts[p_in]
                current_edges_set.remove(idx_in)

                # Add idx_out
                sum_counts[s_out] += 1
                prod_counts[p_out] += 1
                current_edges_set.add(idx_out)

        current_score = _score_from_counts(sum_counts, prod_counts)

        if current_score < best_score:
            best_score = current_score
            best_edges_set = set(current_edges_set)

        # Simulated Annealing
        T = 5.0
        T_min = 0.01
        alpha = 0.995
        iters = 0
        max_iters = 6000
        no_improve = 0
        patience = 1000

        while iters < max_iters:
            if time.time() > deadline:
                break

            in_list = list(current_edges_set)
            out_list = [i for i in range(num_edges) if i not in current_edges_set]
            if not in_list or not out_list:
                break

            idx_in = rng.choice(in_list)
            idx_out = rng.choice(out_list)

            s_in = edge_sums[idx_in]
            p_in = edge_prods[idx_in]
            s_out = edge_sums[idx_out]
            p_out = edge_prods[idx_out]

            # Update counters
            old_s_in = sum_counts[s_in]
            old_p_in = prod_counts[p_in]
            old_s_out = sum_counts[s_out]
            old_p_out = prod_counts[p_out]

            # Remove in
            if old_s_in == 1:
                del sum_counts[s_in]
            else:
                sum_counts[s_in] = old_s_in - 1
            if old_p_in == 1:
                del prod_counts[p_in]
            else:
                prod_counts[p_in] = old_p_in - 1

            # Add out
            sum_counts[s_out] = old_s_out + 1
            prod_counts[p_out] = old_p_out + 1

            new_score = _score_from_counts(sum_counts, prod_counts)

            accept = False
            if new_score < current_score:
                accept = True
            elif new_score == current_score:
                accept = True
            else:
                delta = new_score - current_score
                accept = rng.random() < math.exp(-delta / max(T, 1e-10))

            if accept:
                current_score = new_score
                current_edges_set.remove(idx_in)
                current_edges_set.add(idx_out)
                no_improve = 0
                if current_score < best_score:
                    best_score = current_score
                    best_edges_set = set(current_edges_set)
            else:
                # Revert
                if old_s_out == 0:
                    del sum_counts[s_out]
                else:
                    sum_counts[s_out] = old_s_out
                if old_p_out == 0:
                    del sum_counts[p_out]
                else:
                    prod_counts[p_out] = old_p_out
                sum_counts[s_in] = old_s_in
                prod_counts[p_in] = old_p_in
                no_improve += 1

            if no_improve > patience:
                break

            T *= alpha
            if T < T_min:
                T = T_min
            iters += 1

        # Polish phase: Greedy local search with 1-swap and 2-swap moves
        if best_edges_set and time.time() < deadline:
            current_edges_set = set(best_edges_set)
            sum_counts = Counter()
            prod_counts = Counter()
            for idx in current_edges_set:
                sum_counts[edge_sums[idx]] += 1
                prod_counts[edge_prods[idx]] += 1
            current_score = _score_from_counts(sum_counts, prod_counts)

            improved = True
            while improved and time.time() < deadline:
                improved = False
                in_list = list(current_edges_set)
                out_list = [i for i in range(num_edges) if i not in current_edges_set]

                # 1-swap moves
                for idx_in in in_list:
                    for idx_out in out_list:
                        s_in = edge_sums[idx_in]
                        p_in = edge_prods[idx_in]
                        s_out = edge_sums[idx_out]
                        p_out = edge_prods[idx_out]

                        old_s_in = sum_counts[s_in]
                        old_p_in = prod_counts[p_in]
                        old_s_out = sum_counts[s_out]
                        old_p_out = prod_counts[p_out]

                        if old_s_in == 1:
                            del sum_counts[s_in]
                        else:
                            sum_counts[s_in] = old_s_in - 1
                        if old_p_in == 1:
                            del prod_counts[p_in]
                        else:
                            prod_counts[p_in] = old_p_in - 1

                        sum_counts[s_out] = old_s_out + 1
                        prod_counts[p_out] = old_p_out + 1

                        new_score = _score_from_counts(sum_counts, prod_counts)

                        if new_score < current_score:
                            current_score = new_score
                            current_edges_set.remove(idx_in)
                            current_edges_set.add(idx_out)
                            best_score = new_score
                            best_edges_set = set(current_edges_set)
                            improved = True
                            break
                        else:
                            # Revert
                            if old_s_out == 0:
                                del sum_counts[s_out]
                            else:
                                sum_counts[s_out] = old_s_out
                            if old_p_out == 0:
                                del sum_counts[p_out]
                            else:
                                prod_counts[p_out] = old_p_out
                            sum_counts[s_in] = old_s_in
                            prod_counts[p_in] = old_p_in
                    if improved:
                        break

                # 2-swap moves if time permits
                if not improved and time.time() < deadline - 0.5:
                    in_list = list(current_edges_set)
                    out_list = [i for i in range(num_edges) if i not in current_edges_set]
                    # Sample a subset of 2-swap pairs for speed
                    sample_size = min(len(in_list), 15)
                    sample_in = rng.sample(in_list, sample_size)
                    sample_out = rng.sample(out_list, min(len(out_list), 15))

                    for idx_in1 in sample_in:
                        for idx_in2 in sample_in:
                            if idx_in1 >= idx_in2:
                                continue
                            for idx_out1 in sample_out:
                                for idx_out2 in sample_out:
                                    if idx_out1 >= idx_out2:
                                        continue
                                    if idx_out1 in current_edges_set or idx_out2 in current_edges_set:
                                        continue

                                    s_in1 = edge_sums[idx_in1]
                                    p_in1 = edge_prods[idx_in1]
                                    s_in2 = edge_sums[idx_in2]
                                    p_in2 = edge_prods[idx_in2]
                                    s_out1 = edge_sums[idx_out1]
                                    p_out1 = edge_prods[idx_out1]
                                    s_out2 = edge_sums[idx_out2]
                                    p_out2 = edge_prods[idx_out2]

                                    # Save state
                                    old_s_in1 = sum_counts[s_in1]
                                    old_p_in1 = prod_counts[p_in1]
                                    old_s_in2 = sum_counts[s_in2]
                                    old_p_in2 = prod_counts[p_in2]
                                    old_s_out1 = sum_counts[s_out1]
                                    old_p_out1 = prod_counts[p_out1]
                                    old_s_out2 = sum_counts[s_out2]
                                    old_p_out2 = prod_counts[p_out2]

                                    # Remove in1, in2
                                    if old_s_in1 == 1:
                                        del sum_counts[s_in1]
                                    else:
                                        sum_counts[s_in1] = old_s_in1 - 1
                                    if old_p_in1 == 1:
                                        del prod_counts[p_in1]
                                    else:
                                        prod_counts[p_in1] = old_p_in1 - 1
                                    if old_s_in2 == 1:
                                        del sum_counts[s_in2]
                                    else:
                                        sum_counts[s_in2] = old_s_in2 - 1
                                    if old_p_in2 == 1:
                                        del prod_counts[p_in2]
                                    else:
                                        prod_counts[p_in2] = old_p_in2 - 1

                                    # Add out1, out2
                                    sum_counts[s_out1] = old_s_out1 + 1
                                    prod_counts[p_out1] = old_p_out1 + 1
                                    sum_counts[s_out2] = old_s_out2 + 1
                                    prod_counts[p_out2] = old_p_out2 + 1

                                    new_score = _score_from_counts(sum_counts, prod_counts)

                                    if new_score < current_score:
                                        current_score = new_score
                                        current_edges_set.discard(idx_in1)
                                        current_edges_set.discard(idx_in2)
                                        current_edges_set.add(idx_out1)
                                        current_edges_set.add(idx_out2)
                                        best_score = new_score
                                        best_edges_set = set(current_edges_set)
                                        improved = True
                                        break
                                    else:
                                        # Revert
                                        if old_s_out1 == 0:
                                            del sum_counts[s_out1]
                                        else:
                                            sum_counts[s_out1] = old_s_out1
                                        if old_p_out1 == 0:
                                            del sum_counts[p_out1]
                                        else:
                                            prod_counts[p_out1] = old_p_out1
                                        if old_s_out2 == 0:
                                            del sum_counts[s_out2]
                                        else:
                                            sum_counts[s_out2] = old_s_out2
                                        if old_p_out2 == 0:
                                            del sum_counts[p_out2]
                                        else:
                                            prod_counts[p_out2] = old_p_out2
                                        sum_counts[s_in1] = old_s_in1
                                        prod_counts[p_in1] = old_p_in1
                                        sum_counts[s_in2] = old_s_in2
                                        prod_counts[p_in2] = old_p_in2
                                if improved:
                                    break
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break

    if best_edges_set is None:
        # Fallback
        best_edges_set = set(range(m))

    return [all_edges[idx] for idx in best_edges_set], best_score


def construct():
    start_time = time.time()
    n = 14
    m = 35

    # Candidate value sets - focus on sets with high collision potential
    # Powers of 2 give many product overlaps; small values give sum overlaps
    candidates = [
        # Powers of 2: products are unique but sums have structure
        [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192],
        # Small consecutive
        list(range(1, 15)),
        list(range(2, 16)),
        # Multiples of small numbers
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 18, 24],
        [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32, 36],
        [1, 2, 3, 4, 6, 8, 9, 12, 15, 16, 18, 20, 24, 27],
        [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28],
        [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20],
        [1, 2, 3, 4, 6, 9, 12, 16, 18, 24, 27, 32, 36, 48],
        [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 36, 48],
        [1, 2, 3, 4, 6, 8, 9, 10, 12, 15, 18, 20, 24, 27],
        [1, 2, 3, 4, 6, 9, 12, 15, 18, 20, 24, 27, 30, 36],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        [1, 2, 3, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24],
        [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 18, 20, 24, 25],
        [2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20, 24],
        [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32, 40],
        [1, 2, 3, 4, 6, 8, 9, 12, 15, 18, 20, 24, 25, 30],
        # Highly composite / many factors
        [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128],
        [1, 2, 3, 4, 6, 8, 9, 12, 18, 24, 36, 48, 72, 96],
        # Geometric progressions
        [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192],
        [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384][:14],
        # Mixed: some small, some structured
        [1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 24, 30, 36, 48],
        [1, 2, 3, 4, 6, 10, 12, 15, 20, 24, 30, 36, 40, 48],
    ]

    # Generate random sets with common factors
    rng = random.Random(123)
    for _ in range(15):
        base = rng.randint(1, 4)
        step = rng.randint(1, 2)
        vals = [base + i * step for i in range(n)]
        if len(set(vals)) == n and max(vals) <= 10000:
            candidates.append(vals)

    # Generate sets based on multiples
    for base in [1, 2, 3]:
        vals = [base * (i + 1) for i in range(n)]
        if max(vals) <= 10000:
            candidates.append(vals)

    # Generate sets with many small factors
    for _ in range(15):
        vals = []
        for _ in range(n):
            v = rng.randint(1, 40)
            while v in vals:
                v = rng.randint(1, 40)
            vals.append(v)
        if len(set(vals)) == n and max(vals) <= 10000:
            candidates.append(vals)

    best_score = float('inf')
    best_result = None

    # Sort candidates by a heuristic: prefer smaller values (more sum collisions)
    candidates.sort(key=lambda x: sum(x))

    # Allocate time: give more time to promising candidates
    # First pass: quick evaluation of all
    # Second pass: deeper search on top candidates

    # Quick pass: evaluate each candidate with limited time
    quick_deadline = start_time + 15.0
    for values in candidates:
        if time.time() > quick_deadline:
            break

        if len(values) != n or len(set(values)) != n:
            continue
        if any(v < 1 or v > 10000 for v in values):
            continue

        deadline_for_this = min(quick_deadline, time.time() + 2.0)

        try:
            edges, score = _optimize_edges(values, m, deadline_for_this)
        except Exception:
            continue

        if score is not None and score < best_score:
            best_score = score
            best_result = {
                'values': [int(v) for v in values],
                'edges': [[int(i), int(j)] for i, j in edges]
            }

    # Deep pass: focus on the best few candidates with more time
    if best_result is not None and time.time() < DEADLINE - 5:
        # Re-optimize the best candidate with more time
        deadline_for_deep = DEADLINE - 2
        try:
            edges, score = _optimize_edges(best_result['values'], m, deadline_for_deep)
            if score is not None and score < best_score:
                best_score = score
                best_result = {
                    'values': [int(v) for v in best_result['values']],
                    'edges': [[int(i), int(j)] for i, j in edges]
                }
        except Exception:
            pass

    if best_result is None:
        values = list(range(1, n + 1))
        edges = [list(e) for e in list(combinations(range(n), 2))[:m]]
        best_result = {'values': values, 'edges': edges}

    return best_result
