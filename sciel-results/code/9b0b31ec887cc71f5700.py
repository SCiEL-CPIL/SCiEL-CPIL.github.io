import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    timeout = 55.0  # Safety margin

    n = 14
    m = 35
    max_val = 10000

    all_pairs = list(combinations(range(n), 2))

    def evaluate(values, edges):
        sums = set()
        prods = set()
        for u, v in edges:
            sums.add(values[u] + values[v])
            prods.add(values[u] * values[v])
        return max(len(sums), len(prods))

    def greedy_select_edges(values):
        """Fast greedy: pick edges that minimize growth in max(distinct sums, distinct prods)."""
        # Precompute
        pair_sums = {}
        pair_prods = {}
        for u, v in all_pairs:
            pair_sums[(u, v)] = values[u] + values[v]
            pair_prods[(u, v)] = values[u] * values[v]

        selected = set()
        current_sums = set()
        current_prods = set()
        edges = []

        for _ in range(m):
            best_edge = None
            best_score = float('inf')
            best_s = None
            best_p = None

            for u, v in all_pairs:
                if (u, v) in selected:
                    continue
                s = pair_sums[(u, v)]
                p = pair_prods[(u, v)]
                add_s = 0 if s in current_sums else 1
                add_p = 0 if p in current_prods else 1
                new_sums = len(current_sums) + add_s
                new_prods = len(current_prods) + add_p
                score = max(new_sums, new_prods)
                if score < best_score:
                    best_score = score
                    best_edge = (u, v)
                    best_s = s
                    best_p = p

            if best_edge:
                selected.add(best_edge)
                current_sums.add(best_s)
                current_prods.add(best_p)
                edges.append(best_edge)

        return edges

    def try_values(values):
        """Try a specific value set, return (score, edges)."""
        edges = greedy_select_edges(values)
        score = evaluate(values, edges)
        return score, edges

    # Strategy 1: Consecutive small integers (maximize sum collisions)
    # For 14 distinct positive ints, consecutive starting at a minimizes sum range
    # Sums range from 2a+1 to 2a+25 (25 values)
    # But we only pick 35 edges, so we can restrict to smaller sums

    best_score = float('inf')
    best_values = None
    best_edges = None

    # Try consecutive ranges starting from 1 to 50
    for a in range(1, 51):
        if time.time() - start_time > timeout:
            break
        vals = [a + i for i in range(n)]
        if max(vals) > max_val:
            break
        score, edges = try_values(vals)
        if score < best_score:
            best_score = score
            best_values = vals[:]
            best_edges = edges[:]

    # Strategy 2: Random local search around best found
    random.seed(42)
    if best_values is None:
        best_values = list(range(1, n + 1))
        best_score, best_edges = try_values(best_values)

    # Local search: perturb one value at a time
    while time.time() - start_time < timeout:
        idx = random.randint(0, n - 1)
        old_val = best_values[idx]

        # Try nearby values
        for delta in range(-5, 6):
            if delta == 0:
                continue
            new_val = old_val + delta
            if new_val < 1 or new_val > max_val:
                continue
            if new_val in best_values:
                continue

            new_values = best_values[:]
            new_values[idx] = new_val
            score, edges = try_values(new_values)

            if score < best_score:
                best_score = score
                best_values = new_values
                best_edges = edges
                break  # Restart delta search from new best

        # Also try random larger jumps occasionally
        if random.random() < 0.3:
            new_val = random.randint(1, 200)
            if new_val != old_val and new_val not in best_values:
                new_values = best_values[:]
                new_values[idx] = new_val
                score, edges = try_values(new_values)
                if score < best_score:
                    best_score = score
                    best_values = new_values
                    best_edges = edges

    # Final validation
    if best_values is None:
        best_values = list(range(1, n + 1))
    if best_edges is None:
        best_edges = greedy_select_edges(best_values)

    # Ensure exactly 35 edges
    if len(best_edges) != m:
        # Fallback: take first 35 from all pairs
        best_edges = all_pairs[:m]

    # Ensure types
    return {
        'values': [int(v) for v in best_values],
        'edges': [[int(u), int(v)] for u, v in best_edges]
    }
