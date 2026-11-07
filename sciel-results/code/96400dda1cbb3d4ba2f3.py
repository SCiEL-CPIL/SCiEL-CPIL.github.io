"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations
from collections import Counter

P = {'n': 14, 'm': 35, 'maximum': 10000}

def construct():
    deadline = time.time() + 50
    n = P['n']
    m = P['m']
    all_edges = list(combinations(range(n), 2))

    def evaluate(values, edges):
        sums = set()
        prods = set()
        for i, j in edges:
            sums.add(values[i] + values[j])
            prods.add(values[i] * values[j])
        return max(len(sums), len(prods))

    def compute_edge_maps(values):
        edge_sums = {}
        edge_prods = {}
        for i, j in all_edges:
            edge_sums[(i, j)] = values[i] + values[j]
            edge_prods[(i, j)] = values[i] * values[j]
        return edge_sums, edge_prods

    def greedy_edges(values, edge_sums, edge_prods):
        sum_freq = Counter(edge_sums.values())
        prod_freq = Counter(edge_prods.values())
        sum_count = Counter()
        prod_count = Counter()
        edges = []
        edge_set = set()
        remaining = set(all_edges)
        for _ in range(m):
            best_e = None
            best_key = None
            for e in remaining:
                s = edge_sums[e]
                p = edge_prods[e]
                new_s = 0 if sum_count[s] > 0 else 1
                new_p = 0 if prod_count[p] > 0 else 1
                key = (new_s + new_p, -sum_freq[s], -prod_freq[p])
                if best_key is None or key < best_key:
                    best_key = key
                    best_e = e
            edges.append(list(best_e))
            edge_set.add(best_e)
            sum_count[edge_sums[best_e]] += 1
            prod_count[edge_prods[best_e]] += 1
            remaining.discard(best_e)
        return edges, edge_set

    def local_search(values, edges, max_iters=40000):
        edges = [list(e) for e in edges]
        edge_set = set(tuple(e) for e in edges)

        def score():
            sums = set()
            prods = set()
            for i, j in edge_set:
                sums.add(values[i] + values[j])
                prods.add(values[i] * values[j])
            return max(len(sums), len(prods))

        cur_score = score()
        best_score = cur_score
        best_edges = [list(e) for e in edges]
        best_values = list(values)

        for it in range(max_iters):
            if time.time() > deadline:
                break
            if random.random() < 0.7:
                idx = random.randint(0, m - 1)
                old_edge = tuple(edges[idx])
                new_edge = random.choice(all_edges)
                if new_edge in edge_set:
                    continue
                sums = set()
                prods = set()
                for ei, ej in edge_set:
                    if ei == old_edge[0] and ej == old_edge[1]:
                        continue
                    sums.add(values[ei] + values[ej])
                    prods.add(values[ei] * values[ej])
                sums.add(values[new_edge[0]] + values[new_edge[1]])
                prods.add(values[new_edge[0]] * values[new_edge[1]])
                new_score = max(len(sums), len(prods))
                if new_score <= cur_score:
                    edges[idx] = list(new_edge)
                    edge_set.remove(old_edge)
                    edge_set.add(new_edge)
                    cur_score = new_score
                    if cur_score < best_score:
                        best_score = cur_score
                        best_edges = [list(e) for e in edges]
                        best_values = list(values)
            else:
                v_idx = random.randint(0, n - 1)
                old_val = values[v_idx]
                nv = random.randint(1, P['maximum'])
                if nv == old_val or nv in values:
                    continue
                values[v_idx] = nv
                new_score = score()
                if new_score <= cur_score:
                    cur_score = new_score
                    if cur_score < best_score:
                        best_score = cur_score
                        best_edges = [list(e) for e in edges]
                        best_values = list(values)
                else:
                    values[v_idx] = old_val

        return best_values, best_edges, best_score

    best_overall_score = float('inf')
    best_overall_values = None
    best_overall_edges = None

    structures = [
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20],
        [1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 24, 30, 40, 60],
        [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192],
        [3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42],
    ]

    for init_vals in structures:
        if time.time() > deadline - 8:
            break
        if len(set(init_vals)) != n or any(v < 1 or v > P['maximum'] for v in init_vals):
            continue
        edge_sums, edge_prods = compute_edge_maps(init_vals)
        edges, edge_set = greedy_edges(init_vals, edge_sums, edge_prods)
        vals, eds, sc = local_search(init_vals, edges, max_iters=25000)
        if sc < best_overall_score:
            best_overall_score = sc
            best_overall_values = vals
            best_overall_edges = eds

    if best_overall_values is None:
        best_overall_values = list(range(1, 15))
        edge_sums, edge_prods = compute_edge_maps(best_overall_values)
        best_overall_edges, _ = greedy_edges(best_overall_values, edge_sums, edge_prods)
        best_overall_score = evaluate(best_overall_values, best_overall_edges)

    final_values = [int(v) for v in best_overall_values]
    final_edges = [[int(i), int(j)] for i, j in best_overall_edges]

    assert len(final_values) == n
    assert len(set(final_values)) == n
    assert all(isinstance(v, int) and 1 <= v <= P['maximum'] for v in final_values)
    assert len(final_edges) == m
    edge_tuples = set()
    for e in final_edges:
        i, j = e
        assert 0 <= i < n and 0 <= j < n and i != j
        t = tuple(sorted((i, j)))
        assert t not in edge_tuples
        edge_tuples.add(t)

    return {'values': final_values, 'edges': final_edges}

# EVOLVE-BLOCK-END
