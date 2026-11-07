"""Search-based: minimize max(distinct_sums, distinct_products) over 35 edges of 14 vertices."""
# EVOLVE-BLOCK-START
import time
import itertools
import random

DEADLINE = 85.0  # safety margin before 90s kill

def _raw_score(values, edges):
    """Compute max(#distinct sums, #distinct products) over edges."""
    sums = set()
    prods = set()
    for u, v in edges:
        a, b = values[u], values[v]
        sums.add(a + b)
        prods.add(a * b)
    return max(len(sums), len(prods))

def _all_edges(n):
    return list(itertools.combinations(range(n), 2))

def _greedy_edges(values, target_m):
    """Greedy edge selection with O(1) set-based membership check."""
    n = len(values)
    all_e = _all_edges(n)
    sums = set()
    prods = set()
    edges = []
    edge_set = set()
    for _ in range(target_m):
        best_edge = None
        best_score = float('inf')
        for (u, v) in all_e:
            if (u, v) in edge_set:
                continue
            a, b = values[u], values[v]
            s = a + b
            p = a * b
            new_s = 1 if s not in sums else 0
            new_p = 1 if p not in prods else 0
            score = max(len(sums) + new_s, len(prods) + new_p)
            if score < best_score:
                best_score = score
                best_edge = (u, v)
        if best_edge is None:
            break
        u, v = best_edge
        edges.append((u, v))
        edge_set.add((u, v))
        sums.add(values[u] + values[v])
        prods.add(values[u] * values[v])
    return edges

def _local_search_edges(values, edges, max_iters=50):
    """Local search: try swapping edges to reduce distinct sums/products."""
    n = len(values)
    all_e = _all_edges(n)
    edge_set = set(edges)
    current_score = _raw_score(values, edges)
    
    for _ in range(max_iters):
        improved = False
        # Try removing one edge and adding another
        for idx, (u, v) in enumerate(edges):
            # Compute state without this edge
            sums = set()
            prods = set()
            for (a, b) in edges:
                if (a, b) == (u, v):
                    continue
                sums.add(values[a] + values[b])
                prods.add(values[a] * values[b])
            
            # Try adding each non-edge
            for (x, y) in all_e:
                if (x, y) in edge_set:
                    continue
                s = values[x] + values[y]
                p = values[x] * values[y]
                new_sums = sums | {s}
                new_prods = prods | {p}
                new_score = max(len(new_sums), len(new_prods))
                if new_score < current_score:
                    # Accept the swap
                    edges[idx] = (x, y)
                    edge_set.discard((u, v))
                    edge_set.add((x, y))
                    current_score = new_score
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    
    return edges, current_score

def _evaluate(values):
    """Evaluate a value set with greedy edge selection plus local search polish."""
    edges = _greedy_edges(values, 35)
    if len(edges) < 35:
        return float('inf'), values, edges
    # Polish with local search
    edges, score = _local_search_edges(values, edges, max_iters=30)
    return score, values, edges

def _try_value_set(values):
    """Try a specific value set."""
    if len(set(values)) != 14:
        return None
    if any(not isinstance(v, int) or v < 1 or v > 10000 for v in values):
        return None
    score, vals, edges = _evaluate(values)
    return score, vals, edges

def construct():
    start_time = time.time()
    best_score = float('inf')
    best_result = None
    
    # Strategy 1: Arithmetic progression (minimizes distinct sums)
    for a in range(1, 50):
        for d in range(1, 20):
            vals = [a + i * d for i in range(14)]
            if vals[-1] > 10000:
                break
            if len(set(vals)) != 14:
                continue
            result = _try_value_set(vals)
            if result:
                s, v, e = result
                if s < best_score:
                    best_score = s
                    best_result = (v, e)
            if time.time() > DEADLINE:
                break
        if time.time() > DEADLINE:
            break
    
    # Strategy 2: Values with common factor g, quotients form arithmetic progression
    for g in range(2, 30):
        for a in range(1, 20):
            for d in range(1, 10):
                qs = [a + i * d for i in range(14)]
                vals = [g * q for q in qs]
                if vals[-1] > 10000:
                    break
                if len(set(vals)) != 14:
                    continue
                result = _try_value_set(vals)
                if result:
                    s, v, e = result
                    if s < best_score:
                        best_score = s
                        best_result = (v, e)
                if time.time() > DEADLINE:
                    break
            if time.time() > DEADLINE:
                break
        if time.time() > DEADLINE:
            break
    
    # Strategy 3: Random search in small ranges (increases collision probability)
    if time.time() < DEADLINE - 5:
        random.seed(42)
        for max_val in [20, 30, 50, 100, 200]:
            if time.time() > DEADLINE - 5:
                break
            for _ in range(200):
                if time.time() > DEADLINE - 5:
                    break
                if max_val < 14:
                    continue
                vals = random.sample(range(1, max_val + 1), 14)
                result = _try_value_set(vals)
                if result:
                    s, v, e = result
                    if s < best_score:
                        best_score = s
                        best_result = (v, e)
    
    # Strategy 4: Try structured sets with small common factors
    if time.time() < DEADLINE - 5:
        for g in [2, 3, 4, 5, 6]:
            for a in range(1, 10):
                for d in range(1, 5):
                    qs = [a + i * d for i in range(14)]
                    vals = [g * q for q in qs]
                    if max(vals) > 10000:
                        continue
                    if len(set(vals)) != 14:
                        continue
                    result = _try_value_set(vals)
                    if result:
                        s, v, e = result
                        if s < best_score:
                            best_score = s
                            best_result = (v, e)
            if time.time() > DEADLINE - 5:
                break
    
    if best_result is None:
        vals = list(range(1, 15))
        edges = list(itertools.combinations(range(14), 2))[:35]
        return {'values': vals, 'edges': [list(e) for e in edges]}
    
    vals, edges = best_result
    vals = [int(v) for v in vals]
    edges = [list(e) for e in edges]
    return {'values': vals, 'edges': edges}

# EVOLVE-BLOCK-END
