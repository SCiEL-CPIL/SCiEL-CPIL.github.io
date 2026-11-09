"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
from itertools import combinations

N = 14
M = 35
MAXV = 10000
ALL_EDGES = list(combinations(range(N), 2))

def _score(values, edges):
    sums = set()
    prods = set()
    for i, j in edges:
        a = values[i]
        b = values[j]
        sums.add(a + b)
        prods.add(a * b)
    return max(len(sums), len(prods))

def _edge_score(values, edges):
    # returns (distinct_sums, distinct_prods)
    sums = set()
    prods = set()
    for i, j in edges:
        sums.add(values[i] + values[j])
        prods.add(values[i] * values[j])
    return len(sums), len(prods)

def _greedy_edges(values):
    """Pick M edges greedily to minimize distinct sums/prods."""
    # Precompute sum and product for each edge
    edge_data = []
    for i, j in ALL_EDGES:
        s = values[i] + values[j]
        p = values[i] * values[j]
        edge_data.append((i, j, s, p))
    
    # Greedy: pick edge that adds fewest new distinct sums/prods
    # Prefer edges whose sum and product are already seen
    chosen = []
    chosen_sums = set()
    chosen_prods = set()
    remaining = list(range(len(ALL_EDGES)))
    random.shuffle(remaining)
    
    # First pass: sort by "popularity" - how many other edges share same sum or product
    sum_counts = {}
    prod_counts = {}
    for idx, (i, j, s, p) in enumerate(edge_data):
        sum_counts[s] = sum_counts.get(s, 0) + 1
        prod_counts[p] = prod_counts.get(p, 0) + 1
    
    # Greedy selection
    for _ in range(M):
        best_idx = -1
        best_new = (10**9, 10**9)
        # Try edges in order of high sum_count + prod_count first (more collisions)
        # But we need to be careful about time
        # Just iterate and pick best
        for idx in remaining:
            i, j, s, p = edge_data[idx]
            new_s = 0 if s in chosen_sums else 1
            new_p = 0 if p in chosen_prods else 1
            # We want to minimize max, so minimize the worse one
            score = max(new_s, new_p)
            # Tiebreak: prefer edges that add to both
            if score < best_new[0] or (score == best_new[0] and (new_s + new_p) < best_new[1]):
                best_new = (score, new_s + new_p)
                best_idx = idx
        if best_idx == -1:
            break
        i, j, s, p = edge_data[best_idx]
        chosen.append([i, j])
        chosen_sums.add(s)
        chosen_prods.add(p)
        remaining.remove(best_idx)
    
    return chosen

def _local_search_edges(values, edges, max_iters=500):
    """Local search: swap one edge for another to reduce score."""
    current = _score(values, edges)
    edge_set = set(tuple(sorted(e)) for e in edges)
    edge_list = list(edge_set)
    
    for it in range(max_iters):
        improved = False
        random.shuffle(edge_list)
        for e_out in edge_list:
            if e_out not in edge_set:
                continue
            # Try removing e_out and adding a random other edge
            for _ in range(5):
                e_in = random.choice(ALL_EDGES)
                if e_in in edge_set or e_in == e_out:
                    continue
                new_edges = [list(e) for e in edge_set if e != e_out]
                new_edges.append(list(e_in))
                new_score = _score(values, new_edges)
                if new_score < current:
                    edge_set.discard(e_out)
                    edge_set.add(e_in)
                    current = new_score
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    return [[i, j] for i, j in edge_set]

def _make_values(pattern, base, step, n=N):
    """Generate n distinct positive integers from a pattern."""
    vals = []
    for k in range(n):
        v = base + k * step
        if 1 <= v <= MAXV and v not in vals:
            vals.append(v)
    return vals

def construct():
    start_time = time.time()
    deadline = start_time + 75  # 75s safety margin
    
    random.seed(42)
    
    best_values = None
    best_edges = None
    best_score = 10**9
    
    # Strategy 1: Arithmetic progression values (many sum collisions)
    # For AP with common difference d, sums are also structured
    for d in [1, 2, 3, 4, 5, 7, 10, 15, 20]:
        if time.time() > deadline:
            break
        base = 1
        vals = [base + k * d for k in range(N)]
        if vals[-1] > MAXV:
            continue
        if len(set(vals)) != N:
            continue
        edges = _greedy_edges(vals)
        edges = _local_search_edges(vals, edges, max_iters=100)
        s = _score(vals, edges)
        if s < best_score:
            best_score = s
            best_values = vals[:]
            best_edges = [e[:] for e in edges]
    
    # Strategy 2: Geometric-like (products collide)
    # Use powers or near-powers
    for base_g in [2, 3]:
        if time.time() > deadline:
            break
        # Geometric: base_g^k might exceed 10000 quickly
        # Instead use mixed: some small numbers that multiply to same products
        # E.g., {1,2,3,4,6,12,...} divisors
        # Try sets with many multiplicative relations
        candidate_sets = [
            [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32, 36],
            [1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 24, 30, 40, 48],
            [2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32, 36, 48],
            [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192],
        ]
        for vals in candidate_sets:
            if len(set(vals)) != N or any(v < 1 or v > MAXV for v in vals):
                continue
            edges = _greedy_edges(vals)
            edges = _local_search_edges(vals, edges, max_iters=100)
            s = _score(vals, edges)
            if s < best_score:
                best_score = s
                best_values = vals[:]
                best_edges = [e[:] for e in edges]
    
    # Strategy 3: Random search with local improvement
    if time.time() < deadline - 5:
        for attempt in range(200):
            if time.time() > deadline:
                break
            # Random values with structure: pairs that sum to same value
            # Pick 7 pairs all summing to S
            S = random.randint(10, 50)
            vals = []
            for _ in range(7):
                a = random.randint(1, S - 1)
                b = S - a
                if a != b and a not in vals and b not in vals and 1 <= a <= MAXV and 1 <= b <= MAXV:
                    vals.append(a)
                    vals.append(b)
            if len(vals) != N:
                # Fill with random
                while len(vals) < N:
                    v = random.randint(1, MAXV)
                    if v not in vals:
                        vals.append(v)
                vals = vals[:N]
            
            edges = _greedy_edges(vals)
            edges = _local_search_edges(vals, edges, max_iters=50)
            s = _score(vals, edges)
            if s < best_score:
                best_score = s
                best_values = vals[:]
                best_edges = [e[:] for e in edges]
    
    # Final local search on best
    if best_values is not None and time.time() < deadline - 2:
        best_edges = _local_search_edges(best_values, best_edges, max_iters=200)
        best_score = _score(best_values, best_edges)
    
    # Ensure types are correct
    values = [int(v) for v in best_values]
    edges = [[int(i), int(j)] for i, j in best_edges]
    
    return {'values': values, 'edges': edges}
# EVOLVE-BLOCK-END
