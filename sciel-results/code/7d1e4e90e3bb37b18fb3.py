import random
import time
import math
from itertools import combinations

def _evaluate(values, edges):
    sums = set()
    prods = set()
    for (i, j) in edges:
        a = values[i]
        b = values[j]
        sums.add(a + b)
        prods.add(a * b)
    return max(len(sums), len(prods))

def _build_candidates(n, max_val):
    # Small values create many sum collisions: 1..14 has 13 distinct sums
    # Products 1..14 have 91 distinct products, so we need to pick edges
    # that minimize distinct products
    return list(range(1, n + 1))

def _random_edges(n, m, rng):
    all_edges = list(combinations(range(n), 2))
    rng.shuffle(all_edges)
    return [list(e) for e in all_edges[:m]]

def _edge_swap(edges, all_edges_set, rng):
    """Swap one edge for a random different edge."""
    i = rng.randrange(len(edges))
    e = tuple(edges[i])
    # Pick a random edge not currently in the set
    while True:
        a = rng.randrange(14)
        b = rng.randrange(14)
        if a == b:
            continue
        cand = (min(a, b), max(a, b))
        if cand in all_edges_set:
            continue
        break
    new_edges = list(edges)
    all_edges_set.discard(e)
    all_edges_set.add(cand)
    new_edges[i] = list(cand)
    return new_edges

def _value_swap(values, rng):
    """Swap two values in the array."""
    v = list(values)
    i, j = rng.sample(range(14), 2)
    v[i], v[j] = v[j], v[i]
    return v

def _value_perturb(values, rng):
    """Change one value to a small random number, ensuring distinctness."""
    v = list(values)
    i = rng.randrange(14)
    # Try to set to a small value not already used
    used = set(v)
    used.discard(v[i])
    candidates = [x for x in range(1, 30) if x not in used]
    if not candidates:
        return None
    v[i] = rng.choice(candidates)
    return v

def construct():
    start = time.time()
    deadline = start + 55  # safety margin before 60s

    rng = random.Random(42)

    n, m = 14, 35

    # Start with small values 1..14 which give minimal distinct sums (13)
    best_values = list(range(1, n + 1))
    best_edges = _random_edges(n, m, rng)
    best_score = _evaluate(best_values, best_edges)

    # Greedy initial edge selection: prefer edges that reuse existing sums/products
    def greedy_edges(values, m):
        all_e = list(combinations(range(n), 2))
        rng.shuffle(all_e)
        sums = set()
        prods = set()
        chosen = []
        # Sort by how "colliding" they are - prefer pairs with common sums/prods
        # First pass: just take all, then try to improve
        chosen = [list(e) for e in all_e[:m]]
        return chosen

    best_edges = greedy_edges(best_values, m)
    best_score = _evaluate(best_values, best_edges)

    # Simulated annealing on edges (values fixed at 1..14 for sum optimality)
    current_values = best_values[:]
    current_edges = best_edges[:]
    current_score = best_score

    all_edges_set = set(tuple(e) for e in current_edges)

    T = 50.0
    T_min = 0.01
    cooling = 0.995

    steps = 0
    while time.time() < deadline and T > T_min:
        # Edge swap
        new_edges = _edge_swap(current_edges, all_edges_set, rng)
        new_score = _evaluate(current_values, new_edges)

        if new_score <= current_score:
            current_edges = new_edges
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_edges = new_edges[:]
        else:
            # Accept with probability exp(-delta/T)
            delta = new_score - current_score
            if rng.random() < math.exp(-delta / max(T, 0.001)):
                current_edges = new_edges
                current_score = new_score

        # Occasionally perturb values
        if steps % 50 == 0 and time.time() < deadline - 5:
            new_vals = _value_perturb(current_values, rng)
            if new_vals is not None:
                nv_score = _evaluate(new_vals, current_edges)
                if nv_score <= current_score:
                    current_values = new_vals
                    current_score = nv_score
                    if nv_score < best_score:
                        best_score = nv_score
                        best_values = new_vals[:]
                elif rng.random() < math.exp(-(nv_score - current_score) / max(T, 0.001)):
                    current_values = new_vals
                    current_score = nv_score

        T *= cooling
        steps += 1

    # Final polish: try to improve best with local search
    if time.time() < deadline - 3:
        # Try different initial value sets
        alt_value_sets = [
            list(range(1, 15)),          # 1..14
            list(range(2, 16)),          # 2..15
            [1,2,3,4,5,6,7,8,9,10,11,12,13,14],
            [1,2,3,4,5,6,7,8,9,10,11,12,14,16],  # skip 13, use 16
        ]
        for vals in alt_value_sets:
            if time.time() >= deadline - 2:
                break
            if len(set(vals)) != 14 or any(v < 1 or v > 10000 for v in vals):
                continue
            e = _random_edges(n, m, rng)
            s = _evaluate(vals, e)
            if s < best_score:
                best_score = s
                best_values = vals[:]
                best_edges = e[:]

    # Ensure types are correct
    result = {
        'values': [int(v) for v in best_values],
        'edges': [[int(i), int(j)] for i, j in best_edges]
    }
    return result
