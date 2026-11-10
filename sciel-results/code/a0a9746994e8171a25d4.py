"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _raw_value(values, edges):
    sums = set()
    prods = set()
    for i, j in edges:
        sums.add(values[i] + values[j])
        prods.add(values[i] * values[j])
    return max(len(sums), len(prods))

def _generate_value_candidates():
    """Generate candidate value sets with structure to encourage collisions."""
    candidates = []
    
    # Strategy 1: Symmetric pairs around a center for sum collisions
    # Pairs (a, S-a) all have sum S
    for S in range(100, 500, 10):
        for spread in range(5, 50, 5):
            vals = set()
            # Add symmetric pairs
            for i in range(7):
                a = S // 2 - spread + i * 2
                b = S - a
                if a > 0 and b > 0 and a != b and a <= 10000 and b <= 10000:
                    vals.add(a)
                    vals.add(b)
            if len(vals) == 14:
                candidates.append(sorted(vals))
    
    # Strategy 2: Arithmetic progression
    for start in range(1, 100):
        for step in range(1, 20):
            vals = [start + i * step for i in range(14)]
            if all(0 < v <= 10000 for v in vals) and len(set(vals)) == 14:
                candidates.append(vals)
    
    # Strategy 3: Mix of small values with product structure
    # e.g., {2,3,4,6,8,9,12,16,18,24,27,32,36,48} - powers and multiples
    base_sets = [
        [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32, 36],
        [2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32, 36, 48],
        [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192],
    ]
    candidates.extend(base_sets)
    
    # Strategy 4: Values with many equal-sum pairs
    # Group into pairs that sum to same value
    for S in [50, 80, 100, 120, 150, 200]:
        pairs = []
        for a in range(1, S // 2):
            b = S - a
            if a < b and 0 < a <= 10000 and 0 < b <= 10000:
                pairs.append((a, b))
        if len(pairs) >= 7:
            # Pick 7 pairs
            for combo_idx in range(min(10, len(pairs) - 6)):
                selected = pairs[combo_idx:combo_idx + 7]
                vals = []
                for a, b in selected:
                    vals.extend([a, b])
                if len(set(vals)) == 14:
                    candidates.append(sorted(vals))
    
    return candidates

def _select_edges_local_search(values, m=35, n=14, max_iter=5000, seed=42):
    """Select m edges to minimize max(distinct_sums, distinct_prods) via local search."""
    rng = random.Random(seed)
    all_edges = list(combinations(range(n), 2))
    
    # Initial: random selection
    edges = set(rng.sample(all_edges, m))
    
    def score(edge_set):
        s = set()
        p = set()
        for i, j in edge_set:
            s.add(values[i] + values[j])
            p.add(values[i] * values[j])
        return max(len(s), len(p))
    
    best_score = score(edges)
    best_edges = set(edges)
    
    for it in range(max_iter):
        # Try swapping one edge
        edge_to_remove = rng.choice(list(edges))
        edge_to_add = rng.choice(all_edges)
        if edge_to_add in edges:
            continue
        new_edges = edges - {edge_to_remove} | {edge_to_add}
        sc = score(new_edges)
        if sc < best_score:
            best_score = sc
            best_edges = new_edges
            edges = new_edges
        elif sc == best_score and rng.random() < 0.3:
            # Accept equal to escape local optima
            edges = new_edges
    
    return list(best_edges), best_score

def construct():
    start_time = time.time()
    deadline = start_time + 75.0  # Safety margin
    
    n = 14
    m = 35
    
    best_result = None
    best_score = float('inf')
    
    # Generate candidates
    candidates = _generate_value_candidates()
    random.shuffle(candidates)
    
    for vals in candidates:
        if time.time() > deadline:
            break
        if len(set(vals)) != n or any(v <= 0 or v > 10000 for v in vals):
            continue
        if not all(isinstance(v, int) for v in vals):
            continue
        
        # Try local search with a few different seeds
        for seed in [42, 123, 456]:
            if time.time() > deadline:
                break
            edges, sc = _select_edges_local_search(vals, m=m, n=n, max_iter=3000, seed=seed)
            if sc < best_score:
                best_score = sc
                best_result = {'values': list(vals), 'edges': [list(e) for e in edges]}
    
    # If no valid result, fall back to simple
    if best_result is None:
        vals = list(range(1, n + 1))
        edges = [list(e) for e in list(combinations(range(n), 2))[:m]]
        best_result = {'values': vals, 'edges': edges}
    
    # Ensure types are correct
    best_result['values'] = [int(v) for v in best_result['values']]
    best_result['edges'] = [[int(i), int(j)] for i, j in best_result['edges']]
    
    return best_result
# EVOLVE-BLOCK-END

