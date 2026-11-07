"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations
import sys

def _raw_value(values, edges):
    sums = set()
    prods = set()
    for i, j in edges:
        s = values[i] + values[j]
        p = values[i] * values[j]
        sums.add(s)
        prods.add(p)
    return max(len(sums), len(prods))

def _optimize_edges(values, m=35, n=14):
    """Find m edges minimizing max(distinct_sums, distinct_products)."""
    # Precompute sum and product for each pair
    pairs = list(combinations(range(n), 2))
    pair_sums = {}
    pair_prods = {}
    for i, j in pairs:
        pair_sums[(i, j)] = values[i] + values[j]
        pair_prods[(i, j)] = values[i] * values[j]
    
    # Greedy: pick edges that minimize the growth of distinct sums and products
    # Use a scoring: for each candidate edge, count how many new sums and prods it adds
    # Pick the edge that adds the fewest new distinct values (weighted)
    
    best_edges = None
    best_score = float('inf')
    
    # Try multiple random initial selections then local search
    for attempt in range(20):
        # Random initial selection of m edges
        selected = set()
        for _ in range(m):
            remaining = [p for p in pairs if p not in selected]
            if not remaining:
                break
            # Weighted random: prefer edges whose sum/prod already appears
            current_sums = set()
            current_prods = set()
            for e in selected:
                current_sums.add(pair_sums[e])
                current_prods.add(pair_prods[e])
            
            scores = []
            for p in remaining:
                new_s = 1 if pair_sums[p] not in current_sums else 0
                new_p = 1 if pair_prods[p] not in current_prods else 0
                scores.append(new_s + new_p)
            
            # Pick from low-score edges
            min_score = min(scores)
            candidates = [p for p, s in zip(remaining, scores) if s == min_score]
            chosen = random.choice(candidates)
            selected.add(chosen)
        
        if len(selected) < m:
            continue
        
        edges = list(selected)
        score = _raw_value(values, edges)
        
        # Local search: swap one edge for another
        improved = True
        while improved:
            improved = False
            current_sums = set()
            current_prods = set()
            for e in edges:
                current_sums.add(pair_sums[e])
                current_prods.add(pair_prods[e])
            
            edge_set = set(edges)
            for idx, e in enumerate(edges):
                # Try swapping e with another edge
                for new_e in pairs:
                    if new_e in edge_set:
                        continue
                    new_edges = list(edges)
                    new_edges[idx] = new_e
                    new_sums = set()
                    new_prods = set()
                    for ne in new_edges:
                        new_sums.add(pair_sums[ne])
                        new_prods.add(pair_prods[ne])
                    new_score = max(len(new_sums), len(new_prods))
                    if new_score < score:
                        edges = new_edges
                        edge_set = set(edges)
                        score = new_score
                        improved = True
                        break
                if improved:
                    break
        
        if score < best_score:
            best_score = score
            best_edges = edges
    
    return best_edges, best_score

def _try_value_family(base, n=14, max_val=10000):
    """Try values that are multiples of base with small multipliers."""
    # Use multipliers that create product collisions
    # If values are base * m_i, then product is base^2 * m_i * m_j
    # Distinct products = distinct m_i * m_j values
    # Distinct sums = base * (m_i + m_j), so distinct sums = distinct m_i + m_j
    
    # Use small multipliers to keep products colliding
    # Multipliers like 1,2,3,4,5,6,7,8,9,10,11,12,13,14 give many collisions
    
    # Strategy: use multipliers from a set with many additive/multiplicative collisions
    # e.g., powers of 2: 1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192
    # Products: 2^(a+b), all distinct if a+b distinct -> not great
    # Sums: 2^a + 2^b, all distinct -> not great
    
    # Better: use small integers where many pairs have same sum/product
    # e.g., 1,2,3,4,5,6,7,8,9,10,11,12,13,14
    # Products collide: 1*12=2*6=3*4, 2*10=4*5, etc.
    # Sums collide: 1+14=2+13=3+12=...
    
    multipliers = list(range(1, n+1))
    if base * max(multipliers) > max_val:
        return None
    
    values = [base * m for m in multipliers]
    return values

def _try_arithmetic_progression(start, step, n=14, max_val=10000):
    values = [start + i * step for i in range(n)]
    if max(values) > max_val:
        return None
    return values

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin before 60s
    
    n = 14
    m = 35
    max_val = 10000
    
    best_result = None
    best_score = float('inf')
    
    # Phase 1: Try structured value families
    families = []
    
    # Arithmetic progressions with various steps
    for step in [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 25, 30, 40, 50]:
        for start in [1, 2, 3, 4, 5, 10, 15, 20]:
            vals = _try_arithmetic_progression(start, step, n, max_val)
            if vals is not None:
                families.append(vals)
    
    # Multiples of a base
    for base in [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 15, 20, 25, 30, 40, 50, 70, 100]:
        vals = _try_value_family(base, n, max_val)
        if vals is not None:
            families.append(vals)
    
    # Specific collision-heavy sets
    # Small consecutive integers
    families.append(list(range(1, n+1)))
    families.append(list(range(2, n+2)))
    families.append(list(range(3, n+3)))
    
    # Sets with many factor relationships
    # e.g., 2,4,6,8,10,12,14,16,18,20,22,24,26,28 (all even)
    families.append([2*i for i in range(1, n+1)])
    families.append([3*i for i in range(1, n+1)])
    families.append([4*i for i in range(1, n+1)])
    families.append([5*i for i in range(1, n+1)])
    
    # Geometric + arithmetic mix
    for base in [2, 3, 4, 5]:
        vals = [base**i for i in range(n)]
        if max(vals) <= max_val:
            families.append(vals)
    
    # Remove duplicates
    seen = set()
    unique_families = []
    for vals in families:
        key = tuple(sorted(vals))
        if key not in seen:
            seen.add(key)
            unique_families.append(vals)
    
    # Evaluate each family
    for vals in unique_families:
        if time.time() > deadline:
            break
        
        # Check validity
        if len(set(vals)) != n:
            continue
        if any(v < 1 or v > max_val for v in vals):
            continue
        
        edges, score = _optimize_edges(vals, m, n)
        if edges is not None and score < best_score:
            best_score = score
            best_result = {'values': list(vals), 'edges': [list(e) for e in edges]}
    
    # Phase 2: Local search on values
    if best_result is not None and time.time() < deadline - 5:
        current_vals = best_result['values']
        current_edges = best_result['edges']
        current_score = best_score
        
        for iteration in range(50):
            if time.time() > deadline - 2:
                break
            
            # Try perturbing one value
            improved = False
            for _ in range(20):
                idx = random.randint(0, n-1)
                old_val = current_vals[idx]
                # Try nearby values
                for delta in [-3, -2, -1, 1, 2, 3, -5, 5, -10, 10]:
                    new_val = old_val + delta
                    if new_val < 1 or new_val > max_val:
                        continue
                    if new_val in [v for i, v in enumerate(current_vals) if i != idx]:
                        continue
                    
                    new_vals = list(current_vals)
                    new_vals[idx] = new_val
                    
                    # Re-optimize edges
                    new_edges, new_score = _optimize_edges(new_vals, m, n)
                    if new_edges is not None and new_score < current_score:
                        current_vals = new_vals
                        current_edges = new_edges
                        current_score = new_score
                        best_score = new_score
                        best_result = {'values': list(current_vals), 'edges': [list(e) for e in current_edges]}
                        improved = True
                        break
                if improved:
                    break
            
            if not improved:
                # Try a larger perturbation
                idx = random.randint(0, n-1)
                old_val = current_vals[idx]
                new_val = max(1, min(max_val, old_val + random.randint(-50, 50)))
                if new_val == old_val or new_val in [v for i, v in enumerate(current_vals) if i != idx]:
                    continue
                new_vals = list(current_vals)
                new_vals[idx] = new_val
                new_edges, new_score = _optimize_edges(new_vals, m, n)
                if new_edges is not None and new_score < current_score:
                    current_vals = new_vals
                    current_edges = new_edges
                    current_score = new_score
                    best_score = new_score
                    best_result = {'values': list(current_vals), 'edges': [list(e) for e in current_edges]}
    
    # Ensure types are correct
    if best_result is None:
        # Fallback
        best_result = {'values': list(range(1, n+1)), 'edges': [list(e) for e in list(combinations(range(n), 2))[:m]]}
    
    # Convert to ensure proper types
    best_result['values'] = [int(v) for v in best_result['values']]
    best_result['edges'] = [[int(i), int(j)] for i, j in best_result['edges']]
    
    return best_result
# EVOLVE-BLOCK-END
