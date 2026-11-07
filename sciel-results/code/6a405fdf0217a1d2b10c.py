"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _score(values, edges):
    sums = set()
    prods = set()
    for i, j in edges:
        v_i = values[i]
        v_j = values[j]
        sums.add(v_i + v_j)
        prods.add(v_i * v_j)
    return max(len(sums), len(prods))

def _greedy_edges(values, m=35, deadline=None):
    n = len(values)
    # Precompute all possible edges with their (sum, product)
    all_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            s = values[i] + values[j]
            p = values[i] * values[j]
            all_edges.append((s, p, i, j))
    
    # Greedy: pick edges that minimize the growth of distinct sums and products
    # We want to maximize overlap of sums and products
    # Strategy: sort edges by product (smaller products more likely to collide)
    # Actually, let's try: pick edges where both sum and product are already seen
    
    # Better approach: iteratively pick the edge that adds the fewest new distinct values
    # to the union of sums and products, with tie-breaking
    
    selected = []
    seen_sums = set()
    seen_prods = set()
    remaining = list(all_edges)
    
    # Initial pick: edge with smallest sum+product
    remaining.sort(key=lambda e: (e[0], e[1]))
    if not remaining:
        return []
    s0, p0, i0, j0 = remaining[0]
    selected.append((i0, j0))
    seen_sums.add(s0)
    seen_prods.add(p0)
    remaining.remove(remaining[0])
    
    while len(selected) < m and remaining:
        if deadline is not None and time.time() > deadline:
            break
        best_idx = -1
        best_cost = float('inf')
        # Sample a subset if too many
        if len(remaining) > 200:
            sample = random.sample(remaining, 200)
        else:
            sample = remaining
        
        for idx, (s, p, i, j) in enumerate(sample):
            new_sums = 0 if s in seen_sums else 1
            new_prods = 0 if p in seen_prods else 1
            cost = new_sums + new_prods
            if cost < best_cost:
                best_cost = cost
                best_idx = idx
            elif cost == best_cost:
                # Tie-break: prefer smaller values
                if sample[idx][0] < sample[best_idx][0]:
                    best_idx = idx
        
        if best_idx >= 0:
            s, p, i, j = sample[best_idx]
            selected.append((i, j))
            seen_sums.add(s)
            seen_prods.add(p)
            # Remove from remaining
            for k in range(len(remaining)):
                if remaining[k][2] == i and remaining[k][3] == j:
                    remaining.pop(k)
                    break
    
    return [(i, j) for (_, _, i, j) in selected]

def _try_value_set(values, deadline):
    edges = _greedy_edges(values, 35, deadline)
    if len(edges) < 35:
        return None
    sc = _score(values, edges)
    return sc, edges

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s margin for 90s limit
    
    n = 14
    m = 35
    MAX_VAL = 10000
    
    best_score = float('inf')
    best_values = None
    best_edges = None
    
    # Strategy 1: Use values with shared factors to create product collisions
    # If values share a common factor d, then products are multiples of d^2
    # Better: use values that are all multiples of a small number
    # e.g., all multiples of 2: then all products are multiples of 4
    
    # Strategy 2: Use a structured set where sums collide
    # e.g., symmetric around a center: a and (C-a) give same sum C
    
    # Let's try multiple approaches:
    
    # Approach A: Small multiples of a base
    for base in [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20]:
        if time.time() > deadline:
            break
        # Generate candidate value sets: base * [a1, a2, ..., a14]
        # where a_i are distinct positive integers, base * a_i <= 10000
        max_mult = MAX_VAL // base
        
        # Try different multipliers
        for attempt in range(50):
            if time.time() > deadline:
                break
            if attempt == 0:
                # Sequential
                mults = list(range(1, n + 1))
            elif attempt == 1:
                # Powers-ish
                mults = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
                if mults[-1] > max_mult:
                    continue
            else:
                mults = random.sample(range(1, min(max_mult, 500) + 1), n)
            
            values = [base * x for x in mults]
            if len(set(values)) != n or max(values) > MAX_VAL:
                continue
            
            result = _try_value_set(values, deadline)
            if result is not None:
                sc, edges = result
                if sc < best_score:
                    best_score = sc
                    best_values = values[:]
                    best_edges = edges[:]
    
    # Approach B: Symmetric sets for sum collisions
    # values = [c - d, c - d + 1, ..., c + d - 1, c + d] type sets
    for center in [10, 20, 50, 100, 200, 500, 1000, 2000, 5000]:
        if time.time() > deadline:
            break
        for spread in range(1, 10):
            if time.time() > deadline:
                break
            # Create n values centered around center
            half = n // 2
            values = []
            for k in range(half):
                values.append(center - (spread * (half - k)))
            for k in range(n - half):
                values.append(center + (spread * k))
            
            # Ensure distinct and positive
            values = list(set(values))
            if len(values) < n:
                # Fill in
                used = set(values)
                for v in range(1, MAX_VAL + 1):
                    if v not in used:
                        values.append(v)
                        used.add(v)
                    if len(values) >= n:
                        break
            if len(values) != n or min(values) <= 0 or max(values) > MAX_VAL:
                continue
            
            result = _try_value_set(values, deadline)
            if result is not None:
                sc, edges = result
                if sc < best_score:
                    best_score = sc
                    best_values = values[:]
                    best_edges = edges[:]
    
    # Approach C: Random search with local improvement
    for attempt in range(200):
        if time.time() > deadline:
            break
        # Random values with structure
        strategy = random.choice(['small', 'multiples', 'clustered'])
        if strategy == 'small':
            values = random.sample(range(1, 50), n)
        elif strategy == 'multiples':
            base = random.choice([2, 3, 4, 5, 6, 8, 10])
            max_m = MAX_VAL // base
            if max_m < n:
                continue
            mults = random.sample(range(1, min(max_m, 200) + 1), n)
            values = [base * m for m in mults]
        else:  # clustered
            center = random.randint(10, 5000)
            spread = random.randint(1, 50)
            values = []
            for _ in range(n):
                v = center + random.randint(-spread, spread)
                if 1 <= v <= MAX_VAL and v not in values:
                    values.append(v)
            if len(values) < n:
                # Fill
                used = set(values)
                for v in range(1, MAX_VAL + 1):
                    if v not in used:
                        values.append(v)
                        used.add(v)
                    if len(values) >= n:
                        break
            if len(values) != n:
                continue
        
        values = list(set(values))[:n]
        if len(values) != n or min(values) <= 0 or max(values) > MAX_VAL:
            continue
        
        result = _try_value_set(values, deadline)
        if result is not None:
            sc, edges = result
            if sc < best_score:
                best_score = sc
                best_values = values[:]
                best_edges = edges[:]
    
    # Local search on best found
    if best_values is not None and time.time() < deadline:
        for iteration in range(500):
            if time.time() > deadline:
                break
            # Try swapping one value
            new_values = best_values[:]
            idx = random.randint(0, n - 1)
            new_val = random.randint(1, MAX_VAL)
            if new_val in new_values and new_values[idx] != new_val:
                continue
            new_values[idx] = new_val
            if len(set(new_values)) != n or min(new_values) <= 0 or max(new_values) > MAX_VAL:
                continue
            
            result = _try_value_set(new_values, deadline)
            if result is not None:
                sc, edges = result
                if sc < best_score:
                    best_score = sc
                    best_values = new_values[:]
                    best_edges = edges[:]
    
    # Fallback: if no edges found, use trivial
    if best_edges is None or len(best_edges) < 35:
        # Use the original simple approach
        values = list(range(1, n + 1))
        edges = [list(e) for e in list(combinations(range(n), 2))[:m]]
        best_values = values
        best_edges = edges
        best_score = _score(values, edges)
    
    # Ensure edges are exactly m
    if len(best_edges) > m:
        best_edges = best_edges[:m]
    
    # Convert to proper types
    final_values = [int(v) for v in best_values]
    final_edges = [[int(i), int(j)] for i, j in best_edges]
    
    return {'values': final_values, 'edges': final_edges}
# EVOLVE-BLOCK-END
