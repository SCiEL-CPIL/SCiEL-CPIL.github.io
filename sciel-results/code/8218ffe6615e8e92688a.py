"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit

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

    def select_optimal_edges(values, rng, max_iters=5000):
        """Select 35 edges to minimize max(distinct_sums, distinct_prods) using local search."""
        # Precompute all pair (sum, product)
        pair_sp = {}
        for u, v in all_pairs:
            pair_sp[(u, v)] = (values[u] + values[v], values[u] * values[v])
        
        # Greedy initialization: pick edges that reuse sums/products
        sums = set()
        prods = set()
        edges = []
        remaining = list(all_pairs)
        rng.shuffle(remaining)
        
        for pair in remaining:
            if len(edges) >= m:
                break
            s, p = pair_sp[pair]
            edges.append(pair)
            sums.add(s)
            prods.add(p)
        
        # Local search: swap edges to reduce distinct counts
        edges_set = set(edges)
        best_score = max(len(sums), len(prods))
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            
            # Pick a random edge to remove
            idx = rng.randint(0, m - 1)
            old_edge = edges[idx]
            os, op = pair_sp[old_edge]
            
            # Try to find a replacement
            found = False
            for _ in range(100):
                new_edge = rng.choice(all_pairs)
                if new_edge in edges_set:
                    continue
                
                # Compute effect
                new_sums = set(sums)
                new_prods = set(prods)
                
                # Remove old
                # Count occurrences
                sum_count = 0
                prod_count = 0
                for e in edges:
                    es, ep = pair_sp[e]
                    if es == os:
                        sum_count += 1
                    if ep == op:
                        prod_count += 1
                
                sum_count -= 1  # remove old_edge
                prod_count -= 1
                
                if sum_count == 0:
                    new_sums.discard(os)
                if prod_count == 0:
                    new_prods.discard(op)
                
                # Add new
                ns, np_ = pair_sp[new_edge]
                new_sums.add(ns)
                new_prods.add(np_)
                
                new_score = max(len(new_sums), len(new_prods))
                if new_score < best_score:
                    # Accept
                    edges[idx] = new_edge
                    edges_set.discard(old_edge)
                    edges_set.add(new_edge)
                    sums = new_sums
                    prods = new_prods
                    best_score = new_score
                    found = True
                    break
            
            if not found:
                continue
        
        return [list(e) for e in edges], best_score

    def generate_candidate_values(rng):
        """Generate diverse candidate value sets."""
        strategy = rng.randint(0, 6)
        
        if strategy == 0:
            # Small consecutive
            start = rng.randint(1, 3)
            step = rng.randint(1, 2)
            values = [start + i * step for i in range(14)]
            return values
        
        elif strategy == 1:
            # Geometric
            base = rng.randint(2, 4)
            values = []
            v = 1
            while len(values) < 14 and v <= max_val:
                values.append(v)
                v *= base
            while len(values) < 14:
                v = rng.randint(1, max_val)
                if v not in values:
                    values.append(v)
            return values[:14]
        
        elif strategy == 2:
            # Mixed: some small, some larger
            small = list(range(1, 8))
            large = [rng.randint(50, 200) for _ in range(7)]
            values = small + large
            rng.shuffle(values)
            # Ensure distinct
            seen = set()
            unique = []
            for v in values:
                if v not in seen and 1 <= v <= max_val:
                    seen.add(v)
                    unique.append(v)
            while len(unique) < 14:
                v = rng.randint(1, max_val)
                if v not in seen:
                    seen.add(v)
                    unique.append(v)
            return unique[:14]
        
        elif strategy == 3:
            # Powers of 2 times small odd
            values = set()
            for a in range(0, 6):
                for b in [1, 3, 5, 7]:
                    v = (2 ** a) * b
                    if 1 <= v <= max_val:
                        values.add(v)
            values = list(values)
            rng.shuffle(values)
            values = values[:14]
            while len(values) < 14:
                v = rng.randint(1, max_val)
                if v not in values:
                    values.append(v)
            return values[:14]
        
        elif strategy == 4:
            # Fibonacci-like
            values = [1, 1]
            while len(values) < 14:
                values.append(values[-1] + values[-2])
            # Make distinct
            seen = set()
            unique = []
            for v in values:
                if v not in seen and 1 <= v <= max_val:
                    seen.add(v)
                    unique.append(v)
            while len(unique) < 14:
                v = rng.randint(1, max_val)
                if v not in seen:
                    seen.add(v)
                    unique.append(v)
            return unique[:14]
        
        elif strategy == 5:
            # Random small
            values = set()
            while len(values) < 14:
                v = rng.randint(1, 50)
                values.add(v)
            return list(values)
        
        else:
            # Random medium
            values = set()
            while len(values) < 14:
                v = rng.randint(1, 200)
                values.add(v)
            return list(values)
    
    def simulated_annealing_values(initial_values, edges, rng, max_iters=5000):
        """Simulated annealing on value permutations/assignments."""
        import math
        
        current_values = list(initial_values)
        current_score = evaluate(current_values, edges)
        best_values = list(current_values)
        best_score = current_score
        
        T = 10.0
        cooling = 0.9995
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            
            # Move: swap two values
            i, j = rng.sample(range(n), 2)
            new_values = list(current_values)
            new_values[i], new_values[j] = new_values[j], new_values[i]
            
            # Check validity
            if len(set(new_values)) != n:
                continue
            if any(v < 1 or v > max_val for v in new_values):
                continue
            
            new_score = evaluate(new_values, edges)
            delta = new_score - current_score
            
            if delta < 0 or rng.random() < math.exp(-delta / max(T, 1e-10)):
                current_values = new_values
                current_score = new_score
                
                if current_score < best_score:
                    best_score = current_score
                    best_values = list(current_values)
            
            T *= cooling
        
        return best_values, best_score

    # Main search
    rng = random.Random(42)
    best_result = None
    best_score = float('inf')
    
    for restart in range(50):
        if time.time() > deadline:
            break
        
        values = generate_candidate_values(rng)
        
        # Ensure valid
        values = list(set(values))
        while len(values) < 14:
            v = rng.randint(1, max_val)
            if v not in values:
                values.append(v)
        values = values[:14]
        
        # Select optimal edges for these values
        edges, score = select_optimal_edges(values, rng, max_iters=2000)
        
        # Simulated annealing on values
        values, score = simulated_annealing_values(values, edges, rng, max_iters=3000)
        
        # Re-select edges after value changes
        edges, score = select_optimal_edges(values, rng, max_iters=1500)
        
        if score < best_score:
            best_score = score
            best_result = (list(values), edges)
            if best_score <= 14:
                break
    
    if best_result is None:
        values = list(range(1, 15))
        edges = [list(e) for e in all_pairs[:m]]
    else:
        values, edges = best_result
    
    # Final validation
    assert len(values) == 14
    assert len(set(values)) == 14
    assert all(isinstance(v, int) and 1 <= v <= max_val for v in values)
    assert len(edges) == m
    edge_set = set()
    for e in edges:
        assert len(e) == 2
        assert e[0] != e[1]
        assert 0 <= e[0] < n and 0 <= e[1] < n
        key = tuple(sorted(e))
        assert key not in edge_set
        edge_set.add(key)
    
    return {'values': values, 'edges': edges}
# EVOLVE-BLOCK-END
