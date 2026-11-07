"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s kill
    
    n = 14
    m = 35
    max_val = 10000
    
    def evaluate(values, edges):
        sums = set()
        prods = set()
        for u, v in edges:
            sums.add(values[u] + values[v])
            prods.add(values[u] * values[v])
        return max(len(sums), len(prods))
    
    def greedy_edges(values, rng):
        """Greedy: pick edges that reuse existing sums/products."""
        sums_seen = set()
        prods_seen = set()
        edges = []
        all_pairs = list(combinations(range(n), 2))
        rng.shuffle(all_pairs)
        
        for u, v in all_pairs:
            if len(edges) >= m:
                break
            s = values[u] + values[v]
            p = values[u] * values[v]
            # Prefer edges that add fewer new distinct values
            new_sums = 0 if s in sums_seen else 1
            new_prods = 0 if p in prods_seen else 1
            if new_sums + new_prods <= 1:
                edges.append([u, v])
                sums_seen.add(s)
                prods_seen.add(p)
        
        # If we don't have enough, fill with remaining
        if len(edges) < m:
            used = set(tuple(sorted(e)) for e in edges)
            for u, v in all_pairs:
                if len(edges) >= m:
                    break
                key = tuple(sorted((u, v)))
                if key not in used:
                    edges.append([u, v])
                    used.add(key)
        
        return edges[:m]
    
    def local_search(values, edges, max_iter=5000):
        """Local search to refine both values and edges."""
        current_score = evaluate(values, edges)
        best_values = list(values)
        best_edges = [list(e) for e in edges]
        
        all_pairs = list(combinations(range(n), 2))
        edge_set = set(tuple(sorted(e)) for e in edges)
        
        for _ in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            improved = False
            
            # Try swapping an edge
            if random.random() < 0.5:
                if edges:
                    ei = random.randint(0, len(edges) - 1)
                    old_u, old_v = edges[ei]
                    old_key = tuple(sorted((old_u, old_v)))
                    
                    # Find a new edge to swap in
                    candidates = [p for p in all_pairs if tuple(sorted(p)) not in edge_set]
                    if candidates:
                        new_u, new_v = random.choice(candidates)
                        new_key = tuple(sorted((new_u, new_v)))
                        
                        test_edges = [list(e) for e in edges]
                        test_edges[ei] = [new_u, new_v]
                        test_score = evaluate(values, test_edges)
                        
                        if test_score < current_score:
                            edges[ei] = [new_u, new_v]
                            edge_set.remove(old_key)
                            edge_set.add(new_key)
                            current_score = test_score
                            improved = True
            
            # Try changing a value
            if not improved and random.random() < 0.7:
                vi = random.randint(0, n - 1)
                old_val = values[vi]
                
                # Try nearby values or random
                if random.random() < 0.5:
                    new_val = max(1, min(max_val, old_val + random.randint(-10, 10)))
                else:
                    new_val = random.randint(1, max_val)
                
                if new_val != old_val:
                    # Check distinctness
                    test_values = list(values)
                    test_values[vi] = new_val
                    if len(set(test_values)) == n:
                        test_score = evaluate(test_values, edges)
                        if test_score < current_score:
                            values[vi] = new_val
                            current_score = test_score
                            improved = True
        
        return current_score, values, edges
    
    def small_range_values(rng):
        """Generate values from a small range to cluster sums."""
        base = random.randint(1, 50)
        spread = random.randint(5, 30)
        vals = []
        used = set()
        while len(vals) < n:
            v = base + random.randint(-spread, spread)
            v = max(1, min(max_val, v))
            if v not in used:
                vals.append(v)
                used.add(v)
        return vals
    
    def random_distinct_values(rng):
        """Generate random distinct values."""
        vals = []
        used = set()
        while len(vals) < n:
            v = random.randint(1, max_val)
            if v not in used:
                vals.append(v)
                used.add(v)
        return vals
    
    best_score = float('inf')
    best_result = None
    rng = random.Random(42)
    
    # Multiple restarts
    for restart in range(200):
        if time.time() > deadline - 5:
            break
        
        # Choose value generation strategy
        if restart % 3 == 0:
            values = small_range_values(rng)
        elif restart % 3 == 1:
            values = random_distinct_values(rng)
        else:
            # Arithmetic progression
            a = random.randint(1, 100)
            d = random.randint(1, 20)
            values = []
            for i in range(n):
                v = a + i * d
                if v > max_val:
                    v = max_val - random.randint(0, 50)
                values.append(v)
            # Ensure distinct
            if len(set(values)) < n:
                values = random_distinct_values(rng)
        
        edges = greedy_edges(values, rng)
        score = evaluate(values, edges)
        
        if score < best_score:
            best_score = score
            best_result = (list(values), [list(e) for e in edges])
        
        # Local search
        if time.time() > deadline - 10:
            break
        
        score, values, edges = local_search(values, edges, max_iter=3000)
        
        if score < best_score:
            best_score = score
            best_result = (list(values), [list(e) for e in edges])
    
    if best_result is None:
        # Fallback
        values = list(range(1, n + 1))
        edges = [list(e) for e in list(combinations(range(n), 2))[:m]]
        best_result = (values, edges)
    
    final_values, final_edges = best_result
    
    # Ensure types are correct
    final_values = [int(v) for v in final_values]
    final_edges = [[int(u), int(v)] for u, v in final_edges]
    
    return {'values': final_values, 'edges': final_edges}
# EVOLVE-BLOCK-END
