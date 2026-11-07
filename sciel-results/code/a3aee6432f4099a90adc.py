"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _raw_value(values, edges):
    """Compute max(distinct_sums, distinct_products) for given values and edges."""
    sums = set()
    prods = set()
    for i, j in edges:
        s = values[i] + values[j]
        p = values[i] * values[j]
        sums.add(s)
        prods.add(p)
    return max(len(sums), len(prods))

def _count_sums_prods(values, edges):
    """Return (num_distinct_sums, num_distinct_products) for given values and edges."""
    sums = set()
    prods = set()
    for i, j in edges:
        s = values[i] + values[j]
        p = values[i] * values[j]
        sums.add(s)
        prods.add(p)
    return len(sums), len(prods)

def _optimize_edges_for_values(values, m=35, deadline=None):
    """Find m edges that minimize max(distinct_sums, distinct_products) for given values."""
    n = len(values)
    all_edges = list(combinations(range(n), 2))
    
    # Precompute sum and product for each edge
    edge_data = {}
    for i, j in all_edges:
        edge_data[(i, j)] = (values[i] + values[j], values[i] * values[j])
    
    # Greedy initialization: pick edges that reuse existing sums/products
    current_edges = set()
    current_sums = set()
    current_prods = set()
    
    # Sort edges by "reuse potential" - edges whose sum/product are already common
    # Actually, let's use a frequency-based greedy: at each step, pick the edge
    # that minimizes the increase in max(distinct_sums, distinct_products)
    
    # Better approach: random restarts with greedy + local search
    
    best_raw = float('inf')
    best_edges = None
    
    num_restarts = 50
    for restart in range(num_restarts):
        if deadline is not None and time.time() > deadline:
            break
        
        random.shuffle(all_edges)
        
        # Greedy build: start empty, add edges one by one
        cur_edges = []
        cur_sums = set()
        cur_prods = set()
        
        # Pick first m edges greedily
        remaining = list(all_edges)
        random.shuffle(remaining)
        
        for _ in range(m):
            best_edge = None
            best_metric = float('inf')
            # Sample a subset for speed
            sample = remaining[:min(50, len(remaining))]
            for e in sample:
                s, p = edge_data[e]
                new_sums = len(cur_sums) + (1 if s not in cur_sums else 0)
                new_prods = len(cur_prods) + (1 if p not in cur_prods else 0)
                metric = max(new_sums, new_prods)
                if metric < best_metric:
                    best_metric = metric
                    best_edge = e
            
            if best_edge is not None:
                cur_edges.append(best_edge)
                remaining.remove(best_edge)
                s, p = edge_data[best_edge]
                cur_sums.add(s)
                cur_prods.add(p)
        
        if len(cur_edges) < m:
            continue
        
        cur_edges_set = set(cur_edges)
        cur_raw = max(len(cur_sums), len(cur_prods))
        
        if cur_raw < best_raw:
            best_raw = cur_raw
            best_edges = list(cur_edges_set)
        
        # Local search: try swapping edges
        for iteration in range(200):
            if deadline is not None and time.time() > deadline:
                break
            
            improved = False
            # Try random swaps
            for _ in range(20):
                if deadline is not None and time.time() > deadline:
                    break
                
                # Pick a random edge in current set
                in_idx = random.randint(0, len(cur_edges) - 1)
                out_edge = cur_edges[in_idx]
                
                # Pick a random edge not in current set
                out_candidates = [e for e in all_edges if e not in cur_edges_set]
                if not out_candidates:
                    break
                in_edge = random.choice(out_candidates)
                
                # Compute new metric
                new_sums = set(cur_sums)
                new_prods = set(cur_prods)
                
                # Remove out_edge contribution
                os_, op = edge_data[out_edge]
                # We need to check if other edges still contribute this sum/product
                # This is complex; let's just recompute for simplicity in the inner loop
                # Actually, let's use a counter approach for efficiency
                
                # For now, do a quick estimate: if out_edge's sum is unique to it, removing helps
                # Let's just recompute properly but efficiently
                
                # Count occurrences of each sum and product
                sum_counts = {}
                prod_counts = {}
                for e in cur_edges:
                    s, p = edge_data[e]
                    sum_counts[s] = sum_counts.get(s, 0) + 1
                    prod_counts[p] = prod_counts.get(p, 0) + 1
                
                # Remove out_edge
                os_, op = edge_data[out_edge]
                sum_counts[os_] -= 1
                prod_counts[op] -= 1
                if sum_counts[os_] == 0:
                    del sum_counts[os_]
                if prod_counts[op] == 0:
                    del prod_counts[op]
                
                # Add in_edge
                is_, ip = edge_data[in_edge]
                sum_counts[is_] = sum_counts.get(is_, 0) + 1
                prod_counts[ip] = prod_counts.get(ip, 0) + 1
                
                new_raw = max(len(sum_counts), len(prod_counts))
                
                if new_raw < cur_raw:
                    # Accept the swap
                    cur_edges[in_idx] = in_edge
                    cur_edges_set.remove(out_edge)
                    cur_edges_set.add(in_edge)
                    cur_sums = set(sum_counts.keys())
                    cur_prods = set(prod_counts.keys())
                    cur_raw = new_raw
                    improved = True
                    if cur_raw < best_raw:
                        best_raw = cur_raw
                        best_edges = list(cur_edges_set)
        
        if not improved:
            break
    
    if best_edges is None:
        # Fallback: just take first m edges
        best_edges = list(all_edges[:m])
    
    return best_edges, best_raw

def _optimize_values(values, edges, deadline=None):
    """Try to improve values by swapping or replacing."""
    n = len(values)
    current_raw = _raw_value(values, edges)
    best_raw = current_raw
    best_values = list(values)
    
    # Try random swaps of values
    for iteration in range(500):
        if deadline is not None and time.time() > deadline:
            break
        
        i, j = random.sample(range(n), 2)
        new_values = list(values)
        new_values[i], new_values[j] = new_values[j], new_values[i]
        
        # Check constraints
        if len(set(new_values)) != n:
            continue
        if any(v <= 0 or v > 10000 for v in new_values):
            continue
        
        new_raw = _raw_value(new_values, edges)
        if new_raw < best_raw:
            best_raw = new_raw
            best_values = new_values
            values = new_values
            current_raw = new_raw
    
    # Try replacing a single value
    for iteration in range(500):
        if deadline is not None and time.time() > deadline:
            break
        
        i = random.randint(0, n - 1)
        new_val = random.randint(1, 10000)
        if new_val in values:
            continue
        
        new_values = list(values)
        new_values[i] = new_val
        
        new_raw = _raw_value(new_values, edges)
        if new_raw < best_raw:
            best_raw = new_raw
            best_values = new_values
            values = new_values
            current_raw = new_raw
    
    return best_values, best_raw

def construct():
    deadline = time.time() + 75  # Leave 15s safety margin
    
    n = 14
    m = 35
    
    best_overall_raw = float('inf')
    best_overall_values = None
    best_overall_edges = None
    
    # Start with a reasonable initial value set
    # Use values that are spread out but with some structure
    initial_values = list(range(1, n + 1))
    
    # Optimize edges for initial values
    edges, raw = _optimize_edges_for_values(initial_values, m, deadline)
    
    if raw < best_overall_raw:
        best_overall_raw = raw
        best_overall_values = initial_values
        best_overall_edges = edges
    
    # Now try to improve values with these edges
    values, raw = _optimize_values(initial_values, edges, deadline)
    
    if raw < best_overall_raw:
        best_overall_raw = raw
        best_overall_values = values
        best_overall_edges = edges
    
    # Try a few more random value sets
    for restart in range(20):
        if time.time() > deadline:
            break
        
        # Generate random distinct positive integers <= 10000
        vals = random.sample(range(1, 10001), n)
        
        edges_r, raw_r = _optimize_edges_for_values(vals, m, deadline)
        
        if raw_r < best_overall_raw:
            best_overall_raw = raw_r
            best_overall_values = vals
            best_overall_edges = edges_r
        
        # Try improving values
        vals_improved, raw_improved = _optimize_values(vals, edges_r, deadline)
        
        if raw_improved < best_overall_raw:
            best_overall_raw = raw_improved
            best_overall_values = vals_improved
            best_overall_edges = edges_r
    
    # Ensure types are correct
    final_values = [int(v) for v in best_overall_values]
    final_edges = [[int(i), int(j)] for i, j in best_overall_edges]
    
    # Verify constraints
    assert len(final_values) == n
    assert len(set(final_values)) == n
    assert all(isinstance(v, int) and not isinstance(v, bool) and 1 <= v <= 10000 for v in final_values)
    assert len(final_edges) == m
    for e in final_edges:
        assert len(e) == 2
        assert isinstance(e[0], int) and isinstance(e[1], int)
        assert e[0] != e[1]
        assert 0 <= e[0] < n and 0 <= e[1] < n
    edge_set = set()
    for e in final_edges:
        key = (min(e[0], e[1]), max(e[0], e[1]))
        assert key not in edge_set
        edge_set.add(key)
    
    return {'values': final_values, 'edges': final_edges}
# EVOLVE-BLOCK-END
