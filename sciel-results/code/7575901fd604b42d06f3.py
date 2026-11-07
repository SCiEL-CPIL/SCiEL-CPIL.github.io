"""Search-based: minimize max(distinct_sums, distinct_products) over 35 edges of 14 vertices."""
# EVOLVE-BLOCK-START
import time
import itertools
import random
import heapq
from collections import defaultdict

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

def _precompute_edge_props(values):
    """Precompute sum and product for all possible edges."""
    n = len(values)
    edge_props = {}
    for u, v in itertools.combinations(range(n), 2):
        a, b = values[u], values[v]
        edge_props[(u, v)] = (a + b, a * b)
    return edge_props

def _pq_greedy_edges(values, target_m):
    """Priority-queue greedy: select edges that maximize collision reuse.
    
    Key insight: edges sharing the same sum or product are 'collisions' that
    don't increase the distinct count. We prioritize edges that reuse existing
    sums/products, breaking ties by preferring edges that create the least
    new distinct values overall.
    """
    n = len(values)
    edge_props = _precompute_edge_props(values)
    all_e = list(edge_props.keys())
    
    # Group edges by their sum and product for quick lookup
    sum_to_edges = defaultdict(list)
    prod_to_edges = defaultdict(list)
    for e, (s, p) in edge_props.items():
        sum_to_edges[s].append(e)
        prod_to_edges[p].append(e)
    
    used = set()
    sums = set()
    prods = set()
    edges = []
    
    # Priority: (new_sum, new_prod, edge) - prefer edges with fewer new distinct values
    # Use a heap where lower priority = better
    # Priority = (num_new_sums, num_new_prods, edge_index)
    
    for _ in range(target_m):
        best_edge = None
        best_priority = (999, 999, 999)
        
        for (u, v) in all_e:
            if (u, v) in used:
                continue
            s, p = edge_props[(u, v)]
            new_s = 0 if s in sums else 1
            new_p = 0 if p in prods else 1
            
            # Priority: minimize new distinct counts
            # Tie-break: prefer edges whose sum/product have many potential partners
            # (edges that are part of large collision groups are more valuable)
            sum_group_size = len(sum_to_edges[s])
            prod_group_size = len(prod_to_edges[p])
            
            priority = (new_s + new_p, -sum_group_size, -prod_group_size, u * 100 + v)
            
            if priority < best_priority:
                best_priority = priority
                best_edge = (u, v)
        
        if best_edge is None:
            break
        
        u, v = best_edge
        edges.append((u, v))
        used.add((u, v))
        s, p = edge_props[(u, v)]
        sums.add(s)
        prods.add(p)
    
    return edges

def _fast_local_search(values, edges, target_m, max_iters=300):
    """Fast local search with precomputed properties.
    
    Uses incremental updates instead of recomputing from scratch.
    """
    n = len(values)
    edge_props = _precompute_edge_props(values)
    all_e = list(edge_props.keys())
    current_score = _raw_score(values, edges)
    edge_set = set(edges)
    
    # Precompute for each edge: how many other edges share its sum or product
    sum_counts = defaultdict(int)
    prod_counts = defaultdict(int)
    for e in edges:
        s, p = edge_props[e]
        sum_counts[s] += 1
        prod_counts[p] += 1
    
    for it in range(max_iters):
        if time.time() > DEADLINE - 3:
            break
        improved = False
        
        # Try each edge removal
        for i, (u, v) in enumerate(edges):
            s_rm, p_rm = edge_props[(u, v)]
            
            # Tentatively remove
            sum_counts[s_rm] -= 1
            prod_counts[p_rm] -= 1
            
            # Find best edge to add
            best_add = None
            best_new_score = current_score
            
            for (a, b) in all_e:
                if (a, b) in edge_set:
                    continue
                s_add, p_add = edge_props[(a, b)]
                
                # Would this add a new sum or product?
                new_s = 1 if sum_counts[s_add] == 0 else 0
                new_p = 1 if prod_counts[p_add] == 0 else 0
                
                # Estimate new score
                cur_sums = len([s for s, c in sum_counts.items() if c > 0])
                cur_prods = len([p for p, c in prod_counts.items() if c > 0])
                
                # After adding: sum_counts[s_add] would be +1, prod_counts[p_add] would be +1
                new_cur_sums = cur_sums + new_s
                new_cur_prods = cur_prods + new_p
                
                est_score = max(new_cur_sums, new_cur_prods)
                
                if est_score < best_new_score:
                    best_new_score = est_score
                    best_add = (a, b)
            
            # Restore
            sum_counts[s_rm] += 1
            prod_counts[p_rm] += 1
            
            if best_add is not None and best_new_score < current_score:
                # Apply the move
                edges = edges[:i] + edges[i+1:] + [best_add]
                edge_set = set(edges)
                
                # Update counts
                sum_counts[s_rm] -= 1
                prod_counts[p_rm] -= 1
                s_add, p_add = edge_props[best_add]
                sum_counts[s_add] += 1
                prod_counts[p_add] += 1
                
                current_score = best_new_score
                improved = True
                break
        
        if not improved:
            break
    
    return edges, current_score

def _simulated_annealing(values, edges, target_m, max_iters=500, start_temp=10.0, end_temp=0.1):
    """Simulated annealing for edge selection."""
    n = len(values)
    edge_props = _precompute_edge_props(values)
    all_e = list(edge_props.keys())
    current_score = _raw_score(values, edges)
    best_score = current_score
    best_edges = list(edges)
    
    edge_set = set(edges)
    
    for it in range(max_iters):
        if time.time() > DEADLINE - 2:
            break
        
        temp = start_temp * (end_temp / start_temp) ** (it / max_iters)
        
        # Random move: remove one edge, add another
        i = random.randint(0, len(edges) - 1)
        old_edge = edges[i]
        
        # Find a replacement
        candidates = [e for e in all_e if e not in edge_set]
        if not candidates:
            break
        
        new_edge = random.choice(candidates)
        
        # Apply
        edges[i] = new_edge
        edge_set = set(edges)
        
        new_score = _raw_score(values, edges)
        
        if new_score < current_score:
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_edges = list(edges)
        else:
            # Accept with probability exp(-delta/temp)
            delta = new_score - current_score
            if random.random() < (2.718281828 ** (-delta / max(temp, 0.001))):
                current_score = new_score
            else:
                # Reject: revert
                edges[i] = old_edge
                edge_set = set(edges)
    
    return best_edges, best_score

def _evaluate(values):
    """Evaluate a value set with PQ-greedy + fast local search + SA."""
    edges = _pq_greedy_edges(values, 35)
    if len(edges) < 35:
        return float('inf'), values, edges
    
    # Fast local search
    edges, score = _fast_local_search(values, edges, 35, max_iters=200)
    
    # Simulated annealing for further improvement
    if time.time() < DEADLINE - 5:
        edges, sa_score = _simulated_annealing(values, edges, 35, max_iters=300)
        if sa_score < score:
            score = sa_score
            edges = edges
    
    return score, values, edges

def _try_value_set(values):
    """Try a specific value set."""
    if len(set(values)) != 14:
        return None
    if any(not isinstance(v, int) or v < 1 or v > 10000 for v in values):
        return None
    score, vals, edges = _evaluate(values)
    return score, vals, edges

def _generate_collision_optimized_values():
    """Generate value sets specifically designed to maximize sum/product collisions.
    
    Key insight: if values are multiples of a common factor g, then all products
    are multiples of g^2, creating more structure. Also, arithmetic progressions
    create many sum collisions.
    """
    candidates = []
    
    # Tight arithmetic progressions (more sum collisions)
    for a in range(1, 50):
        for d in range(1, 10):
            vals = [a + i * d for i in range(14)]
            if max(vals) <= 10000 and len(set(vals)) == 14:
                candidates.append(vals)
    
    # Multiples of small numbers (product collisions)
    for g in range(2, 50):
        for offset in range(0, g):
            vals = [g * (i + 1) + offset for i in range(14)]
            if max(vals) <= 10000 and len(set(vals)) == 14:
                candidates.append(vals)
    
    # Pairs: values that are close together create sum collisions
    # e.g., {1,2,3,4,5,6,7,8,9,10,11,12,13,14} has many sum collisions
    for base in range(1, 20):
        vals = list(range(base, base + 14))
        candidates.append(vals)
    
    # Squares of small numbers
    for start in range(1, 5):
        vals = [(start + i) * (start + i) for i in range(14)]
        if max(vals) <= 10000 and len(set(vals)) == 14:
            candidates.append(vals)
    
    # Products of two small factors
    for f1 in range(2, 15):
        for f2 in range(2, 15):
            vals = [f1 * (i + 1) * f2 for i in range(14)]
            if max(vals) <= 10000 and len(set(vals)) == 14:
                candidates.append(vals)
    
    # Mixed: small base with geometric growth
    for a in range(1, 20):
        for r in range(2, 4):
            vals = []
            for i in range(14):
                v = a * (r ** i)
                if v > 10000:
                    break
                vals.append(v)
            if len(vals) == 14:
                candidates.append(vals)
    
    # Random but structured: clusters of close values
    random.seed(123)
    for _ in range(100):
        # Create 2-4 clusters of close values
        num_clusters = random.randint(2, 4)
        cluster_sizes = [14 // num_clusters] * num_clusters
        for i in range(14 - sum(cluster_sizes)):
            cluster_sizes[i] += 1
        
        vals = []
        for size in cluster_sizes:
            center = random.randint(1, 9000)
            spread = random.randint(1, 20)
            for _ in range(size):
                v = center + random.randint(-spread, spread)
                v = max(1, min(10000, v))
                vals.append(v)
        
        if len(set(vals)) == 14:
            candidates.append(vals)
    
    return candidates

def construct():
    start_time = time.time()
    best_score = float('inf')
    best_result = None
    
    # Strategy 1: Try collision-optimized value sets
    candidates = _generate_collision_optimized_values()
    for vals in candidates:
        if time.time() > DEADLINE - 10:
            break
        result = _try_value_set(vals)
        if result:
            s, v, e = result
            if s < best_score:
                best_score = s
                best_result = (v, e)
    
    # Strategy 2: Random search with structured bias
    if time.time() < DEADLINE - 10:
        random.seed(42)
        for _ in range(200):
            if time.time() > DEADLINE - 10:
                break
            
            # Generate values with collision-friendly structure
            method = random.randint(0, 3)
            if method == 0:
                # Arithmetic progression
                a = random.randint(1, 100)
                d = random.randint(1, 30)
                vals = [a + i * d for i in range(14)]
            elif method == 1:
                # Multiples with offset
                g = random.randint(2, 30)
                offset = random.randint(0, g - 1)
                vals = [g * (i + 1) + offset for i in range(14)]
            elif method == 2:
                # Clustered values
                clusters = random.randint(2, 5)
                vals = []
                for _ in range(clusters):
                    center = random.randint(1, 9000)
                    spread = random.randint(1, 30)
                    size = random.randint(2, 7)
                    for _ in range(size):
                        v = center + random.randint(-spread, spread)
                        vals.append(max(1, min(10000, v)))
                # Ensure we have exactly 14
                while len(vals) < 14:
                    vals.append(random.randint(1, 10000))
                vals = vals[:14]
            else:
                # Random
                vals = random.sample(range(1, 10001), 14)
            
            if len(set(vals)) != 14:
                continue
            if any(not isinstance(v, int) or v < 1 or v > 10000 for v in vals):
                continue
            
            result = _try_value_set(vals)
            if result:
                s, v, e = result
                if s < best_score:
                    best_score = s
                    best_result = (v, e)
    
    # Strategy 3: Joint value+edge optimization via perturbation
    if best_result is not None and time.time() < DEADLINE - 5:
        vals, edges = best_result
        current_score = best_score
        vals_list = list(vals)
        edges_list = list(edges)
        
        for _ in range(200):
            if time.time() > DEADLINE - 5:
                break
            
            # Perturb one value
            new_vals = list(vals_list)
            idx = random.randint(0, 13)
            old_val = new_vals[idx]
            new_val = old_val + random.randint(-50, 50)
            new_val = max(1, min(10000, new_val))
            
            if new_val in new_vals:
                continue
            new_vals[idx] = new_val
            
            if len(set(new_vals)) != 14:
                continue
            
            # Re-optimize edges for new values
            new_edges = _pq_greedy_edges(new_vals, 35)
            if len(new_edges) < 35:
                continue
            
            new_score = _raw_score(new_vals, new_edges)
            
            if new_score < current_score:
                current_score = new_score
                vals_list = new_vals
                edges_list = new_edges
                if current_score < best_score:
                    best_score = current_score
                    best_result = (vals_list, edges_list)
            else:
                # Revert
                new_vals[idx] = old_val
    
    if best_result is None:
        # Fallback
        vals = list(range(1, 15))
        edges = list(itertools.combinations(range(14), 2))[:35]
        return {'values': vals, 'edges': [list(e) for e in edges]}
    
    vals, edges = best_result
    # Ensure proper types
    vals = [int(v) for v in vals]
    edges = [list(e) for e in edges]
    return {'values': vals, 'edges': edges}

# EVOLVE-BLOCK-END
