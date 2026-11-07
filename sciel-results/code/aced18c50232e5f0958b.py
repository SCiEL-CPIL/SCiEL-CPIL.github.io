"""Rewritten program using MODEL-AND-OPTIMIZE approach for graph sums and products."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s kill
    
    n = 14
    m = 35
    max_val_limit = 10000
    
    all_edges = list(combinations(range(n), 2))
    
    def evaluate(values, edges):
        """Calculate the score: max(distinct sums, distinct products)."""
        sums = set()
        prods = set()
        for u, v in edges:
            sums.add(values[u] + values[v])
            prods.add(values[u] * values[v])
        return max(len(sums), len(prods))
    
    def find_best_values(rng, time_limit):
        """
        Find values that minimize the number of distinct sums and products 
        across ALL possible edges. This gives us a lower bound on what's achievable.
        """
        best_score = float('inf')
        best_values = None
        
        # Try different value patterns
        patterns = []
        
        # Small consecutive values
        patterns.append(list(range(1, n + 1)))
        patterns.append(list(range(2, 2 * n + 1, 2)))
        patterns.append(list(range(3, 3 * n + 1, 3)))
        
        # Geometric-like sequences (but keep them small)
        base = 2
        for _ in range(5):
            vals = []
            v = base
            while len(vals) < n:
                vals.append(v)
                v *= 2
                if v > max_val_limit:
                    break
            if len(vals) == n and all(1 <= v <= max_val_limit for v in vals):
                patterns.append(vals)
            base += 1
        
        # Random small values
        for _ in range(20):
            if time.time() > time_limit:
                break
            vals = sorted(rng.sample(range(1, 50), n))
            patterns.append(vals)
        
        # More structured: multiples with small offsets
        for mult in [2, 3, 4, 5]:
            for offset in range(0, 5):
                vals = [mult * i + offset + 1 for i in range(n)]
                if all(1 <= v <= max_val_limit for v in vals) and len(set(vals)) == n:
                    patterns.append(vals)
        
        # Evaluate each pattern
        for vals in patterns:
            if len(set(vals)) != n or any(v < 1 or v > max_val_limit for v in vals):
                continue
            
            # Count distinct sums and products across ALL edges
            sums = set()
            prods = set()
            for u, v in all_edges:
                sums.add(vals[u] + vals[v])
                prods.add(vals[u] * vals[v])
            
            score = max(len(sums), len(prods))
            
            if score < best_score:
                best_score = score
                best_values = list(vals)
        
        # Local search to improve best values
        if best_values is not None and time.time() < time_limit - 10:
            current = list(best_values)
            current_score = best_score
            
            for iteration in range(100):
                if time.time() > time_limit - 5:
                    break
                
                improved = False
                for idx in range(n):
                    old_val = current[idx]
                    for delta in range(-10, 11):
                        new_val = old_val + delta
                        if new_val < 1 or new_val > max_val_limit:
                            continue
                        if new_val in current:
                            continue
                        
                        current[idx] = new_val
                        
                        # Quick evaluation
                        sums = set()
                        prods = set()
                        for u, v in all_edges:
                            sums.add(current[u] + current[v])
                            prods.add(current[u] * current[v])
                        
                        new_score = max(len(sums), len(prods))
                        
                        if new_score < current_score:
                            current_score = new_score
                            improved = True
                        else:
                            current[idx] = old_val
                
                if not improved:
                    break
                
                if current_score < best_score:
                    best_score = current_score
                    best_values = list(current)
        
        return best_values, best_score
    
    def build_compatibility_graph(values, target_k):
        """
        Build a graph where vertices are edges, and two vertices are adjacent
        if adding both edges keeps the number of distinct sums and products <= target_k.
        
        Actually, for clique finding, we need: two edges are compatible if
        the union of their contributions doesn't exceed target_k.
        
        For simplicity, let's use a greedy approach instead of full clique.
        """
        pass
    
    def greedy_edge_selection(values, target_k, rng):
        """
        Greedily select 35 edges such that the number of distinct sums and products
        stays at or below target_k.
        """
        selected_edges = []
        current_sums = set()
        current_prods = set()
        
        # Sort all edges by some heuristic (e.g., by how "dense" they are)
        edge_scores = {}
        for e in all_edges:
            u, v = e
            s = values[u] + values[v]
            p = values[u] * values[v]
            # Prefer edges that create values already in our sets
            edge_scores[e] = (0 if s in current_sums else 1, 0 if p in current_prods else 1)
        
        shuffled_edges = list(all_edges)
        rng.shuffle(shuffled_edges)
        
        for e in shuffled_edges:
            if len(selected_edges) >= m:
                break
            
            u, v = e
            s = values[u] + values[v]
            p = values[u] * values[v]
            
            new_sums_count = len(current_sums | {s})
            new_prods_count = len(current_prods | {p})
            
            if max(new_sums_count, new_prods_count) <= target_k:
                selected_edges.append(e)
                current_sums.add(s)
                current_prods.add(p)
        
        return selected_edges, current_sums, current_prods
    
    def local_search_edges(values, edges, target_k, rng, time_limit):
        """
        Local search to try to find a better set of edges that stays within target_k.
        """
        current_edges = [list(e) for e in edges]
        current_score = evaluate(values, current_edges)
        
        if current_score <= target_k:
            return current_edges, current_score
        
        improved = True
        no_improve = 0
        
        while improved and time.time() < time_limit:
            improved = False
            no_improve = 0
            
            for _ in range(500):
                if time.time() > time_limit:
                    break
                
                i = rng.randint(0, len(current_edges) - 1)
                new_edge = rng.choice(all_edges)
                new_edge_tuple = tuple(new_edge)
                
                # Check if new_edge is already in current_edges
                if any(tuple(e) == new_edge_tuple for e in current_edges):
                    continue
                
                old_edge = current_edges[i]
                old_edge_tuple = tuple(old_edge)
                
                # Try the swap
                current_edges[i] = list(new_edge)
                new_score = evaluate(values, current_edges)
                
                if new_score <= target_k:
                    current_score = new_score
                    improved = True
                    no_improve = 0
                elif new_score < current_score:
                    current_score = new_score
                    improved = True
                    no_improve = 0
                else:
                    current_edges[i] = list(old_edge)
                    no_improve += 1
                
                if no_improve > 200:
                    break
        
        return current_edges, current_score
    
    # Phase 1: Find good values
    rng = random.Random(42)
    best_values, values_score = find_best_values(rng, deadline - 30)
    
    if best_values is None:
        best_values = list(range(1, n + 1))
        values_score = evaluate(best_values, all_edges[:m])
    
    # Phase 2: Find minimum K and corresponding edges
    # Start from a reasonable lower bound
    # The theoretical minimum is at least max(ceil(m/2), something)
    # For 35 edges, we need at least ~18 distinct sums if we're lucky
    lower_bound = max(10, values_score // 2)
    upper_bound = min(values_score, 40)
    
    best_result = None
    best_k = float('inf')
    
    # Try to find the smallest K where we can select 35 edges
    for k in range(lower_bound, upper_bound + 1):
        if time.time() > deadline - 15:
            break
        
        found = False
        for attempt in range(20):
            if time.time() > deadline - 10:
                break
            
            edges, sums, prods = greedy_edge_selection(best_values, k, rng)
            
            if len(edges) == m:
                score = max(len(sums), len(prods))
                if score <= k:
                    # Found a valid solution with this K
                    # Try to improve with local search
                    edges_opt, score_opt = local_search_edges(best_values, edges, k, rng, deadline - 5)
                    
                    if score_opt <= k and (best_result is None or score_opt < best_k):
                        best_k = score_opt
                        best_result = {'values': list(best_values), 'edges': [list(e) for e in edges_opt]}
                        found = True
                        break
        
        if found:
            break
    
    # If we didn't find a solution with the initial values, try other value sets
    if best_result is None or time.time() < deadline - 20:
        # Try a few more value configurations
        alternative_values = [
            list(range(1, n + 1)),
            list(range(2, 2 * n + 1, 2)),
            sorted(rng.sample(range(1, 30), n)),
            [2 ** i + 1 for i in range(n)],
        ]
        
        for vals in alternative_values:
            if time.time() > deadline - 10:
                break
            
            if len(set(vals)) != n or any(v < 1 or v > max_val_limit for v in vals):
                continue
            
            for k in range(10, 35):
                if time.time() > deadline - 5:
                    break
                
                for attempt in range(10):
                    if time.time() > deadline - 3:
                        break
                    
                    edges, sums, prods = greedy_edge_selection(vals, k, rng)
                    
                    if len(edges) == m:
                        score = max(len(sums), len(prods))
                        if score <= k:
                            edges_opt, score_opt = local_search_edges(vals, edges, k, rng, deadline - 2)
                            
                            if score_opt <= k and (best_result is None or score_opt < best_k):
                                best_k = score_opt
                                best_result = {'values': list(vals), 'edges': [list(e) for e in edges_opt]}
                                break
                
                if best_result is not None and best_k <= k:
                    break
    
    # Final fallback
    if best_result is None:
        # Just use a simple heuristic
        values = list(range(1, n + 1))
        edges = list(all_edges[:m])
        best_result = {'values': values, 'edges': edges}
    
    # Final validation and formatting
    values = [int(v) for v in best_result['values']]
    edges = [[int(u), int(v)] for u, v in best_result['edges']]
    
    # Check constraints
    assert len(set(values)) == 14, "Values must be distinct"
    assert all(1 <= v <= 10000 for v in values), "Values must be in range"
    assert len(edges) == 35, "Must have 35 edges"
    assert all(u != v for u, v in edges), "No loops"
    assert len(set(tuple(sorted(e)) for e in edges)) == 35, "No duplicate edges"
    assert all(0 <= u < 14 and 0 <= v < 14 for u, v in edges), "Indices in range"
    
    return {'values': values, 'edges': edges}
# EVOLVE-BLOCK-END
