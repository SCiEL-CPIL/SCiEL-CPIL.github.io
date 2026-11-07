"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin for 90s limit
    
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
    
    all_edges = list(combinations(range(n), 2))
    
    # Strategy: 
    # 1. Find good value sets (prefer numbers with shared factors to reduce distinct products)
    # 2. For each value set, find good edge subsets to minimize distinct sums/products
    # 3. Use local search (swap edges) to refine
    
    def optimize_edges_for_values(values, max_iter=5000):
        """Greedy + random restart edge optimization for fixed values"""
        best_score = float('inf')
        best_edges = None
        
        for restart in range(10):
            # Start with random subset of edges
            if restart == 0:
                edges = list(all_edges[:m])
            else:
                edges = list(random.sample(all_edges, m))
            
            edges_set = set(tuple(e) for e in edges)
            current_score = evaluate(values, edges)
            
            # Local search: try swapping one edge for another
            improved = True
            iter_count = 0
            while improved and iter_count < max_iter and (time.time() - start_time) < deadline:
                improved = False
                iter_count += 1
                # Try random swaps
                for _ in range(50):
                    if (time.time() - start_time) > deadline:
                        break
                    i = random.randint(0, len(edges)-1)
                    new_edge = random.choice(all_edges)
                    if new_edge in edges_set:
                        continue
                    
                    # Remove edges[i], add new_edge
                    old_edge = edges[i]
                    edges[i] = new_edge
                    edges_set.discard(tuple(old_edge))
                    edges_set.add(tuple(new_edge))
                    
                    new_score = evaluate(values, edges)
                    if new_score < current_score:
                        current_score = new_score
                        improved = True
                    else:
                        # Revert
                        edges[i] = old_edge
                        edges_set.discard(tuple(new_edge))
                        edges_set.add(tuple(old_edge))
                
                if current_score < best_score:
                    best_score = current_score
                    best_edges = list(edges)
        
        return best_edges, best_score
    
    def optimize_values_for_edges(edges, max_iter=3000):
        """Local search on values for fixed edges"""
        best_score = float('inf')
        best_values = None
        
        for restart in range(5):
            # Generate initial values
            if restart == 0:
                # Start with small numbers that have common factors
                values = [2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20, 24]
            else:
                # Random values with some structure
                base = random.choice([2, 3, 4, 5])
                values = []
                used = set()
                for _ in range(n):
                    while True:
                        v = random.randint(1, max_val)
                        if v not in used:
                            used.add(v)
                            values.append(v)
                            break
            
            current_score = evaluate(values, edges)
            
            # Local search: perturb one value at a time
            improved = True
            iter_count = 0
            while improved and iter_count < max_iter and (time.time() - start_time) < deadline:
                improved = False
                iter_count += 1
                for _ in range(20):
                    if (time.time() - start_time) > deadline:
                        break
                    idx = random.randint(0, n-1)
                    old_val = values[idx]
                    # Try nearby values or values with common factors
                    candidates = set()
                    candidates.add(old_val + random.choice([-3,-2,-1,1,2,3]))
                    candidates.add(old_val + random.choice([-10,-5,5,10]))
                    # Try multiples of small numbers
                    for base in [2, 3, 4, 5, 6, 8, 10]:
                        candidates.add(base * random.randint(1, 100))
                    candidates.add(random.randint(1, max_val))
                    
                    for new_val in candidates:
                        if new_val < 1 or new_val > max_val:
                            continue
                        if new_val in values:
                            continue
                        values[idx] = new_val
                        new_score = evaluate(values, edges)
                        if new_score < current_score:
                            current_score = new_score
                            improved = True
                            break
                        else:
                            values[idx] = old_val
                
                if current_score < best_score:
                    best_score = current_score
                    best_values = list(values)
        
        return best_values, best_score
    
    # Main search loop
    best_overall_score = float('inf')
    best_result = None
    
    # Try several different value initialization strategies
    value_strategies = []
    
    # Strategy 1: Powers of 2 and small multiples
    value_strategies.append([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192])
    
    # Strategy 2: Small numbers with common factors
    value_strategies.append([2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20, 24])
    
    # Strategy 3: Multiples of 2
    value_strategies.append([2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28])
    
    # Strategy 4: Multiples of 3
    value_strategies.append([3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42])
    
    # Strategy 5: Mixed small numbers
    value_strategies.append([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
    
    for strat_idx, init_values in enumerate(value_strategies):
        if (time.time() - start_time) > deadline:
            break
        
        # First, find good edges for this value set
        edges, edge_score = optimize_edges_for_values(init_values, max_iter=2000)
        
        if edge_score < best_overall_score:
            best_overall_score = edge_score
            best_result = {'values': list(init_values), 'edges': edges}
        
        # Now try to improve values with these edges
        if edges is not None:
            improved_values, val_score = optimize_values_for_edges(edges, max_iter=2000)
            if val_score < best_overall_score:
                best_overall_score = val_score
                best_result = {'values': list(improved_values), 'edges': edges}
        
        # Alternate: improve edges again with the new values
        if improved_values is not None:
            edges2, edge_score2 = optimize_edges_for_values(improved_values, max_iter=1000)
            if edge_score2 < best_overall_score:
                best_overall_score = edge_score2
                best_result = {'values': list(improved_values), 'edges': edges2}
    
    # Final polish: if we have time, do more local search
    if best_result is not None and (time.time() - start_time) < deadline - 5:
        values = list(best_result['values'])
        edges = list(best_result['edges'])
        current_score = evaluate(values, edges)
        
        # Try more edge swaps
        for _ in range(2000):
            if (time.time() - start_time) > deadline - 5:
                break
            i = random.randint(0, len(edges)-1)
            new_edge = random.choice(all_edges)
            if new_edge in edges:
                continue
            old_edge = edges[i]
            edges[i] = new_edge
            new_score = evaluate(values, edges)
            if new_score < current_score:
                current_score = new_score
            else:
                edges[i] = old_edge
        
        # Try more value tweaks
        for _ in range(2000):
            if (time.time() - start_time) > deadline - 5:
                break
            idx = random.randint(0, n-1)
            old_val = values[idx]
            new_val = old_val + random.choice([-5,-3,-2,-1,1,2,3,5])
            if new_val < 1 or new_val > max_val:
                continue
            if new_val in values:
                continue
            values[idx] = new_val
            new_score = evaluate(values, edges)
            if new_score < current_score:
                current_score = new_score
            else:
                values[idx] = old_val
        
        best_result = {'values': values, 'edges': edges}
    
    # Ensure validity
    if best_result is None:
        # Fallback to simple solution
        best_result = {
            'values': list(range(1, n+1)),
            'edges': [list(e) for e in list(all_edges)[:m]]
        }
    
    # Ensure types are correct
    values = [int(v) for v in best_result['values']]
    edges = [[int(u), int(v)] for u, v in best_result['edges']]
    
    return {'values': values, 'edges': edges}
# EVOLVE-BLOCK-END
