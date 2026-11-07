"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    n = 14
    m = 35
    max_val = 10000
    
    all_edges = list(combinations(range(n), 2))
    num_all_edges = len(all_edges)  # 91
    
    def evaluate(values, edges):
        sums = set()
        prods = set()
        for u, v in edges:
            sums.add(values[u] + values[v])
            prods.add(values[u] * values[v])
        return max(len(sums), len(prods))
    
    def evaluate_detailed(values, edges):
        sums = set()
        prods = set()
        for u, v in edges:
            sums.add(values[u] + values[v])
            prods.add(values[u] * values[v])
        return max(len(sums), len(prods)), len(sums), len(prods)
    
    def greedy_edges(values, all_edges, rng):
        """Pick edges greedily to maximize reuse of existing sums/products."""
        sums_seen = set()
        prods_seen = set()
        selected = []
        remaining = list(all_edges)
        rng.shuffle(remaining)
        
        for _ in range(m):
            best_edge = None
            best_score = float('inf')
            for edge in remaining:
                u, v = edge
                s = values[u] + values[v]
                p = values[u] * values[v]
                score = 0
                if s not in sums_seen:
                    score += 1
                if p not in prods_seen:
                    score += 1
                if score < best_score:
                    best_score = score
                    best_edge = edge
            if best_edge is None:
                break
            selected.append(best_edge)
            remaining.remove(best_edge)
            u, v = best_edge
            sums_seen.add(values[u] + values[v])
            prods_seen.add(values[u] * values[v])
        
        return selected
    
    def local_search_edges(values, edges, rng, time_limit):
        """Local search on edges: swap one edge for another."""
        edges_set = set(tuple(e) for e in edges)
        current_score = evaluate(values, edges)
        
        improved = True
        while improved and (time.time() - start_time) < time_limit:
            improved = False
            for _ in range(200):
                if (time.time() - start_time) > time_limit:
                    break
                i = rng.randint(0, len(edges)-1)
                new_edge = rng.choice(all_edges)
                if tuple(new_edge) in edges_set:
                    continue
                
                old_edge = edges[i]
                edges[i] = new_edge
                edges_set.discard(tuple(old_edge))
                edges_set.add(tuple(new_edge))
                
                new_score = evaluate(values, edges)
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                else:
                    edges[i] = old_edge
                    edges_set.discard(tuple(new_edge))
                    edges_set.add(tuple(old_edge))
        
        return edges, current_score
    
    def local_search_values(values, edges, rng, time_limit):
        """Local search on values: perturb one value at a time."""
        current_score = evaluate(values, edges)
        
        improved = True
        while improved and (time.time() - start_time) < time_limit:
            improved = False
            for _ in range(80):
                if (time.time() - start_time) > time_limit:
                    break
                idx = rng.randint(0, n-1)
                old_val = values[idx]
                candidates = set()
                candidates.add(old_val + rng.choice([-3,-2,-1,1,2,3]))
                candidates.add(old_val + rng.choice([-10,-5,5,10]))
                for base in [2, 3, 4, 5, 6, 8, 10]:
                    candidates.add(base * rng.randint(1, 100))
                candidates.add(rng.randint(1, max_val))
                
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
        
        return values, current_score
    
    def find_edges_for_values(values, target_k, time_limit):
        """
        Find 35 edges such that the number of distinct sums <= target_k 
        and number of distinct products <= target_k.
        Uses greedy with constraint checking.
        """
        # Compute sum and product for each edge
        edge_sums = []
        edge_prods = []
        for u, v in all_edges:
            edge_sums.append(values[u] + values[v])
            edge_prods.append(values[u] * values[v])
        
        unique_sums = set(edge_sums)
        unique_prods = set(edge_prods)
        
        # If total unique sums or prods already exceeds target_k, impossible
        if len(unique_sums) > target_k or len(unique_prods) > target_k:
            return None
        
        # Group edges by sum
        sum_to_edges = {}
        for i, s in enumerate(edge_sums):
            if s not in sum_to_edges:
                sum_to_edges[s] = []
            sum_to_edges[s].append(i)
        
        # Group edges by product
        prod_to_edges = {}
        for i, p in enumerate(edge_prods):
            if p not in prod_to_edges:
                prod_to_edges[p] = []
            prod_to_edges[p].append(i)
        
        # Greedy with target constraint
        sums_used = set()
        prods_used = set()
        selected = []
        remaining = list(range(num_all_edges))
        
        # Sort remaining by how "popular" their sum and product are
        def edge_priority(i):
            s_count = len(sum_to_edges[edge_sums[i]])
            p_count = len(prod_to_edges[edge_prods[i]])
            return (s_count, p_count)
        
        remaining.sort(key=edge_priority, reverse=True)
        
        for i in remaining:
            if len(selected) >= m:
                break
            s = edge_sums[i]
            p = edge_prods[i]
            if len(sums_used) < target_k or s in sums_used:
                if len(prods_used) < target_k or p in prods_used:
                    selected.append(i)
                    sums_used.add(s)
                    prods_used.add(p)
        
        if len(selected) >= m:
            return [all_edges[i] for i in selected[:m]]
        
        return None
    
    def find_edges_optimal(values, time_limit):
        """Find the best set of 35 edges for given values using iterative deepening on K."""
        best_edges = None
        best_k = float('inf')
        
        for k in range(1, 40):
            if (time.time() - start_time) > time_limit:
                break
            edges = find_edges_for_values(values, k, time_limit)
            if edges is not None:
                actual_score = evaluate(values, edges)
                if actual_score <= k:
                    best_edges = edges
                    best_k = k
                    break
        
        if best_edges is None:
            best_edges = greedy_edges(values, all_edges, random.Random(42))
            best_k = evaluate(values, best_edges)
        
        return best_edges, best_k
    
    # Main search loop
    best_overall_score = float('inf')
    best_result = None
    rng = random.Random(42)
    
    # Initial value strategies - focus on values that create many sum/product collisions
    value_strategies = [
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192],
        [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28],
        [6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58],
        [1, 2, 3, 6, 12, 24, 48, 96, 192, 384, 768, 1536, 3072, 6144],
        [2, 3, 4, 6, 12, 18, 24, 36, 48, 72, 96, 144, 192, 288],
        [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27],
    ]
    
    for init_values in value_strategies:
        if (time.time() - start_time) > deadline:
            break
        
        # 1. Find optimal edges for these values
        edges, score = find_edges_optimal(init_values, deadline - 5.0)
        
        if score < best_overall_score:
            best_overall_score = score
            best_result = {'values': list(init_values), 'edges': edges}
        
        # 2. Local search on edges
        if (time.time() - start_time) < deadline - 5.0:
            edges2, score2 = local_search_edges(list(init_values), edges, rng, deadline - 5.0)
            if score2 < best_overall_score:
                best_overall_score = score2
                best_result = {'values': list(init_values), 'edges': edges2}
        
        # 3. Local search on values
        if (time.time() - start_time) < deadline - 5.0:
            values, score3 = local_search_values(list(init_values), edges2, rng, deadline - 5.0)
            if score3 < best_overall_score:
                best_overall_score = score3
                best_result = {'values': list(values), 'edges': edges2}
            
            # 4. Refine edges with new values
            if (time.time() - start_time) < deadline - 5.0:
                edges4, score4 = local_search_edges(values, edges2, rng, deadline - 5.0)
                if score4 < best_overall_score:
                    best_overall_score = score4
                    best_result = {'values': list(values), 'edges': edges4}
    
    # Final polish
    if best_result is not None and (time.time() - start_time) < deadline - 2.0:
        values = list(best_result['values'])
        edges = list(best_result['edges'])
        
        for _ in range(15):
            if (time.time() - start_time) > deadline - 2.0:
                break
            values, s1 = local_search_values(values, edges, rng, deadline - 2.0)
            edges, s2 = local_search_edges(values, edges, rng, deadline - 2.0)
            if s2 < best_overall_score:
                best_overall_score = s2
                best_result = {'values': list(values), 'edges': edges}
    
    if best_result is None:
        best_result = {
            'values': list(range(1, n+1)),
            'edges': [list(e) for e in list(all_edges)[:m]]
        }
    
    values = [int(v) for v in best_result['values']]
    edges = [[int(u), int(v)] for u, v in best_result['edges']]
    
    return {'values': values, 'edges': edges}
# EVOLVE-BLOCK-END
