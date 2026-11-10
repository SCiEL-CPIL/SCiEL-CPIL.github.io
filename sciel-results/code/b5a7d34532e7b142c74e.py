"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    
    n = 14
    m = 35
    maximum = 10000
    
    def _raw_value(values, edges):
        sums = set()
        prods = set()
        for i, j in edges:
            sums.add(values[i] + values[j])
            prods.add(values[i] * values[j])
        return max(len(sums), len(prods))
    
    # Strategy: Use values that encourage sum collisions
    # Try several value configurations and pick best edges
    
    best_result = None
    best_score = float('inf')
    
    # Candidate value sets
    # 1) Consecutive integers
    # 2) Powers of 2 (products unique, sums may collide)
    # 3) Symmetric values around center
    # 4) Random sets
    
    candidate_values = []
    
    # Consecutive
    candidate_values.append(list(range(1, n + 1)))
    
    # Powers of 2
    candidate_values.append([2**i for i in range(n)])
    
    # Symmetric: 1,2,3,4,5,6,7,8,9,10,11,12,13,14 -> already tried
    # Try: values clustered to create sum collisions
    # e.g., 1,2,3,4, 10,11,12,13, 20,21,22,23, 30,31
    candidate_values.append([1,2,3,4,10,11,12,13,20,21,22,23,30,31])
    
    # Another: all odd numbers
    candidate_values.append(list(range(1, 2*n+1, 2)))
    
    # Random sets
    random.seed(42)
    for _ in range(50):
        vals = random.sample(range(1, maximum+1), n)
        candidate_values.append(sorted(vals))
    
    for vals in candidate_values:
        if time.time() > deadline - 5.0:
            break
        
        # For this value set, find best 35 edges
        # Greedy: sort all edges by (sum, product) and pick to minimize distinct sums/products
        all_edges = list(combinations(range(n), 2))
        
        # Compute sum and product for each edge
        edge_info = []
        for i, j in all_edges:
            s = vals[i] + vals[j]
            p = vals[i] * vals[j]
            edge_info.append((i, j, s, p))
        
        # Try different edge selection strategies
        
        # Strategy A: Pick edges that share common sums
        # Group edges by sum, pick from most common sums first
        from collections import defaultdict
        sum_groups = defaultdict(list)
        for i, j, s, p in edge_info:
            sum_groups[s].append((i, j, p))
        
        # Sort sums by group size (descending)
        sorted_sums = sorted(sum_groups.keys(), key=lambda s: -len(sum_groups[s]))
        
        edges_a = []
        prods_a = set()
        for s in sorted_sums:
            if len(edges_a) >= m:
                break
            for i, j, p in sum_groups[s]:
                if len(edges_a) >= m:
                    break
                if (i, j) not in [(a, b) for a, b, *_ in edges_a]:
                    edges_a.append((i, j))
                    prods_a.add(p)
        
        if len(edges_a) == m:
            score_a = max(len(set(vals[i]+vals[j] for i,j in edges_a)), len(prods_a))
            if score_a < best_score:
                best_score = score_a
                best_result = {'values': list(vals), 'edges': [[i, j] for i, j in edges_a]}
        
        # Strategy B: Pick edges that share common products (harder)
        # Strategy C: Random sampling with local search
        for attempt in range(200):
            if time.time() > deadline - 5.0:
                break
            
            # Random start
            random.shuffle(all_edges)
            edges_c = [list(e) for e in all_edges[:m]]
            
            # Local search: try swapping edges to reduce objective
            current_score = _raw_value(vals, edges_c)
            
            for _ in range(50):
                if time.time() > deadline - 5.0:
                    break
                improved = False
                for _swap in range(100):
                    idx1 = random.randint(0, m-1)
                    idx2 = random.randint(0, m-1)
                    if idx1 == idx2:
                        continue
                    new_edge = list(random.choice(all_edges))
                    if new_edge in edges_c:
                        continue
                    old_edge = edges_c[idx1]
                    edges_c[idx1] = new_edge
                    new_score = _raw_value(vals, edges_c)
                    if new_score < current_score:
                        current_score = new_score
                        improved = True
                        break
                    else:
                        edges_c[idx1] = old_edge
                
                if not improved:
                    break
            
            if current_score < best_score:
                best_score = current_score
                best_result = {'values': list(vals), 'edges': [list(e) for e in edges_c]}
    
    if best_result is None:
        # Fallback
        vals = list(range(1, n+1))
        edges = [list(e) for e in list(combinations(range(n), 2))[:m]]
        best_result = {'values': vals, 'edges': edges}
    
    # Ensure types are correct
    best_result['values'] = [int(v) for v in best_result['values']]
    best_result['edges'] = [[int(i), int(j)] for i, j in best_result['edges']]
    
    return best_result
# EVOLVE-BLOCK-END
