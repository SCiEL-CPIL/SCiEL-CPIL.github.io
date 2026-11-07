import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s kill
    
    n = 14
    m = 35
    
    # Helper to compute score
    def _score(vals, edges):
        sums = set()
        prods = set()
        for u, v in edges:
            a, b = vals[u], vals[v]
            sums.add(a + b)
            prods.add(a * b)
        return max(len(sums), len(prods))

    # Helper to generate all possible edges
    all_possible_edges = list(combinations(range(n), 2))
    
    # Initial candidate set of values
    # 1. Consecutive small integers (good for sums)
    # 2. Random samples (good for products potentially)
    # 3. Structured sets (powers, etc.)
    
    candidates = []
    candidates.append(list(range(1, n + 1)))
    
    # Add some random sets
    for _ in range(10):
        candidates.append(sorted(random.sample(range(1, 10001), n)))
        
    # Add a set with many small factors to maximize product collisions
    # e.g. multiples of 2, 3, 4...
    candidates.append([2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28])
    candidates.append([3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42])
    
    best_result = None
    best_score = float('inf')
    
    for vals in candidates:
        if time.time() > deadline:
            break
            
        # Validate values
        if len(set(vals)) != n:
            continue
        if max(vals) > 10000:
            continue
            
        # Strategy A: Greedy Edge Selection
        # We want to pick m edges such that the number of distinct sums and products is minimized.
        # Greedy: at each step, pick the edge that adds the fewest new distinct sums/products.
        # Since we need to balance both, we can look at the increase in max(count_sums, count_prods).
        
        edges = []
        used_sums = set()
        used_prods = set()
        
        # Precompute sum and product for each edge for this value set
        edge_props = []
        for i, j in all_possible_edges:
            s = vals[i] + vals[j]
            p = vals[i] * vals[j]
            edge_props.append((i, j, s, p))
            
        # Sort edges by some heuristic? Or just iterate?
        # A common heuristic is to prefer edges that share sums/products with already selected edges.
        # Let's do a greedy loop.
        
        remaining = list(range(len(edge_props)))
        random.shuffle(remaining) # Break ties randomly
        
        for _ in range(m):
            if not remaining:
                break
            best_idx = -1
            best_cost = float('inf')
            best_s, best_p = None, None
            best_i, best_j = None, None
            
            # To speed this up, we can't check all remaining edges against all previous ones in O(N^2) per step if N is large.
            # N=91, m=35. 35 * 91 * (cost of update) is fine.
            
            for idx in remaining:
                i, j, s, p = edge_props[idx]
                
                # Check if this edge introduces new sums or products
                new_sums = 0 if s in used_sums else 1
                new_prods = 0 if p in used_prods else 1
                
                # We want to minimize the final max.
                # Current max is M = max(len(used_sums), len(used_prods))
                # New max would be max(len(used_sums)+new_sums, len(used_prods)+new_prods)
                
                cur_max = max(len(used_sums), len(used_prods))
                new_max = max(len(used_sums) + new_sums, len(used_prods) + new_prods)
                
                # Prefer edges that don't increase the max.
                # If they do, prefer the one that increases it less? Or breaks ties by adding fewer total new items?
                cost = new_max
                if cost == best_cost:
                    # Tie breaker: prefer edges that add fewer total new distinct values
                    total_new = new_sums + new_prods
                    if best_s is None or total_new < (1 if best_s not in used_sums else 0) + (1 if best_p not in used_prods else 0):
                         # This logic is getting messy. Let's simplify.
                         pass
                         
                if cost < best_cost:
                    best_cost = cost
                    best_idx = idx
                    best_s = s
                    best_p = p
                    best_i = i
                    best_j = j
            
            if best_idx != -1:
                i, j, s, p = edge_props[best_idx]
                edges.append((i, j))
                used_sums.add(s)
                used_prods.add(p)
                remaining.remove(best_idx)
                
        if len(edges) < m:
            # Fill with any remaining
            for idx in remaining:
                if len(edges) >= m:
                    break
                i, j, s, p = edge_props[idx]
                edges.append((i, j))
                used_sums.add(s)
                used_prods.add(p)
                
        score = _score(vals, edges)
        if score < best_score:
            best_score = score
            best_result = {'values': vals[:], 'edges': [list(e) for e in edges]}
            
        # Strategy B: Local Search / Simulated Annealing on Edges
        # Start with the greedy result and try to swap edges to improve score.
        
        if best_result is not None:
            cur_vals = vals
            cur_edges = [tuple(e) for e in edges]
            cur_score = score
            
            # Precompute properties for all edges again for speed
            edge_props_dict = {}
            for i, j in all_possible_edges:
                s = cur_vals[i] + cur_vals[j]
                p = cur_vals[i] * cur_vals[j]
                edge_props_dict[(i, j)] = (s, p)
                
            # Local search
            for _ in range(500):
                if time.time() > deadline:
                    break
                
                # Pick a random edge to remove
                rem_idx = random.randint(0, len(cur_edges) - 1)
                rem_edge = cur_edges[rem_idx]
                
                # Pick a random edge to add (not currently in set)
                add_edge = random.choice(all_possible_edges)
                if add_edge in cur_edges:
                    continue
                    
                # Create new edge set
                new_edges = cur_edges[:]
                new_edges[rem_idx] = add_edge
                
                # Check for duplicates (shouldn't happen if we pick from all_possible_edges not in set, but just in case)
                if len(set(new_edges)) != m:
                    continue
                    
                new_score = _score(cur_vals, new_edges)
                
                if new_score < cur_score:
                    cur_edges = new_edges
                    cur_score = new_score
                    if cur_score < best_score:
                        best_score = cur_score
                        best_result = {'values': cur_vals[:], 'edges': [list(e) for e in cur_edges]}
                        
                # Allow some worsening moves? Maybe not necessary if greedy is good.
                # Let's stick to hill climbing for now.
                
            # Also try perturbing values slightly?
            # Given time constraints, maybe just stick to edge optimization for fixed values.
            # But we can try a few random value perturbations on the best found so far.
            
    # Final Fallback
    if best_result is None:
        vals = list(range(1, n + 1))
        edges = [list(e) for e in all_possible_edges[:m]]
        best_result = {'values': vals, 'edges': edges}
        
    # Ensure types are correct
    best_result['values'] = [int(v) for v in best_result['values']]
    best_result['edges'] = [[int(a), int(b)] for a, b in best_result['edges']]
    
    return best_result
