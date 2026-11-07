import time
import random
from itertools import combinations
from collections import Counter

P = {'n': 14, 'm': 35, 'maximum': 10000}

def construct():
    deadline = time.time() + 55
    n = P['n']
    m = P['m']
    all_edges = list(combinations(range(n), 2))
    
    # Candidate value structures
    # We want values that create few distinct sums and products.
    # AP: sums depend on i+j. Products depend on i*j (scaled).
    # Multiples of a large constant C: sums = C*(i+j), prods = C^2*i*j.
    # This decouples the scale from the combinatorial structure of i,j.
    
    def evaluate(values, edges):
        sums = set()
        prods = set()
        for i, j in edges:
            sums.add(values[i] + values[j])
            prods.add(values[i] * values[j])
        return max(len(sums), len(prods))

    def greedy_edges(values):
        # Precompute sums and products for all edges
        edge_data = []
        for i, j in all_edges:
            s = values[i] + values[j]
            p = values[i] * values[j]
            edge_data.append(((i, j), s, p))
        
        # Sort edges? No, greedy selection based on current state.
        current_sums = set()
        current_prods = set()
        selected_edges = []
        selected_set = set()
        remaining = list(edge_data)
        
        for _ in range(m):
            best_idx = -1
            best_key = None
            
            # To speed up, we can just iterate. 91 edges is small.
            for idx, (e, s, p) in enumerate(remaining):
                if e in selected_set:
                    continue
                
                new_s = 0 if s in current_sums else 1
                new_p = 0 if p in current_prods else 1
                
                # We want to minimize the final max.
                # Heuristic:
                # If current_sums < current_prods, prefer edges that don't add new sums.
                # If current_prods < current_sums, prefer edges that don't add new prods.
                # If equal, prefer edges that add neither, then one, then two.
                
                len_s = len(current_sums)
                len_p = len(current_prods)
                
                # Key: (new_sum_if_len_s_le_len_p, new_prod_if_len_p_le_len_s, new_s, new_p)
                # Simpler: Just minimize (new_s + new_p, new_s, new_p)
                # But balancing is important.
                
                # Let's try: cost = max(len_s + new_s, len_p + new_p)
                # We want to minimize this projected cost.
                proj_s = len_s + new_s
                proj_p = len_p + new_p
                proj_max = max(proj_s, proj_p)
                
                # Tie breaker: prefer adding to the smaller set to keep them balanced?
                # Or prefer adding to the larger set to catch up?
                # If len_s < len_p, adding to sums (new_s=1) is good because it catches up.
                # Adding to prods (new_p=1) makes the gap larger.
                
                # So:
                # If len_s <= len_p:
                #   Prefer new_s=1 (catch up) over new_s=0?
                #   Actually, if we add to the smaller one, the max might not increase.
                #   If len_s = 5, len_p = 10.
                #   Add to sums: 6, 10 -> max 10.
                #   Add to prods: 5, 11 -> max 11.
                #   So adding to the smaller one is better.
                
                # If len_s == len_p:
                #   Add to either: k+1, k -> max k+1.
                #   Add to both: k+1, k+1 -> max k+1.
                
                # So the cost function max(proj_s, proj_p) handles this naturally.
                
                key = (proj_max, new_s + new_p, new_s, new_p)
                
                if best_key is None or key < best_key:
                    best_key = key
                    best_idx = idx
            
            if best_idx == -1:
                break
            
            e, s, p = remaining.pop(best_idx)
            selected_edges.append(list(e))
            selected_set.add(e)
            current_sums.add(s)
            current_prods.add(p)
            
        return selected_edges

    def local_search(values, edges, max_iters=200000):
        edges = [tuple(e) for e in edges]
        edge_set = set(edges)
        
        current_sums = set()
        current_prods = set()
        for i, j in edges:
            current_sums.add(values[i] + values[j])
            current_prods.add(values[i] * values[j])
            
        cur_score = max(len(current_sums), len(current_prods))
        best_score = cur_score
        best_edges = [tuple(e) for e in edges]
        
        # Precompute all edge sums/prods for quick lookup
        all_edge_sums = {}
        all_edge_prods = {}
        for i, j in all_edges:
            all_edge_sums[(i, j)] = values[i] + values[j]
            all_edge_prods[(i, j)] = values[i] * values[j]
            
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
                
            # Pick a random edge to remove
            idx = random.randint(0, m - 1)
            old_edge = edges[idx]
            
            # Pick a random edge to add (not in current set)
            # To be efficient, pick from all_edges and check membership
            new_edge = random.choice(all_edges)
            if new_edge in edge_set:
                continue
                
            # Calculate delta
            old_s = all_edge_sums[old_edge]
            old_p = all_edge_prods[old_edge]
            new_s = all_edge_sums[new_edge]
            new_p = all_edge_prods[new_edge]
            
            # Simulate removal
            # We need counts to know if a value disappears from the set
            # For simplicity, we can just recalculate the sets if the change is small?
            # No, recalculating 35 edges is fast.
            
            # Let's just recalculate the score for the candidate move.
            # It's O(m) which is 35. 200k iters * 35 ops is 7M ops. Should be fast enough in Python?
            # Maybe. Let's try to optimize with counts if needed.
            
            # Actually, maintaining counts is better.
            
            # Let's restart local search with counts for speed.
            break # Placeholder, will implement proper count-based LS below

        # Proper Count-Based Local Search
        sum_count = Counter()
        prod_count = Counter()
        for i, j in edges:
            sum_count[all_edge_sums[(i, j)]] += 1
            prod_count[all_edge_prods[(i, j)]] += 1
            
        cur_score = max(len(sum_count), len(prod_count))
        best_score = cur_score
        best_edges = [tuple(e) for e in edges]
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
                
            idx = random.randint(0, m - 1)
            old_edge = edges[idx]
            new_edge = random.choice(all_edges)
            if new_edge in edge_set:
                continue
                
            old_s = all_edge_sums[old_edge]
            old_p = all_edge_prods[old_edge]
            new_s = all_edge_sums[new_edge]
            new_p = all_edge_prods[new_edge]
            
            # Remove old
            sum_count[old_s] -= 1
            if sum_count[old_s] == 0:
                del sum_count[old_s]
            prod_count[old_p] -= 1
            if prod_count[old_p] == 0:
                del prod_count[old_p]
                
            # Add new
            sum_count[new_s] += 1
            prod_count[new_p] += 1
            
            new_score = max(len(sum_count), len(prod_count))
            
            if new_score <= cur_score:
                edges[idx] = new_edge
                edge_set.remove(old_edge)
                edge_set.add(new_edge)
                cur_score = new_score
                if cur_score < best_score:
                    best_score = cur_score
                    best_edges = [tuple(e) for e in edges]
            else:
                # Revert
                sum_count[new_s] -= 1
                if sum_count[new_s] == 0:
                    del sum_count[new_s]
                prod_count[new_p] -= 1
                if prod_count[new_p] == 0:
                    del prod_count[new_p]
                    
                sum_count[old_s] += 1
                prod_count[old_p] += 1
                
        return best_edges, best_score

    best_overall_score = float('inf')
    best_overall_edges = None
    
    # Try different value structures
    structures = [
        list(range(1, 15)),
        list(range(10, 150, 10)),
        list(range(100, 1500, 100)),
        list(range(1000, 15000, 1000))[:14], # Check max
        [i * 7 for i in range(1, 15)],
        [i * 11 for i in range(1, 15)],
        [i * 13 for i in range(1, 15)],
    ]
    
    # Filter valid structures
    valid_structures = []
    for s in structures:
        if len(set(s)) == n and all(1 <= v <= P['maximum'] for v in s):
            valid_structures.append(s)
            
    # Add some random structures
    for _ in range(10):
        vals = random.sample(range(1, P['maximum'] + 1), n)
        valid_structures.append(vals)
        
    for vals in valid_structures:
        if time.time() > deadline - 10:
            break
            
        edges = greedy_edges(vals)
        if len(edges) < m:
            continue
            
        best_edges, score = local_search(vals, edges, max_iters=50000)
        
        if score < best_overall_score:
            best_overall_score = score
            best_overall_edges = best_edges
            best_values = vals
            
    if best_overall_edges is None:
        # Fallback
        best_values = list(range(1, 15))
        best_overall_edges = greedy_edges(best_values)
        best_overall_score = evaluate(best_values, best_overall_edges)
        
    final_values = [int(v) for v in best_values]
    final_edges = [[int(i), int(j)] for i, j in best_overall_edges]
    
    # Assertions
    assert len(final_values) == n
    assert len(set(final_values)) == n
    assert all(isinstance(v, int) and 1 <= v <= P['maximum'] for v in final_values)
    assert len(final_edges) == m
    edge_tuples = set()
    for e in final_edges:
        i, j = e
        assert 0 <= i < n and 0 <= j < n and i != j
        t = tuple(sorted((i, j)))
        assert t not in edge_tuples
        edge_tuples.add(t)
        
    return {'values': final_values, 'edges': final_edges}
