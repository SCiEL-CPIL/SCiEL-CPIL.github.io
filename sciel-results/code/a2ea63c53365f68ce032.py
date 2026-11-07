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

def _optimize_edges(values, m=35, deadline=None):
    n = len(values)
    all_edges = list(combinations(range(n), 2))
    
    # Precompute sums and products for all edges
    edge_data = {}
    for i, j in all_edges:
        s = values[i] + values[j]
        p = values[i] * values[j]
        edge_data[(i, j)] = (s, p)
        
    def get_score(edges):
        sums = set()
        prods = set()
        for e in edges:
            s, p = edge_data[e]
            sums.add(s)
            prods.add(p)
        return max(len(sums), len(prods)), len(sums), len(prods)

    best_score = float('inf')
    best_edges = None
    
    # Multiple restarts with Greedy Init
    for restart in range(50):
        if deadline and time.time() > deadline:
            break
            
        # Greedy Init
        current_edges = []
        current_sums = set()
        current_prods = set()
        
        available_edges = list(all_edges)
        random.shuffle(available_edges) # Randomize tie-breaking
        
        for _ in range(m):
            best_edge = None
            best_score_candidate = float('inf')
            best_sums_candidate = set()
            best_prods_candidate = set()
            
            for e in available_edges:
                if e in current_edges: continue
                s, p = edge_data[e]
                new_sums = current_sums | {s}
                new_prods = current_prods | {p}
                score = max(len(new_sums), len(new_prods))
                # Tie break: minimize sum of sizes
                tie_breaker = len(new_sums) + len(new_prods)
                
                if (score, tie_breaker) < (best_score_candidate, len(best_sums_candidate) + len(best_prods_candidate)):
                    best_score_candidate = score
                    best_edge = e
                    best_sums_candidate = new_sums
                    best_prods_candidate = new_prods
                    
            if best_edge is None:
                break
                
            current_edges.append(best_edge)
            current_sums = best_sums_candidate
            current_prods = best_prods_candidate
            available_edges.remove(best_edge)
            
        if len(current_edges) < m:
            continue
            
        current_score, _, _ = get_score(current_edges)
        
        # Local Search: Swaps
        improved = True
        while improved and (not deadline or time.time() < deadline):
            improved = False
            # Shuffle order to avoid bias
            edges_list = list(current_edges)
            other_edges = [e for e in all_edges if e not in current_edges]
            random.shuffle(edges_list)
            random.shuffle(other_edges)
            
            for i, e_in in enumerate(edges_list):
                if deadline and time.time() > deadline:
                    break
                for e_out in other_edges:
                    # Try swap
                    trial_edges = list(current_edges)
                    trial_edges[i] = e_out
                    trial_score, _, _ = get_score(trial_edges)
                    
                    if trial_score < current_score:
                        current_score = trial_score
                        current_edges = trial_edges
                        improved = True
                        # Update best if better than global best
                        if current_score < best_score:
                            best_score = current_score
                            best_edges = list(current_edges)
                        # Break inner loop to restart outer with new state
                        break
                if improved:
                    break
                    
        if current_score < best_score:
            best_score = current_score
            best_edges = list(current_edges)
            
    return best_edges, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    # Candidate values
    # 1. Consecutive small integers
    vals_1 = list(range(1, 15))
    
    # 2. Maybe some other pattern?
    # Small integers are best for collisions.
    # Let's just use vals_1 for now, it's the most promising.
    
    best_raw = float('inf')
    best_result = None
    
    edges, raw = _optimize_edges(vals_1, 35, deadline)
    
    if raw < best_raw:
        best_raw = raw
        best_result = {'values': vals_1, 'edges': edges}
        
    if best_result is None:
        # Fallback
        vals = list(range(1, 15))
        edges = list(combinations(range(14), 2))[:35]
        best_result = {'values': vals, 'edges': edges}
        
    # Format output
    best_result['values'] = [int(v) for v in best_result['values']]
    best_result['edges'] = [[int(i), int(j)] for i, j in best_result['edges']]
    
    return best_result
