import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 50s safety margin
    
    n = 14
    m = 35
    values = list(range(1, n + 1))
    
    # Precompute sums and products for all edges
    all_edges = list(combinations(range(n), 2))
    edge_sums = {}
    edge_prods = {}
    for u, v in all_edges:
        s = values[u] + values[v]
        p = values[u] * values[v]
        edge_sums[(u, v)] = s
        edge_prods[(u, v)] = p

    # Helper to calculate score from a set of edges
    def calc_score(edges):
        sums = set()
        prods = set()
        for u, v in edges:
            sums.add(edge_sums[(u, v)])
            prods.add(edge_prods[(u, v)])
        return max(len(sums), len(prods))

    # 1. Greedy Initialization
    current_edges = []
    
    available_edges = list(all_edges)
    for _ in range(m):
        best_edge = None
        best_score = float('inf')
        for e in available_edges:
            # Simulate adding
            temp_edges = current_edges + [e]
            score = calc_score(temp_edges)
            if score < best_score:
                best_score = score
                best_edge = e
        
        if best_edge is not None:
            current_edges.append(best_edge)
            available_edges.remove(best_edge)

    current_score = calc_score(current_edges)
    best_score = current_score
    best_edges = list(current_edges)
    current_edge_set = set(current_edges)
    
    # 2. Local Search with Simulated Annealing
    T = 5.0
    cooling = 0.999
    min_T = 0.01
    
    while time.time() < deadline and T > min_T:
        # Pick a random edge to remove
        idx_remove = random.randint(0, m - 1)
        e_remove = current_edges[idx_remove]
        
        # Pick a random edge to add
        while True:
            e_add = random.choice(all_edges)
            if e_add not in current_edge_set:
                break
                
        # Perform swap
        current_edges[idx_remove] = e_add
        current_edge_set.remove(e_remove)
        current_edge_set.add(e_add)
        
        new_score = calc_score(current_edges)
        
        # Acceptance criterion
        if new_score < current_score:
            accept = True
        elif new_score == current_score:
            accept = random.random() < 0.5 # Plateau wandering
        else:
            # Simulated Annealing
            delta = new_score - current_score
            accept = random.random() < pow(2.71828, -delta / T)
            
        if accept:
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_edges = list(current_edges)
        else:
            # Revert
            current_edges[idx_remove] = e_remove
            current_edge_set.remove(e_add)
            current_edge_set.add(e_remove)
            
        T *= cooling

    return {
        'values': values,
        'edges': [list(e) for e in best_edges]
    }
