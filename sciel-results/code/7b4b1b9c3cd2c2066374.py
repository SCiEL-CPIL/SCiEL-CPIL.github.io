import random
import time

def construct():
    start_time = time.time()
    timeout = 55  # Leave 5s margin for safety
    
    # 1. Define the values. 
    # Using 1..14 is a strong baseline because it has many additive collisions (sums).
    # We can also try other sets, but 1..14 is standard and effective for sums.
    values = list(range(1, 15))
    
    # 2. Pre-calculate properties for all possible edges
    all_edges = []
    for i in range(14):
        for j in range(i + 1, 14):
            s = values[i] + values[j]
            p = values[i] * values[j]
            all_edges.append((i, j, s, p))
    
    # 3. Greedy Edge Selection
    # We want to pick 35 edges such that the set of sums and products is as small as possible.
    # Heuristic: At each step, pick the edge that minimizes the resulting max(|S|, |P|).
    # To break ties, we can look at how "popular" the sum/product is among remaining edges,
    # or just use randomization to avoid getting stuck in a specific local minimum of the greedy choice.
    
    candidates = list(all_edges)
    current_S = set()
    current_P = set()
    selected_edges = []
    
    # To improve on pure greedy, we can run multiple passes with slight randomization
    # or a "lookahead" of 1-2 steps, but pure greedy is a good start for a new approach.
    # Let's add a small randomization: among edges that yield the same minimal cost,
    # pick one randomly. This helps explore different topologies.
    
    best_edges = None
    best_cost = float('inf')
    
    # Run a few iterations with different random seeds for tie-breaking to find a good topology
    for iteration in range(20):
        cands = list(all_edges)
        S = set()
        P = set()
        edges = []
        
        while len(edges) < 35:
            best_cost_for_step = float('inf')
            best_edge_for_step = None
            best_sum_for_step = None
            best_prod_for_step = None
            
            # To speed up, we can group or just iterate. 91 edges is small.
            # We need to find the edge that minimizes max(len(S | {s}), len(P | {p}))
            
            # Optimization: 
            # Adding an edge with sum s and prod p changes |S| by 0 or 1, and |P| by 0 or 1.
            # new_len_S = len(S) + (0 if s in S else 1)
            # new_len_P = len(P) + (0 if p in P else 1)
            # cost = max(new_len_S, new_len_P)
            
            min_cost = float('inf')
            viable_edges = []
            
            for e in cands:
                i, j, s, p = e
                new_len_S = len(S) + (0 if s in S else 1)
                new_len_P = len(P) + (0 if p in P else 1)
                cost = max(new_len_S, new_len_P)
                
                if cost < min_cost:
                    min_cost = cost
                    viable_edges = [e]
                elif cost == min_cost:
                    viable_edges.append(e)
            
            if not viable_edges:
                break
                
            # Pick randomly from viable edges to diversify
            chosen = random.choice(viable_edges)
            edges.append(chosen)
            S.add(chosen[2])
            P.add(chosen[3])
            
            # Remove chosen edge from candidates
            cands.remove(chosen)
            
        current_cost = max(len(S), len(P))
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_edges = [e[:2] for e in edges]
            
        if time.time() - start_time > timeout - 5:
            break

    # If greedy failed to produce 35 edges (unlikely), fallback
    if not best_edges or len(best_edges) != 35:
        best_edges = [e[:2] for e in random.sample(all_edges, 35)]
        # Recalculate cost for fallback if needed, but structure is valid

    final_values = [int(v) for v in values]
    final_edges = [[int(u), int(v)] for u, v in best_edges]
    
    # Validation
    if len(final_values) != 14 or len(set(final_values)) != 14:
        final_values = list(range(1, 15))
    if len(final_edges) != 35:
        final_edges = [list(e) for e in random.sample(all_edges, 35)]
        
    return {'values': final_values, 'edges': final_edges}
