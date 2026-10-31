import numpy as np

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return destination_node
        
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
        
    if n_unvisited == 2:
        c1 = unvisited_nodes[0]
        c2 = unvisited_nodes[1]
        cost1 = distance_matrix[current_node, c1] + distance_matrix[c1, c2] + distance_matrix[c2, destination_node]
        cost2 = distance_matrix[current_node, c2] + distance_matrix[c2, c1] + distance_matrix[c1, destination_node]
        return int(c1) if cost1 <= cost2 else int(c2)
        
    if n_unvisited == 3:
        c1, c2, c3 = unvisited_nodes[0], unvisited_nodes[1], unvisited_nodes[2]
        # 6 permutations
        perms = [
            (c1, c2, c3),
            (c1, c3, c2),
            (c2, c1, c3),
            (c2, c3, c1),
            (c3, c1, c2),
            (c3, c2, c1)
        ]
        best_cost = float('inf')
        best_next = c1
        for p in perms:
            cost = distance_matrix[current_node, p[0]] + distance_matrix[p[0], p[1]] + distance_matrix[p[1], p[2]] + distance_matrix[p[2], destination_node]
            if cost < best_cost:
                best_cost = cost
                best_next = p[0]
        return int(best_next)

    # For n > 3, use 2-step look-ahead with adaptive destination bias
    # We want to avoid getting stuck far from the destination.
    # Let's define lambda based on how many nodes are left.
    # When n=50, lambda=0.1 (mostly local)
    # When n=10, lambda=0.9 (strongly destination)
    # Linear interpolation: lambda = 0.1 + 0.8 * (1 - n/50)
    lam = 0.1 + 0.8 * (1.0 - n_unvisited / 50.0)
    
    best_cost = float('inf')
    best_next = unvisited_nodes[0]
    
    # Extract relevant rows for speed if needed, but loop is fine for N=50
    dm = distance_matrix
    
    for c in unvisited_nodes:
        # Cost to reach c from current
        cost_to_c = dm[current_node, c]
        
        # Find best next node c' from remaining unvisited
        # To avoid O(N^2) inside the loop being too slow, we can pre-compute or just loop.
        # N is small (<= 50), so O(N^2) per step is ~2500 ops, total ~100k ops. Very fast.
        
        min_next_cost = float('inf')
        for c2 in unvisited_nodes:
            if c2 == c:
                continue
            # Cost of c -> c2 + lambda * distance(c2, dest)
            # This approximates the future cost if we go c -> c2
            next_cost = dm[c, c2] + lam * dm[c2, destination_node]
            if next_cost < min_next_cost:
                min_next_cost = next_cost
        
        total_cost = cost_to_c + min_next_cost
        
        if total_cost < best_cost:
            best_cost = total_cost
            best_next = c
            
    return int(best_next)
