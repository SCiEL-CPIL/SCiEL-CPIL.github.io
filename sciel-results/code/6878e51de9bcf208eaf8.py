"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import random
from itertools import combinations

# EVOLVE-BLOCK-START

def construct():
    n = 16
    target_edges = 65
    
    # QUBO-inspired Simulated Annealing
    # Objective: Minimize sum of squared triangle degree sums + penalty for max triangle sum
    # This encourages uniform low-degree distribution across triangles.
    
    def compute_triangles(adj):
        """
        adj: numpy array n x n, 0/1 symmetric.
        Returns list of (u, v, w) for all triangles with u < v < w.
        """
        triangles = []
        for u in range(n):
            for v in range(u+1, n):
                if adj[u, v] == 1:
                    for w in range(v+1, n):
                        if adj[u, w] == 1 and adj[v, w] == 1:
                            triangles.append((u, v, w))
        return triangles

    def compute_objective(adj, lambda_max=10.0):
        """
        Compute the QUBO objective: sum of squared triangle degree sums + lambda * max_triangle_sum^2
        Returns (objective_value, max_triangle_sum)
        """
        triangles = compute_triangles(adj)
        if not triangles:
            return 0, 0
        
        degrees = adj.sum(axis=1)
        triangle_sums = []
        for (u, v, w) in triangles:
            s = degrees[u] + degrees[v] + degrees[w]
            triangle_sums.append(s)
        
        sum_of_squares = sum(s * s for s in triangle_sums)
        max_sum = max(triangle_sums)
        objective = sum_of_squares + lambda_max * max_sum * max_sum
        
        return objective, max_sum

    # Initial state: K_{8,8} + edge (0,1)
    adj = np.zeros((n, n), dtype=int)
    for u in range(8):
        for v in range(8, 16):
            adj[u, v] = 1
            adj[v, u] = 1
    adj[0, 1] = 1
    adj[1, 0] = 1
    
    current_edges = int(adj.sum() // 2)
    assert current_edges == 65, f"Initial edges {current_edges} != 65"
    
    current_obj, current_max = compute_objective(adj)
    
    # SA Parameters
    T = 20000
    t_start = 50.0
    t_end = 0.5
    cooling = (t_end / t_start) ** (1.0 / T)
    
    best_adj = adj.copy()
    best_obj = current_obj
    best_max = current_max
    
    # Precompute all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    num_edges = len(all_edges)
    
    for step in range(T):
        t = t_start * (cooling ** step)
        
        # Move: Swap an existing edge with a non-existing edge
        # Pick random edge to remove
        edge_list = [(i, j) for (i, j) in all_edges if adj[i, j] == 1]
        if not edge_list:
            continue
        a, b = random.choice(edge_list)
        
        # Pick random non-edge to add
        non_edge_list = [(i, j) for (i, j) in all_edges if adj[i, j] == 0]
        if not non_edge_list:
            continue
        c, d = random.choice(non_edge_list)
        
        # Apply move
        adj[a, b] = 0
        adj[b, a] = 0
        adj[c, d] = 1
        adj[d, c] = 1
        
        new_obj, new_max = compute_objective(adj)
        
        # Acceptance criterion
        delta = new_obj - current_obj
        if delta < 0:
            current_obj = new_obj
            if new_obj < best_obj:
                best_obj = new_obj
                best_max = new_max
                best_adj = adj.copy()
        else:
            if delta == 0:
                current_obj = new_obj
                if new_max < best_max:
                    best_max = new_max
                    best_adj = adj.copy()
            else:
                prob = np.exp(-delta / t)
                if random.random() < prob:
                    current_obj = new_obj
                else:
                    # Revert
                    adj[a, b] = 1
                    adj[b, a] = 1
                    adj[c, d] = 0
                    adj[d, c] = 0
        
        # Early stopping if we've achieved a very low max sum
        if best_max <= 23:
            break
    
    # Extract edges from best_adj
    edges = []
    for u in range(n):
        for v in range(u+1, n):
            if best_adj[u, v] == 1:
                edges.append([u, v])
                
    return {'n': n, 'edges': edges}

# EVOLVE-BLOCK-END
