"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import random
from itertools import combinations

# EVOLVE-BLOCK-START

def construct():
    n = 16
    target_edges = 65
    
    # QUBO-inspired approach:
    # Minimize sum of squared triangle degree sums to encourage uniform low-degree distribution.
    # Use simulated annealing with this objective.
    
    def compute_triangle_sums(adj):
        """
        adj: numpy array n x n, 0/1 symmetric.
        Returns list of (u, v, w) for all triangles.
        """
        triangles = []
        for u in range(n):
            for v in range(u+1, n):
                if adj[u, v] == 1:
                    for w in range(v+1, n):
                        if adj[u, w] == 1 and adj[v, w] == 1:
                            triangles.append((u, v, w))
        return triangles

    def compute_qubo_objective(adj):
        """
        Compute sum of squared triangle degree sums.
        This encourages uniform low degrees across all triangles.
        """
        triangles = compute_triangle_sums(adj)
        if not triangles:
            return 0.0
        
        total_sq_sum = 0.0
        for (u, v, w) in triangles:
            s = adj[u].sum() + adj[v].sum() + adj[w].sum()
            total_sq_sum += s * s
        
        return total_sq_sum

    def get_max_triangle_sum(adj):
        """
        Returns the maximum triangle degree sum (the actual score).
        """
        triangles = compute_triangle_sums(adj)
        if not triangles:
            return 0
        max_sum = 0
        for (u, v, w) in triangles:
            s = adj[u].sum() + adj[v].sum() + adj[w].sum()
            if s > max_sum:
                max_sum = s
        return max_sum

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
    
    current_qubo = compute_qubo_objective(adj)
    current_max_sum = get_max_triangle_sum(adj)
    
    # SA Parameters
    T = 8000  # More steps for better exploration
    t_start = 50.0
    t_end = 0.5
    cooling = (t_end / t_start) ** (1.0 / T)
    
    best_adj = adj.copy()
    best_max_sum = current_max_sum
    best_qubo = current_qubo
    
    # Precompute edge indices for fast access
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    edge_count = len(all_edges)  # 120
    
    for step in range(T):
        t = t_start * (cooling ** step)
        
        # Pick random edge to remove and random non-edge to add
        # Maintain lists of edges and non-edges for efficiency
        edge_indices = [idx for idx, (i, j) in enumerate(all_edges) if adj[i, j] == 1]
        non_edge_indices = [idx for idx, (i, j) in enumerate(all_edges) if adj[i, j] == 0]
        
        if not edge_indices or not non_edge_indices:
            continue
        
        a_idx = random.choice(edge_indices)
        c_idx = random.choice(non_edge_indices)
        
        a, b = all_edges[a_idx]
        c, d = all_edges[c_idx]
        
        # Apply move
        adj[a, b] = 0
        adj[b, a] = 0
        adj[c, d] = 1
        adj[d, c] = 1
        
        new_qubo = compute_qubo_objective(adj)
        new_max_sum = get_max_triangle_sum(adj)
        
        # Use QUBO objective for acceptance, but track max sum as the true score
        delta_qubo = new_qubo - current_qubo
        
        accepted = False
        if delta_qubo < 0:
            accepted = True
        else:
            # Accept with probability exp(-delta / t)
            if t > 1e-10:
                prob = np.exp(-delta_qubo / t)
            else:
                prob = 0.0
            if random.random() < prob:
                accepted = True
        
        if accepted:
            current_qubo = new_qubo
            current_max_sum = new_max_sum
            
            # Update best based on max sum (the actual metric)
            if new_max_sum < best_max_sum:
                best_max_sum = new_max_sum
                best_adj = adj.copy()
                best_qubo = new_qubo
        else:
            # Revert move
            adj[a, b] = 1
            adj[b, a] = 1
            adj[c, d] = 0
            adj[d, c] = 0
        
        # Early stopping if we reach theoretical minimum
        if best_max_sum <= 23:
            break
    
    # Extract edges from best_adj
    edges = []
    for u in range(n):
        for v in range(u+1, n):
            if best_adj[u, v] == 1:
                edges.append([u, v])
                
    return {'n': n, 'edges': edges}

# EVOLVE-BLOCK-END
