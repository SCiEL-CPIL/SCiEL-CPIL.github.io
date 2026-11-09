"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import random
from itertools import combinations

# EVOLVE-BLOCK-START

def construct():
    n = 16
    target_edges = 65
    
    # Precompute all possible edges and triangle triples for speed
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    all_pairs = all_edges
    num_pairs = len(all_pairs)
    
    # Map edge index to pair
    edge_to_idx = {e: i for i, e in enumerate(all_pairs)}
    
    # Precompute triangles as list of (u, v, w)
    # and for each triangle, the list of edge indices involved
    triangles = []
    tri_edges = []
    for u in range(n):
        for v in range(u + 1, n):
            for w in range(v + 1, n):
                # Triangle (u, v, w) exists if edges (u,v), (v,w), (u,w) are present
                # We store the indices of these edges
                e_uv = edge_to_idx[(u, v)]
                e_vw = edge_to_idx[(v, w)]
                e_uw = edge_to_idx[(u, w)]
                triangles.append((u, v, w))
                tri_edges.append((e_uv, e_vw, e_uw))
                
    num_triangles = len(triangles)
    
    # Objective function:
    # Minimize sum(TriangleSum^2) + Penalty * (edge_count - target_edges)^2
    # TriangleSum = deg(u) + deg(v) + deg(w)
    # deg(v) = sum of adj[v, :]
    
    def get_scores(adj):
        """
        Returns (max_sum, sum_sq, edge_count)
        adj: numpy array n x n
        """
        degrees = adj.sum(axis=1)
        max_sum = 0
        sum_sq = 0
        edge_count = int(adj.sum() // 2)
        
        # Iterate over triangles
        # For speed, we can vectorize or just loop. n=16 is small.
        for u, v, w in triangles:
            s = degrees[u] + degrees[v] + degrees[w]
            if s > max_sum:
                max_sum = s
            sum_sq += s * s
            
        return max_sum, sum_sq, edge_count

    # Initial state: Random graph with target_edges edges
    # Or start with K8,8 + 1 edge as a good baseline
    adj = np.zeros((n, n), dtype=np.int8)
    
    # Start with K8,8
    for u in range(8):
        for v in range(8, 16):
            adj[u, v] = 1
            adj[v, u] = 1
    # Add edge (0,1)
    adj[0, 1] = 1
    adj[1, 0] = 1
    
    current_max, current_sq, current_edges = get_scores(adj)
    assert current_edges == target_edges, f"Initial edges {current_edges} != {target_edges}"
    
    best_adj = adj.copy()
    best_max = current_max
    best_sq = current_sq
    
    # Simulated Annealing Parameters
    T = 8000
    t_start = 10.0
    t_end = 0.1
    cooling = (t_end / t_start) ** (1.0 / T)
    
    # Precompute list of edge indices for random selection
    edge_indices = np.arange(num_pairs)
    
    # To pick random edge/non-edge efficiently, we maintain a list of current edges?
    # Or just pick random pairs and check.
    # Given n=16, num_pairs=120. Picking random pairs is fast.
    
    for step in range(T):
        t = t_start * (cooling ** step)
        
        # Move: Swap one edge for another
        # Pick random pair 1 and random pair 2
        idx1 = random.choice(edge_indices)
        idx2 = random.choice(edge_indices)
        
        if idx1 == idx2:
            continue
            
        e1 = all_pairs[idx1]
        e2 = all_pairs[idx2]
        
        state1 = adj[e1[0], e1[1]]
        state2 = adj[e2[0], e2[1]]
        
        # If both are 0 or both are 1, swapping does nothing to the graph structure (symmetry)
        # We want to change the graph. So we require one to be 0 and the other 1.
        if state1 == state2:
            continue
            
        # Apply move
        # If state1 is 1, remove e1, add e2
        # If state1 is 0, add e1, remove e2
        
        # Update adj
        if state1 == 1:
            # Remove e1
            adj[e1[0], e1[1]] = 0
            adj[e1[1], e1[0]] = 0
            # Add e2
            adj[e2[0], e2[1]] = 1
            adj[e2[1], e2[0]] = 1
        else:
            # Add e1
            adj[e1[0], e1[1]] = 1
            adj[e1[1], e1[0]] = 1
            # Remove e2
            adj[e2[0], e2[1]] = 0
            adj[e2[1], e2[0]] = 0
            
        new_max, new_sq, new_edges = get_scores(adj)
        
        # Objective: We want to minimize max_sum primarily.
        # But to avoid local minima, we use a combined score.
        # Score = max_sum * 100 + sum_sq / 100? 
        # Or just minimize max_sum, and use sum_sq as tie-breaker or secondary SA energy.
        # Let's define energy E = max_sum * 10 + new_sq / 10.0
        # Actually, the task is to minimize max_sum.
        # If we minimize sum_sq, we might reduce max_sum indirectly.
        # Let's try: Energy = max_sum * 100 + sum_sq
        # This heavily prioritizes reducing the max sum.
        
        curr_energy = current_max * 100 + current_sq
        new_energy = new_max * 100 + new_sq
        
        delta = new_energy - curr_energy
        
        accept = False
        if delta < 0:
            accept = True
        else:
            if t > 0:
                prob = np.exp(-delta / t)
                if random.random() < prob:
                    accept = True
        
        if accept:
            current_max = new_max
            current_sq = new_sq
            current_edges = new_edges
            if current_max < best_max:
                best_max = current_max
                best_sq = current_sq
                best_adj = adj.copy()
        else:
            # Revert
            if state1 == 1:
                adj[e1[0], e1[1]] = 1
                adj[e1[1], e1[0]] = 1
                adj[e2[0], e2[1]] = 0
                adj[e2[1], e2[0]] = 0
            else:
                adj[e1[0], e1[1]] = 0
                adj[e1[1], e1[0]] = 0
                adj[e2[0], e2[1]] = 1
                adj[e2[1], e2[0]] = 1
                
    # Extract edges from best_adj
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_adj[u, v] == 1:
                edges.append([u, v])
                
    return {'n': n, 'edges': edges}

# EVOLVE-BLOCK-END
