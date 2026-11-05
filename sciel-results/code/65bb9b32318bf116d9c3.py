"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def partition(adjacency):
    # Simulated Annealing with vectorized energy computation and deadline guard.
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=np.int8)
    
    # Convert to numpy array of int for efficient computation
    adj = np.array(adjacency, dtype=np.int8)
    
    # Deadline: 140 seconds (safety margin before 150s limit)
    deadline = time.time() + 140.0
    
    # Random initialization
    labels = np.random.randint(0, 2, size=n, dtype=np.int8)
    
    # Helper to compute cut size
    def cut_size(lb):
        # Cut edges are those where labels differ
        # For each edge (i,j), it's in the cut if lb[i] != lb[j]
        # We can compute this as: sum over all edges of (lb[i] != lb[j])
        # Using matrix: for each i, sum of adj[i, j] where lb[j] != lb[i]
        # Vectorized: for each i, sum adj[i, lb != lb[i]]
        # More efficient: use the fact that cut = total_edges - same_side_edges
        # same_side_edges[i] = sum of adj[i, j] where lb[j] == lb[i]
        # Let's use: for each i, count neighbors with same label
        # same_count[i] = sum(adj[i] * (lb == lb[i]))
        # But we want cut, which is sum(adj[i] * (lb != lb[i]))
        # cut = sum over all i of sum(adj[i] * (1 - (lb == lb[i])))
        # But each edge counted twice, so cut_size = 0.5 * that sum
        # Actually, let's just compute directly:
        # For each i, the number of edges from i to opposite side
        # We can do: for each i, sum of adj[i, lb != lb[i]]
        # This is O(n^2) per call but vectorized
        
        # Efficient vectorized approach:
        # Create a matrix where M[i,j] = 1 if lb[i] != lb[j], else 0
        # Then cut = sum(adj * M)
        # M[i,j] = 1 - (lb[i] == lb[j])
        # lb[i] == lb[j] can be computed as outer comparison
        diff = (lb[:, None] != lb[None, :]).astype(np.int8)
        return int(np.sum(adj * diff))
    
    current_cut = cut_size(labels)
    best_labels = labels.copy()
    best_cut = current_cut
    
    # Temperature schedule
    T_start = 1.0
    T_end = 0.01
    # We'll use a geometric cooling schedule
    # Number of iterations depends on time budget
    
    # Precompute degree for potential optimization
    degree = adj.sum(axis=1).astype(np.float64)
    
    # Simulated Annealing
    T = T_start
    no_improve_count = 0
    max_no_improve = 2000  # Restart if stuck
    
    while time.time() < deadline:
        # Choose a random vertex to flip
        v = np.random.randint(0, n)
        
        # Compute energy change if we flip vertex v
        # Currently, v is in partition labels[v]
        # If we flip v, edges from v to same side become crossing, and edges to other side become non-crossing
        # delta_cut = (edges from v to opposite side) - (edges from v to same side)
        # = sum(adj[v, lb != lb[v]]) - sum(adj[v, lb == lb[v]])
        # Note: adj[v, v] should be 0 (no self-loops typically)
        
        same_mask = (labels == labels[v])
        other_mask = (labels != labels[v])
        
        same_count = adj[v, same_mask].sum()
        other_count = adj[v, other_mask].sum()
        
        # If we flip v:
        # - Edges that were same-side (same_count) become crossing: +same_count to cut
        # - Edges that were other-side (other_count) become same-side: -other_count to cut
        # delta = same_count - other_count
        delta = same_count - other_count
        
        # Accept if delta > 0 (improvement) or with probability exp(delta/T) if delta < 0
        if delta >= 0 or (T > 1e-10 and np.random.random() < np.exp(delta / T)):
            labels[v] = 1 - labels[v]
            current_cut += delta
            if current_cut > best_cut:
                best_cut = current_cut
                best_labels = labels.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            no_improve_count += 1
        
        # Cool down
        T *= 0.9999
        
        # If stuck for too long, restart with a new random initialization
        if no_improve_count > max_no_improve:
            # Try a few more restarts
            labels = np.random.randint(0, 2, size=n, dtype=np.int8)
            current_cut = cut_size(labels)
            if current_cut > best_cut:
                best_cut = current_cut
                best_labels = labels.copy()
            T = T_start * 0.5  # Restart with lower temperature
            no_improve_count = 0
    
    # Final local improvement pass on the best solution
    # Greedy hill climbing from best_labels
    lb = best_labels.copy()
    cur_cut = best_cut
    for _ in range(3):
        improved = False
        for v in range(n):
            same_mask = (lb == lb[v])
            other_mask = (lb != lb[v])
            same_count = adj[v, same_mask].sum()
            other_count = adj[v, other_mask].sum()
            if same_count > other_count:
                lb[v] = 1 - lb[v]
                cur_cut += same_count - other_count
                improved = True
        if not improved:
            break
    
    if cur_cut > best_cut:
        best_cut = cur_cut
        best_labels = lb
    
    return best_labels.astype(np.int8)
# EVOLVE-BLOCK-END
