"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _cut_size(adj, labels):
    """Compute the number of edges crossing the partition."""
    # A cut edge is one where labels differ.
    # For a symmetric adjacency matrix, we can compute:
    # sum over all (i,j) where i<j and labels[i] != labels[j] of adj[i,j]
    # Efficient way: use matrix multiplication with sign vectors
    signs = np.where(labels == 0, 1.0, -1.0)
    # The number of crossing edges = (n^2 - signs @ A @ signs) / 4
    # But simpler: for each vertex, count neighbors with different label
    # Vectorized: 
    # cut = 0.5 * sum_v sum_u adj[v,u] * (1 - signs[v]*signs[u]) / 2 ... let's just do it directly
    # Actually: cut edges = 0.5 * (sum of all edges - sum of edges within same partition)
    # Even simpler: 
    # For each vertex v, the number of neighbors in the other partition is:
    #   if labels[v] == 0: count of neighbors u where labels[u] == 1
    #   if labels[v] == 1: count of neighbors u where labels[u] == 0
    # We can compute this as:
    #   Let S0 = set of vertices with label 0, S1 = set with label 1
    #   cut = sum_{v in S0} sum_{u in S1} adj[v,u]
    # This equals: (signs @ adj @ signs_complement) ... let me just use a clean method
    
    # Method: 
    # Let x = signs (1 for label 0, -1 for label 1)
    # For edge (i,j): it crosses iff x[i] != x[j], i.e., x[i]*x[j] = -1
    # Number of crossing edges = 0.5 * sum_{i,j} adj[i,j] * (1 - x[i]*x[j])
    # Since adj is symmetric and no self-loops (or even if there are, they don't cross):
    # = 0.5 * (total_edges_sum - x @ adj @ x)
    # where total_edges_sum = sum of all entries in adj (counts each edge twice)
    total = adj.sum()
    x_adj_x = signs @ adj @ signs
    cut = 0.5 * (total - x_adj_x)
    return int(round(cut))


def _local_search(adj, labels, rng, deadline):
    """Randomized local search: repeatedly pick a random vertex, flip if it improves cut."""
    n = len(labels)
    labels = labels.copy()
    current_cut = _cut_size(adj, labels)
    
    # Precompute degree for quick gain calculation
    # Gain of flipping vertex v: 
    #   If v is in S0: new contribution = (deg(v) - 2 * same_count(v)) where same_count = neighbors in S0
    #   Actually: flipping v changes cut by: 
    #     edges from v to same side become crossing (+1 each)
    #     edges from v to other side become non-crossing (-1 each)
    #   net gain = (neighbors in same side) - (neighbors in other side)
    #             = 2*(neighbors in same side) - deg(v)
    #   We flip if gain > 0, i.e., neighbors_in_same > deg/2
    
    best_labels = labels.copy()
    best_cut = current_cut
    
    while time.time() < deadline:
        improved = False
        # Random order for this pass
        order = list(range(n))
        rng.shuffle(order)
        
        for v in order:
            if time.time() > deadline:
                break
            
            # Count neighbors with same label
            v_label = labels[v]
            same_count = adj[v][labels == v_label].sum()
            deg = adj[v].sum()
            
            # Gain if we flip: same_count - (deg - same_count) = 2*same_count - deg
            gain = 2 * same_count - deg
            
            if gain > 0:
                labels[v] = 1 - v_label
                current_cut += gain
                improved = True
                if current_cut > best_cut:
                    best_cut = current_cut
                    best_labels = labels.copy()
        
        if not improved:
            break
    
    return best_labels, best_cut


def partition(adjacency):
    n = len(adjacency)
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    adj = np.array(adjacency, dtype=np.int8)
    
    rng = random.Random(42)
    np_rng = np.random.RandomState(42)
    
    best_labels = None
    best_cut = -1
    
    # Multiple diverse restarts
    restarts = 0
    while time.time() < deadline:
        restarts += 1
        
        # Choose initialization strategy
        strategy = restarts % 3
        
        if strategy == 0:
            # Alternating
            labels = np.arange(n) % 2
        elif strategy == 1:
            # Random
            labels = np_rng.randint(0, 2, size=n)
        else:
            # Greedy random: start with random, then do a few greedy passes
            labels = np_rng.randint(0, 2, size=n)
            # Quick greedy improvement
            for _ in range(2):
                for v in range(n):
                    v_label = labels[v]
                    same_count = adj[v][labels == v_label].sum()
                    deg = adj[v].sum()
                    if 2 * same_count > deg:
                        labels[v] = 1 - v_label
        
        # Run local search
        result_labels, result_cut = _local_search(adj, labels, rng, deadline)
        
        if result_cut > best_cut:
            best_cut = result_cut
            best_labels = result_labels.copy()
        
        # Check if we have time for another restart
        if time.time() > deadline - 2.0:
            break
    
    if best_labels is None:
        # Fallback
        best_labels = np.arange(n) % 2
    
    return best_labels
# EVOLVE-BLOCK-END
