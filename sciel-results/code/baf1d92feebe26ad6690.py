"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _cut_size(adjacency, labels):
    """Compute the number of edges crossing the partition."""
    n = len(labels)
    # A edge (i,j) crosses if labels[i] != labels[j]
    # We can compute this efficiently using matrix operations
    # cut = sum over i<j where labels differ of adjacency[i,j]
    # Using: for each i, sum of adjacency[i, j] where labels[j] != labels[i]
    # Then divide by 2 since each edge counted twice
    same_mask = labels[:, None] == labels[None, :]
    crossing = adjacency & (~same_mask)
    return np.sum(crossing) // 2


def _flip_gain(adjacency, labels, v):
    """Compute the gain in cut size if we flip vertex v."""
    # Current state: v is in group labels[v]
    # Neighbors in same group: contribute to non-cut edges
    # Neighbors in other group: contribute to cut edges
    same_count = np.sum(adjacency[v] & (labels == labels[v]))
    other_count = np.sum(adjacency[v] & (labels != labels[v]))
    # If we flip v, same becomes other and vice versa
    # Gain = other_count - same_count
    return other_count - same_count


def _local_search(adjacency, labels, max_iters=1000, deadline=None):
    """Iterative local search flipping vertices that improve the cut."""
    n = len(adjacency)
    current_cut = _cut_size(adjacency, labels)
    
    for iteration in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        improved = False
        # Randomize the order of vertices to avoid bias
        order = np.random.permutation(n)
        
        for v in order:
            if deadline is not None and time.time() > deadline:
                break
            
            gain = _flip_gain(adjacency, labels, v)
            if gain > 0:
                labels[v] = 1 - labels[v]
                current_cut += gain
                improved = True
        
        if not improved:
            break
    
    return labels, current_cut


def _simulated_annealing(adjacency, labels, start_temp=1.0, end_temp=0.01, 
                          max_iters=500, deadline=None):
    """Simulated annealing for maximum cut."""
    n = len(adjacency)
    current_cut = _cut_size(adjacency, labels)
    best_labels = labels.copy()
    best_cut = current_cut
    
    for iteration in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute temperature
        t = start_temp * (end_temp / start_temp) ** (iteration / max_iters)
        
        # Pick a random vertex to flip
        v = np.random.randint(n)
        gain = _flip_gain(adjacency, labels, v)
        
        # Accept if improvement, or with probability exp(gain/t) if worsening
        if gain >= 0 or np.random.random() < np.exp(gain / max(t, 1e-10)):
            labels[v] = 1 - labels[v]
            current_cut += gain
            
            if current_cut > best_cut:
                best_cut = current_cut
                best_labels = labels.copy()
    
    return best_labels, best_cut


def partition(adjacency):
    """
    Maximum Cut partition using multi-start local search with simulated annealing.
    
    Args:
        adjacency: bool matrix (n x n) representing the graph
        
    Returns:
        0/1 label per vertex (numpy array of length n)
    """
    n = len(adjacency)
    
    # Set a safety deadline (leave margin before 150s limit)
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    # Convert to numpy array if needed
    adj = np.asarray(adjacency, dtype=bool)
    
    best_labels = None
    best_cut = -1
    
    # Multi-start strategy: try multiple initializations
    num_starts = 20  # Will be limited by time
    
    for start_idx in range(num_starts):
        if time.time() > deadline:
            break
        
        # Different initialization strategies
        if start_idx == 0:
            # Alternating start
            labels = np.arange(n) % 2
        elif start_idx == 1:
            # Random start
            labels = np.random.randint(0, 2, n)
        elif start_idx == 2:
            # Greedy initialization
            labels = np.zeros(n, dtype=int)
            # Process vertices in random order, assign to side with more external edges
            order = np.random.permutation(n)
            for v in order:
                # Count neighbors in each group
                group0 = np.sum(adj[v] & (labels == 0))
                group1 = np.sum(adj[v] & (labels == 1))
                labels[v] = 1 if group0 > group1 else 0
        else:
            # Random start with different seeds
            labels = np.random.randint(0, 2, n)
        
        # Run local search
        labels_ls, cut_ls = _local_search(adj, labels.copy(), max_iters=500, deadline=deadline)
        
        if cut_ls > best_cut:
            best_cut = cut_ls
            best_labels = labels_ls.copy()
        
        # If we have time, try simulated annealing from the best local solution
        if time.time() < deadline - 5.0:
            labels_sa, cut_sa = _simulated_annealing(adj, labels_ls.copy(), 
                                                       start_temp=2.0, end_temp=0.01,
                                                       max_iters=300, deadline=deadline)
            if cut_sa > best_cut:
                best_cut = cut_sa
                best_labels = labels_sa.copy()
    
    # Final polish: run local search on the best solution found
    if time.time() < deadline - 1.0 and best_labels is not None:
        best_labels, best_cut = _local_search(adj, best_labels.copy(), max_iters=100, deadline=deadline)
    
    if best_labels is None:
        # Fallback: return alternating labels
        best_labels = np.arange(n) % 2
    
    return best_labels
# EVOLVE-BLOCK-END
