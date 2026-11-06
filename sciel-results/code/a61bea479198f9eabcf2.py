"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _local_search(labels, adjacency, time_limit, rng):
    """
    Perform local search (flip individual vertices) to improve cut size.
    Uses efficient gain computation via maintained opp_edges array.
    """
    n = len(labels)
    labels = labels.copy()
    adj = adjacency.astype(np.int8)
    
    # Compute initial cut size
    diff_mask = (labels[:, None] != labels[None, :])
    current_cut = int((adj * diff_mask).sum() // 2)
    
    # Precompute degree
    degrees = adj.sum(axis=1).astype(np.int32)
    
    # opp_edges[v] = number of neighbors of v with opposite label
    opp_edges = np.zeros(n, dtype=np.int32)
    for v in range(n):
        opp_edges[v] = adj[v][labels != labels[v]].sum()
    
    start_time = time.time()
    improved = True
    while improved:
        improved = False
        for v in range(n):
            if time.time() - start_time > time_limit:
                return labels, current_cut
            # Gain from flipping v = 2 * opp_edges[v] - degrees[v]
            # (flip moves all edges to same side to opposite, and vice versa)
            gain = 2 * opp_edges[v] - degrees[v]
            if gain > 0:
                labels[v] = 1 - labels[v]
                current_cut += gain
                # Update opp_edges for v and its neighbors
                # For v: opp_edges[v] becomes degrees[v] - opp_edges[v]
                old_opp_v = opp_edges[v]
                opp_edges[v] = degrees[v] - old_opp_v
                # For each neighbor u of v:
                # If u was opposite to v (before flip), after flip u is same as v
                # So opp_edges[u] decreases by 1
                # If u was same as v (before flip), after flip u is opposite to v
                # So opp_edges[u] increases by 1
                neighbors = np.where(adj[v] != 0)[0]
                for u in neighbors:
                    if labels[u] != labels[v]:  # u is now opposite to v (was same before)
                        opp_edges[u] += 1
                    else:  # u is now same as v (was opposite before)
                        opp_edges[u] -= 1
                improved = True
    
    return labels, current_cut


def _two_flip_search(labels, adjacency, time_limit, rng):
    """
    Perform 2-flip local search to escape local optima.
    """
    n = len(labels)
    labels = labels.copy()
    adj = adjacency.astype(np.int8)
    
    diff_mask = (labels[:, None] != labels[None, :])
    current_cut = int((adj * diff_mask).sum() // 2)
    
    degrees = adj.sum(axis=1).astype(np.int32)
    opp_edges = np.zeros(n, dtype=np.int32)
    for v in range(n):
        opp_edges[v] = adj[v][labels != labels[v]].sum()
    
    start_time = time.time()
    improved = True
    while improved:
        improved = False
        # Try all pairs (i, j) with i < j
        for i in range(n):
            if time.time() - start_time > time_limit:
                return labels, current_cut
            for j in range(i + 1, n):
                if labels[i] == labels[j]:
                    continue
                # Compute gain from flipping both i and j
                # This is complex; approximate by checking if both have positive gain
                gain_i = 2 * opp_edges[i] - degrees[i]
                gain_j = 2 * opp_edges[j] - degrees[j]
                if gain_i > 0 and gain_j > 0:
                    # Flip both
                    labels[i] = 1 - labels[i]
                    labels[j] = 1 - labels[j]
                    current_cut += gain_i + gain_j
                    # Update opp_edges
                    old_opp_i = opp_edges[i]
                    old_opp_j = opp_edges[j]
                    opp_edges[i] = degrees[i] - old_opp_i
                    opp_edges[j] = degrees[j] - old_opp_j
                    # Update neighbors of i and j
                    neighbors_i = np.where(adj[i] != 0)[0]
                    neighbors_j = np.where(adj[j] != 0)[0]
                    all_neighbors = np.unique(np.concatenate([neighbors_i, neighbors_j]))
                    for u in all_neighbors:
                        if u == i or u == j:
                            continue
                        # Determine if u's relationship to i and j changed
                        # This is getting complex; for simplicity, recompute opp_edges for all neighbors
                        pass
                    # Recompute opp_edges for affected vertices
                    for u in all_neighbors:
                        opp_edges[u] = adj[u][labels != labels[u]].sum()
                    improved = True
                    break
            if improved:
                break
        if time.time() - start_time > time_limit:
            break
    
    return labels, current_cut


def partition(adjacency):
    n = len(adjacency)
    # Deadline guard
    global_deadline = time.time() + 110  # 110 seconds out of 120s, leave safety margin
    rng = np.random.default_rng(seed=42)
    
    best_labels = None
    best_cut = -1
    
    num_restarts = 0
    max_restarts = 200  # Upper bound, but time will limit it
    
    while time.time() < global_deadline - 5 and num_restarts < max_restarts:
        # Generate random initial labels
        if num_restarts == 0:
            # First start: alternating
            labels = np.arange(n) % 2
        elif num_restarts == 1:
            # Second start: all zeros
            labels = np.zeros(n, dtype=int)
        elif num_restarts == 2:
            # Third start: all ones
            labels = np.ones(n, dtype=int)
        elif num_restarts == 3:
            # Random but biased: 60/40 split
            labels = (rng.random(n) < 0.6).astype(int)
        elif num_restarts == 4:
            # Random but biased: 40/60 split
            labels = (rng.random(n) < 0.4).astype(int)
        else:
            labels = rng.integers(0, 2, size=n)
        
        # Run local search with a time limit that shrinks as we approach the deadline
        remaining_time = max(0.5, (global_deadline - 5) - time.time())
        ls_time_limit = max(0.02, remaining_time / (num_restarts + 20))
        
        refined_labels, cut = _local_search(labels, adjacency, ls_time_limit, rng)
        
        if cut > best_cut:
            best_cut = cut
            best_labels = refined_labels.copy()
        
        # Occasionally try 2-flip search on the best solution found so far
        if num_restarts > 0 and num_restarts % 5 == 0 and best_labels is not None:
            if time.time() < global_deadline - 10:
                two_flip_time = min(1.0, (global_deadline - 10) - time.time())
                if two_flip_time > 0.1:
                    tf_labels, tf_cut = _two_flip_search(best_labels, adjacency, two_flip_time, rng)
                    if tf_cut > best_cut:
                        best_cut = tf_cut
                        best_labels = tf_labels.copy()
        
        num_restarts += 1
    
    # Fallback if no labels found (should not happen)
    if best_labels is None:
        best_labels = np.zeros(n, dtype=int)
    
    return best_labels
# EVOLVE-BLOCK-END
