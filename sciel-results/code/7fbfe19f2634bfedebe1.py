"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _local_search(labels, adjacency, time_limit, rng):
    """
    Perform local search (flip individual vertices) to improve cut size.
    Uses a time limit to stop before exceeding the budget.
    Optimized: maintain cut value incrementally.
    """
    n = len(labels)
    labels = labels.copy()
    adj = adjacency.astype(np.int32)
    
    # Compute initial cut size: count edges crossing the partition
    # For each vertex i, count neighbors j with different label
    # Cut = 0.5 * sum_i sum_j adj[i,j] * (labels[i] != labels[j])
    # Use vectorized computation
    diff = labels[:, None] != labels[None, :]
    current_cut = int((adj * diff).sum() // 2)
    
    # Precompute degrees
    degrees = adj.sum(axis=1).astype(np.int32)
    
    # Track number of edges from each vertex to the opposite set
    # opp_count[v] = number of neighbors of v with label != labels[v]
    # This can be updated incrementally when a vertex flips
    
    # Initialize opp_count: for each v, count neighbors u where labels[u] != labels[v]
    opp_count = np.zeros(n, dtype=np.int32)
    for v in range(n):
        opp_count[v] = adj[v][labels != labels[v]].sum()
    
    # Precompute neighbor lists for faster access
    neighbor_lists = [np.where(adj[v] > 0)[0] for v in range(n)]
    
    start_time = time.time()
    improved = True
    while improved:
        improved = False
        for v in range(n):
            if time.time() - start_time > time_limit:
                break
            # Gain from flipping v:
            # Currently, opp_count[v] edges from v go to opposite set (these are cut)
            # After flipping, those edges become non-cut, and (degrees[v] - opp_count[v]) become cut
            # New cut contribution from v's edges: degrees[v] - opp_count[v]
            # Old cut contribution from v's edges: opp_count[v]
            # Change in cut = (degrees[v] - opp_count[v]) - opp_count[v] = degrees[v] - 2*opp_count[v]
            gain = degrees[v] - 2 * opp_count[v]
            if gain > 0:
                # Flip vertex v
                labels[v] = 1 - labels[v]
                current_cut += gain
                
                # Update opp_count for v and its neighbors
                # For v: its opp_count becomes degrees[v] - opp_count[v]
                opp_count[v] = degrees[v] - opp_count[v]
                
                # For each neighbor u of v:
                # Since labels[v] has already been flipped, we can check:
                #   For neighbor u: if labels[u] == labels[v] (new), then u was in opposite before -> opp_count[u] -= 1
                #   For neighbor u: if labels[u] != labels[v] (new), then u was in same before -> opp_count[u] += 1
                
                neighbors = neighbor_lists[v]
                for u in neighbors:
                    if labels[u] == labels[v]:
                        # u and v are now in same set, so u's edge to v is no longer cut
                        opp_count[u] -= 1
                    else:
                        # u and v are in different sets, so u's edge to v is now cut
                        opp_count[u] += 1
                
                improved = True
        if time.time() - start_time > time_limit:
            break
    
    return labels, current_cut


def _simulated_annealing(labels, adjacency, time_limit, rng):
    """
    Simulated annealing to escape local optima.
    Starts from a greedy local search result and uses SA to potentially find better solutions.
    """
    n = len(labels)
    labels = labels.copy()
    adj = adjacency.astype(np.int32)
    
    # Compute initial cut size
    diff = labels[:, None] != labels[None, :]
    current_cut = int((adj * diff).sum() // 2)
    
    # Precompute degrees
    degrees = adj.sum(axis=1).astype(np.int32)
    
    # Track number of edges from each vertex to the opposite set
    opp_count = np.zeros(n, dtype=np.int32)
    for v in range(n):
        opp_count[v] = adj[v][labels != labels[v]].sum()
    
    # Precompute neighbor lists
    neighbor_lists = [np.where(adj[v] > 0)[0] for v in range(n)]
    
    start_time = time.time()
    
    # Temperature schedule: start high, cool down
    T0 = max(1.0, degrees.max() if n > 0 else 1.0)
    T_min = 0.01
    cooling_rate = 0.995
    
    T = T0
    best_labels = labels.copy()
    best_cut = current_cut
    
    while T > T_min:
        if time.time() - start_time > time_limit:
            break
        
        # Pick a random vertex to flip
        v = rng.integers(0, n)
        
        # Compute gain from flipping v
        gain = degrees[v] - 2 * opp_count[v]
        
        # Decide whether to accept the flip
        if gain >= 0:
            # Always accept improving or neutral moves
            accept = True
        else:
            # Accept worsening moves with probability exp(gain / T)
            # gain is negative, so gain/T is negative
            accept_prob = np.exp(gain / T)
            accept = rng.random() < accept_prob
        
        if accept:
            # Flip vertex v
            labels[v] = 1 - labels[v]
            current_cut += gain
            
            # Update opp_count for v and its neighbors
            opp_count[v] = degrees[v] - opp_count[v]
            
            neighbors = neighbor_lists[v]
            for u in neighbors:
                if labels[u] == labels[v]:
                    opp_count[u] -= 1
                else:
                    opp_count[u] += 1
            
            # Track best solution
            if current_cut > best_cut:
                best_cut = current_cut
                best_labels = labels.copy()
        
        # Cool down
        T *= cooling_rate
    
    return best_labels, best_cut


def partition(adjacency):
    n = len(adjacency)
    # Deadline guard
    global_deadline = time.time() + 110  # 110 seconds out of 120s, leave safety margin
    rng = np.random.default_rng(seed=42)
    
    best_labels = None
    best_cut = -1
    
    # Strategy: Multiple random restarts with local search + simulated annealing.
    # First do greedy local search, then use SA to escape local optima.
    
    num_restarts = 0
    max_restarts = 100  # Upper bound, but time will limit it
    
    # Phase 1: Greedy local search with multiple starts
    while time.time() < global_deadline - 30 and num_restarts < max_restarts:
        # Generate random initial labels
        if num_restarts == 0:
            # First start: alternating
            labels = np.arange(n) % 2
        elif num_restarts == 1:
            # Second start: all zeros (will be improved by local search)
            labels = np.zeros(n, dtype=int)
        elif num_restarts == 2:
            # Third start: all ones
            labels = np.ones(n, dtype=int)
        else:
            labels = rng.integers(0, 2, size=n)
        
        # Run local search with a time limit that shrinks as we approach the deadline
        remaining_time = max(1.0, (global_deadline - 30) - time.time())
        # Give each restart a portion of the remaining time, but at least 0.05s
        ls_time_limit = max(0.05, remaining_time / (num_restarts + 10))
        
        refined_labels, cut = _local_search(labels, adjacency, ls_time_limit, rng)
        
        if cut > best_cut:
            best_cut = cut
            best_labels = refined_labels.copy()
        
        num_restarts += 1
    
    # Phase 2: Simulated annealing to escape local optima
    # Use the best solution found so far as starting point
    if best_labels is not None and time.time() < global_deadline - 5:
        sa_time_limit = max(0.1, global_deadline - 5 - time.time())
        sa_labels, sa_cut = _simulated_annealing(best_labels, adjacency, sa_time_limit, rng)
        
        if sa_cut > best_cut:
            best_cut = sa_cut
            best_labels = sa_labels.copy()
    
    # Fallback if no labels found (should not happen)
    if best_labels is None:
        best_labels = np.zeros(n, dtype=int)
    
    return best_labels
# EVOLVE-BLOCK-END
