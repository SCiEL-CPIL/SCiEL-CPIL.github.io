"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    GLS perturbation using 2-opt move regret model.
    
    For each edge in the current tour, compute the minimal length increase
    required to remove it via a single 2-opt swap. Edges with low regret
    (easy to break) get higher penalty to encourage the modified 2-opt
    to break them, while edges with high regret (structurally important)
    get lower penalty.
    """
    start_time = time.time()
    deadline = start_time + 280.0  # safety margin before 330s limit
    
    N = edge_distance.shape[0]
    tour = local_opt_tour.copy()
    
    # Ensure tour is a valid permutation of 0..N-1
    if len(tour) != N:
        return edge_distance.copy()
    
    # Compute tour edges: (tour[i], tour[(i+1) % N])
    # For each tour edge, compute 2-opt regret
    
    # Precompute tour edge list
    tour_edges = [(tour[i], tour[(i + 1) % N]) for i in range(N)]
    
    # Compute regret for each tour edge
    # regret[i] = min over all other edges j of min(delta1, delta2)
    # where delta is the length change of the 2-opt move
    
    # For efficiency, precompute distances between all pairs of tour nodes
    # d_tour[i][j] = edge_distance[tour[i], tour[j]]
    d_tour = edge_distance[np.ix_(tour, tour)]
    
    # Tour edge lengths: d_tour[i][(i+1)%N]
    tour_edge_lengths = np.array([d_tour[i, (i + 1) % N] for i in range(N)])
    
    # For each edge i, compute regret = min over j != i of:
    #   min( d(i,j) + d(i+1,j+1) - d(i,i+1) - d(j,j+1),
    #        d(i,j+1) + d(i+1,j) - d(i,i+1) - d(j,j+1) )
    
    regrets = np.full(N, np.inf)
    
    for i in range(N):
        i_next = (i + 1) % N
        edge_len_i = d_tour[i, i_next]
        
        # Vectorized computation over all j
        # j ranges over 0..N-1, j != i
        # delta1[j] = d(i,j) + d(i+1,j+1) - edge_len_i - d(j,j+1)
        # delta2[j] = d(i,j+1) + d(i+1,j) - edge_len_i - d(j,j+1)
        
        # For all j:
        d_i_j = d_tour[i, :]           # d(i, j) for all j
        d_i1_j1 = d_tour[i_next, (np.arange(N) + 1) % N]  # d(i+1, j+1) for all j
        d_i_j1 = d_tour[i, (np.arange(N) + 1) % N]        # d(i, j+1) for all j
        d_i1_j = d_tour[i_next, :]                  # d(i+1, j) for all j
        d_j_j1 = tour_edge_lengths  # d(j, j+1) for all j
        
        delta1 = d_i_j + d_i1_j1 - edge_len_i - d_j_j1
        delta2 = d_i_j1 + d_i1_j - edge_len_i - d_j_j1
        
        # Mask out j == i
        mask = np.ones(N, dtype=bool)
        mask[i] = False
        # Also need to be careful: when j == i_next, the edges are adjacent
        # but 2-opt on adjacent edges in a cycle of size > 2 is degenerate
        # For N >= 4, we should exclude j == i and j == i_next (and j == i-1 for the other side)
        # Actually in a cycle, edge i and edge i_next share a node. A 2-opt move on
        # two adjacent edges in a cycle of N>=4 just reverses a segment of length 1,
        # which is a no-op or creates a 2-cycle. Let's exclude adjacent edges.
        mask[i_next] = False
        if N > 2:
            mask[(i - 1) % N] = False
        
        delta1_masked = np.where(mask, delta1, np.inf)
        delta2_masked = np.where(mask, delta2, np.inf)
        
        min_delta = np.minimum(delta1_masked, delta2_masked)
        regrets[i] = np.min(min_delta)
    
    # Handle case where no valid 2-opt move exists (shouldn't happen for N>=4)
    if np.any(np.isinf(regrets)):
        regrets = np.where(np.isinf(regrets), 0.0, regrets)
    
    # Compute penalty: penalty_factor * (max_regret - edge_regret)
    max_regret = np.max(regrets)
    min_regret = np.min(regrets)
    
    # Scale factor: use a moderate penalty factor
    # The penalty should be significant enough to influence 2-opt decisions
    # but not so large as to completely distort the landscape
    penalty_factor = 0.15
    
    # Create the perturbed distance matrix
    perturbed = edge_distance.copy()
    
    # Apply penalty to tour edges
    for i in range(N):
        a = tour[i]
        b = tour[(i + 1) % N]
        penalty = penalty_factor * (max_regret - regrets[i])
        # Ensure penalty is non-negative
        penalty = max(0.0, penalty)
        perturbed[a, b] += penalty
        perturbed[b, a] += penalty
    
    # Also incorporate edge_n_used as a secondary signal
    # Edges used many times get additional penalty to encourage exploration
    max_used = edge_n_used.max()
    if max_used > 0:
        used_penalty_factor = 0.05
        for i in range(N):
            a = tour[i]
            b = tour[(i + 1) % N]
            used_penalty = used_penalty_factor * edge_n_used[a, b] / (1.0 + max_used)
            perturbed[a, b] += used_penalty
            perturbed[b, a] += used_penalty
    
    # Ensure finite values
    perturbed = np.where(np.isfinite(perturbed), perturbed, edge_distance)
    
    # Check deadline
    if time.time() > deadline:
        # Return the best we have (already computed)
        pass
    
    return perturbed
# EVOLVE-BLOCK-END
