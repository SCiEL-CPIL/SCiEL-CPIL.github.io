"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(adjacency):
    n = adjacency.shape[0]
    adj = adjacency.astype(bool)
    
    # New Approach: Stochastic Ensemble Voting
    # Run multiple randomized greedy trials and count frequency of each vertex
    # appearing in the resulting independent set.
    
    num_trials = 100
    freq = np.zeros(n, dtype=float)
    
    for _ in range(num_trials):
        # State for this trial
        remaining = np.ones(n, dtype=bool)
        # Degrees within the remaining subgraph
        deg = np.zeros(n, dtype=int)
        
        # Initialize degrees
        deg[:] = adj.sum(axis=1)
        
        while remaining.any():
            # Find min degree among remaining
            d = np.where(remaining, deg, np.inf)
            min_deg = np.min(d)
            candidates = np.where(d == min_deg)[0]
            
            # Pick one randomly to add diversity
            v = candidates[np.random.randint(len(candidates))]
            
            # v is selected in this trial's independent set
            freq[v] += 1
            
            # Remove v and its neighbors from remaining
            neighbors = adj[v] & remaining
            remaining[v] = False
            remaining[neighbors] = False
            
            # Update degrees of remaining vertices
            # Recalculate degrees for the induced subgraph of remaining vertices
            if remaining.any():
                # adj[remaining][:, remaining] gets the submatrix
                # Summing axis=1 gives the degree of each row (vertex) in that subgraph
                deg[remaining] = np.sum(adj[remaining][:, remaining], axis=1)
        
    # Use frequency as the primary score.
    # Add a small negative degree component to break ties deterministically
    # in favor of lower-degree vertices (which are generally less constraining).
    initial_degrees = adj.sum(axis=1)
    scores = freq - 0.001 * initial_degrees
    
    return scores
# EVOLVE-BLOCK-END
