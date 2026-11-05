"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def partition(adjacency):
    """
    Solves Maximum Cut using Simulated Annealing with multiple restarts.
    
    Strategy:
    1. Convert adjacency to a numpy boolean matrix for fast vectorized cut calculations.
    2. Use a time-bounded loop to perform multiple restarts of Simulated Annealing.
    3. Each SA run starts with a random binary partition.
    4. The cooling schedule is geometric, tuned to finish within a fraction of the total time budget per restart.
    5. Track the global best cut size found across all restarts.
    6. Return the labels corresponding to the best cut.
    """
    n = len(adjacency)
    # Convert to numpy array for speed if not already
    if not isinstance(adjacency, np.ndarray):
        adj = np.array(adjacency, dtype=bool)
    else:
        adj = adjacency
        
    # Precompute degrees to help estimate max possible cut? 
    # Actually, for cut calculation: 
    # cut_size = 0.5 * sum( (1 - labels[i]*labels[j]) * adj[i][j] ) 
    # Or more efficiently using vectorization:
    # Let L be vector of labels (+1/-1). 
    # Cut edges are those where L[i] != L[j].
    # Sum of adj[i][j] where L[i] != L[j] = 0.5 * (sum_all_adj - sum_adj_same_side)
    # Alternatively: 
    # Let x be 0/1 labels.
    # Cross edges = sum( x[i]*(1-x[j]) + (1-x[i])*x[j] ) * adj[i][j]
    # This can be computed as: 
    # Let A be adjacency matrix.
    # Let x be column vector of 0/1.
    # Cut = x^T (J - I - A) x / 2 + ... wait, standard trick:
    # Cut(x) = 0.5 * ( ||x||^2 * M - x^T A x ) where M is total edges? No.
    # Let's stick to a robust vectorized update for a single flip.
    
    # For a vertex v, if we flip its label, the change in cut size is:
    # delta = (number of neighbors on opposite side) - (number of neighbors on same side)
    # If delta > 0, cut increases.
    
    # We will maintain:
    # labels: array of 0/1
    # cut_val: current cut size
    
    # Helper to compute initial cut for a label vector
    def compute_cut(labels):
        # labels: 0/1 array
        # For each edge (i,j), it is cut if labels[i] != labels[j]
        # Vectorized: 
        # Create matrix of differences? n=100 is small.
        # diff = labels[:, None] != labels[None, :]
        # cut = np.sum(adj & diff)
        # This is O(n^2) memory and time per call, which is 10k ops. Very fast.
        diff = labels[:, None] != labels[None, :]
        return np.sum(adj & diff)

    # Helper to get delta for flipping vertex v
    def get_delta(v, labels):
        # Neighbors of v
        neighbors = adj[v]
        # Neighbors with same label as v
        same_side = np.sum(neighbors & (labels == labels[v]))
        # Neighbors with different label than v
        diff_side = np.sum(neighbors & (labels != labels[v]))
        # If we flip v, same becomes diff and diff becomes same.
        # New cut = Old cut - same_side + diff_side
        # Delta = diff_side - same_side
        return diff_side - same_side

    # Time management
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    best_labels = None
    best_cut = -1
    
    # Random seed for reproducibility in local tests, but evaluator might use fixed seed or random.
    # Using random module is fine.
    
    # Number of restarts is dynamic based on time.
    # We will run restarts until time is up.
    
    while time.time() < deadline:
        # Initialize random partition
        labels = np.random.randint(0, 2, n)
        current_cut = compute_cut(labels)
        
        # If this random start is already better, update best
        if current_cut > best_cut:
            best_cut = current_cut
            best_labels = labels.copy()
            
        # Simulated Annealing Parameters
        # We want to finish this SA run in roughly 5-10 seconds max to allow restarts?
        # Or use a fixed number of steps?
        # With n=100, a step is very fast (vectorized delta calc is ~O(n)).
        # 1000 steps is trivial. Let's aim for ~100,000 steps per restart if possible.
        
        # Dynamic cooling:
        # Start temp high, end temp low.
        # T = T0 * lambda^t
        # We want T_end to be small enough to converge.
        # Let's fix the number of steps for this restart based on remaining time.
        remaining_time = deadline - time.time()
        # If we have lots of time, do more steps. If little, fewer.
        # Estimate time per step: ~1 microsecond? 
        # Let's just set a max steps per restart to be safe, e.g., 50,000.
        # 50,000 steps * ~2us = 0.1s. So we can do many restarts.
        
        max_steps = 50000
        T0 = 10.0
        T_end = 0.1
        # lambda = (T_end / T0) ** (1 / max_steps)
        if max_steps > 0:
            alpha = (T_end / T0) ** (1.0 / max_steps)
        else:
            alpha = 1.0
            
        T = T0
        
        for _ in range(max_steps):
            if time.time() >= deadline:
                break
                
            # Pick a random vertex
            v = np.random.randint(n)
            
            # Calculate delta
            delta = get_delta(v, labels)
            
            if delta > 0:
                # Accept
                labels[v] = 1 - labels[v]
                current_cut += delta
                if current_cut > best_cut:
                    best_cut = current_cut
                    best_labels = labels.copy()
            else:
                # Metropolis criterion
                if T > 1e-6:
                    # Accept with probability exp(delta / T)
                    # delta is negative
                    prob = np.exp(delta / T)
                    if np.random.random() < prob:
                        labels[v] = 1 - labels[v]
                        current_cut += delta
                        if current_cut > best_cut:
                            best_cut = current_cut
                            best_labels = labels.copy()
            
            T *= alpha
        
        # Check time again after loop
        if time.time() >= deadline:
            break

    # Fallback if no labels were set (should not happen)
    if best_labels is None:
        best_labels = np.arange(n) % 2

    return best_labels
# EVOLVE-BLOCK-END
