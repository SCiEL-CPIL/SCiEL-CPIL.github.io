"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def partition(adjacency):
    """
    Find a maximum cut in an unweighted graph using:
    1. Multiple random restarts with greedy local improvement
    2. Simulated annealing for escaping local optima
    3. Vectorized operations for speed
    4. Explicit deadline guard to ensure we finish in time
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    adj = np.asarray(adjacency, dtype=np.int8)
    n = len(adj)
    
    # Precompute degrees
    degrees = adj.sum(axis=1)
    
    def cut_size(labels):
        """Compute cut size efficiently."""
        # An edge (i,j) is in the cut if labels[i] != labels[j]
        # Sum over all pairs where labels differ
        # Using: for each vertex, count neighbors with different label
        # cut = sum_v sum_u adj[v,u] * (labels[v] != labels[u]) / 2
        # Vectorized: (labels[:, None] != labels[None, :]) & adj, sum, // 2
        mask = (labels[:, None] != labels[None, :]) & adj
        return mask.sum() // 2
    
    def greedy_improve(labels):
        """Greedy local improvement: flip vertices if it increases cut size."""
        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False
            for v in range(n):
                # Count neighbors in same group vs other group
                same_count = adj[v][labels == labels[v]].sum()
                other_count = adj[v][labels != labels[v]].sum()
                if same_count > other_count:
                    labels[v] = 1 - labels[v]
                    improved = True
        return labels
    
    def simulated_annealing(labels, max_iter=5000):
        """Simulated annealing to escape local optima."""
        current = labels.copy()
        current_cut = cut_size(current)
        best = current.copy()
        best_cut = current_cut
        
        T = 1.0
        T_min = 0.01
        cooling = 0.999
        
        for iteration in range(max_iter):
            if time.time() > deadline:
                break
            
            # Pick a random vertex to flip
            v = np.random.randint(n)
            new_labels = current.copy()
            new_labels[v] = 1 - new_labels[v]
            new_cut = cut_size(new_labels)
            
            delta = new_cut - current_cut
            
            if delta >= 0:
                current = new_labels
                current_cut = new_cut
                if current_cut > best_cut:
                    best = current.copy()
                    best_cut = current_cut
            else:
                # Accept with probability exp(delta / T)
                if np.random.random() < np.exp(delta / T):
                    current = new_labels
                    current_cut = new_cut
            
            T *= cooling
            if T < T_min:
                T = T_min
        
        return best, best_cut
    
    # Strategy: multiple random restarts with greedy + optional annealing
    best_labels = None
    best_cut = -1
    
    # First: try a few random starts with greedy
    n_restarts = 20
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Random initialization
        labels = np.random.randint(0, 2, size=n)
        
        # Greedy improvement
        labels = greedy_improve(labels)
        
        # Evaluate
        c = cut_size(labels)
        if c > best_cut:
            best_cut = c
            best_labels = labels.copy()
    
    # If we have time, try simulated annealing from the best solution found
    if time.time() < deadline - 30 and best_labels is not None:
        sa_labels, sa_cut = simulated_annealing(best_labels, max_iter=10000)
        if sa_cut > best_cut:
            best_cut = sa_cut
            best_labels = sa_labels
    
    # Final greedy polish
    if best_labels is not None and time.time() < deadline - 5:
        final_labels = greedy_improve(best_labels)
        final_cut = cut_size(final_labels)
        if final_cut > best_cut:
            best_cut = final_cut
            best_labels = final_labels
    
    if best_labels is None:
        # Fallback: alternating pattern
        best_labels = np.arange(n) % 2
    
    return best_labels
# EVOLVE-BLOCK-END
