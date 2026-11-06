"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def partition(adjacency):
    """
    Maximum Cut via Parallel Simulated Annealing.
    
    1. Multiple restarts of Simulated Annealing to escape local optima.
    2. Efficient O(n) delta calculation for vertex flips.
    3. Geometric cooling schedule.
    """
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=int)
    
    # Precompute adjacency as a numpy array
    adj = np.asarray(adjacency, dtype=np.float64)
    
    # Ensure symmetry (just in case)
    adj = (adj + adj.T) / 2.0
    np.fill_diagonal(adj, 0)
    
    # Precompute total edge weight (sum of upper triangle)
    total_edges = 0.5 * np.sum(adj)
    
    def calc_cut(labels):
        # Map 0/1 to -1/1
        s = 2 * labels - 1
        val = s @ adj @ s
        return total_edges - 0.5 * val

    def simulated_annealing(seed):
        rng = np.random.default_rng(seed)
        
        # Initialize randomly
        s = rng.integers(0, 2, size=n) * 2 - 1  # -1 or 1
        labels = (s + 1) // 2  # 0 or 1
        
        # Precompute As = adj @ s for O(n) delta updates
        As = adj @ s
        
        current_cut = calc_cut(labels)
        
        # Parameters
        T_start = 10.0
        T_end = 0.1
        cooling = 0.995
        max_iter = 20000
        T = T_start
        
        best_cut = current_cut
        best_labels = labels.copy()
        
        for _ in range(max_iter):
            if T < T_end:
                break
                
            # Pick a random vertex to flip
            v = rng.integers(0, n)
            
            # Calculate delta in cut size
            # delta = 2 * s[v] * As[v]
            delta = 2 * s[v] * As[v]
            
            # Acceptance probability
            if delta > 0:
                accept = True
            else:
                # Avoid overflow in exp
                if T > 0:
                    accept = rng.random() < np.exp(delta / T)
                else:
                    accept = False
            
            if accept:
                # Update state
                old_s_v = s[v]
                s[v] = -old_s_v
                labels[v] = 1 - labels[v]
                
                # Update As vector
                # As[u] changes by -2 * old_s_v * A[v, u]
                As -= 2 * old_s_v * adj[v]
                
                current_cut += delta
                
                if current_cut > best_cut:
                    best_cut = current_cut
                    best_labels = labels.copy()
            
            # Cool down
            T *= cooling
            
        return best_labels, best_cut

    # Run multiple parallel restarts
    # For n=100, 10 restarts with 20k iterations each is fast (< 1-2s per restart)
    n_restarts = 15
    best_overall_labels = None
    best_overall_cut = -np.inf
    
    start_time = time.time()
    time_limit = 110.0  # Leave 10s buffer
    
    for i in range(n_restarts):
        labels, cut_val = simulated_annealing(i)
        
        if cut_val > best_overall_cut:
            best_overall_cut = cut_val
            best_overall_labels = labels
        
        # Check time
        if time.time() - start_time > time_limit:
            break
    
    if best_overall_labels is None:
        # Fallback to random if something went wrong
        best_overall_labels = np.random.randint(0, 2, size=n).astype(np.int8)
        
    return best_overall_labels
# EVOLVE-BLOCK-END
