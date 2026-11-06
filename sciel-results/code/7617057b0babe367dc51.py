"""Combined program for Maximum Cut: Simulated Annealing with multiple restarts."""
# EVOLVE-BLOCK-START
import time
import numpy as np


def partition(adjacency):
    """
    Assign each vertex a 0/1 label to maximize the number of cut edges.
    Uses Simulated Annealing with efficient gain updates and multiple restarts.
    """
    adj = np.asarray(adjacency, dtype=np.int8)
    n = len(adj)
    
    if n <= 1:
        return np.zeros(n, dtype=np.int8)
    
    # Precompute degrees and neighbor lists for efficient updates
    degrees = adj.sum(axis=1)
    neighbors = [np.nonzero(adj[i])[0] for i in range(n)]
    
    deadline = time.time() + 115.0  # Safety margin before 150s kill
    
    best_labels = None
    best_cut = -1
    
    def compute_cut(labels):
        """Number of edges crossing the partition."""
        mask0 = labels == 0
        mask1 = labels == 1
        cut = adj[mask0][:, mask1].sum()
        return int(cut)
    
    def simulated_annealing(labels, T_start, T_end, steps):
        """
        Simulated Annealing for Maximum Cut.
        """
        labels = labels.copy()
        # delta[v] = change in cut size if v is flipped
        # delta[v] = 2 * (number of neighbors with same label) - degree[v]
        delta = np.zeros(n, dtype=np.int32)
        
        # Initialize delta
        for v in range(n):
            if labels[v] == 0:
                same_count = adj[v, 0].sum()
            else:
                same_count = adj[v, 1].sum()
            delta[v] = 2 * same_count - degrees[v]
        
        current_cut = compute_cut(labels)
        
        rng = np.random.default_rng()
        
        for step in range(steps):
            if time.time() > deadline - 2.0:
                break
            
            # Cool temperature
            # Linear cooling is simple and effective for short runs
            T = T_start + (T_end - T_start) * (step / steps)
            
            # Pick a random vertex
            v = rng.integers(0, n)
            
            gain = delta[v]
            
            # Accept if gain > 0 or with probability exp(gain / T)
            accept = False
            if gain > 0:
                accept = True
            else:
                # Avoid division by zero or underflow
                if T > 1e-6:
                    prob = np.exp(gain / T)
                    if rng.random() < prob:
                        accept = True
            
            if accept:
                # Flip v
                old_label = labels[v]
                labels[v] = 1 - old_label
                current_cut += gain
                
                # Update delta for v
                delta[v] = -delta[v]
                
                # Update delta for neighbors
                for u in neighbors[v]:
                    # If u had same label as old v, it becomes different -> delta[u] decreases by 2
                    # If u had different label as old v, it becomes same -> delta[u] increases by 2
                    if labels[u] == old_label:
                        delta[u] -= 2
                    else:
                        delta[u] += 2
                
                if current_cut > best_cut:
                    best_cut = current_cut
                    best_labels = labels.copy()
        
        return best_labels, best_cut
    
    rng_main = np.random.default_rng(seed=42)
    
    # Strategy 1: Random restarts with Simulated Annealing
    # We will run several restarts with different temperatures and steps
    # Based on N=100, 5000 steps per restart should be fast enough to fit many in 115s
    
    restarts = 0
    max_restarts = 5000 # High cap, time guard will stop it
    
    while time.time() < deadline - 5.0:
        restarts += 1
        
        # Random initial labels
        labels_rand = rng_main.integers(0, 2, size=n).astype(np.int8)
        
        # Temperature schedule
        # Start high enough to allow exploration, end low for refinement
        T_start = 10.0
        T_end = 0.1
        steps = 2000
        
        # Run SA
        labels_sa, cut_sa = simulated_annealing(labels_rand, T_start, T_end, steps)
        
        if cut_sa > best_cut:
            best_cut = cut_sa
            best_labels = labels_sa.copy()
            
        # Occasionally try a greedy local search from the SA result to polish
        if restarts % 10 == 0:
            labels_polish = best_labels.copy()
            # Simple greedy polish
            improved = True
            while improved:
                improved = False
                for v in range(n):
                    if time.time() > deadline - 2.0:
                        break
                    # Calculate gain
                    if labels_polish[v] == 0:
                        same = adj[v, 0].sum()
                    else:
                        same = adj[v, 1].sum()
                    gain = 2 * same - degrees[v]
                    if gain > 0:
                        labels_polish[v] = 1 - labels_polish[v]
                        improved = True
                if time.time() > deadline - 2.0:
                    break
            
            cut_polish = compute_cut(labels_polish)
            if cut_polish > best_cut:
                best_cut = cut_polish
                best_labels = labels_polish.copy()

    if best_labels is None:
        best_labels = np.zeros(n, dtype=np.int8)
    
    return best_labels.astype(np.int8)
# EVOLVE-BLOCK-END
