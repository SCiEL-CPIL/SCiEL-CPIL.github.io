"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def partition(adjacency):
    """
    Solve Maximum Cut using Simulated Annealing with spin-glass formulation.
    
    Args:
        adjacency: bool matrix of shape (n, n)
        
    Returns:
        labels: array of 0/1 labels per vertex
    """
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=int)
    
    # Convert to numpy array for efficient operations
    adj = np.asarray(adjacency, dtype=np.int8)
    
    # Best solution tracking
    best_labels = None
    best_cut = -1
    
    # Parameters for Simulated Annealing
    n_restarts = 30
    n_iterations = 20000
    T_start = 10.0
    T_end = 0.0001
    cooling_rate = (T_end / T_start) ** (1.0 / n_iterations)
    
    rng = np.random.default_rng(42)
    
    for restart in range(n_restarts):
        # Random initialization
        if restart == 0:
            # Start with alternating pattern
            labels = np.arange(n) % 2
        else:
            # Random initialization
            labels = rng.integers(0, 2, size=n)
            
        # Compute initial cut size
        # cut = sum of edges between 0s and 1s
        mask0 = labels == 0
        mask1 = labels == 1
        current_cut = adj[np.ix_(mask0, mask1)].sum()
        
        best_local_cut = current_cut
        best_local_labels = labels.copy()
        
        current_T = T_start
        
        for it in range(n_iterations):
            # Pick a random vertex
            v = rng.integers(0, n)
            
            # Calculate delta in cut size if we flip v
            # Using efficient computation
            row = adj[v]
            
            # Sum of edges to same side vs different side
            same_label = labels == labels[v]
            diff_label = labels != labels[v]
            
            same_count = row[same_label].sum()
            diff_count = row[diff_label].sum()
            
            delta = same_count - diff_count
            
            # Acceptance criterion
            if delta > 0:
                # Improvement
                labels[v] = 1 - labels[v]
                current_cut += delta
            else:
                # Deterioration or no change
                # Accept with probability exp(delta / T)
                if current_T > 1e-10:
                    prob = np.exp(delta / current_T)
                    if rng.random() < prob:
                        labels[v] = 1 - labels[v]
                        current_cut += delta
            
            # Track best local
            if current_cut > best_local_cut:
                best_local_cut = current_cut
                best_local_labels = labels.copy()
                
            # Update temperature
            current_T *= cooling_rate
            
        # Multi-pass greedy local search polish
        improved = True
        max_passes = 10
        pass_count = 0
        while improved and pass_count < max_passes:
            improved = False
            pass_count += 1
            for v in range(n):
                row = adj[v]
                same_label = best_local_labels == best_local_labels[v]
                diff_label = best_local_labels != best_local_labels[v]
                
                same_count = row[same_label].sum()
                diff_count = row[diff_label].sum()
                
                if same_count > diff_count:
                    best_local_labels[v] = 1 - best_local_labels[v]
                    best_local_cut += (same_count - diff_count)
                    improved = True
        
        # Additional local search with random order to escape local optima
        for _ in range(5):
            order = rng.permutation(n)
            improved = True
            while improved:
                improved = False
                for v in order:
                    row = adj[v]
                    same_label = best_local_labels == best_local_labels[v]
                    diff_label = best_local_labels != best_local_labels[v]
                    
                    same_count = row[same_label].sum()
                    diff_count = row[diff_label].sum()
                    
                    if same_count > diff_count:
                        best_local_labels[v] = 1 - best_local_labels[v]
                        best_local_cut += (same_count - diff_count)
                        improved = True
        
        # Update global best
        if best_local_cut > best_cut:
            best_cut = best_local_cut
            best_labels = best_local_labels.copy()
            
    return best_labels
# EVOLVE-BLOCK-END
