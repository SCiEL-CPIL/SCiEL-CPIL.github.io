"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def partition(adjacency):
    """
    Solve Maximum Cut using simulated annealing with multiple restarts
    and greedy pre-improvement.
    
    adjacency: bool matrix (n x n)
    returns: 0/1 label per vertex (numpy array of int)
    """
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=int)
    
    # Convert to numpy array for faster operations
    adj = np.array(adjacency, dtype=np.int8)
    
    # Deadline guard: return best result before 150s wall-clock
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin of 20s
    
    def cut_size(labels):
        """Compute the cut size: number of edges crossing the partition."""
        # Edge (i,j) crosses if labels[i] != labels[j]
        # Using matrix operations for speed
        # For each edge (i,j), it contributes if adj[i,j] * (labels[i] != labels[j])
        # Efficient way: sum of adj * (labels[:, None] != labels[None, :]) / 2
        # But let's use a simpler approach:
        # cut = sum over all i<j where adj[i,j]=1 and labels[i]!=labels[j]
        # Using: 0.5 * sum_{i,j} adj[i,j] * (labels[i] - labels[j])^2
        # Since labels are 0/1, (labels[i]-labels[j])^2 is 1 iff they differ
        diff = (labels[:, None] - labels[None, :]) ** 2
        return int(np.sum(adj * diff) // 2)
    
    def greedy_flip_improve(labels, max_iters=10):
        """Greedy local search: flip vertices that increase cut size."""
        best_labels = labels.copy()
        best_cut = cut_size(best_labels)
        
        for _ in range(max_iters):
            if time.time() > deadline - 5.0:
                break
            improved = False
            for v in range(n):
                # Check if flipping v improves the cut
                new_labels = best_labels.copy()
                new_labels[v] = 1 - new_labels[v]
                new_cut = cut_size(new_labels)
                if new_cut > best_cut:
                    best_labels = new_labels
                    best_cut = new_cut
                    improved = True
            if not improved:
                break
        
        return best_labels, best_cut
    
    def simulated_annealing(initial_labels, start_temp=1.0, end_temp=0.01, n_iters=5000):
        """Simulated annealing to escape local optima."""
        labels = initial_labels.copy()
        current_cut = cut_size(labels)
        best_labels = labels.copy()
        best_cut = current_cut
        
        for i in range(n_iters):
            if time.time() > deadline - 5.0:
                break
            
            # Cooldown schedule
            temp = start_temp * (end_temp / start_temp) ** (i / n_iters)
            
            # Pick a random vertex to flip
            v = random.randint(0, n - 1)
            new_labels = labels.copy()
            new_labels[v] = 1 - new_labels[v]
            new_cut = cut_size(new_labels)
            
            delta = new_cut - current_cut
            if delta > 0 or random.random() < np.exp(delta / (temp + 1e-10)):
                labels = new_labels
                current_cut = new_cut
                if current_cut > best_cut:
                    best_labels = labels.copy()
                    best_cut = current_cut
        
        return best_labels, best_cut
    
    # Multiple restarts with diverse initializations
    best_labels_overall = None
    best_cut_overall = -1
    
    num_restarts = 8
    for restart in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
        
        # Diverse initializations
        if restart == 0:
            # Alternating start
            labels = np.arange(n) % 2
        elif restart == 1:
            # Random start
            labels = np.random.randint(0, 2, n)
        elif restart == 2:
            # Random with bias
            labels = (np.random.rand(n) > 0.4).astype(int)
        else:
            # More random starts
            labels = np.random.randint(0, 2, n)
        
        # Greedy pre-improvement
        labels, cut_val = greedy_flip_improve(labels, max_iters=5)
        
        if cut_val > best_cut_overall:
            best_cut_overall = cut_val
            best_labels_overall = labels.copy()
        
        # Simulated annealing to escape local optima
        sa_labels, sa_cut = simulated_annealing(labels, start_temp=1.0, end_temp=0.01, n_iters=3000)
        
        if sa_cut > best_cut_overall:
            best_cut_overall = sa_cut
            best_labels_overall = sa_labels.copy()
        
        # Final greedy polish
        polished_labels, polished_cut = greedy_flip_improve(sa_labels, max_iters=10)
        
        if polished_cut > best_cut_overall:
            best_cut_overall = polished_cut
            best_labels_overall = polished_labels.copy()
    
    # If no restarts completed, use a simple greedy from alternating start
    if best_labels_overall is None:
        labels = np.arange(n) % 2
        best_labels_overall, _ = greedy_flip_improve(labels, max_iters=10)
    
    return best_labels_overall
# EVOLVE-BLOCK-END
