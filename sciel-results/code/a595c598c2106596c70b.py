"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import random


def partition(adjacency):
    """
    Solve Maximum Cut using simulated annealing with multiple restarts.
    adjacency: bool matrix (n x n)
    returns: 0/1 label per vertex (numpy array of int)
    """
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=int)
    
    # Convert to numpy array for faster operations
    adj = np.array(adjacency, dtype=np.int8)
    
    def cut_size(labels):
        # Count edges crossing the partition
        # For each edge (i,j) where i<j, it crosses if labels[i] != labels[j]
        # Using matrix multiplication trick:
        # cut = 0.5 * sum over i,j of adj[i,j] * (1 if labels differ else 0)
        # (labels[i] != labels[j]) is 1 - (labels[i] == labels[j])
        # Since labels are 0/1, labels[i] == labels[j] is 1 - abs(labels[i]-labels[j])
        # So 1 - (labels[i]==labels[j]) = abs(labels[i]-labels[j])
        # cut = 0.5 * sum_{i,j} adj[i,j] * abs(labels[i]-labels[j])
        # But adj is symmetric, so we can compute:
        diff = np.abs(labels[:, None] - labels[None, :])
        return 0.5 * np.sum(adj * diff)
    
    def greedy_flip_improve(labels, max_iter=100):
        """Greedy local search: flip a vertex if it improves the cut."""
        labels = labels.copy()
        improved = True
        iter_count = 0
        while improved and iter_count < max_iter:
            improved = False
            iter_count += 1
            # For each vertex, compute gain from flipping
            for v in range(n):
                # Current contribution: edges from v to opposite side
                # After flip: edges from v to same side become crossing
                # Gain = (edges to same side) - (edges to opposite side)
                # Actually, let's compute directly
                same_side = (labels == labels[v])
                opp_side = (labels != labels[v])
                edges_to_same = adj[v, same_side].sum()
                edges_to_opp = adj[v, opp_side].sum()
                # If we flip v, the new cut size changes by:
                # Edges that were crossing (v to opp) become non-crossing: -edges_to_opp
                # Edges that were not crossing (v to same) become crossing: +edges_to_same
                # Net change = edges_to_same - edges_to_opp
                if edges_to_same > edges_to_opp:
                    labels[v] = 1 - labels[v]
                    improved = True
        return labels
    
    def simulated_annealing(labels, T_start=1.0, T_end=0.01, alpha=0.99, max_iter=5000):
        """Simulated annealing for maximum cut."""
        labels = labels.copy()
        current_cut = cut_size(labels)
        best_labels = labels.copy()
        best_cut = current_cut
        T = T_start
        
        for i in range(max_iter):
            # Pick a random vertex to flip
            v = random.randint(0, n - 1)
            new_labels = labels.copy()
            new_labels[v] = 1 - new_labels[v]
            new_cut = cut_size(new_labels)
            delta = new_cut - current_cut
            
            if delta > 0 or random.random() < np.exp(delta / T):
                labels = new_labels
                current_cut = new_cut
                if current_cut > best_cut:
                    best_cut = current_cut
                    best_labels = labels.copy()
            
            T = max(T_end, T * alpha)
        
        return best_labels
    
    # Multiple restarts with different initializations
    best_labels = None
    best_cut = -1
    
    num_restarts = 5
    
    for r in range(num_restarts):
        # Different initializations
        if r == 0:
            # Alternating
            labels = np.arange(n) % 2
        elif r == 1:
            # Random
            labels = np.random.randint(0, 2, n)
        elif r == 2:
            # All zeros
            labels = np.zeros(n, dtype=int)
        elif r == 3:
            # First half 0, second half 1
            labels = np.zeros(n, dtype=int)
            labels[n // 2:] = 1
        else:
            # Random with bias
            labels = (np.random.rand(n) > 0.3).astype(int)
        
        # First do greedy improvement
        labels = greedy_flip_improve(labels)
        
        # Then simulated annealing
        labels = simulated_annealing(labels)
        
        current_cut = cut_size(labels)
        if current_cut > best_cut:
            best_cut = current_cut
            best_labels = labels.copy()
    
    return best_labels
# EVOLVE-BLOCK-END
