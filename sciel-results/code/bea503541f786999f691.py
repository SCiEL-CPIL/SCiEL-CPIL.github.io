"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
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
    
    # Deadline guard: 120s limit, use 110s to be safe
    deadline = time.time() + 110.0
    
    def cut_size(labels):
        # Count edges crossing the cut
        # For each vertex, count neighbors in the other partition
        # Sum over all vertices and divide by 2 (each edge counted twice)
        pos = labels == 1
        neg = ~pos
        # Edges from pos to neg
        cut = adj[pos, :][:, neg].sum()
        return cut
    
    def local_improve(labels, max_passes=5):
        """Greedy local improvement: flip vertex if it increases cut."""
        best_labels = labels.copy()
        best_cut = cut_size(best_labels)
        
        for _ in range(max_passes):
            improved = False
            # Randomize order for diversity
            order = np.random.permutation(n)
            for v in order:
                if time.time() > deadline:
                    return best_labels, best_cut
                
                # Calculate gain from flipping vertex v
                current_label = best_labels[v]
                # Neighbors in same partition
                same_partition = best_labels == current_label
                # Neighbors in other partition
                other_partition = best_labels != current_label
                
                # Edges from v to same partition (these become crossing if we flip)
                edges_to_same = adj[v, same_partition].sum()
                # Edges from v to other partition (these stop being crossing if we flip)
                edges_to_other = adj[v, other_partition].sum()
                
                # Gain = edges_to_same - edges_to_other
                # If positive, flipping increases cut
                gain = edges_to_same - edges_to_other
                if gain > 0:
                    best_labels[v] = 1 - current_label
                    best_cut += gain
                    improved = True
            
            if not improved:
                break
        
        return best_labels, best_cut
    
    def simulated_annealing(initial_labels, T_start=1.0, T_end=0.01, cooling=0.995, max_iter=5000):
        """Simulated annealing to escape local optima."""
        labels = initial_labels.copy()
        current_cut = cut_size(labels)
        best_labels = labels.copy()
        best_cut = current_cut
        
        T = T_start
        for it in range(max_iter):
            if time.time() > deadline:
                break
            
            # Pick a random vertex to flip
            v = np.random.randint(n)
            labels[v] = 1 - labels[v]
            
            # Calculate new cut
            new_cut = cut_size(labels)
            delta = new_cut - current_cut
            
            if delta >= 0 or random.random() < np.exp(delta / T):
                current_cut = new_cut
                if current_cut > best_cut:
                    best_cut = current_cut
                    best_labels = labels.copy()
            
            # Cool down
            T *= cooling
            if T < T_end:
                break
        
        return best_labels, best_cut
    
    # Multiple restarts with different initializations
    best_labels = None
    best_cut = -1
    
    num_restarts = 10
    restarts_done = 0
    
    while restarts_done < num_restarts:
        if time.time() > deadline:
            break
        
        restarts_done += 1
        
        # Different initialization strategies
        if restarts_done == 1:
            # Alternating start
            initial = np.arange(n) % 2
        elif restarts_done == 2:
            # Random start
            initial = np.random.randint(0, 2, n)
        else:
            # Perturbed version of best solution found so far
            if best_labels is not None:
                initial = best_labels.copy()
                # Flip some random vertices
                num_flips = max(1, n // 10)
                flip_idx = np.random.choice(n, num_flips, replace=False)
                initial[flip_idx] = 1 - initial[flip_idx]
            else:
                initial = np.random.randint(0, 2, n)
        
        # First, do local improvement
        improved_labels, improved_cut = local_improve(initial, max_passes=10)
        
        if improved_cut > best_cut:
            best_cut = improved_cut
            best_labels = improved_labels.copy()
        
        # Then, try simulated annealing from the locally optimal solution
        sa_labels, sa_cut = simulated_annealing(improved_labels, T_start=1.0, T_end=0.005, cooling=0.998, max_iter=8000)
        
        if sa_cut > best_cut:
            best_cut = sa_cut
            best_labels = sa_labels.copy()
        
        # Final local improvement from SA result
        final_labels, final_cut = local_improve(sa_labels, max_passes=10)
        
        if final_cut > best_cut:
            best_cut = final_cut
            best_labels = final_labels.copy()
    
    # If we still have time, do more restarts
    while time.time() < deadline - 5.0:
        # Perturb best solution
        initial = best_labels.copy()
        num_flips = max(1, n // 8)
        flip_idx = np.random.choice(n, num_flips, replace=False)
        initial[flip_idx] = 1 - initial[flip_idx]
        
        improved_labels, improved_cut = local_improve(initial, max_passes=10)
        
        if improved_cut > best_cut:
            best_cut = improved_cut
            best_labels = improved_labels.copy()
        
        sa_labels, sa_cut = simulated_annealing(improved_labels, T_start=0.8, T_end=0.005, cooling=0.998, max_iter=5000)
        
        if sa_cut > best_cut:
            best_cut = sa_cut
            best_labels = sa_labels.copy()
        
        final_labels, final_cut = local_improve(sa_labels, max_passes=10)
        
        if final_cut > best_cut:
            best_cut = final_cut
            best_labels = final_labels.copy()
    
    return best_labels.astype(int)
# EVOLVE-BLOCK-END
