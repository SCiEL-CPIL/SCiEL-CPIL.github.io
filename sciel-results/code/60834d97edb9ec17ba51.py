"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def partition(adjacency):
    adj = np.asarray(adjacency, dtype=np.int8)
    n = adj.shape[0]
    if n == 0:
        return np.array([], dtype=int)

    start_time = time.time()
    # Use 145s as a safe deadline for 150s limit
    DEADLINE = start_time + 145.0

    def time_left():
        return DEADLINE - time.time()

    E = int(np.sum(adj)) // 2
    # Precompute nothing else needed, adj is small enough for full mat-vec
    
    def cut_size_from_labels(labels):
        x = 2 * labels.astype(np.float64) - 1
        # Cut size = E - (1/4) * sum_{i,j} adj_{ij} x_i x_j = E - 0.25 * x @ (adj @ x)
        # Wait, sum_{i,j} adj_{ij} x_i x_j = x^T A x.
        # Cut = (1/2) sum_{i,j} adj_{ij} (1 - x_i x_j)/2 = (1/4) sum adj - (1/4) x^T A x.
        # sum adj = 2E. So Cut = E - 0.25 * x^T A x.
        return int(E - 0.25 * (x @ (adj @ x)))

    def vectorized_local_search(labels):
        """
        Performs vectorized best-improvement local search.
        Flips all vertices that increase the cut size simultaneously.
        """
        labels = labels.copy()
        x = 2 * labels.astype(np.float64) - 1
        best_labels = labels.copy()
        best_cut = cut_size_from_labels(labels)
        
        # Limit iterations to avoid infinite loops in case of oscillation (shouldn't happen with strict > 0)
        for _ in range(100):
            if time_left() < 1.0:
                break
                
            Ax = adj @ x
            # Gain for flipping v is 2 * x[v] * Ax[v]
            gains = 2.0 * x * Ax
            flip_mask = gains > 0
            
            if not np.any(flip_mask):
                break
                
            x[flip_mask] = -x[flip_mask]
            labels[flip_mask] = 1 - labels[flip_mask]
            
            # Update cut size
            current_cut = cut_size_from_labels(labels)
            if current_cut > best_cut:
                best_cut = current_cut
                best_labels = labels.copy()
                
        return best_labels, best_cut

    def perturb(labels, num_flips):
        """
        Perturbs the labels by flipping a random subset of vertices.
        """
        perturbed = labels.copy()
        if num_flips > 0:
            # Select random indices to flip
            flip_indices = np.random.choice(n, size=min(num_flips, n), replace=False)
            perturbed[flip_indices] = 1 - perturbed[flip_indices]
        return perturbed

    def iterated_local_search(num_restarts):
        """
        Runs ILS: Start with random, local search, perturb, local search.
        """
        best_labels = np.random.randint(0, 2, size=n)
        best_labels, best_cut = vectorized_local_search(best_labels)
        
        # Adaptive perturbation strength
        base_perturbation = max(1, int(n * 0.05))
        
        for i in range(num_restarts):
            if time_left() < 5.0:
                break
                
            # Increase perturbation slightly as we go deeper? Or keep random.
            # Let's vary perturbation size
            p_size = random.randint(base_perturbation, base_perturbation * 3)
            
            perturbed_labels = perturb(best_labels, p_size)
            new_labels, new_cut = vectorized_local_search(perturbed_labels)
            
            if new_cut > best_cut:
                best_cut = new_cut
                best_labels = new_labels.copy()
                
        return best_labels, best_cut

    # Strategy:
    # 1. Quick initial local search from random
    # 2. ILS with many restarts
    # 3. Final polish
    
    # Initial start
    labels_init = np.random.randint(0, 2, size=n)
    best_labels, best_cut = vectorized_local_search(labels_init)
    
    # Run ILS
    # Estimate how many ILS iterations we can do.
    # Each ILS iter takes ~2 local searches.
    # Local search for n=100 is very fast.
    # Let's just loop until deadline.
    
    # We can also try a deterministic start like alternating
    labels_alt = np.arange(n) % 2
    labels_alt, cut_alt = vectorized_local_search(labels_alt)
    if cut_alt > best_cut:
        best_cut = cut_alt
        best_labels = labels_alt
        
    # Main ILS loop
    # Use a dynamic restart count based on time, but just loop while time remains
    while time_left() > 2.0:
        # Randomize perturbation size between 5% and 15% of nodes
        p_size = random.randint(max(1, int(n * 0.05)), max(2, int(n * 0.15)))
        perturbed_labels = perturb(best_labels, p_size)
        new_labels, new_cut = vectorized_local_search(perturbed_labels)
        
        if new_cut > best_cut:
            best_cut = new_cut
            best_labels = new_labels.copy()
            
    # Final polish: run local search one more time to be sure
    final_labels, final_cut = vectorized_local_search(best_labels)
    if final_cut > best_cut:
        best_cut = final_cut
        best_labels = final_labels

    return best_labels.astype(int)

# EVOLVE-BLOCK-END
