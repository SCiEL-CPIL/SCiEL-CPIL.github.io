"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(demands, capacity):
    """
    Computes a heuristic matrix based on a co-occurrence frequency prior 
    derived from multiple runs of a Randomized Best-Fit heuristic.
    
    Strategy:
    1. Run Randomized Best-Fit (RBF) multiple times to generate a set of 
       high-quality packing solutions.
       - RBF shuffles item order and places each item in the bin with the 
         least remaining space (where it fits).
    2. For each solution, record which items are packed together in the same bin.
    3. Aggregate these co-occurrences into a matrix.
    4. The heuristic value eta[i, j] is proportional to the frequency with which 
       item j was placed in the same bin as item i across all runs.
    5. This provides the ACO with a 'learned' guide based on empirical success 
       of simple greedy strategies, rather than a static geometric rule.
    """
    n = len(demands)
    demands = np.asarray(demands, dtype=float)
    
    # Deadline guard
    start_time = time.time()
    max_time = 280.0  # Keep well under 300s limit
    
    # Number of simulation runs to build the prior.
    # We can adjust this if time allows, but 100-200 runs of O(N^2) greedy 
    # on N=120 is very fast.
    n_runs = 200
    
    # Matrix to store co-occurrence counts
    co_occurrence = np.zeros((n, n), dtype=np.float64)
    
    # We need to track bin assignments for each run to know who is with whom.
    # Efficient way: For each run, we assign items to bins. 
    # Then for each bin, we add 1 to all pairs in that bin.
    # Since N=120, max bins is ~120 (if no packing) but usually ~20-30.
    # Pairs per bin is small. Total pairs per run is sum(k_b^2) for bin sizes k_b.
    
    for run in range(n_runs):
        if time.time() - start_time > max_time:
            break
            
        # Shuffle item indices
        indices = np.random.permutation(n)
        
        # Best-Fit logic
        # bins: list of lists of item indices
        # remaining: list of remaining capacities
        bins = []
        remaining = []
        
        for idx in indices:
            item_size = demands[idx]
            best_bin_idx = -1
            min_remaining_after = np.inf
            
            # Check existing bins
            for b_idx, rem in enumerate(remaining):
                if rem >= item_size:
                    new_rem = rem - item_size
                    if new_rem < min_remaining_after:
                        min_remaining_after = new_rem
                        best_bin_idx = b_idx
                        
            if best_bin_idx != -1:
                bins[best_bin_idx].append(idx)
                remaining[best_bin_idx] = min_remaining_after
            else:
                # Open new bin
                bins.append([idx])
                remaining.append(capacity - item_size)
        
        # Update co-occurrence matrix
        for bin_items in bins:
            if len(bin_items) > 1:
                # Create a dense submatrix for this bin's items
                # Or just loop pairs. N is small.
                for i_idx in bin_items:
                    for j_idx in bin_items:
                        if i_idx != j_idx:
                            co_occurrence[i_idx, j_idx] += 1.0
    
    # Convert counts to heuristic values
    # We want larger values for more attractive pairs.
    # If co_occurrence is 0, the heuristic should be low (1e-9).
    # If co_occurrence is high, heuristic should be high.
    
    # Normalize or scale?
    # The ACO uses pheromone * heuristic. 
    # If heuristic values are large (e.g., 100), they dominate pheromone initially.
    # If they are small, pheromone dominates.
    # A good heuristic should be O(1) or slightly larger than 1.
    
    # Let's normalize by the maximum count to keep values in [0, 1] or [1e-9, 1].
    # But we want to preserve the relative differences.
    max_count = np.max(co_occurrence)
    
    if max_count > 0:
        eta = co_occurrence / max_count
    else:
        eta = np.ones((n, n))
        
    # Ensure diagonal is low (can't place item with itself)
    np.fill_diagonal(eta, 1e-9)
    
    # Entries that are 0 (never co-occurred) should be 1e-9 to avoid zero probability
    # unless we want to strictly forbid them. The task says "entries that are not positive become 1e-9".
    # So 0 becomes 1e-9.
    eta[eta == 0] = 1e-9
    
    # Optional: Add a small base value to all valid pairs (sum <= capacity) to allow exploration?
    # The task says "Heuristic entries that are not positive become 1e-9".
    # It doesn't say we must set invalid pairs to 1e-9, but the ACO logic "picks a fitting item" 
    # usually implies the mask is applied in the ACO loop or the heuristic is 0/1e-9 for invalid.
    # The current program explicitly sets invalid pairs to 1e-9.
    # We should do the same to be safe and consistent with "fitting item" logic if the ACO 
    # doesn't mask by capacity itself (though the prompt says "picks a fitting item", 
    # suggesting the ACO checks fit. But if heuristic is high for non-fitting items, 
    # it might be selected and then rejected? Or does the ACO only consider fitting items?
    # "repeatedly picks a fitting item with probability proportional to..."
    # This implies the probability distribution is over *fitting* items only.
    # So the heuristic values for non-fitting items don't strictly matter for the choice, 
    # but setting them to 1e-9 is good practice.
    
    pair_sums = demands[:, None] + demands[None, :]
    valid = pair_sums <= capacity
    
    # Set invalid pairs to 1e-9
    eta[~valid] = 1e-9
    
    return eta
# EVOLVE-BLOCK-END
