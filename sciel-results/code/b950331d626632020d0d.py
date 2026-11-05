"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def partition(adjacency):
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=int)
    
    # Convert to numpy array for fast operations
    A = np.array(adjacency, dtype=np.int8)
    
    # Deadline guard
    start_time = time.time()
    deadline = 140.0
    
    # Helper to calculate cut size for a given label array
    def calc_cut_size(labels):
        # Cut size = sum of edges where labels differ
        # labels is 0/1. 
        # mask[i,j] = 1 if labels[i] != labels[j]
        # Efficient: A * (1 - (labels[:, None] == labels[None, :]))
        # Or simply: sum(A * (labels[:, None] != labels[None, :]))
        # For n=100, this is very fast.
        return np.sum(A * (labels[:, None] != labels[None, :]))

    # Helper to perform greedy local search (polishing)
    # Returns the improved labels and cut size
    def local_search(labels, current_cut):
        # We flip vertices that improve the cut until no improvement is possible.
        # To be efficient, we can use a priority queue or just scan.
        # Scanning is O(n^2) per pass. n=100, so 10k ops.
        # Number of passes is usually small.
        
        improved = True
        while improved:
            improved = False
            # To avoid bias, we could shuffle order, but for polishing, 
            # sequential or random is fine. Let's use random order to break symmetry.
            order = np.random.permutation(n)
            for v in order:
                # Calculate gain of flipping v
                # gain = (neighbors in other set) - (neighbors in same set)
                # neighbors_same = sum(A[v, labels == labels[v]])
                # neighbors_other = sum(A[v, labels != labels[v]])
                
                # Using boolean indexing
                same_mask = (labels == labels[v])
                # A[v] is the row.
                # Note: A[v, v] is 0 (no self loops usually, but even if not, 
                # labels[v]==labels[v] so it's in same_mask, but A[v,v] contributes 0 to cut change 
                # because it doesn't cross partition. Actually, if self loop existed, 
                # flipping doesn't change cut for self loop. So it's safe.)
                
                # We can compute this faster with precomputed sums?
                # Let's stick to simple numpy for clarity and correctness, 
                # n=100 is small enough.
                sum_same = np.sum(A[v][same_mask])
                sum_other = np.sum(A[v][~same_mask])
                gain = sum_other - sum_same
                
                if gain > 0:
                    labels[v] = 1 - labels[v]
                    current_cut += gain
                    improved = True
                    
        return labels, current_cut

    # Main ILS Loop
    best_labels = None
    best_cut = -1
    
    # Initial random solution
    labels = np.random.randint(0, 2, size=n)
    current_cut = calc_cut_size(labels)
    
    # Initial polish
    labels, current_cut = local_search(labels, current_cut)
    
    if current_cut > best_cut:
        best_cut = current_cut
        best_labels = labels.copy()
    
    # ILS Parameters
    # Perturbation strength: number of random flips
    # We want to move far enough to escape local minima but not destroy the solution completely.
    # For n=100, flipping 10-20% of vertices is a good start.
    perturb_size = int(0.15 * n)
    
    while time.time() < deadline:
        # 1. Perturb
        # Pick random vertices to flip
        if perturb_size > 0:
            flip_indices = np.random.choice(n, size=perturb_size, replace=False)
            labels[flip_indices] = 1 - labels[flip_indices]
            # Recalculate cut? 
            # It's faster to just recalculate from scratch for n=100 (10k ops) 
            # than to track deltas for 15 flips (15 * 100 = 1500 ops + overhead).
            # Actually, tracking is better if we do many flips. But for one batch, 
            # recalc is simple.
            current_cut = calc_cut_size(labels)
            
        # 2. Local Search (Polish)
        labels, current_cut = local_search(labels, current_cut)
        
        # 3. Acceptance Criterion
        # Standard ILS: Accept if better than best, or with some probability if worse (to escape).
        # Or: Accept if better than current best. If not, we stay in current basin?
        # Actually, ILS usually works on a "current" solution.
        # If new solution is better than best, update best.
        # Should we update our "current" working solution?
        # If we always return to best, we might get stuck in one basin.
        # Strategy: 
        #   - Keep a 'current' solution.
        #   - If perturbed+polished is better than 'current', update 'current'.
        #   - If perturbed+polished is better than 'best', update 'best'.
        #   - This allows moving to slightly worse local optima if they lead to better ones later.
        
        # For simplicity and robustness in short time:
        # Accept if better than best. 
        # Also, with some probability, accept if better than the solution we started this iteration from (which is best).
        # Let's try: Accept if better than best. 
        # To diversify, we can occasionally restart from a random solution if we stagnate.
        
        if current_cut > best_cut:
            best_cut = current_cut
            best_labels = labels.copy()
            
        # Optional: Adaptive perturbation?
        # If we haven't improved in a while, increase perturbation size.
        # For now, fixed size is a good structural change from SA.
        
    return best_labels
# EVOLVE-BLOCK-END
