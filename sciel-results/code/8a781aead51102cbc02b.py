"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def partition(adjacency):
    # Convert to numpy array for efficient operations
    A = np.asarray(adjacency, dtype=np.int8)
    n = len(A)
    
    # Calculate total edges for normalization if needed, but we maximize raw cut size
    # Total edges = sum(A) / 2
    
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    best_labels = None
    best_score = -1
    
    # Multi-start local search
    # We will run random restarts until time runs out
    # Each restart: random init -> local search (flip if improves)
    
    # Precompute degrees? Not strictly necessary for the flip check, but good for speed.
    # For vertex v, gain from flipping = 2 * (num_neighbors_in_other_partition) - degree(v)
    # Or simply: current_cut_contrib = num_neighbors_in_other_partition
    # new_cut_contrib = num_neighbors_in_same_partition
    # Gain = new - current = same - other
    
    # Efficient local search implementation
    def local_search(labels):
        # labels is 0/1 array
        # We iterate until no improvement or a fixed number of passes
        # To be fast, we can do single pass sweeps until convergence
        # Or randomized order to break symmetry
        
        # Precompute neighbor lists? For 100 nodes, dense matrix ops are fine.
        # Let's use vectorized checks for speed in the inner loop if possible, 
        # but simple loops over 100 nodes are very fast in Python if optimized.
        
        # Actually, let's stick to a simple loop but make it robust.
        # n=100 is small.
        
        improved = True
        while improved:
            improved = False
            # Random order to avoid bias? Or fixed order? 
            # Fixed order is deterministic and fast. Random order might help escape plateaus within a run?
            # Usually, simple steepest ascent or first-improvement is fine.
            # Let's use first-improvement in a random order for robustness.
            order = np.random.permutation(n)
            for v in order:
                # Calculate gain of flipping v
                # Current label: labels[v]
                # Neighbors in same partition: A[v, labels == labels[v]].sum()
                # Neighbors in other partition: A[v, labels != labels[v]].sum()
                # If same > other, flipping increases cut size by (same - other)
                
                same = np.count_nonzero(A[v] & (labels == labels[v]))
                other = np.count_nonzero(A[v] & (labels != labels[v]))
                
                if same > other:
                    labels[v] = 1 - labels[v]
                    improved = True
        return labels

    # Run multiple restarts
    # Estimate cost per restart: 
    # n=100. Local search converges quickly, usually < 10 passes.
    # 10 passes * 100 nodes * (2 vector sums of size 100) = 20,000 ops. Very fast.
    # We can probably do thousands of restarts in 115 seconds.
    
    # Let's do a fixed number of high-quality restarts first, then fill remaining time.
    # Or just loop until deadline.
    
    # To maximize diversity, we use random initializations.
    # We can also try "spectral" init if scipy is available, but random is simpler and robust.
    # Given n=100, random init is surprisingly effective with local search.
    
    # Optimization: Use integer arrays for labels to speed up comparisons?
    # numpy bool/int comparisons are fast.
    
    # Let's implement a faster local search using matrix multiplication?
    # Let x be the label vector in {-1, 1}.
    # Cut size = (1/4) * sum_{i,j} A_{ij} (1 - x_i x_j)
    # This is equivalent to minimizing x^T L x / 4 + constant?
    # Maximizing cut is equivalent to minimizing x^T L x.
    # Local search in {-1, 1} space:
    # Flip x_i to -x_i.
    # Change in objective (x^T L x) when flipping i:
    # Delta = 2 * (sum_{j != i} L_{ij} x_j) * x_i ?
    # Let's stick to the combinatorial gain which is easier to verify.
    
    # Vectorized local search step:
    # For all i, compute gain_i.
    # If gain_i > 0, flip.
    # We can do this in parallel for all i?
    # If we flip all improving nodes simultaneously, it's parallel hill climbing.
    # This is faster than sequential.
    
    def parallel_local_search(labels):
        # labels: 0/1 array
        # Convert to -1/1 for easier math? Or stick to 0/1.
        # Let's use 0/1.
        
        # same_count[i] = sum_j A[i,j] * (labels[j] == labels[i])
        # other_count[i] = sum_j A[i,j] * (labels[j] != labels[i])
        
        # Matrix multiplication approach:
        # Let M be the adjacency matrix.
        # Let L be the label matrix (n x n) where L[i,j] = 1 if labels[i]==labels[j] else 0.
        # This is expensive to build.
        
        # Alternative:
        # same_count = (A * (labels == labels[:, None])).sum(axis=1)
        # This creates an n x n boolean matrix. For n=100, that's 10k elements. Very fast.
        
        while True:
            # Build comparison matrix: labels[i] == labels[j]
            # labels is 1D. labels[:, None] is (n,1). labels[None, :] is (1,n).
            same_mask = (labels[:, None] == labels[None, :])
            
            # Count neighbors in same partition
            # A is (n,n). same_mask is (n,n).
            # We want sum over j of A[i,j] * same_mask[i,j]
            same_counts = (A * same_mask).sum(axis=1)
            
            # Total degree
            degrees = A.sum(axis=1)
            
            # Neighbors in other partition
            other_counts = degrees - same_counts
            
            # Gain from flipping i is (same_counts[i] - other_counts[i])
            # If gain > 0, flip is beneficial.
            gains = same_counts - other_counts
            
            # Find indices to flip
            to_flip = np.where(gains > 0)[0]
            
            if len(to_flip) == 0:
                break
            
            # Flip all beneficial nodes simultaneously
            labels[to_flip] = 1 - labels[to_flip]
            
            # Check time inside the loop? 
            # The loop is fast, but let's be safe.
            if time.time() > deadline:
                break
        
        return labels

    # Main loop
    restart_count = 0
    while time.time() < deadline:
        # Random initialization
        labels = np.random.randint(0, 2, size=n)
        
        # Local search
        labels = parallel_local_search(labels)
        
        # Calculate score
        # Cut size = sum of edges crossing the partition
        # Crossing if labels[i] != labels[j]
        # cross_mask = (labels[:, None] != labels[None, :])
        # score = (A * cross_mask).sum() / 2  # Divide by 2 because each edge counted twice
        
        # More efficient:
        # same_mask = (labels[:, None] == labels[None, :])
        # same_edges = (A * same_mask).sum() / 2
        # total_edges = A.sum() / 2
        # cut_size = total_edges - same_edges
        
        total_edges = A.sum() // 2 # Integer division since undirected, sum is 2*E
        same_mask = (labels[:, None] == labels[None, :])
        same_edges = (A * same_mask).sum() // 2
        cut_size = total_edges - same_edges
        
        if cut_size > best_score:
            best_score = cut_size
            best_labels = labels.copy()
        
        restart_count += 1
        
        # Optional: If we are stuck at a very high score, we might want to break early?
        # No, keep searching until deadline.
        
    return best_labels
# EVOLVE-BLOCK-END
