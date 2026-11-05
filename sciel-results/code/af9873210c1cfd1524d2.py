import itertools
import time

def construct(v, k, t):
    """
    Constructs a covering design C(v, k, t) using a greedy set cover algorithm
    with bitmasks for efficiency.
    """
    # Generate all t-subsets and map them to indices
    t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {s: i for i, s in enumerate(t_subsets)}
    n_t = len(t_subsets)
    
    # Generate all k-subsets and precompute their coverage bitmasks
    k_subsets = list(itertools.combinations(range(v), k))
    n_k = len(k_subsets)
    
    # block_masks[i] is an integer bitmask where bit j is set if t_subset j is contained in k_subset i
    block_masks = [0] * n_k
    
    for i, k_sub in enumerate(k_subsets):
        mask = 0
        # A k-subset contains C(k, t) t-subsets
        for t_sub in itertools.combinations(k_sub, t):
            idx = t_subset_to_idx[t_sub]
            mask |= (1 << idx)
        block_masks[i] = mask

    # Initial state: all t-subsets are uncovered
    # Using an integer as a bitset for uncovered t-subsets
    # Since n_t can be up to a few thousands, Python's arbitrary precision int handles this efficiently.
    uncovered_mask = (1 << n_t) - 1
    
    blocks = []
    
    # Greedy Set Cover
    while uncovered_mask:
        best_block_idx = -1
        max_covered = 0
        
        # Find the block that covers the most currently uncovered t-subsets
        for i in range(n_k):
            # Count bits in the intersection of uncovered_mask and block_masks[i]
            # Python 3.10+ has int.bit_count(). If older, use bin(x).count('1').
            # Assuming modern Python environment.
            covered = (uncovered_mask & block_masks[i]).bit_count()
            
            if covered > max_covered:
                max_covered = covered
                best_block_idx = i
                
        if max_covered == 0:
            # Should not happen if a covering exists, but safety check
            break
            
        # Add the best block to the solution
        blocks.append(list(k_subsets[best_block_idx]))
        
        # Update uncovered mask: remove all t-subsets covered by this block
        uncovered_mask &= ~block_masks[best_block_idx]
        
    return blocks
