import itertools
import random
import time

def construct(v, k, t):
    # Precompute all t-subsets and map to indices
    all_t_subs = list(itertools.combinations(range(v), t))
    n_t_subs = len(all_t_subs)
    
    # Full mask of all t-subsets
    full_mask = (1 << n_t_subs) - 1
    
    # Precompute all k-subsets
    all_k_subs = list(itertools.combinations(range(v), k))
    n_k_subs = len(all_k_subs)
    
    # Precompute bitmask for each k-subset
    # block_masks[i] is a bitmask where bit j is 1 if k-subset i covers t-subset j
    # We need a fast way to check if a k-subset covers a t-subset.
    # Since v=19, k=7, t=4.
    # We can represent each point as a bit in a 19-bit integer.
    # k_sub_mask: 19-bit int
    # t_sub_mask: 19-bit int
    # t_sub is covered by k_sub if (t_sub_mask & k_sub_mask) == t_sub_mask
    
    t_sub_masks = []
    for tsub in all_t_subs:
        m = 0
        for p in tsub:
            m |= (1 << p)
        t_sub_masks.append(m)
    
    k_sub_masks = []
    for ksub in all_k_subs:
        m = 0
        for p in ksub:
            m |= (1 << p)
        k_sub_masks.append(m)
    
    # Build block_masks
    # This is O(n_k_subs * n_t_subs * t) which is ~50k * 3876 * 4 ~ 770M ops. Too slow in Python.
    # We need a faster way.
    # For each t-subset index j, find all k-subsets that cover it?
    # Or for each k-subset, iterate its C(7,4)=35 t-subsets.
    # We need a mapping from t-subset tuple to index.
    t_sub_to_idx = {tsub: i for i, tsub in enumerate(all_t_subs)}
    
    block_masks = [0] * n_k_subs
    for i, ksub in enumerate(all_k_subs):
        mask = 0
        # Generate all t-subsets of ksub
        for tsub in itertools.combinations(ksub, t):
            idx = t_sub_to_idx[tsub]
            mask |= (1 << idx)
        block_masks[i] = mask
    
    # Greedy Set Cover
    uncovered = full_mask
    selected = []
    selected_set = set()
    
    while uncovered:
        best_idx = -1
        best_gain = 0
        for i in range(n_k_subs):
            if i in selected_set:
                continue
            gain = (block_masks[i] & uncovered).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_idx = i
        if best_idx == -1:
            break # Should not happen if full cover exists
        selected.append(best_idx)
        selected_set.add(best_idx)
        uncovered &= ~block_masks[best_idx]
        
    current_blocks = selected[:]
    
    # Local Search
    rng = random.Random(42)
    start_time = time.time()
    time_limit = 100.0 # Leave 20s buffer
    
    def get_union(indices):
        u = 0
        for i in indices:
            u |= block_masks[i]
        return u
    
    current_union = get_union(current_blocks)
    
    improved = True
    while improved:
        if time.time() - start_time > time_limit:
            break
        improved = False
        
        # Phase 1: Remove redundant blocks
        removed_any = False
        for i in range(len(current_blocks)):
            if time.time() - start_time > time_limit:
                break
            # Compute union of all others
            others_union = 0
            for j in range(len(current_blocks)):
                if i != j:
                    others_union |= block_masks[current_blocks[j]]
            if others_union == full_mask:
                # Block i is redundant
                current_blocks.pop(i)
                current_union = others_union
                improved = True
                removed_any = True
                break # Restart loop as indices changed
        if removed_any:
            continue
            
        # Phase 2: Try swaps to enable removal
        # For each block i in current_blocks, find a replacement j not in current_blocks
        # such that (others_union | block_masks[j]) == full_mask
        # This swap keeps the count same. Then we re-enter the loop to check for redundancy.
        # To be efficient, we only swap if it seems promising.
        swapped_any = False
        for i in range(len(current_blocks)):
            if time.time() - start_time > time_limit:
                break
            
            # Compute union of others
            others_union = 0
            for j in range(len(current_blocks)):
                if i != j:
                    others_union |= block_masks[current_blocks[j]]
            
            missing = full_mask & ~others_union
            if missing == 0:
                # Redundant, should have been caught in Phase 1
                # But maybe Phase 1 didn't run or failed?
                # If missing is 0, we can just remove it.
                current_blocks.pop(i)
                current_union = others_union
                improved = True
                swapped_any = True
                break
                
            # Find a block j not in current_blocks that covers 'missing'
            # (block_masks[j] & missing) == missing
            # To speed up, we can randomize the order of candidates or just iterate.
            # n_k_subs is ~50k. Iterating 50k * 150 blocks * iterations is too slow.
            # We need a faster way to find a candidate.
            
            # Heuristic: Pick a random t-subset from 'missing'.
            # Find all k-subsets that cover that t-subset.
            # One of them might cover all of 'missing'.
            
            # Let's get a random bit from missing
            if missing == 0:
                continue
                
            # Pick a random set bit from missing
            # Convert to binary string? No, use bit operations.
            # Find index of one of the bits
            # This is a bit tricky in Python without a popcount/ctz.
            # We can just pick the lowest set bit.
            # idx = (missing & -missing).bit_length() - 1
            target_t_idx = (missing & -missing).bit_length() - 1
            target_t_sub = all_t_subs[target_t_idx]
            
            # Find all k-subsets containing target_t_sub
            # Precomputed? No.
            # We can iterate all k-subsets? 50k.
            # Better: Precompute for each t-subset, the list of k-subsets covering it.
            # This was done in the original code but slow.
            # Let's build it once.
            
            # Actually, let's just try a few random k-subsets that cover the target_t_sub.
            # We need a list of k-subsets for each t-subset.
            # Let's build `t_to_ksubs` efficiently.
            
            # This is a one-time cost of ~50k * 35 lookups?
            # No, for each k-sub, add to 35 t-sub lists.
            # t_to_ksubs = [[] for _ in range(n_t_subs)]
            # for i, ksub in enumerate(all_k_subs):
            #    for tsub in itertools.combinations(ksub, t):
            #        idx = t_sub_to_idx[tsub]
            #        t_to_ksubs[idx].append(i)
            
            # Let's do this outside the loop if we haven't already.
            # Since we are inside the function, we can build it now if not built.
            # But building it takes time.
            # Let's assume we build it once before the local search.
            
            pass # Placeholder for logic below

    # We need to restructure to build t_to_ksubs before the loop.
    
    return [all_k_subs[i] for i in current_blocks]
