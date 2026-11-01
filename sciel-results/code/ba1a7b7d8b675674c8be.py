import itertools
import random
import time

def construct(v, k, t):
    all_t_subs = list(itertools.combinations(range(v), t))
    n_t_subs = len(all_t_subs)
    tsub_to_idx = {tsub: i for i, tsub in enumerate(all_t_subs)}
    
    all_k_subs = list(itertools.combinations(range(v), k))
    n_k_subs = len(all_k_subs)
    
    # ksub_to_tsubs[ksub_idx] = list of t-subset indices covered
    ksub_to_tsubs = []
    for ksub in all_k_subs:
        covered = []
        for tsub in itertools.combinations(ksub, t):
            covered.append(tsub_to_idx[tsub])
        ksub_to_tsubs.append(covered)
    
    # tsub_to_ksubs[tsub_idx] = list of k-subset indices that cover it
    tsub_to_ksubs = [[] for _ in range(n_t_subs)]
    for kidx, covered in enumerate(ksub_to_tsubs):
        for tidx in covered:
            tsub_to_ksubs[tidx].append(kidx)
    
    # Greedy initialization using Focused Greedy
    # Pick a random uncovered t-subset, then pick the k-subset covering it that maximizes total uncovered coverage
    uncovered = set(range(n_t_subs))
    blocks = []
    block_to_ksub = {}
    rng = random.Random(42)
    
    while uncovered:
        # Pick a random uncovered t-subset to focus on
        target_idx = rng.choice(list(uncovered))
        
        # Among k-subsets that cover target_idx, pick the one that covers most uncovered t-subsets
        best_kidx = -1
        best_gain = -1
        for kidx in tsub_to_ksubs[target_idx]:
            gain = 0
            for tidx in ksub_to_tsubs[kidx]:
                if tidx in uncovered:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_kidx = kidx
        
        blocks.append(all_k_subs[best_kidx])
        block_to_ksub[all_k_subs[best_kidx]] = best_kidx
        uncovered -= set(ksub_to_tsubs[best_kidx])
        
    # Local Search
    current_blocks = blocks[:]
    current_count = len(current_blocks)
    
    # Map block tuple to ksub index for quick lookup
    # Note: blocks are lists, convert to tuple for dict key
    
    # Helper to update cover counts
    def get_cover_counts(blk_list):
        counts = [0] * n_t_subs
        for b in blk_list:
            kidx = block_to_ksub[tuple(b)]
            for tidx in ksub_to_tsubs[kidx]:
                counts[tidx] += 1
        return counts

    cover_counts = get_cover_counts(current_blocks)
    
    # Helper to get exclusive t-subsets for a block
    def get_exclusive(blk_list, b_idx):
        b = blk_list[b_idx]
        kidx = block_to_ksub[tuple(b)]
        excl = set()
        for tidx in ksub_to_tsubs[kidx]:
            # Check if any OTHER block covers tidx
            # This is O(N) per tidx, so O(N * 35) per block
            # We can optimize by checking cover_counts and seeing if b is the only one?
            # No, cover_counts > 1 doesn't tell us if b is the only one.
            # We need to know if cover_counts[tidx] == 1 AND b covers tidx.
            if cover_counts[tidx] == 1:
                # Since b covers tidx, and count is 1, b is the only one
                excl.add(tidx)
        return excl

    # Main local search loop
    improved = True
    while improved:
        improved = False
        
        # 1. Try to remove any block
        for i in range(len(current_blocks) - 1, -1, -1):
            b = current_blocks[i]
            kidx = block_to_ksub[tuple(b)]
            # Check if any t-subset covered by b is exclusively covered by b
            removable = True
            for tidx in ksub_to_tsubs[kidx]:
                if cover_counts[tidx] == 1:
                    removable = False
                    break
            if removable:
                # Remove b
                for tidx in ksub_to_tsubs[kidx]:
                    cover_counts[tidx] -= 1
                current_blocks.pop(i)
                current_count -= 1
                improved = True
                break
        
        if improved:
            continue
            
        # 2. Try to swap a block to enable a future removal
        # Pick a block with small exclusive set
        # We want to find a block B_i such that its exclusive set E_i can be covered by some other block B_new
        # If we swap B_i with B_new, the size stays same, but the structure changes.
        
        # To be efficient, we only try a few swaps per iteration
        swaps_attempted = 0
        max_swaps = 10
        
        # Sort blocks by size of exclusive set (smallest first) to find easy swaps
        # This is expensive to compute for all blocks every time, but N is small (~150)
        # Let's just pick a random block with small exclusive set
        
        candidates = []
        for i in range(len(current_blocks)):
            b = current_blocks[i]
            kidx = block_to_ksub[tuple(b)]
            # Quick check: if any tidx has count 1, it's exclusive
            excl_count = 0
            for tidx in ksub_to_tsubs[kidx]:
                if cover_counts[tidx] == 1:
                    excl_count += 1
            if excl_count > 0 and excl_count < 10: # Only try small exclusive sets
                candidates.append((excl_count, i))
                
        if not candidates:
            break
            
        candidates.sort()
        
        for _, i in candidates[:max_swaps]:
            b = current_blocks[i]
            kidx = block_to_ksub[tuple(b)]
            
            # Get exclusive t-subsets
            excl = set()
            for tidx in ksub_to_tsubs[kidx]:
                if cover_counts[tidx] == 1:
                    excl.add(tidx)
            
            # Find a k-subset that covers all of excl
            # It should NOT be the same as current block (or it doesn't help)
            # Actually, if it's the same, it's not a swap.
            # We want a DIFFERENT block.
            
            # To speed up, we can intersect the lists of k-subsets covering each tidx in excl
            # Common k-subsets = intersection of tsub_to_ksubs[tidx] for tidx in excl
            
            common_ksubs = set(tsub_to_ksubs[excl[0]])
            for tidx in list(excl)[1:]:
                common_ksubs &= set(tsub_to_ksubs[tidx])
                if not common_ksubs:
                    break
            
            # Remove current block's kidx from candidates if it's there
            common_ksubs.discard(kidx)
            
            if not common_ksubs:
                continue
                
            # Pick a random new block from common_ksubs
            new_kidx = rng.choice(list(common_ksubs))
            
            # Perform swap
            # 1. Remove old block's coverage
            for tidx in ksub_to_tsubs[kidx]:
                cover_counts[tidx] -= 1
            
            # 2. Add new block's coverage
            for tidx in ksub_to_tsubs[new_kidx]:
                cover_counts[tidx] += 1
                
            # 3. Update block list
            current_blocks[i] = all_k_subs[new_kidx]
            block_to_ksub[all_k_subs[new_kidx]] = new_kidx
            # Note: We don't strictly need to update block_to_ksub for the old one, 
            # but it helps if we need to lookup. Since blocks are unique, 
            # and we replaced one, the new one is in the list.
            
            improved = True
            swaps_attempted += 1
            break # Restart the while loop to try removals again
            
        if swaps_attempted == 0:
            break
            
    return current_blocks
