import itertools
import random
import time
import math

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # Pre-compute all t-subsets (triples)
    all_triples = list(itertools.combinations(range(v), t))
    num_triples = len(all_triples)
    triple_to_idx = {tri: i for i, tri in enumerate(all_triples)}
    
    # Pre-compute all k-subsets (blocks)
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # Pre-compute bitmask for each block
    # bit i is set if block covers triple i
    block_masks = [0] * num_blocks
    for bi, b in enumerate(all_blocks):
        mask = 0
        for tri in itertools.combinations(b, t):
            idx = triple_to_idx[tri]
            mask |= (1 << idx)
        block_masks[bi] = mask
        
    # Pre-compute for each triple, the list of blocks that cover it
    # This is needed for fast repair in local search
    triple_to_blocks = [[] for _ in range(num_triples)]
    for bi in range(num_blocks):
        mask = block_masks[bi]
        # Iterate through set bits
        m = mask
        while m:
            # Get lowest set bit
            lb = m & -m
            idx = lb.bit_length() - 1
            triple_to_blocks[idx].append(bi)
            m ^= lb
            
    # Helper to get popcount
    def popcount(x):
        return x.bit_count()
        
    # Helper to find an uncovered triple index
    def find_uncovered(uncovered_mask):
        if uncovered_mask == 0:
            return -1
        # Get lowest set bit
        lb = uncovered_mask & -uncovered_mask
        return lb.bit_length() - 1

    best_blocks = None
    best_len = float('inf')
    
    # Multi-Start Greedy
    # We run multiple greedy constructions with different random seeds/orderings
    # Since we have time, we can do many restarts.
    
    num_restarts = 50 # Estimate: Greedy is fast with bitmasks
    
    for restart in range(num_restarts):
        if time.time() > deadline - 20: # Leave time for SA
            break
            
        random.seed(42 + restart)
        
        uncovered_mask = (1 << num_triples) - 1
        current_blocks = []
        
        # Greedy Phase
        # To add diversity, we can randomize the order of triples or the choice among best blocks
        while uncovered_mask != 0:
            # Pick a random uncovered triple to focus on
            # This helps escape local biases of a deterministic "first uncovered"
            # We can also just pick the first one, but random is better for diversity
            # Efficient random selection from set bits is tricky, so we pick a random index 
            # and check if it's uncovered, or just iterate. 
            # Given num_triples ~ 2024, iterating is fast enough.
            # Actually, picking the "hardest" triple (fewest candidates) is a standard heuristic.
            # Let's stick to random for diversity in restarts.
            
            # Find a random uncovered triple
            # We can just pick a random index from 0 to num_triples-1 until we find an uncovered one
            # Or better: extract a random set bit.
            # Python doesn't have a built-in random bit select.
            # Let's just pick the first uncovered one for speed, but randomize the 
            # tie-breaking in block selection.
            
            target_idx = find_uncovered(uncovered_mask)
            if target_idx == -1:
                break
                
            candidates = triple_to_blocks[target_idx]
            
            best_gain = -1
            best_block_idx = -1
            best_block_mask = 0
            
            # Evaluate all blocks covering this triple
            for bi in candidates:
                bm = block_masks[bi]
                gain = popcount(uncovered_mask & bm)
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = bi
                    best_block_mask = bm
                # Tie-breaking: randomize if gain is same
                elif gain == best_gain and random.random() < 0.5:
                    best_block_idx = bi
                    best_block_mask = bm
            
            if best_block_idx != -1:
                current_blocks.append(best_block_idx)
                uncovered_mask &= ~best_block_mask
        
        if len(current_blocks) < best_len:
            best_len = len(current_blocks)
            best_blocks = current_blocks[:]
            
    # If we haven't found a solution or ran out of time, use the best found
    if best_blocks is None:
        # Fallback: just return a valid cover (might be large)
        # Greedy without restarts
        uncovered_mask = (1 << num_triples) - 1
        current_blocks = []
        while uncovered_mask != 0:
            target_idx = find_uncovered(uncovered_mask)
            if target_idx == -1: break
            candidates = triple_to_blocks[target_idx]
            best_gain = -1
            best_block_idx = -1
            best_block_mask = 0
            for bi in candidates:
                bm = block_masks[bi]
                gain = popcount(uncovered_mask & bm)
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = bi
                    best_block_mask = bm
            if best_block_idx != -1:
                current_blocks.append(best_block_idx)
                uncovered_mask &= ~best_block_mask
        best_blocks = current_blocks
        best_len = len(current_blocks)

    # Simulated Annealing / Local Search
    # State: list of block indices
    # We maintain coverage counts to quickly check redundancy
    # But with bitmasks, we can check if a block is redundant by seeing if 
    # its mask is a subset of the union of other blocks' masks.
    # Union of others = (1 << num_triples) - 1 ^ (intersection of complements)? No.
    # Let's maintain a list of masks.
    
    current_state = best_blocks[:]
    # Optimization: Store masks for current state for faster access
    current_masks = [block_masks[bi] for bi in current_state]
    
    # To check redundancy of block i:
    # Is current_masks[i] subset of OR(current_masks[j] for j != i)?
    # This is O(N) per check.
    
    # Better Local Search Move:
    # 1. Try to remove a block. If valid, accept.
    # 2. Try to swap a block with another.
    
    # We will use a simple Hill Climbing with occasional random restarts or 
    # "kick" moves if stuck.
    
    T = 1.0
    cooling_rate = 0.999
    T_min = 0.01
    
    # We need to be careful with time.
    # Bitmask operations are fast, but O(N) redundancy checks are not.
    # N is around 70-80.
    # Redundancy check for one block: O(N * 1) bitwise ORs? 
    # Actually, we can precompute the "coverage union" of all blocks except one?
    # That's expensive to update.
    
    # Alternative: 
    # Maintain `coverage_counts` array of size num_triples.
    # This allows O(1) redundancy check (if all counts > 1, redundant).
    # Updating counts on add/remove is O(35) per block.
    # This is likely faster than O(N) bitwise ORs for N~80.
    
    # Let's switch back to counts for the local search phase, as it's more standard and easy to parallelize logic (though we are single threaded).
    
    # Re-initialize counts for current_state
    counts = [0] * num_triples
    for bi in current_state:
        mask = block_masks[bi]
        m = mask
        while m:
            lb = m & -m
            idx = lb.bit_length() - 1
            counts[idx] += 1
            m ^= lb
            
    # Helper to remove redundant blocks
    def remove_redundant(state, masks, counts):
        changed = True
        while changed:
            changed = False
            for i in range(len(state) - 1, -1, -1):
                mask = masks[i]
                m = mask
                redundant = True
                while m:
                    lb = m & -m
                    idx = lb.bit_length() - 1
                    if counts[idx] == 1:
                        redundant = False
                        break
                    m ^= lb
                if redundant:
                    # Remove block i
                    m = mask
                    while m:
                        lb = m & -m
                        idx = lb.bit_length() - 1
                        counts[idx] -= 1
                        m ^= lb
                    state.pop(i)
                    masks.pop(i)
                    changed = True
                    if len(state) < best_len:
                        best_len = len(state)
                        best_blocks = state[:]
                        # Update best state reference
                        # Note: best_blocks is a list of indices
                    break # Restart scan as indices changed
    
    # Initial cleanup
    remove_redundant(current_state, current_masks, counts)
    
    # Local Search Loop
    no_improve = 0
    max_no_improve = 500
    
    while time.time() < deadline - 2 and no_improve < max_no_improve:
        if not current_state:
            break
            
        # Choose a move
        # 1. Try to remove a random block
        # 2. If fails, try to swap a random block with a better one
        
        move_type = random.random()
        
        if move_type < 0.3:
            # Try removal
            idx = random.randint(0, len(current_state) - 1)
            bi = current_state[idx]
            mask = current_masks[idx]
            
            # Check if removable
            m = mask
            can_remove = True
            while m:
                lb = m & -m
                t_idx = lb.bit_length() - 1
                if counts[t_idx] == 1:
                    can_remove = False
                    break
                m ^= lb
            
            if can_remove:
                # Accept removal
                m = mask
                while m:
                    lb = m & -m
                    t_idx = lb.bit_length() - 1
                    counts[t_idx] -= 1
                    m ^= lb
                current_state.pop(idx)
                current_masks.pop(idx)
                
                # Remove further redundancies
                remove_redundant(current_state, current_masks, counts)
                
                if len(current_state) < best_len:
                    best_len = len(current_state)
                    best_blocks = current_state[:]
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                no_improve += 1
                
        else:
            # Try swap
            idx = random.randint(0, len(current_state) - 1)
            bi_remove = current_state[idx]
            mask_remove = current_masks[idx]
            
            # Temporarily remove
            m = mask_remove
            while m:
                lb = m & -m
                t_idx = lb.bit_length() - 1
                counts[t_idx] -= 1
                m ^= lb
                
            # Find uncovered triples
            uncovered_list = []
            m = mask_remove
            while m:
                lb = m & -m
                t_idx = lb.bit_length() - 1
                if counts[t_idx] == 0:
                    uncovered_list.append(t_idx)
                m ^= lb
                
            if not uncovered_list:
                # Pure removal, handle like above
                current_state.pop(idx)
                current_masks.pop(idx)
                remove_redundant(current_state, current_masks, counts)
                if len(current_state) < best_len:
                    best_len = len(current_state)
                    best_blocks = current_state[:]
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                # Need to repair
                # Find a block that covers all uncovered_list
                # Pick the most constrained uncovered triple
                min_cand = float('inf')
                target_tri = -1
                for t_idx in uncovered_list:
                    cands = triple_to_blocks[t_idx]
                    if len(cands) < min_cand:
                        min_cand = len(cands)
                        target_tri = t_idx
                
                best_swap_idx = -1
                best_swap_mask = 0
                best_overlap = -1 # Prefer block that overlaps most with current (minimizes new redundancy? or maximizes?)
                # Actually, we just need ANY valid block. 
                # To improve quality, we might prefer one that covers "many" of the other blocks' unique triples?
                # For now, find first valid one, or randomize among valid ones.
                
                valid_swaps = []
                for bi in triple_to_blocks[target_tri]:
                    if bi == bi_remove: continue
                    bm = block_masks[bi]
                    # Check if bm covers all uncovered_list
                    # Efficient check: 
                    # Create mask of uncovered_list
                    unc_mask = 0
                    for t_idx in uncovered_list:
                        unc_mask |= (1 << t_idx)
                    if (bm & unc_mask) == unc_mask:
                        valid_swaps.append(bi)
                        
                if valid_swaps:
                    # Pick one
                    # Heuristic: Pick one that maximizes popcount(bm & (union of current masks))
                    # This might help reduce total "waste"?
                    # Or just random.
                    bi_add = random.choice(valid_swaps)
                    mask_add = block_masks[bi_add]
                    
                    # Perform swap
                    current_state[idx] = bi_add
                    current_masks[idx] = mask_add
                    
                    # Update counts
                    m = mask_add
                    while m:
                        lb = m & -m
                        t_idx = lb.bit_length() - 1
                        counts[t_idx] += 1
                        m ^= lb
                        
                    # Remove redundancies
                    remove_redundant(current_state, current_masks, counts)
                    
                    if len(current_state) < best_len:
                        best_len = len(current_state)
                        best_blocks = current_state[:]
                        no_improve = 0
                    else:
                        no_improve += 1
                else:
                    # Revert
                    m = mask_remove
                    while m:
                        lb = m & -m
                        t_idx = lb.bit_length() - 1
                        counts[t_idx] += 1
                        m ^= lb
                    no_improve += 1

    # Final validation and cleanup
    # Rebuild counts for best_blocks to be absolutely sure
    final_counts = [0] * num_triples
    for bi in best_blocks:
        mask = block_masks[bi]
        m = mask
        while m:
            lb = m & -m
            t_idx = lb.bit_length() - 1
            final_counts[t_idx] += 1
            m ^= lb
            
    final_blocks = []
    for bi in best_blocks:
        mask = block_masks[bi]
        m = mask
        can_remove = True
        while m:
            lb = m & -m
            t_idx = lb.bit_length() - 1
            if final_counts[t_idx] == 1:
                can_remove = False
                break
            m ^= lb
        if not can_remove:
            final_blocks.append(bi)
            # Decrement counts for final check? Not strictly necessary if we just output
            # But good for logic consistency
            m = mask
            while m:
                lb = m & -m
                t_idx = lb.bit_length() - 1
                final_counts[t_idx] -= 1
                m ^= lb
                
    # Convert indices to actual blocks
    return [list(all_blocks[bi]) for bi in final_blocks]
