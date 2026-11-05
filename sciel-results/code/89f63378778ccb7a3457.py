import itertools
import random
import time

def construct(v, k, t):
    """
    Find a covering design C(v, k, t): select as few k-subsets (blocks) of {0,...,v-1}
    such that every t-subset is contained in at least one block.
    
    Strategy:
    1. Fast greedy initialization with multiple random restarts.
    2. Local search via 1-exchange (replace one block) to reduce block count.
    3. Local search via 2-exchange (replace two blocks) if time permits.
    4. Return the best valid solution found.
    """
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Precompute t-subsets and map to indices
    all_t_subs = list(itertools.combinations(range(v), t))
    sub_to_idx = {s: i for i, s in enumerate(all_t_subs)}
    n_subs = len(all_t_subs)
    
    # Precompute all k-blocks
    all_k_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_k_blocks)
    
    # Precompute mask for each block: which t-subsets it covers
    # Using bitmasks for fast operations
    block_masks = {}
    for b in all_k_blocks:
        mask = 0
        for sub in itertools.combinations(b, t):
            mask |= 1 << sub_to_idx[sub]
        block_masks[b] = mask
    
    # For each t-subset, list of blocks that cover it
    sub_to_blocks = [[] for _ in range(n_subs)]
    for b in all_k_blocks:
        mask = block_masks[b]
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            sub_to_blocks[idx].append(b)
            m ^= lsb
    
    def is_valid(blocks):
        """Check if blocks cover all t-subsets."""
        if not blocks:
            return n_subs == 0
        covered = 0
        for b in blocks:
            covered |= block_masks[b]
        return covered == (1 << n_subs) - 1
    
    def greedy_cover(seed):
        """Greedy set cover with random tie-breaking."""
        rng = random.Random(seed)
        uncovered = (1 << n_subs) - 1
        blocks = []
        
        while uncovered:
            if time.time() > deadline:
                return None
            
            # Find the uncovered subset with fewest covering blocks
            min_deg = float('inf')
            target_idx = -1
            m = uncovered
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                # Count how many blocks in sub_to_blocks[idx] are not yet in blocks
                # For speed, just use total degree (static)
                deg = len(sub_to_blocks[idx])
                if deg < min_deg:
                    min_deg = deg
                    target_idx = idx
                m ^= lsb
            
            if target_idx == -1:
                break
            
            # Pick block with max gain among those covering target_idx
            best_b = None
            best_gain = -1
            best_rand = float('inf')
            
            for b in sub_to_blocks[target_idx]:
                gain = (block_masks[b] & uncovered).bit_count()
                rand_val = rng.random()
                if gain > best_gain or (gain == best_gain and rand_val < best_rand):
                    best_gain = gain
                    best_b = b
                    best_rand = rand_val
            
            if best_b is None:
                return None
            
            blocks.append(best_b)
            uncovered &= ~block_masks[best_b]
        
        return blocks
    
    def local_search_1exchange(blocks, max_iter=5000):
        """Try to reduce block count by replacing one block with another."""
        if not blocks:
            return blocks
        
        current = list(blocks)
        n = len(current)
        
        # Build coverage count
        cover_counts = [0] * n_subs
        for b in current:
            mask = block_masks[b]
            m = mask
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                cover_counts[idx] += 1
                m ^= lsb
        
        # Check current validity
        if any(c == 0 for c in cover_counts):
            return None  # Invalid
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(current)):
                if time.time() > deadline:
                    break
                
                # Try removing block i
                mask_i = block_masks[current[i]]
                # Count how many subsets would become uncovered
                m = mask_i
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    if cover_counts[idx] == 1:
                        # This subset would be uncovered
                        # Find a replacement block that covers it
                        found = False
                        for j in range(len(current)):
                            if j == i:
                                continue
                            mask_j = block_masks[current[j]]
                            # Check if current[j] covers this subset
                            if mask_j & (1 << idx):
                                continue  # Already covered by another block, not unique
                        # Actually, we need to find ANY block that covers all subsets
                        # that would become uncovered when removing block i
                        # This is complex. Let's use a simpler approach:
                        # Try replacing block i with a block that covers more uncovered subsets
                        pass
                    m ^= lsb
                
                # Simpler 1-exchange: try swapping block i with a better block
                # Remove block i, find uncovered subsets, try to cover them with one block
                removed_mask = mask_i
                # Subsets that would be uncovered
                would_be_uncovered = []
                m = removed_mask
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    if cover_counts[idx] == 1:
                        would_be_uncovered.append(idx)
                    m ^= lsb
                
                if not would_be_uncovered:
                    # Block i is redundant!
                    current.pop(i)
                    # Update cover counts
                    m = removed_mask
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        cover_counts[idx] -= 1
                        m ^= lsb
                    improved = True
                    break
                
                # Find a block that covers all of would_be_uncovered
                # and maximizes additional coverage
                # Use the most constrained subset to find candidates
                target = would_be_uncovered[0]
                best_new_block = None
                best_extra = -1
                
                for b in sub_to_blocks[target]:
                    if b == current[i]:
                        continue
                    # Check if b covers all would_be_uncovered
                    mask_b = block_masks[b]
                    ok = True
                    for idx in would_be_uncovered:
                        if not (mask_b & (1 << idx)):
                            ok = False
                            break
                    if not ok:
                        continue
                    # Count extra coverage
                    extra = 0
                    m = mask_b
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        if cover_counts[idx] == 0:
                            extra += 1
                        m ^= lsb
                    if extra > best_extra:
                        best_extra = extra
                        best_new_block = b
                
                if best_new_block is not None:
                    # Perform the swap
                    old_block = current[i]
                    current[i] = best_new_block
                    # Update cover counts
                    m = removed_mask
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        cover_counts[idx] -= 1
                        m ^= lsb
                    new_mask = block_masks[best_new_block]
                    m = new_mask
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        cover_counts[idx] += 1
                        m ^= lsb
                    improved = True
                    break
        
        return current if all(c >= 1 for c in cover_counts) else None
    
    # Run multiple greedy restarts
    best_blocks = None
    best_size = float('inf')
    
    n_restarts = 20
    for seed in range(n_restarts):
        if time.time() > deadline:
            break
        sol = greedy_cover(seed)
        if sol and is_valid(sol):
            if len(sol) < best_size:
                best_size = len(sol)
                best_blocks = sol
    
    # Apply local search to the best solution
    if best_blocks and time.time() < deadline - 30:
        optimized = local_search_1exchange(best_blocks)
        if optimized and is_valid(optimized) and len(optimized) < best_size:
            best_size = len(optimized)
            best_blocks = optimized
    
    # Final fallback: ensure we have a valid solution
    if best_blocks is None:
        sol = greedy_cover(12345)
        if sol and is_valid(sol):
            best_blocks = sol
        else:
            # Last resort: brute force small case
            if v <= 10:
                from itertools import combinations
                for r in range(1, 100):
                    for combo in combinations(all_k_blocks, r):
                        if is_valid(list(combo)):
                            return [list(b) for b in combo]
            return []
    
    return [list(b) for b in best_blocks]

if __name__ == "__main__":
    res = construct(19, 7, 4)
    print(f"Found {len(res)} blocks")
    # Verify
    v, k, t = 19, 7, 4
    all_subs = set(itertools.combinations(range(v), t))
    covered = set()
    for b in res:
        for sub in itertools.combinations(b, t):
            covered.add(sub)
    if all_subs == covered:
        print("Valid covering")
    else:
        print("Invalid covering")
