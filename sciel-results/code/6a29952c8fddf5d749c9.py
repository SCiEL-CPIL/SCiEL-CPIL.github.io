import itertools
import time
import random
from collections import defaultdict

def construct(v, k, t):
    """
    Construct a covering design C(v,k,t) using bitmask-based greedy set cover
    with local search refinement.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subsets)
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subsets)}
    
    # Precompute all k-blocks and their coverage masks
    # v=19, k=8 => 75582 blocks. n_t=11628 bits.
    # Python ints handle this efficiently.
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    block_masks = [0] * n_blocks
    for bi, b in enumerate(all_blocks):
        mask = 0
        for sub in itertools.combinations(b, t):
            idx = t_sub_to_idx[sub]
            mask |= (1 << idx)
        block_masks[bi] = mask
        
    # Full mask of all t-subsets uncovered
    full_mask = (1 << n_t) - 1
    
    def greedy_cover(noise_seed=None, alpha=0.0):
        """
        Greedy set cover with optional random noise.
        alpha: noise magnitude (0 = pure greedy, >0 = stochastic)
        Returns list of block indices or None if timeout.
        """
        if noise_seed is not None:
            rng = random.Random(noise_seed)
        else:
            rng = random.Random()
            
        uncovered = full_mask
        selected_blocks = []
        
        while uncovered:
            if time.time() > deadline:
                return None
                
            best_block_idx = -1
            best_gain = -1
            best_score = -1.0
            
            # Find block with maximum coverage of uncovered t-subsets
            # Optimization: We can skip if best_gain is already max possible? 
            # Max possible is popcount(uncovered). If we hit that, we can break.
            current_uncovered_count = uncovered.bit_count()
            
            for bi in range(n_blocks):
                gain = (uncovered & block_masks[bi]).bit_count()
                if gain == 0:
                    continue
                
                # If we found a block that covers all remaining, take it
                if gain == current_uncovered_count:
                    best_block_idx = bi
                    best_gain = gain
                    break
                
                if alpha > 0:
                    score = gain + rng.random() * alpha
                else:
                    score = float(gain)
                
                if score > best_score:
                    best_score = score
                    best_gain = gain
                    best_block_idx = bi
                    
            if best_block_idx == -1:
                # Should not happen if uncovered is not empty
                break
                
            selected_blocks.append(best_block_idx)
            uncovered &= ~block_masks[best_block_idx]
            
        return selected_blocks

    def prune_blocks(block_indices):
        """
        Remove redundant blocks.
        A block is redundant if all t-subsets it covers are covered by other blocks.
        """
        if not block_indices:
            return block_indices
            
        current = list(block_indices)
        changed = True
        
        while changed and time.time() < deadline:
            changed = False
            n = len(current)
            if n == 0:
                break
            
            # Count how many blocks cover each t-subset
            # Instead of full array, we can just check essentiality directly?
            # Efficient way: 
            # For each block i, check if there is any t-subset s such that 
            # block i covers s and no other block j covers s.
            
            # Let's use the mask intersection method.
            # A block i is essential if (block_masks[i] & ~(OR of all other masks)) != 0
            
            # Computing OR of all others is O(n) bitwise ORs. 
            # Doing this for all i is O(n^2) bitwise ops.
            # n is ~300. 300^2 = 90k ops. Bitwise OR on 11628-bit ints is fast.
            
            # Alternative:
            # 1. Compute total_union = OR of all masks
            # 2. For each i, check if (mask_i & ~(total_union ^ mask_i)) != 0? 
            #    No, XOR removes bits present in both, but if bit is in mask_i and mask_j, 
            #    total_union has it. total_union ^ mask_i clears it if mask_j has it.
            #    So if (mask_i & (total_union ^ mask_i)) == 0, then mask_i is subset of union of others.
            
            total_union = 0
            for bi in current:
                total_union |= block_masks[bi]
                
            new_current = []
            for bi in current:
                m_i = block_masks[bi]
                # Bits in m_i that are NOT in (total_union ^ m_i)
                # total_union ^ m_i contains bits covered by others but not i, 
                # AND bits covered by both? No.
                # Let A = union(all). Let B = union(all except i).
                # We want to know if m_i \ B is empty.
                # B = A ^ m_i is only true if m_i is disjoint from B? No.
                # Actually, B = A & ~m_i is wrong if m_i has unique bits.
                # Correct logic: 
                # B = OR of all masks except m_i.
                # m_i is redundant if m_i & ~B == 0.
                
                # Computing B for each i is expensive if done naively.
                # But we can approximate: 
                # If we just remove one at a time, we can recompute union.
                # Let's try a simpler pass:
                # Check if m_i is subset of (total_union ^ m_i) is FALSE.
                
                # Let's just do the O(n) OR for the rest, it's fast enough for n~300.
                union_others = 0
                for j, bj in enumerate(current):
                    if j != bi:
                        union_others |= block_masks[bj]
                        
                if (m_i & ~union_others) == 0:
                    # Redundant
                    changed = True
                else:
                    new_current.append(bi)
                    
            current = new_current
            
        return current

    def validate_blocks(block_indices):
        """Check if the set of blocks covers all t-subsets."""
        union = 0
        for bi in block_indices:
            union |= block_masks[bi]
        return union == full_mask

    def convert_to_lists(block_indices):
        """Convert indices back to lists of points."""
        return [list(all_blocks[bi]) for bi in block_indices]

    # Main search loop
    best_result = None
    best_len = float('inf')
    
    # Phase 1: Pure Greedy
    pure_result = greedy_cover(noise_seed=0, alpha=0.0)
    if pure_result is not None:
        pruned = prune_blocks(pure_result)
        if validate_blocks(pruned):
            if len(pruned) < best_len:
                best_len = len(pruned)
                best_result = pruned

    # Phase 2: Stochastic Greedy with various noise levels
    # Higher alpha allows escaping local minima in the greedy choice
    alphas = [0.1, 0.5, 1.0, 2.0, 5.0]
    seeds = range(1, 50)
    
    for alpha in alphas:
        for seed in seeds:
            if time.time() > deadline - 10:
                break
                
            result = greedy_cover(noise_seed=seed, alpha=alpha)
            if result is not None:
                pruned = prune_blocks(result)
                if validate_blocks(pruned):
                    if len(pruned) < best_len:
                        best_len = len(pruned)
                        best_result = pruned

    # Phase 3: Local Search / Perturbation on the best result
    if best_result is not None and time.time() < deadline - 5:
        current = list(best_result)
        
        # Try to improve by swapping or removing
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            n = len(current)
            
            # 1. Try removing any redundant block (re-run prune to be sure)
            pruned = prune_blocks(current)
            if len(pruned) < len(current):
                current = pruned
                improved = True
                continue
                
            # 2. Try replacing a block with another block that covers the same "critical" parts
            # This is complex. Instead, let's try a "kick" strategy:
            # Remove a random block, then re-run greedy to fill the gap.
            if len(current) > 0:
                rng = random.Random(int(time.time() * 1000) % 10000)
                # Remove 1-2 blocks
                remove_count = min(2, len(current))
                to_remove = rng.sample(range(len(current)), remove_count)
                to_remove_set = set(to_remove)
                
                # Get current coverage
                union = 0
                for bi in current:
                    if bi not in to_remove_set:
                        union |= block_masks[bi]
                
                # Uncovered parts
                new_uncovered = full_mask & ~union
                
                # Run a mini-greedy to cover new_uncovered
                # We can reuse greedy_cover but it starts from full_mask.
                # Let's write a quick inline greedy for the gap.
                gap_blocks = []
                temp_uncovered = new_uncovered
                while temp_uncovered:
                    best_bi = -1
                    best_gain = -1
                    for bi in range(n_blocks):
                        gain = (temp_uncovered & block_masks[bi]).bit_count()
                        if gain > best_gain:
                            best_gain = gain
                            best_bi = bi
                    if best_bi == -1:
                        break
                    gap_blocks.append(best_bi)
                    temp_uncovered &= ~block_masks[best_bi]
                    
                candidate = [bi for i, bi in enumerate(current) if i not in to_remove_set] + gap_blocks
                # Prune candidate
                candidate = prune_blocks(candidate)
                
                if validate_blocks(candidate) and len(candidate) < len(current):
                    current = candidate
                    improved = True
                    best_len = len(current)
                    best_result = current
                    continue
                    
                # If not improved, don't keep the change.
                
    if best_result is not None:
        return convert_to_lists(best_result)
        
    # Fallback: Return pure greedy result if available, else empty
    if pure_result is not None:
        return convert_to_lists(pure_result)
        
    return []
