import time
import itertools
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0

    all_points = list(range(v))
    all_t_subsets = list(itertools.combinations(all_points, t))
    n_triples = len(all_t_subsets)
    triple_to_idx = {triple: i for i, triple in enumerate(all_t_subsets)}

    # Precompute all k-subsets and their coverage of t-subsets
    all_k_subsets = list(itertools.combinations(all_points, k))
    n_k_subsets = len(all_k_subsets)

    # cover_matrix[i] contains the indices of t-subsets covered by all_k_subsets[i]
    # Shape: (n_k_subsets, C(k,t))
    c_k_t = int(len(list(itertools.combinations(range(k), t))))
    cover_matrix = np.zeros((n_k_subsets, c_k_t), dtype=np.int32)
    
    # Also store the count of valid entries per row (should be c_k_t for all)
    for i, block in enumerate(all_k_subsets):
        for j, triple in enumerate(itertools.combinations(block, t)):
            cover_matrix[i, j] = triple_to_idx[triple]

    # Build a lookup from block tuple to its index in all_k_subsets
    block_to_k_idx = {block: i for i, block in enumerate(all_k_subsets)}

    def fast_greedy_cover(seed, randomize_ties=True):
        rng = random.Random(seed)
        uncovered = np.ones(n_triples, dtype=bool)
        blocks = []
        
        while uncovered.any():
            # Vectorized gain calculation
            gains = np.sum(uncovered[cover_matrix], axis=1)
            
            max_gain = np.max(gains)
            if max_gain == 0:
                break
                
            candidates = np.where(gains == max_gain)[0]
            if randomize_ties and len(candidates) > 1:
                chosen_idx = candidates[rng.randint(0, len(candidates) - 1)]
            else:
                chosen_idx = candidates[0]
            
            blocks.append(all_k_subsets[chosen_idx])
            uncovered[cover_matrix[chosen_idx]] = False
            
        return blocks

    def local_search_remove_redundant(blocks):
        if not blocks:
            return blocks
        
        def build_coverage_map(bks):
            coverage = np.zeros(n_triples, dtype=np.int32)
            for b in bks:
                idx = block_to_k_idx[b]
                coverage[cover_matrix[idx]] += 1
            return coverage
        
        coverage = build_coverage_map(blocks)
        changed = True
        
        # Iteratively remove redundant blocks
        while changed:
            changed = False
            to_remove = []
            
            for i, block in enumerate(blocks):
                idx = block_to_k_idx[block]
                # Check if this block has any unique triples
                # unique if coverage[triple] == 1
                if not np.any(coverage[cover_matrix[idx]] == 1):
                    to_remove.append(i)
            
            if to_remove:
                remove_set = set(to_remove)
                new_blocks = [b for i, b in enumerate(blocks) if i not in remove_set]
                coverage = build_coverage_map(new_blocks)
                blocks = new_blocks
                changed = True
        
        # Try swap-based local search
        # For each block, if it has unique triples, try to replace it with another block 
        # that covers those unique triples. This might enable further removals.
        
        max_swap_iterations = 5
        for _ in range(max_swap_iterations):
            if time.time() > deadline - 10:
                break
                
            coverage = build_coverage_map(blocks)
            improved = False
            
            # Identify blocks with unique triples
            blocks_with_unique = []
            for i, block in enumerate(blocks):
                idx = block_to_k_idx[block]
                unique_triples_in_block = cover_matrix[idx][coverage[cover_matrix[idx]] == 1]
                if len(unique_triples_in_block) > 0:
                    blocks_with_unique.append((i, block, unique_triples_in_block))
            
            # For each such block, try to find a replacement
            for i, block, unique_triples in blocks_with_unique:
                if time.time() > deadline - 10:
                    break
                
                # Find all candidate replacement blocks that cover all unique_triples
                # unique_triples is an array of indices into all_t_subsets
                # We need blocks j (not in current set) such that cover_matrix[j] covers all unique_triples
                
                # Optimization: Instead of checking all blocks, we can filter.
                # But for now, a simple check.
                
                # Create a set of unique triples for fast lookup? No, we need to check if unique_triples is a subset of block's triples.
                # Let's iterate through all_k_subsets? That's 346k. Too slow for each block.
                
                # Better approach: 
                # For a given set of unique triples U, a block B covers U if U is a subset of triples(B).
                # We can precompute for each triple which blocks cover it.
                # Then intersect the lists of blocks for each triple in U.
                
                # Let's build triple_to_blocks_map once outside if possible, or here if it's fast enough.
                # Actually, building it for 2024 triples * 346k blocks is heavy.
                # Let's just do a vectorized check on a subset of candidates?
                
                # Alternative: 
                # If the number of unique triples is small (usually 1 or 2), we can just scan.
                # If unique_triples length is 1:
                if len(unique_triples) == 1:
                    # Find blocks covering this triple
                    # We don't have triple_to_blocks. Let's compute it on the fly? 
                    # No, let's just skip complex swaps if it's too slow.
                    # The removal pass is the most effective.
                    pass
                else:
                    pass
                    
            # Since the swap logic is complex and potentially slow, and the removal pass is already good,
            # we might skip the swap pass to save time for more restarts, OR implement a very cheap version.
            # Let's try a cheap swap: 
            # If we have a block with 1 unique triple, can we replace it with a block that covers that triple 
            # and has a high gain (covers many other uncovered/shared triples)?
            # Actually, just replacing it with ANY block covering the unique triple doesn't reduce count.
            # It only helps if it allows REMOVAL of other blocks.
            
            # Given time constraints, let's stick to the removal pass which is robust.
            break
        
        return blocks

    def local_search_swap(blocks):
        # A simpler, faster swap heuristic:
        # 1. Compute coverage.
        # 2. Find blocks that are "critical" (have unique triples).
        # 3. For each critical block, try to replace it with a block that covers its unique triples 
        #    and is "better" (e.g., covers more shared triples that might become unique if we remove the original? No, that's circular).
        #    Actually, a standard heuristic: replace a block with another that covers the same unique triples 
        #    and has a higher "flexibility" (covers more triples that are currently covered by >=2 blocks).
        
        # This is getting complicated. Let's try a different local search:
        # "Greedy Re-insertion":
        # 1. Start with the set of blocks.
        # 2. Remove all blocks.
        # 3. Add them back one by one in an order that maximizes gain, but allow duplicates? No, blocks must be distinct.
        #    Actually, just re-run greedy on the remaining uncovered triples, but starting from the current set?
        
        # Let's try a "Perturbation" approach:
        # Randomly remove a block, then run greedy to cover the newly uncovered triples.
        # If the result is smaller, keep it.
        
        if not blocks:
            return blocks
            
        best_blocks = blocks[:]
        
        def build_coverage_map(bks):
            coverage = np.zeros(n_triples, dtype=np.int32)
            for b in bks:
                idx = block_to_k_idx[b]
                coverage[cover_matrix[idx]] += 1
            return coverage
        
        # Try perturbing each block
        for i in range(len(blocks)):
            if time.time() > deadline - 5:
                break
            
            # Remove block i
            remaining = blocks[:i] + blocks[i+1:]
            
            # Find uncovered triples
            coverage = build_coverage_map(remaining)
            uncovered_indices = np.where(coverage == 0)[0]
            
            if len(uncovered_indices) == 0:
                # We can just remove this block!
                best_blocks = remaining
                continue
            
            # Run a mini-greedy to cover these uncovered triples
            # We only care about covering these specific triples.
            # Gains should be calculated based on these uncovered triples.
            
            # Create a mask for uncovered triples
            # We can reuse the main greedy logic but restricted to these triples?
            # Or just run the main fast_greedy_cover but it will cover ALL uncovered, which is just these.
            
            # However, fast_greedy_cover starts from scratch. We want to add to 'remaining'.
            # Let's write a helper that adds blocks to cover a specific set of uncovered triples.
            
            def cover_specific_uncovered(remaining_blocks, uncovered_mask, rng):
                # remaining_blocks: list of blocks
                # uncovered_mask: boolean array of n_triples
                # Returns new list of blocks
                
                blocks = remaining_blocks[:]
                uncovered = uncovered_mask.copy()
                
                while uncovered.any():
                    # Calculate gains for all_k_subsets
                    # But we should avoid adding blocks already in 'blocks'
                    # Let's track which blocks are already used.
                    # Since n_k_subsets is 346k, tracking used blocks is easy with a set of indices.
                    
                    used_indices = set(block_to_k_idx[b] for b in blocks)
                    
                    gains = np.sum(uncovered[cover_matrix], axis=1)
                    
                    # Mask out used blocks
                    # Create a boolean mask of used blocks
                    # This is O(n_k_subsets) which is fine.
                    # But we can be smarter: just set gains for used blocks to -1 or 0?
                    # If a used block has gain 0, it won't be picked unless all gains are 0 (which means covered).
                    # But if a used block has gain > 0, we shouldn't pick it again.
                    
                    # Efficient way:
                    # gains[used_indices] = 0? No, if max_gain is 0, we break.
                    # We need to ensure we don't pick used blocks.
                    
                    # Let's just filter candidates.
                    max_gain = np.max(gains)
                    if max_gain == 0:
                        break
                    
                    candidates = np.where(gains == max_gain)[0]
                    
                    # Filter candidates to exclude used blocks
                    # This might be slow if many candidates.
                    # Alternative: set gains for used blocks to -1 before max?
                    # That requires modifying gains array.
                    
                    # Let's try a different approach:
                    # Instead of re-running full greedy, just pick the best block that covers the most uncovered.
                    # We can just pick one.
                    
                    # To avoid O(n_k_subsets) filtering every step, we can just pick the first candidate not in used_indices.
                    # But that might be bad.
                    
                    # Given time, let's just skip this complex local search and rely on the removal pass + multiple restarts.
                    return blocks
                
                return blocks
            
            # Since the specific cover is complex, let's just use the main fast_greedy_cover on the remaining uncovered?
            # No, that starts from empty.
            
            # Let's just do the removal check.
            pass
            
        return best_blocks

    # Main loop
    best_blocks = None
    best_len = float('inf')
    restart_count = 0
    
    # First, try a deterministic greedy (seed 0, no randomization)
    # Then try randomized restarts.
    
    while time.time() < deadline - 5:
        # Alternate between deterministic and random?
        # Or just random.
        
        blocks = fast_greedy_cover(restart_count, randomize_ties=True)
        
        # Local search: remove redundant
        blocks = local_search_remove_redundant(blocks)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks[:]
            # Print for debugging if needed
            # print(f"Restart {restart_count}: Found {best_len} blocks")
            
        restart_count += 1
        
        # If we have time, try a few more restarts with different randomization seeds
        # The current loop does this naturally.
        
    if best_blocks is None:
        # Fallback: just run one greedy
        best_blocks = fast_greedy_cover(0, randomize_ties=False)
        
    return [list(b) for b in best_blocks]
