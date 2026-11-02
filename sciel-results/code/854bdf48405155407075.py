import time
import random
import itertools
import numpy as np

def construct(v, k, t):
    t0 = time.time()
    deadline = t0 + 145.0  # Safety margin for 150s limit
    
    # Precompute all t-subsets
    all_t = list(itertools.combinations(range(v), t))
    N = len(all_t)
    idx = {s: i for i, s in enumerate(all_t)}
    
    # Precompute all k-subsets (blocks)
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks_total = len(all_blocks)
    
    # For each block, precompute the list of t-subset indices it covers
    block_coverages = []
    for b in all_blocks:
        covers = []
        for s in itertools.combinations(b, t):
            covers.append(idx[s])
        block_coverages.append(np.array(covers, dtype=np.int32))
        
    # Precompute for each t-subset, the list of blocks that cover it.
    t_sub_to_blocks = [[] for _ in range(N)]
    for bi, cov in enumerate(block_coverages):
        for ti in cov:
            t_sub_to_blocks[ti].append(bi)

    def get_initial_solution():
        """
        Fast greedy initialization to get a valid (though likely suboptimal) covering.
        """
        uncovered_mask = np.ones(N, dtype=bool)
        selected_blocks = []
        
        # Greedy set cover
        while np.any(uncovered_mask):
            best_block_idx = -1
            best_score = -1
            
            # To speed up, we can sample or use a heuristic.
            # For initialization, let's do a full scan but optimized.
            # Actually, full scan of 75k blocks * 56 checks is ~4M ops.
            # Doing this ~300 times (depth) is 1.2B ops. Might be slow in Python.
            # Let's use the "pick a random uncovered, then scan its blocks" heuristic.
            
            uncovered_indices = np.where(uncovered_mask)[0]
            if len(uncovered_indices) == 0:
                break
                
            # Pick a random uncovered subset to anchor the search
            target_ti = random.choice(uncovered_indices)
            candidate_blocks = t_sub_to_blocks[target_ti]
            
            if not candidate_blocks:
                # Should not happen
                break
            
            # Find the best block among candidates
            for bi in candidate_blocks:
                if bi in selected_blocks:
                    continue
                score = np.sum(uncovered_mask[block_coverages[bi]])
                if score > best_score:
                    best_score = score
                    best_block_idx = bi
            
            if best_block_idx == -1:
                # Fallback: pick any unused block
                for bi in candidate_blocks:
                    if bi not in selected_blocks:
                        best_block_idx = bi
                        break
            
            if best_block_idx == -1:
                # No valid block? This implies infeasibility or logic error.
                # For C(19,8,5), it's always solvable.
                break
            
            selected_blocks.append(best_block_idx)
            uncovered_mask[block_coverages[best_block_idx]] = False
            
        return selected_blocks

    def calculate_cover_counts(blocks, N):
        """
        Count how many times each t-subset is covered by the given blocks.
        """
        counts = np.zeros(N, dtype=np.int32)
        for bi in blocks:
            cov = block_coverages[bi]
            # Increment counts for all t-subsets in this block
            # Using numpy fancy indexing increment is tricky for duplicates within same block?
            # No, a block has distinct t-subsets.
            # But multiple blocks can cover the same t-subset.
            # We just need the total count.
            for ti in cov:
                counts[ti] += 1
        return counts

    def is_redundant(block_idx, blocks, N):
        """
        Check if removing block_idx leaves all t-subsets covered by the remaining blocks.
        This is equivalent to checking if block_idx covers any t-subset that is covered
        by NO other block in 'blocks'.
        
        To do this efficiently:
        1. Compute cover counts for 'blocks'.
        2. For each t-subset covered by block_idx, check if count[ti] > 1.
        3. If all such ti have count[ti] > 1, then block_idx is redundant.
        
        However, computing full counts for every check is O(N * |blocks|).
        We can optimize by maintaining counts incrementally, but that's complex.
        
        Alternative:
        Just check the specific t-subsets covered by block_idx.
        For each ti in block_coverages[block_idx]:
            Check if there exists another block in blocks (excluding block_idx) that covers ti.
            This is O(56 * |blocks| * (cost to check)).
            Cost to check: scanning t_sub_to_blocks[ti] for a match in blocks?
            Or just iterating blocks?
            
            Let's use the count array approach but recompute it only when needed?
            No, let's just iterate.
            For a block b, it covers 56 t-subsets.
            For each of these 56, we need to know if it's covered by another block.
            We can precompute a set of blocks for each ti? No, t_sub_to_blocks[ti] is a list.
            
            Let's try a simpler check:
            Remove b from blocks.
            For each ti in block_coverages[b]:
                Is there any other block in blocks that covers ti?
                We can check this by:
                for other_b in blocks:
                    if other_b == b: continue
                    if ti in block_coverages[other_b]: # This is O(56) or O(1) if set
                        break
                If no other_b found, b is not redundant.
                
            To make this fast, we can represent block_coverages as sets?
            Or just use the fact that 56 is small.
            
            Let's optimize by precomputing a set of t-subsets for each block?
            block_t_sets = [set(cov) for cov in block_coverages]
            
            Then:
            def is_redundant(block_idx, blocks):
                for ti in block_coverages[block_idx]:
                    covered = False
                    for other_b in blocks:
                        if other_b == block_idx:
                            continue
                        if ti in block_t_sets[other_b]:
                            covered = True
                            break
                    if not covered:
                        return False
                return True
            
            Complexity: 56 * |blocks| * (set lookup).
            If |blocks| ~ 300, this is ~16k set lookups. Very fast.
        """
        for ti in block_coverages[block_idx]:
            covered = False
            for other_b in blocks:
                if other_b == block_idx:
                    continue
                if ti in block_t_sets[other_b]:
                    covered = True
                    break
            if not covered:
                return False
        return True

    # Precompute sets for faster redundancy checks
    block_t_sets = [set(cov) for cov in block_coverages]

    def remove_redundant(blocks):
        """
        Iteratively remove redundant blocks.
        """
        changed = True
        while changed:
            changed = False
            for i in range(len(blocks) - 1, -1, -1):
                if is_redundant(blocks[i], blocks):
                    blocks.pop(i)
                    changed = True
        return blocks

    def local_search(blocks, deadline):
        """
        Perform local search:
        1. Try to remove redundant blocks.
        2. Try to swap one block with another to allow further removal?
           Or simply:
           - Remove all redundant blocks.
           - Try to replace a block with a "better" one that allows more redundancy removal?
           
           A common heuristic:
           - Pick a random block.
           - Try to remove it.
           - If it's redundant, remove it.
           - If not, try to find a block that, when swapped, reduces the count?
           
           Let's stick to:
           1. Remove redundant blocks.
           2. Randomly perturb: remove a non-redundant block, see if we can cover the gaps
              by adding fewer blocks? This is complex.
              
           Simpler perturbation:
           - Remove a random block.
           - Run greedy to cover the newly uncovered t-subsets.
           - If the total count decreases, accept.
           
           Let's implement "Remove and Repair".
        """
        blocks = list(blocks)
        remove_redundant(blocks)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Remove redundant blocks again to be safe
            remove_redundant(blocks)
            
            # Try to remove a random block and repair
            if not blocks:
                break
                
            # Pick a random block to try removing
            # Heuristic: Pick a block that covers many t-subsets (likely to be hard to remove)
            # Or pick one that covers few?
            # Let's pick a random one.
            idx_to_remove = random.randint(0, len(blocks) - 1)
            block_to_remove = blocks[idx_to_remove]
            
            # Tentatively remove it
            remaining_blocks = [b for i, b in enumerate(blocks) if i != idx_to_remove]
            
            # Find uncovered t-subsets
            # We need to know which t-subsets are now uncovered.
            # We can compute coverage of remaining_blocks.
            # But we want to add minimal blocks to cover them.
            
            # Compute uncovered mask
            # This is expensive if done from scratch.
            # Let's maintain a coverage count?
            # For now, let's just check.
            
            # Count coverage of remaining blocks
            # We can use the precomputed block_t_sets.
            # But we need to find which t-subsets are NOT covered by any block in remaining_blocks.
            
            # Optimization:
            # Only t-subsets covered by block_to_remove are candidates for being uncovered.
            # Let's check those.
            candidates = block_coverages[block_to_remove]
            uncovered = []
            for ti in candidates:
                is_covered = False
                for b in remaining_blocks:
                    if ti in block_t_sets[b]:
                        is_covered = True
                        break
                if not is_covered:
                    uncovered.append(ti)
            
            if not uncovered:
                # Block was redundant (should have been caught by remove_redundant, but maybe not if we didn't check all)
                blocks = remaining_blocks
                improved = True
                continue
            
            # We need to cover 'uncovered' t-subsets.
            # Use greedy to add blocks.
            new_blocks = list(remaining_blocks)
            uncovered_set = set(uncovered)
            
            while uncovered_set:
                # Pick a random uncovered t-subset
                target_ti = random.choice(list(uncovered_set))
                
                # Find best block to cover it
                best_b = -1
                best_score = -1
                for b in t_sub_to_blocks[target_ti]:
                    if b in new_blocks:
                        continue
                    # Score: number of uncovered t-subsets it covers
                    score = 0
                    for ti in block_coverages[b]:
                        if ti in uncovered_set:
                            score += 1
                    if score > best_score:
                        best_score = score
                        best_b = b
                
                if best_b == -1:
                    break # Should not happen
                
                new_blocks.append(best_b)
                # Update uncovered_set
                for ti in block_coverages[best_b]:
                    uncovered_set.discard(ti)
                
                if time.time() > deadline:
                    return blocks # Return best so far
            
            # Check if we improved
            if len(new_blocks) < len(blocks):
                blocks = new_blocks
                improved = True
                remove_redundant(blocks)
            
            # If not improved, we just continue.
            # To avoid getting stuck, we could do a random restart or accept worse solutions (SA).
            # For now, let's just loop.
            
        return blocks

    # Main Logic
    best_blocks = None
    best_size = float('inf')
    
    # 1. Get initial solution
    if time.time() < deadline:
        init_blocks = get_initial_solution()
        best_blocks = init_blocks
        best_size = len(init_blocks)
        
        # 2. Local Search
        if time.time() < deadline:
            improved_blocks = local_search(best_blocks, deadline)
            if len(improved_blocks) < best_size:
                best_blocks = improved_blocks
                best_size = len(best_blocks)
                
    # 3. Try a few more random restarts if time permits
    restarts = 0
    while time.time() < deadline - 10:
        restarts += 1
        random.seed(restarts)
        
        init_blocks = get_initial_solution()
        if len(init_blocks) >= best_size:
            # Prune if initial is already worse?
            # But local search might improve it.
            # Let's still try local search if it's close.
            if len(init_blocks) > best_size + 5:
                continue
            
        improved_blocks = local_search(init_blocks, deadline)
        if len(improved_blocks) < best_size:
            best_blocks = improved_blocks
            best_size = len(best_blocks)

    if best_blocks is None:
        # Fallback if something went wrong
        best_blocks = []
        
    # Convert indices back to lists of points
    final_blocks = [list(all_blocks[bi]) for bi in best_blocks]
    
    return final_blocks
