import itertools
import time
import random
import numpy as np

# Precomputation for v=24, t=3
V = 24
T = 3
K = 7

def _precompute():
    global triple_to_idx, idx_to_triple, all_triples
    all_triples = list(itertools.combinations(range(V), T))
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    idx_to_triple = all_triples

_precompute()

def _greedy_init(v, k, t, deadline, rng):
    """
    Quick greedy to get a valid starting covering.
    We want this to be fast so we have time for local search.
    """
    start_time = time.time()
    uncovered = [True] * len(idx_to_triple)
    blocks = []
    
    # Precompute triples for each potential block addition? 
    # For speed, we just do standard greedy but stop early if we hit deadline.
    while True:
        if time.time() > deadline:
            break
        
        # Find an uncovered triple
        uncovered_idx = -1
        for i in range(len(uncovered)):
            if uncovered[i]:
                uncovered_idx = i
                break
        
        if uncovered_idx == -1:
            break
        
        seed_triple = idx_to_triple[uncovered_idx]
        block = list(seed_triple)
        
        # Greedily add points to block
        while len(block) < k:
            best_x = None
            best_gain = -1
            
            # Candidates are all points not in block
            for x in range(v):
                if x in block:
                    continue
                
                gain = 0
                # Triples formed by x and any 2 points in block
                for pair in itertools.combinations(block, t - 1):
                    tri = tuple(sorted((x, pair[0], pair[1])))
                    idx = triple_to_idx[tri]
                    if uncovered[idx]:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            
            if best_x is None:
                break
            block.append(best_x)
        
        block.sort()
        blocks.append(block)
        
        # Mark covered
        for tri in itertools.combinations(block, t):
            idx = triple_to_idx[tri]
            uncovered[idx] = False
            
    return blocks

def _count_coverage(blocks, v, t):
    """
    Returns cover_count array and a list of triple indices for each block.
    """
    cover_count = [0] * len(idx_to_triple)
    block_triples = []
    
    for b in blocks:
        tris = []
        for tri in itertools.combinations(b, t):
            idx = triple_to_idx[tri]
            cover_count[idx] += 1
            tris.append(idx)
        block_triples.append(tris)
        
    return cover_count, block_triples

def _local_search_sa(blocks, v, k, t, deadline):
    """
    Simulated Annealing / Local Search to minimize block count.
    Moves:
    1. Remove a block if it doesn't uncover any triple.
    2. Swap a point in a block with another point (from 0..v-1) if:
       a. The new block is valid (distinct elements).
       b. The swap doesn't uncover any triple that was previously covered ONLY by this block.
          (i.e., for every triple in the old block, if cover_count was 1, it must still be covered by the new block OR by other blocks. 
           Actually, simpler: Calculate change in coverage.
           Let old_tris = triples in old block.
           Let new_tris = triples in new block.
           For any tri in old_tris not in new_tris: if cover_count[tri] == 1, swap is INVALID (uncover).
           For any tri in new_tris not in old_tris: cover_count[tri] increases.
           For any tri in intersection: cover_count unchanged.
           If valid, we can accept the swap.
           Does this help reduce block count? Not directly.
           However, swaps can create redundancy in OTHER blocks, allowing them to be removed.
           So the primary driver is "Remove Block". Swaps are auxiliary to enable removals.
    """
    start_time = time.time()
    blocks = [list(b) for b in blocks] # Deep copy
    
    # Initial state
    cover_count, block_triples = _count_coverage(blocks, v, t)
    
    # We will try to remove blocks. 
    # To enable removal, we might need to swap points to shift coverage.
    
    # Temperature for SA (if we allow temporary increases in 'difficulty'?)
    # Actually, let's stick to strict local search first: 
    # 1. Try to remove any redundant block.
    # 2. If stuck, try a swap that doesn't break coverage, hoping it creates a new redundant block.
    
    best_blocks = blocks
    best_count = len(blocks)
    
    # Precompute list of all possible points
    all_points = list(range(v))
    
    # For speed, keep track of which blocks are active
    active = [True] * len(blocks)
    
    # We need to update cover_count efficiently.
    
    # Helper to check if a block can be removed
    def can_remove(block_idx):
        if not active[block_idx]:
            return False
        for tri_idx in block_triples[block_idx]:
            if cover_count[tri_idx] == 1:
                return False
        return True

    # Helper to remove a block
    def remove_block(block_idx):
        active[block_idx] = False
        for tri_idx in block_triples[block_idx]:
            cover_count[tri_idx] -= 1

    # Helper to perform a swap in a block
    # swap_block_idx: index of block to modify
    # old_point: point to remove
    # new_point: point to add
    def try_swap(block_idx, old_point, new_point):
        if new_point in blocks[block_idx]:
            return False # Already there, or invalid
            
        # Construct new block
        new_block = [p for p in blocks[block_idx] if p != old_point]
        new_block.append(new_point)
        new_block.sort()
        
        # Check if new block is valid (distinct, size k)
        if len(new_block) != k:
            return False
            
        # Calculate coverage impact
        old_tris_set = set(block_triples[block_idx])
        
        new_tris_list = []
        for tri in itertools.combinations(new_block, t):
            new_tris_list.append(triple_to_idx[tri])
            
        new_tris_set = set(new_tris_list)
        
        # Triples that will lose coverage (in old, not in new)
        lost = old_tris_set - new_tris_set
        # Triples that will gain coverage (in new, not in old)
        gained = new_tris_set - old_tris_set
        
        # Check if any lost triple is uniquely covered by this block
        for tri_idx in lost:
            if cover_count[tri_idx] == 1:
                return False # Invalid swap, would uncover a triple
        
        # If we reach here, the swap is valid.
        # Update state
        for tri_idx in lost:
            cover_count[tri_idx] -= 1
        for tri_idx in gained:
            cover_count[tri_idx] += 1
            
        blocks[block_idx] = new_block
        block_triples[block_idx] = new_tris_list
        
        return True

    # Main Loop
    # Strategy:
    # 1. Scan for removable blocks. Remove them.
    # 2. If no removable blocks found in a full scan, perform a random swap on a random block.
    #    This might break the "stuck" state.
    # 3. Repeat.
    
    stuck_count = 0
    
    while time.time() < deadline:
        # Phase 1: Greedy Removal
        removed_any = False
        for i in range(len(blocks)):
            if not active[i]:
                continue
            if can_remove(i):
                remove_block(i)
                removed_any = True
                # After removal, other blocks might become removable. 
                # We could continue scanning, but restarting the scan is safer for correctness of "can_remove" logic?
                # No, can_remove checks cover_count which is updated. 
                # However, removing a block might make ANOTHER block redundant.
                # So we can just continue the loop, but we might miss a block that became redundant 
                # because we passed it. 
                # To be efficient, we can just let the loop continue. 
                # If we remove block 5, block 2 (already passed) might now be removable. 
                # We will catch it in the next outer iteration.
        
        if removed_any:
            stuck_count = 0
            # Update best
            current_count = sum(active)
            if current_count < best_count:
                best_count = current_count
                best_blocks = [blocks[i] for i in range(len(blocks)) if active[i]]
                # We can compact the list to speed up future scans? 
                # Or just keep active flags. Keeping flags is easier for index consistency.
        else:
            # No block could be removed. We are stuck.
            # Try a random swap to perturb the state.
            # Pick a random active block
            active_indices = [i for i in range(len(blocks)) if active[i]]
            if not active_indices:
                break
                
            b_idx = random.choice(active_indices)
            
            # Pick a random point in the block to remove
            old_point = random.choice(blocks[b_idx])
            
            # Pick a random new point to add
            # Exclude points already in the block
            candidates = [p for p in all_points if p not in blocks[b_idx]]
            if not candidates:
                continue
            new_point = random.choice(candidates)
            
            if try_swap(b_idx, old_point, new_point):
                stuck_count += 1
                # Swaps don't reduce count directly, but might enable removal.
                # Limit swaps to avoid infinite loops of useless swaps
                if stuck_count > 100:
                    # If we've done 100 swaps and still no removal, 
                    # maybe this configuration is a local minimum.
                    # We could restart from a different greedy init, 
                    # but we don't have time for that.
                    # Let's just keep trying different swaps.
                    pass
            else:
                # Swap failed, try another
                pass

    return best_blocks

def construct(v, k, t):
    start = time.time()
    # Safety margin: 10s for final validation/cleanup? 
    # The evaluator kills at 150s. 
    # We need to return a valid list.
    deadline = start + 145.0
    
    rng = random.Random(42)
    
    # 1. Generate a valid covering using greedy
    # We give greedy a tight deadline to ensure we have time for local search
    greedy_deadline = start + 20.0 
    blocks = _greedy_init(v, k, t, greedy_deadline, rng)
    
    # If greedy didn't finish in time, it might be invalid. 
    # _greedy_init breaks on deadline. 
    # We must ensure the returned blocks form a valid covering.
    # If time ran out, the covering is likely incomplete.
    # So, we should only use the result if it's valid, or finish it.
    # Given v=24, k=7, greedy is very fast (usually < 1s).
    # So 20s is plenty.
    
    # Verify validity of greedy output (just in case)
    # This is O(N * C(7,3)) which is small.
    covered = set()
    for b in blocks:
        for tri in itertools.combinations(b, t):
            covered.add(tri)
            
    if len(covered) < len(idx_to_triple):
        # Greedy failed to cover everything in time? Unlikely.
        # Force finish greedy?
        # For now, if it's invalid, we return what we have (which scores 0) or try a simpler fallback.
        # Let's assume greedy works.
        pass

    # 2. Local Search
    # We have plenty of time left.
    ls_deadline = deadline
    optimized_blocks = _local_search_sa(blocks, v, k, t, ls_deadline)
    
    # Final validation to be safe
    # The local search maintains validity by construction (only valid swaps/removals).
    # But let's double check if we have time.
    if time.time() < deadline - 5.0:
        covered = set()
        for b in optimized_blocks:
            for tri in itertools.combinations(b, t):
                covered.add(tri)
        if len(covered) < len(idx_to_triple):
            # Fallback to greedy blocks if local search broke something (shouldn't happen)
            optimized_blocks = blocks
            
    return optimized_blocks
