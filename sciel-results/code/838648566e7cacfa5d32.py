import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    points = list(range(v))
    all_triples = list(itertools.combinations(points, t))
    num_triples = len(all_triples)
    triple_to_idx = {tri: i for i, tri in enumerate(all_triples)}
    
    all_blocks = list(itertools.combinations(points, k))
    num_blocks = len(all_blocks)
    
    # Precompute coverage structure
    # block_triple_indices[b] = list of triple indices covered by block b
    block_triple_indices = [[] for _ in range(num_blocks)]
    # triple_to_blocks[idx] = list of block indices covering this triple
    triple_to_blocks = [[] for _ in range(num_triples)]
    
    # For fast bitset operations
    block_bitsets = [0] * num_blocks
    
    for i, b in enumerate(all_blocks):
        tris_in_block = []
        bs = 0
        for tri in itertools.combinations(b, t):
            idx = triple_to_idx[tri]
            tris_in_block.append(idx)
            triple_to_blocks[idx].append(i)
            bs |= (1 << idx)
        block_triple_indices[i] = tris_in_block
        block_bitsets[i] = bs

    all_triples_bitset = (1 << num_triples) - 1

    # Helper: Fast greedy set cover
    # Returns a list of block indices that cover all triples in the target bitset
    def fast_greedy_cover(target_bitset):
        selected = []
        current_uncovered = target_bitset
        while current_uncovered:
            best_gain = -1
            best_idx = -1
            for i in range(num_blocks):
                gain = (block_bitsets[i] & current_uncovered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
                    if gain == 0:
                        break # No block covers anything new? Should not happen if target is coverable
            if best_gain <= 0:
                break # Should not happen for valid inputs if we have full block list
                
            selected.append(best_idx)
            current_uncovered &= ~block_bitsets[best_idx]
            
        return selected

    # Helper: Check if a set of blocks is a valid cover
    def is_valid_cover(block_indices):
        covered = 0
        for b in block_indices:
            covered |= block_bitsets[b]
        return covered == all_triples_bitset

    # Helper: Count uncovered triples for a set of blocks
    def count_uncovered(block_indices):
        covered = 0
        for b in block_indices:
            covered |= block_bitsets[b]
        return all_triples_bitset ^ covered

    # Initial Solution: Strong Greedy
    initial_sol = fast_greedy_cover(all_triples_bitset)
    
    if not initial_sol:
        # Fallback, likely invalid but prevents crash
        return [list(b) for b in all_blocks[:100]]

    current_sol = list(initial_sol)
    best_sol = list(current_sol)
    best_size = len(current_sol)

    # Helper: Remove redundant blocks
    def try_remove_redundant(sol):
        covered_counts = [0] * num_triples
        for b in sol:
            for tri_idx in block_triple_indices[b]:
                covered_counts[tri_idx] += 1
        
        new_sol = []
        for b in sol:
            is_essential = False
            for tri_idx in block_triple_indices[b]:
                if covered_counts[tri_idx] == 1:
                    is_essential = True
                    break
            if not is_essential:
                for tri_idx in block_triple_indices[b]:
                    covered_counts[tri_idx] -= 1
            else:
                new_sol.append(b)
        return new_sol

    # Helper: Compute the bitset of triples that are uniquely covered by a given block in the solution
    # This helps identify "critical" blocks
    def get_critical_blocks(sol):
        critical = []
        # Count coverage
        counts = [0] * num_triples
        for b in sol:
            for tri_idx in block_triple_indices[b]:
                counts[tri_idx] += 1
        
        for b in sol:
            for tri_idx in block_triple_indices[b]:
                if counts[tri_idx] == 1:
                    critical.append(b)
                    break
        return critical

    # Main Search Loop: Variable Neighborhood Search (VNS) style
    # We will systematically increase the "neighborhood size" (number of blocks to remove/perturb)
    # and use a fast greedy repair.
    
    stagnation = 0
    max_stagnation = 20
    neighborhood_size = 1
    
    while time.time() < deadline - 5:
        # 1. Local Improvement: Remove redundant blocks
        current_sol = try_remove_redundant(current_sol)
        
        if len(current_sol) < best_size:
            best_sol = list(current_sol)
            best_size = len(current_sol)
            stagnation = 0
            neighborhood_size = 1 # Reset neighborhood size on improvement
        
        # 2. Check time
        if time.time() > deadline - 10:
            break
            
        # 3. Generate a perturbation (Shake) based on neighborhood_size
        # Strategy: Remove 'neighborhood_size' random blocks, then repair.
        # If we get stuck, increase neighborhood_size.
        
        if len(current_sol) <= neighborhood_size:
            # If solution is small, just restart from scratch?
            # Or just break.
            # Let's try a full restart from a new random greedy?
            # Random greedy is slow.
            # Let's just break and try a different initial?
            # For now, break.
            break
            
        # Select k blocks to remove
        # To make it interesting, remove blocks that are "least essential"?
        # Or random?
        # Random is good for diversity.
        
        removed_blocks = random.sample(current_sol, neighborhood_size)
        remaining_blocks = [b for b in current_sol if b not in removed_blocks]
        
        # Compute covered bitset of remaining blocks
        covered_remaining = 0
        for b in remaining_blocks:
            covered_remaining |= block_bitsets[b]
            
        # The triples that need to be covered are those not in covered_remaining
        needed_cover = all_triples_bitset & ~covered_remaining
        
        # Repair: Find min blocks to cover needed_cover
        repair_blocks = fast_greedy_cover(needed_cover)
        
        new_sol = remaining_blocks + repair_blocks
        
        # Remove duplicates?
        # fast_greedy_cover might pick a block already in remaining_blocks?
        # No, because fast_greedy_cover picks blocks that cover needed_cover.
        # If a block in remaining_blocks covers a triple in needed_cover,
        # that triple would be in covered_remaining, so it wouldn't be in needed_cover.
        # So repair_blocks are disjoint from remaining_blocks in terms of "essential coverage".
        # However, they might be the same block index?
        # If a block is in remaining_blocks, its bits are in covered_remaining.
        # So it covers no triple in needed_cover.
        # So fast_greedy_cover will not pick it (gain 0).
        # So new_sol should have distinct blocks.
        
        # Apply local improvement to new_sol
        new_sol = try_remove_redundant(new_sol)
        
        # Acceptance Criterion
        if len(new_sol) < len(current_sol):
            current_sol = new_sol
            stagnation = 0
            neighborhood_size = 1
        else:
            # Stagnation
            stagnation += 1
            if stagnation > max_stagnation:
                # Try a bigger shake? Or restart from scratch?
                # Let's try a bigger shake
                neighborhood_size += 1
                if neighborhood_size > 5:
                    # Give up on this path, try a completely new random initial?
                    # Generating a new random initial is expensive.
                    # Let's just break and hope best_sol is good.
                    break
                max_stagnation = 20 # Reset stagnation counter for the new neighborhood size

    # Final Polish on best_sol
    if time.time() < deadline - 2:
        best_sol = try_remove_redundant(best_sol)
        
    # Ensure validity
    if not is_valid_cover(best_sol):
        # Fallback to initial if it was valid
        if is_valid_cover(initial_sol):
            best_sol = initial_sol
        else:
            # Last resort: return something valid?
            # If initial is invalid, we have a problem.
            # But initial was constructed by greedy to cover all.
            # So it should be valid.
            best_sol = initial_sol
            
    result_blocks = [list(all_blocks[i]) for i in best_sol]
    
    return result_blocks
