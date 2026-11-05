import itertools
import random
import time
import math

def construct(v, k, t):
    # Parameters for the specific task are v=24, k=7, t=3
    # But we keep it general for robustness if possible, though optimized for this case.
    
    # Precompute triple to index mapping for fast lookup
    # We use a dictionary for O(1) lookup
    all_subs = list(itertools.combinations(range(v), t))
    sub_to_idx = {s: i for i, s in enumerate(all_subs)}
    n_subs = len(all_subs)
    
    # Helper to get triple indices for a block
    def get_block_triple_indices(block):
        # block is a tuple or list of k distinct ints
        return [sub_to_idx[tr] for tr in itertools.combinations(block, t)]

    # --- Phase 1: Greedy Construction ---
    def greedy_construct(seed):
        random.seed(seed)
        uncovered = set(range(n_subs))
        blocks = []
        
        # To speed up, we track uncovered set
        # Standard Greedy: Pick a random uncovered triple, build a block around it maximizing gain
        
        while uncovered:
            # Pick a random uncovered triple
            triple_idx = random.choice(list(uncovered))
            triple = all_subs[triple_idx]
            
            # Build a block containing this triple
            # Start with the triple
            block = list(triple)
            
            # Greedily add points to maximize coverage of uncovered triples
            while len(block) < k:
                best_x = -1
                max_gain = -1
                
                # Candidates: points not in block
                # Optimization: Precompute combinations of block of size t-1
                block_t_minus_1 = list(itertools.combinations(block, t-1))
                
                for x in range(v):
                    if x in block:
                        continue
                    
                    # Gain = number of uncovered triples formed by adding x to block
                    # Triples formed: (t-1) elements from block + x
                    gain = 0
                    for rest in block_t_minus_1:
                        # Create sorted tuple for lookup
                        new_triple = tuple(sorted(rest + (x,)))
                        idx = sub_to_idx.get(new_triple)
                        if idx is not None and idx in uncovered:
                            gain += 1
                    
                    if gain > max_gain:
                        max_gain = gain
                        best_x = x
                        
                if best_x == -1:
                    break # Should not happen
                block.append(best_x)
            
            block.sort()
            blocks.append(block)
            
            # Update uncovered
            for tr_idx in get_block_triple_indices(block):
                uncovered.discard(tr_idx)
                
        return blocks

    # --- Phase 2: Local Search (Simulated Annealing) ---
    def local_search(initial_blocks, time_limit):
        current_blocks = initial_blocks[:]
        current_count = len(current_blocks)
        
        # Maintain coverage counts
        cover_counts = [0] * n_subs
        # Also maintain a list of triple indices for each block to speed up updates
        block_triple_indices = [get_block_triple_indices(b) for b in current_blocks]
        
        for tr_indices in block_triple_indices:
            for idx in tr_indices:
                cover_counts[idx] += 1
        
        # Verify validity
        if any(c == 0 for c in cover_counts):
            return None, float('inf')
            
        best_blocks = current_blocks[:]
        best_block_triple_indices = block_triple_indices[:]
        best_count = current_count
        
        start_time = time.time()
        
        # Simulated Annealing Parameters
        T = 1.0
        T_min = 0.01
        cooling_rate = 0.99
        
        while time.time() - start_time < time_limit:
            
            # --- Move 1: Try to Remove a Redundant Block ---
            # A block is redundant if all its triples have count > 1
            removed = False
            for i in range(len(current_blocks)):
                redundant = True
                for idx in block_triple_indices[i]:
                    if cover_counts[idx] == 1:
                        redundant = False
                        break
                if redundant:
                    # Remove it
                    for idx in block_triple_indices[i]:
                        cover_counts[idx] -= 1
                    current_blocks.pop(i)
                    block_triple_indices.pop(i)
                    current_count -= 1
                    removed = True
                    break
                    
            if removed:
                if current_count < best_count:
                    best_count = current_count
                    best_blocks = current_blocks[:]
                    best_block_triple_indices = block_triple_indices[:]
                continue # Restart loop to check for more removals
            
            # --- Move 2: Try a Swap (Replace a block) ---
            if not current_blocks:
                break
                
            # Pick a random block to replace
            i = random.randint(0, len(current_blocks) - 1)
            b_old = current_blocks[i]
            old_tr_indices = block_triple_indices[i]
            
            # Identify critical triples in b_old (those with count == 1)
            critical_triples = []
            for idx in old_tr_indices:
                if cover_counts[idx] == 1:
                    critical_triples.append(all_subs[idx])
            
            # If no critical triples, b_old is redundant? 
            # Wait, if no critical triples, all triples in b_old have count > 1.
            # Then b_old is redundant. Why didn't Move 1 catch it?
            # Move 1 checks if *all* triples in b have count > 1.
            # If critical_triples is empty, then all have count > 1. So it IS redundant.
            # So if we reach here, critical_triples should not be empty.
            
            if not critical_triples:
                # This is a redundant block. Remove it.
                for idx in old_tr_indices:
                    cover_counts[idx] -= 1
                current_blocks.pop(i)
                block_triple_indices.pop(i)
                current_count -= 1
                if current_count < best_count:
                    best_count = current_count
                    best_blocks = current_blocks[:]
                    best_block_triple_indices = block_triple_indices[:]
                continue
            
            # We need to find a new block b_new such that:
            # 1. It contains all points in the critical triples? 
            #    i.e., for each tr in critical_triples, tr is a subset of b_new.
            
            # Union of points in critical triples
            U = set()
            for tr in critical_triples:
                U.update(tr)
                
            if len(U) > k:
                # Impossible to cover all critical triples with one block
                # Skip this swap
                continue
            
            # We need to pick k - len(U) additional points from the remaining v - len(U) points.
            # Heuristic: Pick points that appear in many "critical" triples of OTHER blocks.
            # This increases redundancy elsewhere.
            
            # Count how many critical triples (from other blocks) contain each point
            # Note: We only consider triples with count == 1.
            # We need to be careful not to count triples from b_old itself (though they are critical, 
            # and we are replacing b_old, so their status will change).
            
            # Let's build a score for each point x not in U
            point_scores = {p: 0 for p in range(v) if p not in U}
            
            # Iterate over all blocks except i
            for j in range(len(current_blocks)):
                if j == i:
                    continue
                for idx in block_triple_indices[j]:
                    if cover_counts[idx] == 1:
                        tr = all_subs[idx]
                        for p in tr:
                            if p in point_scores:
                                point_scores[p] += 1
            
            # Sort points by score descending
            available_points = sorted([p for p in point_scores.keys()], key=lambda x: point_scores[x], reverse=True)
            remaining_slots = k - len(U)
            
            # Sample a few candidate new blocks
            # We prioritize high-score points, but add some randomness
            found_new_block = False
            for _ in range(10): # Try 10 random completions
                if remaining_slots > len(available_points):
                    break
                # Pick top 'remaining_slots' with some noise
                # Or just random sample from top 2*remaining_slots
                top_candidates = available_points[:max(remaining_slots, 2*remaining_slots)]
                extra_points = random.sample(top_candidates, remaining_slots)
                b_new = sorted(list(U) + extra_points)
                
                # Calculate new triple indices
                new_tr_indices = get_block_triple_indices(b_new)
                
                # Simulate swap
                # Remove old
                for idx in old_tr_indices:
                    cover_counts[idx] -= 1
                # Add new
                for idx in new_tr_indices:
                    cover_counts[idx] += 1
                
                # Check if valid (should be, as we covered criticals of b_old)
                # But we might have uncovered some other triple?
                # No, we only removed b_old's contribution. The criticals of b_old are covered by b_new.
                # Other triples covered by b_old had count > 1, so count >= 1 remains.
                # So validity is preserved.
                
                # Accept the swap (SA would check energy, but here we accept if it doesn't break validity)
                # Actually, we should check if this swap enables a removal?
                # For now, accept all valid swaps to explore the space.
                
                current_blocks[i] = b_new
                block_triple_indices[i] = new_tr_indices
                found_new_block = True
                break
            
            if not found_new_block:
                # Try to add a block? 
                # If we can't swap, maybe we can add a block that covers critical triples of MULTIPLE blocks?
                # That increases count. We only do this if we expect to remove 2+ blocks later.
                # For simplicity, we skip adding blocks in this basic SA.
                pass
            
            # Cooling
            T *= cooling_rate
            if T < T_min:
                T = T_min
                
        return best_blocks, best_count

    # --- Main Execution ---
    
    start_time = time.time()
    total_time_limit = 110 # Leave 10s for safety
    
    best_blocks = None
    best_count = float('inf')
    
    # Run multiple iterations with different random seeds
    seed = 42
    while time.time() - start_time < total_time_limit:
        # Greedy
        blocks = greedy_construct(seed)
        
        # Local Search
        # Allocate remaining time to LS, but ensure we finish
        remaining = total_time_limit - (time.time() - start_time)
        if remaining < 1:
            break
            
        ls_blocks, ls_count = local_search(blocks, time_limit=remaining)
        
        if ls_blocks is not None and ls_count < best_count:
            best_count = ls_count
            best_blocks = ls_blocks[:]
            
        seed += 1
        
        # If we are getting close to the lower bound or time is running out, break
        if best_count <= 70: # Lower bound is 69
            break
            
    if best_blocks is None:
        # Fallback: return the last greedy result
        blocks = greedy_construct(seed)
        return blocks
        
    return best_blocks
