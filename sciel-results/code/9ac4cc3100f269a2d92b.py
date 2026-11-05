import itertools
import random
import time
import heapq
import math

def construct(v, k, t):
    # Parameters for the specific task are v=24, k=7, t=3
    # But we keep it general for robustness if possible, though optimized for this case.
    
    all_subs = list(itertools.combinations(range(v), t))
    sub_to_idx = {s: i for i, s in enumerate(all_subs)}
    n_subs = len(all_subs)
    
    def get_block_triple_indices(block):
        return [sub_to_idx[tr] for tr in itertools.combinations(block, t)]

    def greedy_construct(seed):
        random.seed(seed)
        uncovered = set(range(n_subs))
        blocks = []
        
        while uncovered:
            # Pick a random uncovered triple
            triple_idx = random.choice(list(uncovered))
            triple = all_subs[triple_idx]
            
            # Build a block containing this triple
            block = list(triple)
            
            # Greedily add points to maximize coverage of uncovered triples
            # Use top-K sampling to diversify
            top_k = 5
            while len(block) < k:
                gains = []
                for x in range(v):
                    if x in block:
                        continue
                    gain = 0
                    for rest in itertools.combinations(block, t-1):
                        new_triple = tuple(sorted(rest + (x,)))
                        idx = sub_to_idx[new_triple]
                        if idx in uncovered:
                            gain += 1
                    gains.append((gain, x))
                
                # Sort by gain descending
                gains.sort(reverse=True)
                
                # Pick from top K
                if len(gains) > top_k:
                    candidates = gains[:top_k]
                else:
                    candidates = gains
                    
                # Weighted random choice or just random from top
                # Let's pick the best one, but if there are ties, pick randomly
                best_gain = candidates[0][0]
                best_candidates = [x for g, x in candidates if g == best_gain]
                best_x = random.choice(best_candidates)
                
                block.append(best_x)
            
            block.sort()
            blocks.append(block)
            
            # Update uncovered
            for idx in get_block_triple_indices(block):
                uncovered.discard(idx)
                
        return blocks

    def local_search(initial_blocks, time_limit):
        current_blocks = initial_blocks[:]
        current_count = len(current_blocks)
        
        # Maintain coverage counts
        cover_counts = [0] * n_subs
        for b in current_blocks:
            for idx in get_block_triple_indices(b):
                cover_counts[idx] += 1
        
        # Verify validity
        if any(c == 0 for c in cover_counts):
            return None, float('inf')
            
        best_blocks = current_blocks[:]
        best_count = current_count
        
        start_time = time.time()
        
        # Local Search Loop
        while time.time() - start_time < time_limit:
            improvement_made = False
            
            # --- Move 1: Remove Redundant Blocks ---
            # A block is redundant if all its triples are covered by >= 2 blocks.
            # We scan and remove one if found.
            for i in range(len(current_blocks)):
                b = current_blocks[i]
                redundant = True
                for idx in get_block_triple_indices(b):
                    if cover_counts[idx] == 1:
                        redundant = False
                        break
                if redundant:
                    for idx in get_block_triple_indices(b):
                        cover_counts[idx] -= 1
                    current_blocks.pop(i)
                    current_count -= 1
                    improvement_made = True
                    if current_count < best_count:
                        best_count = current_count
                        best_blocks = current_blocks[:]
                    break # Restart scan after removal
            
            if improvement_made:
                continue # Loop again to check for more removals
            
            # --- Move 2: Swap Block to Improve Redundancy ---
            # If no redundant block, try to swap a block to make others redundant.
            if not current_blocks:
                break
            
            # Pick a random block to replace
            i = random.randint(0, len(current_blocks) - 1)
            b_old = current_blocks[i]
            
            # Identify critical triples in b_old (count == 1)
            critical_indices = []
            for idx in get_block_triple_indices(b_old):
                if cover_counts[idx] == 1:
                    critical_indices.append(idx)
            
            if not critical_indices:
                # This block is redundant, but Move 1 didn't catch it?
                # Move 1 checks all. If we are here, it means no block was redundant.
                # So critical_indices cannot be empty if no block was redundant?
                # Wait. If no block is redundant, then EVERY block has at least one critical triple.
                # So critical_indices should not be empty.
                # If it is empty, it's redundant. Remove it.
                for idx in get_block_triple_indices(b_old):
                    cover_counts[idx] -= 1
                current_blocks.pop(i)
                current_count -= 1
                if current_count < best_count:
                    best_count = current_count
                    best_blocks = current_blocks[:]
                continue
            
            # We need a new block b_new containing the union of points in critical_indices.
            U = set()
            for idx in critical_indices:
                U.update(all_subs[idx])
                
            if len(U) > k:
                continue # Cannot cover all criticals with one block
            
            available_points = [p for p in range(v) if p not in U]
            remaining_slots = k - len(U)
            
            # Find best swap
            # Score for a candidate block B_new:
            # We remove b_old. The criticals are now uncovered.
            # B_new covers them.
            # For other triples in B_new, we want them to be covered by *other* blocks (in current_blocks \ {b_old}).
            # This increases the redundancy of the system.
            
            # Pre-calculate cover counts without b_old for evaluation?
            # cover_counts_no_old = cover_counts - cover(b_old)
            # But we only care about triples in B_new.
            
            best_score = -1
            best_b_new = None
            
            # Sample candidates
            # If remaining_slots is small, we can enumerate.
            # C(24-|U|, k-|U|).
            # |U| is at least t=3. k=7. So remaining_slots <= 4.
            # Available points = 24 - |U|.
            # If |U|=3, slots=4, avail=21. C(21,4) = 5985. This is small enough to enumerate!
            
            if remaining_slots <= 5: # Enumerate all completions
                candidates = itertools.combinations(available_points, remaining_slots)
            else:
                # Random sampling
                candidates = [tuple(random.sample(available_points, remaining_slots)) for _ in range(100)]
            
            for extra in candidates:
                b_new = sorted(list(U) + list(extra))
                
                # Check validity: distinct, size k (guaranteed)
                # Calculate score:
                # Number of triples in b_new (excluding criticals) that are covered by other blocks.
                # "Other blocks" means cover_counts[idx] - 1 (since b_old is removed) > 0.
                # i.e., cover_counts[idx] > 1 in the current state.
                
                score = 0
                for idx in get_block_triple_indices(b_new):
                    # If this triple is one of the criticals, it's "forced". 
                    # We don't count it for redundancy gain because it *was* covered by b_old and now by b_new.
                    # But wait, if it was covered by b_old AND another block, it wasn't critical.
                    # Criticals are count==1. So for criticals, cover_counts[idx] is 1.
                    # After removing b_old, count becomes 0.
                    # b_new adds 1. So it's covered.
                    # For non-criticals in b_new:
                    # If cover_counts[idx] > 1, then it's covered by at least one other block.
                    # This is good.
                    if idx not in critical_indices:
                        if cover_counts[idx] > 1:
                            score += 1
                    else:
                        # Criticals are always "saved" by b_new.
                        # But do we get a bonus? No, they are mandatory.
                        pass
                        
                # We also want to minimize the number of "new" uncovered triples? 
                # No, b_new covers the criticals. 
                # What about triples in b_old that are NOT in b_new?
                # If they were critical (count 1), they are now uncovered -> INVALID.
                # But we ensured b_new contains ALL criticals.
                # So all criticals are covered.
                # Non-criticals in b_old (count > 1) become count - 1 (still >= 1). So they remain covered.
                # So the swap is always VALID if b_new contains all criticals.
                
                if score > best_score:
                    best_score = score
                    best_b_new = b_new
            
            if best_b_new is not None:
                # Perform swap
                for idx in get_block_triple_indices(b_old):
                    cover_counts[idx] -= 1
                for idx in get_block_triple_indices(best_b_new):
                    cover_counts[idx] += 1
                current_blocks[i] = best_b_new
                
                # Check if this swap enabled a removal? 
                # We don't know yet. The loop will check for removals in the next iteration.
                # But we can do a quick check: is best_b_new redundant?
                # Or did we make another block redundant?
                # The loop structure handles this by restarting the scan.
                
            # Cooling / Randomization
            # If we are stuck, we might need to accept a "worse" swap?
            # For now, we only accept if it's valid. 
            # To escape local minima, we can randomly replace a block with a random valid block 
            # that covers its criticals, even if the score is lower.
            # Let's add a small probability of random swap.
            if random.random() < 0.05:
                 # Random swap
                 i = random.randint(0, len(current_blocks) - 1)
                 b_old = current_blocks[i]
                 critical_indices = [idx for idx in get_block_triple_indices(b_old) if cover_counts[idx] == 1]
                 if critical_indices:
                     U = set()
                     for idx in critical_indices:
                         U.update(all_subs[idx])
                     if len(U) <= k:
                         available_points = [p for p in range(v) if p not in U]
                         remaining_slots = k - len(U)
                         if remaining_slots <= len(available_points):
                             extra = random.sample(available_points, remaining_slots)
                             b_new = sorted(list(U) + extra)
                             # Validity check is implicit if U contains criticals
                             for idx in get_block_triple_indices(b_old):
                                 cover_counts[idx] -= 1
                             for idx in get_block_triple_indices(b_new):
                                 cover_counts[idx] += 1
                             current_blocks[i] = b_new
                             
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
        ls_blocks, ls_count = local_search(blocks, time_limit=total_time_limit - (time.time() - start_time))
        
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
