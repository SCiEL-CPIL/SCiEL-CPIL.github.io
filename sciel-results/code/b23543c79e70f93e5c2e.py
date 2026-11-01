import itertools
import random
import time
import sys

def construct(v, k, t):
    # Map t-subsets to IDs for efficient counting
    t_subs = list(itertools.combinations(range(v), t))
    sub_to_id = {sub: i for i, sub in enumerate(t_subs)}
    num_subs = len(t_subs)
    
    def get_sub_ids(block):
        """Get IDs of all t-subsets contained in block."""
        return [sub_to_id[sub] for sub in itertools.combinations(block, t)]

    def randomized_greedy_cover(v, k, t, sub_to_id, t_subs, num_subs, seed):
        random.seed(seed)
        uncovered = set(range(num_subs)) # Set of uncovered subset IDs
        blocks = []
        
        while uncovered:
            # Pick a random uncovered subset to start the block
            start_sub_id = random.choice(list(uncovered))
            start_sub = t_subs[start_sub_id]
            block = list(start_sub)
            
            while len(block) < k:
                # Calculate gain for each candidate point
                gains = []
                for x in range(v):
                    if x in block:
                        continue
                    # Count how many uncovered subsets would be covered
                    # A subset is covered if it contains all of block + x
                    # It must be a t-subset of (block + [x])
                    # So we look for uncovered subsets S such that block U {x} subseteq S
                    # This is equivalent to S containing block and x.
                    # Since S has size t, and block has size b, x is new.
                    # S must contain b+1 specific elements.
                    # The number of such S is C(v - (b+1), t - (b+1)).
                    # We need to check which of these are in `uncovered`.
                    
                    # Optimization: 
                    # Instead of iterating all uncovered, we can precompute?
                    # No, dynamic.
                    # Let's just iterate candidates for x, and for each, check subsets.
                    # To speed up, we can precompute for each block prefix?
                    
                    # Simple approach:
                    # For a candidate x, the subsets covered are those in uncovered 
                    # that contain block and x.
                    # We can iterate through combinations of (range(v) - block - {x}) of size (t - len(block) - 1)
                    # and form the subset.
                    
                    remaining_points = [p for p in range(v) if p not in block and p != x]
                    need = t - len(block) - 1
                    count = 0
                    if need >= 0:
                        for rest in itertools.combinations(remaining_points, need):
                            s = tuple(sorted(tuple(block) + (x,) + rest))
                            sid = sub_to_id.get(s)
                            if sid is not None and sid in uncovered:
                                count += 1
                    gains.append((count, x))
                
                if not gains or max(g[0] for g in gains) == 0:
                    # No point covers any new subset? Should not happen if uncovered is not empty
                    # and we started with an uncovered subset.
                    # But if k is small? 
                    # Actually, if we are stuck, just pick random.
                    candidates = [x for x in range(v) if x not in block]
                    if candidates:
                        block.append(random.choice(candidates))
                    else:
                        break
                else:
                    # Select from top candidates
                    gains.sort(reverse=True)
                    top_k = min(5, len(gains))
                    # Weighted random selection or just random from top?
                    # Random from top K is a good heuristic (GRASP)
                    best_gain = gains[0][0]
                    # Filter to those with gain close to best? Or just top K?
                    # Let's take top K.
                    candidates = [g[1] for g in gains[:top_k]]
                    block.append(random.choice(candidates))
            
            block.sort()
            blocks.append(block)
            
            # Update uncovered
            for sid in get_sub_ids(block):
                uncovered.discard(sid)
        
        return blocks

    def local_search(blocks, v, k, t, sub_to_id, t_subs, num_subs, max_iters=500):
        # Initialize cover counts
        cover_count = [0] * num_subs
        block_sub_ids = []
        for b in blocks:
            ids = get_sub_ids(b)
            block_sub_ids.append(ids)
            for sid in ids:
                cover_count[sid] += 1
        
        current_blocks = list(blocks)
        current_sub_ids = list(block_sub_ids)
        
        for _ in range(max_iters):
            # Check for redundant blocks
            removed_any = False
            for i in range(len(current_blocks)):
                # Check if block i is redundant
                # i.e., all its subsets are covered by others
                redundant = True
                for sid in current_sub_ids[i]:
                    if cover_count[sid] == 1:
                        redundant = False
                        break
                if redundant:
                    # Remove block i
                    for sid in current_sub_ids[i]:
                        cover_count[sid] -= 1
                    current_blocks.pop(i)
                    current_sub_ids.pop(i)
                    removed_any = True
                    break # Restart loop as indices changed
            
            if removed_any:
                continue
            
            # If no block is redundant, try to replace one to enable future removals
            # Find block with minimum unique coverage
            min_unique = float('inf')
            target_idx = -1
            for i in range(len(current_blocks)):
                unique_count = 0
                for sid in current_sub_ids[i]:
                    if cover_count[sid] == 1:
                        unique_count += 1
                if unique_count < min_unique:
                    min_unique = unique_count
                    target_idx = i
            
            if target_idx == -1:
                break
            
            # Try to replace current_blocks[target_idx]
            old_block = current_blocks[target_idx]
            old_ids = current_sub_ids[target_idx]
            
            # Identify critical subsets (only covered by old_block)
            critical_ids = [sid for sid in old_ids if cover_count[sid] == 1]
            
            if not critical_ids:
                # Should have been caught as redundant
                continue
            
            # Try to construct a new block that covers all critical_ids
            # We can use a greedy builder constrained to cover critical_ids
            # Start with a point from the first critical subset
            start_sub = t_subs[critical_ids[0]]
            new_block = list(start_sub)
            
            success = False
            # Greedily add points to cover remaining critical subsets
            remaining_critical = set(critical_ids)
            # Remove those covered by initial points? 
            # Actually, we just need the final block to cover them.
            
            while len(new_block) < k:
                best_gain = -1
                best_point = -1
                for x in range(v):
                    if x in new_block:
                        continue
                    # Count how many remaining_critical subsets are covered by adding x
                    # A critical subset S is covered if new_block U {x} subseteq S
                    gain = 0
                    for sid in remaining_critical:
                        s = t_subs[sid]
                        # Check if new_block + [x] is subset of s
                        # Since sizes are small, we can just check
                        # But this is O(|critical| * k * v). 
                        # |critical| can be large.
                        # Optimization: 
                        # For each x, we can check which critical subsets contain new_block + [x].
                        # This is still heavy.
                        
                        # Let's approximate: just pick a point that is in many critical subsets.
                        # Frequency of points in critical subsets.
                        pass
                
                # Better heuristic for replacement:
                # We want a block B' such that critical_ids subseteq get_sub_ids(B').
                # This is a hitting set problem.
                # Simple greedy:
                # 1. Pick a point x that appears in the most critical subsets.
                # 2. Add to new_block.
                # 3. Remove critical subsets that contain x? No, they must contain ALL of new_block.
                
                # Let's refine the greedy for replacement:
                # State: current partial block P.
                # Goal: Cover all critical subsets C.
                # A subset c in C is "satisfied" if P subseteq c.
                # We want to find x such that P U {x} satisfies many c.
                
                satisfied = set()
                for sid in remaining_critical:
                    s = t_subs[sid]
                    if all(p in s for p in new_block):
                        satisfied.add(sid)
                
                # Remove satisfied from remaining?
                # Actually, we just need to ensure final block covers them.
                # If P subseteq c, then any superset of P will also be subseteq c? No.
                # If P subseteq c, then adding x keeps P U {x} subseteq c only if x in c.
                
                # So, for a candidate x, it "preserves" the possibility of covering c
                # if x is in c.
                # It "satisfies" c immediately if P U {x} subseteq c.
                
                # Let's just score x by how many critical subsets contain x AND are consistent with P.
                # Consistent means: P subseteq c.
                # If P is not subseteq c, we've already failed to cover c with this branch.
                
                # So, first filter critical subsets to those that contain P.
                valid_critical = [sid for sid in remaining_critical if all(p in t_subs[sid] for p in new_block)]
                
                if not valid_critical:
                    # Dead end for this start
                    break
                
                # Score x by number of valid_critical subsets that contain x
                counts = {}
                for sid in valid_critical:
                    s = t_subs[sid]
                    for p in s:
                        if p not in new_block:
                            counts[p] = counts.get(p, 0) + 1
                
                if not counts:
                    break
                    
                best_x = max(counts, key=counts.get)
                new_block.append(best_x)
                new_block.sort()
                
                # Check if we covered all critical
                # If all critical subsets are contained in new_block, we succeeded
                # But we need to finish the block to size k.
                # If we've covered all critical, we can fill the rest randomly.
                
                # Check coverage of critical by new_block
                all_covered = True
                for sid in critical_ids:
                    s = t_subs[sid]
                    if not all(p in s for p in new_block):
                        all_covered = False
                        break
                
                if all_covered:
                    # Fill rest of block
                    while len(new_block) < k:
                        candidates = [x for x in range(v) if x not in new_block]
                        if not candidates:
                            break
                        new_block.append(random.choice(candidates))
                    new_block.sort()
                    success = True
                    break
            
            if success:
                # Replace old block with new_block
                # Update cover counts
                for sid in old_ids:
                    cover_count[sid] -= 1
                new_ids = get_sub_ids(new_block)
                for sid in new_ids:
                    cover_count[sid] += 1
                
                current_blocks[target_idx] = new_block
                current_sub_ids[target_idx] = new_ids
            else:
                # If we can't replace, try a different target or break?
                # To avoid infinite loop, we can randomize the target selection
                # or just break if we stuck.
                # Let's try a different random block if this one fails.
                if random.random() < 0.5:
                    continue
                else:
                    break

        return current_blocks

    # Run multiple iterations to find best
    best_blocks = None
    best_len = float('inf')
    
    start_time = time.time()
    timeout = 110.0 # Leave 10s buffer
    
    seed = 42
    while time.time() - start_time < timeout:
        blocks = randomized_greedy_cover(v, k, t, sub_to_id, t_subs, num_subs, seed)
        # Local search
        optimized = local_search(blocks, v, k, t, sub_to_id, t_subs, num_subs, max_iters=100)
        
        if len(optimized) < best_len:
            best_len = len(optimized)
            best_blocks = optimized
        
        seed += 1
        
        # If we reached a very low number, we might stop early?
        # Lower bound is 264. If we get close, keep going.
        
    return best_blocks

# EVOLVE-BLOCK-END
