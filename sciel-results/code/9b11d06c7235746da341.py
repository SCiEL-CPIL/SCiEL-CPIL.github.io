import itertools
import random
import time
import sys

def construct(v, k, t):
    # Precompute all t-subsets and their IDs
    t_subs = list(itertools.combinations(range(v), t))
    sub_to_id = {sub: i for i, sub in enumerate(t_subs)}
    num_subs = len(t_subs)
    
    # Generate all possible k-blocks
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # Map block index to list of t-subset IDs it covers
    block_to_subs = []
    # Map t-subset ID to list of block indices covering it
    sub_to_blocks = [[] for _ in range(num_subs)]
    
    for i, blk in enumerate(all_blocks):
        subs = [sub_to_id[sub] for sub in itertools.combinations(blk, t)]
        block_to_subs.append(subs)
        for sid in subs:
            sub_to_blocks[sid].append(i)

    def greedy_init():
        """Initialize with a greedy set cover."""
        uncovered = set(range(num_subs))
        selected = set()
        cover_count = [0] * num_subs
        
        while uncovered:
            target_sid = random.choice(list(uncovered))
            candidates = sub_to_blocks[target_sid]
            
            best_block = -1
            best_gain = -1
            for b_idx in candidates:
                gain = 0
                for sid in block_to_subs[b_idx]:
                    if sid in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_block = b_idx
            
            if best_block != -1:
                selected.add(best_block)
                for sid in block_to_subs[best_block]:
                    cover_count[sid] += 1
                    if cover_count[sid] == 1:
                        uncovered.discard(sid)
        
        return selected, cover_count, uncovered

    start_time = time.time()
    timeout = 110.0
    
    best_size = float('inf')
    best_selected = None
    
    num_restarts = 20
    for r in range(num_restarts):
        if time.time() - start_time > timeout:
            break
        
        random.seed(1000 + r)
        
        selected, cover_count, uncovered = greedy_init()
        current_size = len(selected)
        
        # Local Search
        while time.time() - start_time < timeout:
            if current_size < best_size:
                best_size = current_size
                best_selected = set(selected)
                
            if uncovered:
                # Repair Phase
                target_sid = random.choice(list(uncovered))
                candidates = [b for b in sub_to_blocks[target_sid] if b not in selected]
                
                if not candidates:
                    # Should not happen with full pool
                    break
                
                # Score candidates
                best_cand = -1
                best_gain = -1
                # Optimization: only check top few or random sample if candidates list is large?
                # sub_to_blocks size is ~100-200, so full scan is fine.
                for b_idx in candidates:
                    gain = 0
                    for sid in block_to_subs[b_idx]:
                        if sid in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_cand = b_idx
                
                if best_cand != -1:
                    selected.add(best_cand)
                    for sid in block_to_subs[best_cand]:
                        cover_count[sid] += 1
                        if cover_count[sid] == 1:
                            uncovered.discard(sid)
                    current_size += 1
            else:
                # Optimization Phase (uncovered is empty)
                # Try to reduce size
                if random.random() < 0.5:
                    # Try Delete
                    if selected:
                        b = random.choice(list(selected))
                        can_delete = True
                        for sid in block_to_subs[b]:
                            if cover_count[sid] == 1:
                                can_delete = False
                                break
                        
                        if can_delete:
                            selected.discard(b)
                            for sid in block_to_subs[b]:
                                cover_count[sid] -= 1
                            current_size -= 1
                else:
                    # Try Swap
                    if selected:
                        b = random.choice(list(selected))
                        
                        # Temporarily remove b
                        selected.discard(b)
                        lost_subs = []
                        for sid in block_to_subs[b]:
                            cover_count[sid] -= 1
                            if cover_count[sid] == 0:
                                lost_subs.append(sid)
                        
                        if not lost_subs:
                            # This was actually a delete
                            current_size -= 1
                        else:
                            # Need to find a block b' covering all lost_subs
                            # Intersection of sub_to_blocks for each sid in lost_subs
                            # Start with the smallest list to minimize intersection size
                            lost_subs_sorted = sorted(lost_subs, key=lambda s: len(sub_to_blocks[s]))
                            candidates = set(sub_to_blocks[lost_subs_sorted[0]])
                            for sid in lost_subs_sorted[1:]:
                                candidates.intersection_update(sub_to_blocks[sid])
                                if not candidates:
                                    break
                            
                            # Filter out already selected blocks
                            candidates.difference_update(selected)
                            
                            if candidates:
                                # Pick a random candidate from the intersection
                                b_prime = random.choice(list(candidates))
                                selected.add(b_prime)
                                for sid in block_to_subs[b_prime]:
                                    cover_count[sid] += 1
                                    # Since we just restored coverage for lost_subs,
                                    # and b_prime might cover others, we don't need to update 'uncovered' 
                                    # because it's empty. But we should ensure cover_count is correct.
                                # current_size is unchanged (removed 1, added 1)
                                
                                # After swap, try to delete redundant blocks
                                # This is crucial to actually reduce size
                                while True:
                                    if not selected:
                                        break
                                    b_del = random.choice(list(selected))
                                    can_delete = True
                                    for sid in block_to_subs[b_del]:
                                        if cover_count[sid] == 1:
                                            can_delete = False
                                            break
                                    
                                    if can_delete:
                                        selected.discard(b_del)
                                        for sid in block_to_subs[b_del]:
                                            cover_count[sid] -= 1
                                        current_size -= 1
                                    else:
                                        break
                            else:
                                # No candidate found, revert b
                                selected.add(b)
                                for sid in block_to_subs[b]:
                                    cover_count[sid] += 1

    if best_selected is None:
        # Fallback
        return [list(blk) for blk in all_blocks[:10]] # Just some blocks, will be invalid but shouldn't happen
        
    return [list(all_blocks[i]) for i in best_selected]
