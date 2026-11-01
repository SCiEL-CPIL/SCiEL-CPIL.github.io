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
    
    # Greedy initialization to get a good starting point
    def greedy_init():
        """Initialize with a greedy set cover."""
        uncovered = set(range(num_subs))
        selected = set()
        cover_count = [0] * num_subs
        
        while uncovered:
            # Pick an uncovered subset, prefer those with fewest covering blocks
            min_len = min(len(sub_to_blocks[s]) for s in list(uncovered)[:100])
            candidates_uncovered = [s for s in uncovered if len(sub_to_blocks[s]) == min_len]
            target_sid = random.choice(candidates_uncovered)
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
    
    # WalkSAT-style local search for Max-SAT
    # Variables: selected (True) or not (False) for each block
    # Clauses: each t-subset must be covered by at least one selected block
    def walk_sat_search(initial_selected, initial_cover_count, max_time=105.0):
        """
        WalkSAT-style local search.
        State: set of selected blocks, cover_count array
        Objective: minimize number of uncovered t-subsets, then minimize |selected|
        """
        selected = set(initial_selected)
        cover_count = list(initial_cover_count)
        
        # Track unsatisfied clauses (uncovered t-subsets)
        uncovered = set(sid for sid in range(num_subs) if cover_count[sid] == 0)
        
        best_size = len(selected)
        best_selected = set(selected)
        best_uncovered_count = len(uncovered)
        
        start = time.time()
        no_improve_count = 0
        max_no_improve = 5000
        
        while time.time() - start < max_time:
            # If we have a valid cover, try to reduce size
            if not uncovered:
                # Try to delete a redundant block
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
                        if len(selected) < best_size:
                            best_size = len(selected)
                            best_selected = set(selected)
                            best_uncovered_count = 0
                            no_improve_count = 0
                        continue
                    
                # Try swap: remove one block, add another that covers what was lost
                if selected and len(selected) > 1:
                    b_remove = random.choice(list(selected))
                    
                    # Calculate what would be lost
                    lost_subs = []
                    for sid in block_to_subs[b_remove]:
                        if cover_count[sid] == 1:
                            lost_subs.append(sid)
                    
                    if lost_subs:
                        # Find blocks that cover all lost_subs
                        # Use the rarest lost subset to find candidates
                        lost_sorted = sorted(lost_subs, key=lambda s: len(sub_to_blocks[s]))
                        candidates = set(sub_to_blocks[lost_sorted[0]])
                        for sid in lost_sorted[1:]:
                            candidates.intersection_update(sub_to_blocks[sid])
                            if not candidates:
                                break
                        
                        candidates.difference_update(selected)
                        
                        if candidates:
                            # Pick a candidate that maximizes additional coverage
                            b_add = random.choice(list(candidates))
                            
                            # Perform the swap
                            selected.discard(b_remove)
                            for sid in block_to_subs[b_remove]:
                                cover_count[sid] -= 1
                            
                            selected.add(b_add)
                            for sid in block_to_subs[b_add]:
                                cover_count[sid] += 1
                            
                            # Update uncovered
                            new_uncovered = set()
                            for sid in range(num_subs):
                                if cover_count[sid] == 0:
                                    new_uncovered.add(sid)
                            uncovered = new_uncovered
                            
                            # Try to delete redundant blocks after swap
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
                                else:
                                    break
                            
                            if len(selected) < best_size and not uncovered:
                                best_size = len(selected)
                                best_selected = set(selected)
                                best_uncovered_count = 0
                                no_improve_count = 0
                            elif len(selected) < best_size:
                                no_improve_count = 0
                            else:
                                no_improve_count += 1
                            continue
                
                # No improvement possible via delete/swap, try random perturbation
                if random.random() < 0.1:
                    # Randomly add or remove a block
                    if random.random() < 0.5 and selected:
                        b = random.choice(list(selected))
                        selected.discard(b)
                        for sid in block_to_subs[b]:
                            cover_count[sid] -= 1
                    else:
                        # Add a random block
                        b = random.randint(0, num_blocks - 1)
                        if b not in selected:
                            selected.add(b)
                            for sid in block_to_subs[b]:
                                cover_count[sid] += 1
                    
                    # Update uncovered
                    uncovered = set(sid for sid in range(num_subs) if cover_count[sid] == 0)
                    no_improve_count += 1
                else:
                    no_improve_count += 1
                
            else:
                # We have uncovered subsets, need to add blocks
                # Pick an uncovered subset to target
                target_sid = random.choice(list(uncovered))
                candidates = [b for b in sub_to_blocks[target_sid] if b not in selected]
                
                if not candidates:
                    # Should not happen, but break if it does
                    break
                
                # Score candidates by how many uncovered subsets they cover
                best_cand = -1
                best_gain = -1
                # For efficiency, sample if candidates list is large
                if len(candidates) > 50:
                    sample = random.sample(candidates, 50)
                else:
                    sample = candidates
                
                for b_idx in sample:
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
                    
                    if not uncovered:
                        if len(selected) < best_size:
                            best_size = len(selected)
                            best_selected = set(selected)
                            best_uncovered_count = 0
                            no_improve_count = 0
                    else:
                        no_improve_count += 1
            
            # Check for timeout or stagnation
            if no_improve_count > max_no_improve:
                # Perturb the solution
                if selected:
                    # Remove some random blocks
                    num_remove = max(1, len(selected) // 20)
                    for _ in range(num_remove):
                        if selected:
                            b = random.choice(list(selected))
                            selected.discard(b)
                            for sid in block_to_subs[b]:
                                cover_count[sid] -= 1
                    uncovered = set(sid for sid in range(num_subs) if cover_count[sid] == 0)
                no_improve_count = 0
        
        return best_selected, best_size
    
    start_time = time.time()
    timeout = 110.0
    
    best_size = float('inf')
    best_selected = None
    
    num_restarts = 15
    for r in range(num_restarts):
        if time.time() - start_time > timeout:
            break
        
        random.seed(2000 + r)
        
        selected, cover_count, uncovered = greedy_init()
        current_size = len(selected)
        
        # Run WalkSAT search
        ws_selected, ws_size = walk_sat_search(selected, cover_count, max_time=min(105.0, timeout - (time.time() - start_time)))
        
        if ws_size < best_size:
            best_size = ws_size
            best_selected = set(ws_selected)
    
    if best_selected is None:
        # Fallback: use greedy result
        selected, cover_count, uncovered = greedy_init()
        return [list(all_blocks[i]) for i in selected]
        
    return [list(all_blocks[i]) for i in best_selected]
