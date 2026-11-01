"""Improved covering design C(24,7,3) via greedy + deletion heuristic with time guard."""
import itertools
import time
import sys

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill

    # Precompute all t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    all_tsubs_set = set(all_tsubs)
    n_tsubs = len(all_tsubs)
    
    # Map each t-subset to an index for fast lookup
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    
    # Coverage count: how many blocks cover each t-subset
    # We'll use a list of ints for speed
    cover_count = [0] * n_tsubs
    
    def add_block(block):
        """Add a block and update coverage counts."""
        for ts in itertools.combinations(block, t):
            idx = tsub_to_idx[ts]
            cover_count[idx] += 1
    
    def remove_block(block):
        """Remove a block and update coverage counts."""
        for ts in itertools.combinations(block, t):
            idx = tsub_to_idx[ts]
            cover_count[idx] -= 1
    
    def is_uncovered(idx):
        return cover_count[idx] == 0
    
    def uncovered_count():
        """Count how many t-subsets are uncovered."""
        c = 0
        for i in range(n_tsubs):
            if cover_count[i] == 0:
                c += 1
        return c
    
    # Greedy phase: build initial covering
    blocks = []
    
    # For efficiency, track which t-subsets are uncovered
    # Start with all uncovered
    uncovered_mask = [False] * n_tsubs  # True if uncovered
    
    def update_uncovered_mask():
        for i in range(n_tsubs):
            uncovered_mask[i] = (cover_count[i] == 0)
    
    update_uncovered_mask()
    
    while True:
        if time.time() > deadline - 20:
            break
        
        # Find first uncovered t-subset
        found_idx = -1
        for i in range(n_tsubs):
            if uncovered_mask[i]:
                found_idx = i
                break
        
        if found_idx == -1:
            break  # All covered
        
        # Start a new block with the points of this uncovered t-subset
        block = list(all_tsubs[found_idx])
        
        # Greedily add points to maximize coverage of uncovered t-subsets
        while len(block) < k:
            if time.time() > deadline - 20:
                break
            
            best_point = -1
            best_gain = -1
            
            for x in range(v):
                if x in block:
                    continue
                
                # Calculate gain: number of uncovered t-subsets that would be covered
                # by adding x to the current block
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    ts = tuple(sorted(rest + (x,)))
                    if ts in tsub_to_idx:
                        idx = tsub_to_idx[ts]
                        if cover_count[idx] == 0:
                            gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
            
            if best_point == -1 or best_gain <= 0:
                # No point adds coverage; just pick an arbitrary unused point
                for x in range(v):
                    if x not in block:
                        best_point = x
                        break
                if best_point == -1:
                    break
            
            block.append(best_point)
        
        if len(block) == k:
            block.sort()
            blocks.append(block)
            add_block(block)
            update_uncovered_mask()
        
        # Check if we're running out of time
        if time.time() > deadline - 10:
            break
    
    # If we didn't finish the greedy phase due to time, ensure we have a valid covering
    # by continuing without time checks but with a hard cap
    while True:
        found_idx = -1
        for i in range(n_tsubs):
            if cover_count[i] == 0:
                found_idx = i
                break
        
        if found_idx == -1:
            break
        
        block = list(all_tsubs[found_idx])
        while len(block) < k:
            best_point = -1
            best_gain = -1
            for x in range(v):
                if x in block:
                    continue
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    ts = tuple(sorted(rest + (x,)))
                    if ts in tsub_to_idx:
                        idx = tsub_to_idx[ts]
                        if cover_count[idx] == 0:
                            gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
            if best_point == -1:
                for x in range(v):
                    if x not in block:
                        best_point = x
                        break
                if best_point == -1:
                    break
            block.append(best_point)
        
        if len(block) == k:
            block.sort()
            blocks.append(block)
            add_block(block)
        
        if time.time() > deadline - 5:
            # Force completion quickly
            while True:
                found_idx2 = -1
                for i in range(n_tsubs):
                    if cover_count[i] == 0:
                        found_idx2 = i
                        break
                if found_idx2 == -1:
                    break
                block2 = list(all_tsubs[found_idx2])
                while len(block2) < k:
                    for x in range(v):
                        if x not in block2:
                            block2.append(x)
                            break
                if len(block2) == k:
                    block2.sort()
                    blocks.append(block2)
                    add_block(block2)
            break
    
    # Deletion heuristic: try to remove redundant blocks
    if time.time() < deadline - 10:
        changed = True
        while changed and time.time() < deadline - 5:
            changed = False
            for i in range(len(blocks)):
                if time.time() > deadline - 5:
                    break
                # Check if block i is redundant
                # Temporarily remove it
                remove_block(blocks[i])
                
                # Check if all t-subsets are still covered
                redundant = True
                for j in range(n_tsubs):
                    if cover_count[j] == 0:
                        redundant = False
                        break
                
                if redundant:
                    blocks.pop(i)
                    changed = True
                    # Don't re-add; just continue
                else:
                    # Restore the block
                    add_block(blocks[i])
                
                if time.time() > deadline - 5:
                    break
    
    # Final verification and return
    # Ensure all t-subsets are covered
    # If not, add more blocks (shouldn't happen but safety)
    for i in range(n_tsubs):
        if cover_count[i] == 0:
            block = list(all_tsubs[i])
            while len(block) < k:
                for x in range(v):
                    if x not in block:
                        block.append(x)
                        break
            if len(block) == k:
                block.sort()
                # Check if this block already exists
                if block not in blocks:
                    blocks.append(block)
                    add_block(block)
    
    return blocks
