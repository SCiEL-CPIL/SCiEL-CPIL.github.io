"""Simulated Annealing for covering design C(24,7,3)."""
import itertools
import random
import time
import sys
import numpy as np

def _cover_count(blocks, v, t):
    """Count how many t-sets are NOT covered by the given blocks."""
    # Use a boolean array for speed
    all_tsets = list(itertools.combinations(range(v), t))
    n_tsets = len(all_tsets)
    # Map each t-set to an index
    tset_index = {}
    for i, ts in enumerate(all_tsets):
        tset_index[ts] = i
    
    covered = np.zeros(n_tsets, dtype=np.uint8)
    for b in blocks:
        bset = set(b)
        for ts in itertools.combinations(b, t):
            if all(x in bset for x in ts):
                covered[tset_index[ts]] = 1
    
    return n_tsets - covered.sum()

def _uncovered_tsets(blocks, v, t, all_tsets, tset_index):
    """Return list of uncovered t-sets."""
    covered = set()
    for b in blocks:
        bset = set(b)
        for ts in itertools.combinations(b, t):
            if all(x in bset for x in ts):
                covered.add(ts)
    uncovered = []
    for ts in all_tsets:
        if ts not in covered:
            uncovered.append(ts)
    return uncovered

def _sa_covering(v, k, t, rng, time_limit):
    """
    Simulated annealing: start with a random covering, then iteratively
    modify blocks (add, remove, replace) to minimize uncovered t-sets.
    """
    deadline = time.time() + time_limit
    
    all_tsets = list(itertools.combinations(range(v), t))
    n_tsets = len(all_tsets)
    tset_index = {ts: i for i, ts in enumerate(all_tsets)}
    
    # Initial solution: greedy to get a reasonable starting point
    uncovered = set(all_tsets)
    blocks = []
    while uncovered and time.time() < deadline - 5.0:
        first = min(uncovered)
        block = list(first)
        while len(block) < k:
            # Pick element that covers most uncovered t-sets
            best_x = None
            best_gain = -1
            for x in range(v):
                if x in block:
                    continue
                gain = 0
                for S in itertools.combinations(block, t-1):
                    tset = tuple(sorted(S + (x,)))
                    if tset in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            if best_x is None:
                break
            block.append(best_x)
        if len(block) < k:
            break
        block.sort()
        blocks.append(tuple(block))
        covered = set(itertools.combinations(block, t))
        uncovered -= covered
    
    if not blocks:
        # Fallback: random blocks
        blocks = [tuple(sorted(rng.sample(range(v), k))) for _ in range(80)]
    
    # Convert to mutable list
    blocks = [list(b) for b in blocks]
    n_blocks = len(blocks)
    
    # Compute initial uncovered count
    cur_uncovered = _cover_count(blocks, v, t)
    cur_obj = cur_uncovered  # We'll minimize uncovered count
    
    best_blocks = [list(b) for b in blocks]
    best_uncovered = cur_uncovered
    best_obj = cur_obj
    
    # SA parameters
    T = 10.0
    T_min = 0.01
    cooling = 0.999
    max_steps = 50000
    
    step = 0
    while time.time() < deadline and T > T_min and step < max_steps:
        step += 1
        
        # Choose a random move
        move_type = rng.random()
        
        if move_type < 0.4 and n_blocks > 1:
            # Remove a block
            idx = rng.randint(0, n_blocks - 1)
            removed_block = blocks.pop(idx)
            new_uncovered = _cover_count(blocks, v, t)
            delta = new_uncovered - cur_uncovered
            new_obj = new_uncovered
            
            if delta <= 0 or rng.random() < np.exp(-delta / T):
                cur_uncovered = new_uncovered
                cur_obj = new_obj
                if cur_uncovered < best_uncovered:
                    best_uncovered = cur_uncovered
                    best_obj = cur_obj
                    best_blocks = [list(b) for b in blocks]
            else:
                blocks.append(removed_block)
        
        elif move_type < 0.7:
            # Replace a block with a new one
            idx = rng.randint(0, n_blocks - 1)
            old_block = blocks[idx]
            new_block = sorted(rng.sample(range(v), k))
            blocks[idx] = new_block
            new_uncovered = _cover_count(blocks, v, t)
            delta = new_uncovered - cur_uncovered
            new_obj = new_uncovered
            
            if delta <= 0 or rng.random() < np.exp(-delta / T):
                cur_uncovered = new_uncovered
                cur_obj = new_obj
                if cur_uncovered < best_uncovered:
                    best_uncovered = cur_uncovered
                    best_obj = cur_obj
                    best_blocks = [list(b) for b in blocks]
            else:
                blocks[idx] = old_block
        
        elif move_type < 0.9:
            # Add a new block (only if we have room and it helps)
            if n_blocks < 100:
                new_block = sorted(rng.sample(range(v), k))
                blocks.append(new_block)
                n_blocks += 1
                new_uncovered = _cover_count(blocks, v, t)
                delta = new_uncovered - cur_uncovered
                new_obj = new_uncovered
                
                if delta <= 0 or rng.random() < np.exp(-delta / T):
                    cur_uncovered = new_uncovered
                    cur_obj = new_obj
                    if cur_uncovered < best_uncovered:
                        best_uncovered = cur_uncovered
                        best_obj = cur_obj
                        best_blocks = [list(b) for b in blocks]
                else:
                    blocks.pop()
                    n_blocks -= 1
        
        else:
            # Modify a block: swap one element
            idx = rng.randint(0, n_blocks - 1)
            old_block = blocks[idx][:]
            if len(old_block) >= 2:
                pos = rng.randint(0, k - 1)
                old_elem = old_block[pos]
                new_elem = rng.choice([x for x in range(v) if x not in old_block])
                old_block[pos] = new_elem
                old_block.sort()
                blocks[idx] = old_block
                new_uncovered = _cover_count(blocks, v, t)
                delta = new_uncovered - cur_uncovered
                new_obj = new_uncovered
                
                if delta <= 0 or rng.random() < np.exp(-delta / T):
                    cur_uncovered = new_uncovered
                    cur_obj = new_obj
                    if cur_uncovered < best_uncovered:
                        best_uncovered = cur_uncovered
                        best_obj = cur_obj
                        best_blocks = [list(b) for b in blocks]
                else:
                    blocks[idx] = old_block
        
        T *= cooling
    
    # After SA, try to remove redundant blocks from the best solution
    best_blocks = [list(b) for b in best_blocks]
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i in range(len(best_blocks)):
            if time.time() >= deadline:
                break
            test_blocks = best_blocks[:i] + best_blocks[i+1:]
            if _cover_count(test_blocks, v, t) == 0:
                best_blocks = test_blocks
                changed = True
                break
    
    return best_blocks

def _is_valid_covering(blocks, v, k, t):
    """Check if the covering is valid."""
    covered = set()
    for b in blocks:
        bset = set(b)
        for ts in itertools.combinations(b, t):
            if all(x in bset for x in ts):
                covered.add(ts)
    for ts in itertools.combinations(range(v), t):
        if ts not in covered:
            return False
    return True

def construct(v, k, t):
    start_time = time.time()
    total_budget = 135.0  # Safety margin for 150s kill
    
    best_blocks = None
    best_len = float('inf')
    
    # Run SA with different seeds
    num_runs = 5
    per_run_time = total_budget / (num_runs + 1)
    
    for run in range(num_runs):
        if time.time() - start_time > total_budget - 10.0:
            break
        rng = random.Random(100 + run * 7919)
        blocks = _sa_covering(v, k, t, rng, per_run_time)
        
        if _is_valid_covering(blocks, v, k, t):
            if len(blocks) < best_len:
                best_len = len(blocks)
                best_blocks = blocks
    
    # If no valid covering found, do a simple greedy fallback
    if best_blocks is None:
        uncovered = set(itertools.combinations(range(v), t))
        blocks = []
        while uncovered:
            first = min(uncovered)
            block = list(first)
            while len(block) < k:
                best_x = None
                best_gain = -1
                for x in range(v):
                    if x in block:
                        continue
                    gain = 0
                    for S in itertools.combinations(block, t-1):
                        tset = tuple(sorted(S + (x,)))
                        if tset in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                if best_x is None:
                    break
                block.append(best_x)
            if len(block) < k:
                break
            block.sort()
            blocks.append(block)
            covered = set(itertools.combinations(block, t))
            uncovered -= covered
        best_blocks = blocks
        best_len = len(blocks)
    
    # Final safety check
    if time.time() - start_time > total_budget - 5.0:
        return best_blocks
    
    return best_blocks
