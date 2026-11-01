"""Covering design C(24,7,3) via optimized SA + greedy repair."""
import itertools
import random
import time
import sys

# Precompute combinations and index maps for speed
def _precompute(v, k, t):
    """Precompute all t-subsets, their indices, and mapping from k-blocks to covered t-subsets."""
    all_tsubs = list(itertools.combinations(range(v), t))
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    num_tsubs = len(all_tsubs)
    
    # For each possible k-block, precompute which t-subsets it covers
    # This is too large to precompute fully, so we'll compute on the fly but cache
    return all_tsubs, tsub_to_idx, num_tsubs


def _block_covers_tsubs(block, t, tsub_to_idx):
    """Return set of t-subset indices covered by a k-block."""
    covered = set()
    for ts in itertools.combinations(block, t):
        idx = tsub_to_idx.get(ts)
        if idx is not None:
            covered.add(idx)
    return covered


def _greedy_cover(v, k, t, time_limit=30.0):
    """Fast greedy covering to get initial solution."""
    all_tsubs, tsub_to_idx, num_tsubs = _precompute(v, k, t)
    
    # Track coverage count for each t-subset
    cov_count = [0] * num_tsubs
    blocks = []
    
    start_time = time.time()
    
    while True:
        # Find uncovered t-subset with highest "difficulty" (fewest candidate blocks containing it)
        # For speed, just pick the first uncovered one
        uncovered = [i for i in range(num_tsubs) if cov_count[i] == 0]
        if not uncovered:
            break
        
        if time.time() - start_time > time_limit:
            break
        
        # Pick a random uncovered t-subset to start the block
        start_ts = uncovered[random.randint(0, len(uncovered) - 1)]
        block = list(all_tsubs[start_ts])
        
        # Greedily add points that cover the most uncovered t-subsets
        while len(block) < k:
            best_point = None
            best_gain = -1
            for x in range(v):
                if x in block:
                    continue
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    ts = tuple(sorted(rest + (x,)))
                    idx = tsub_to_idx.get(ts)
                    if idx is not None and cov_count[idx] == 0:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
            if best_point is not None:
                block.append(best_point)
        
        block.sort()
        
        # Add block and update coverage
        blocks.append(block)
        for ts in itertools.combinations(block, t):
            idx = tsub_to_idx.get(ts)
            if idx is not None:
                cov_count[idx] += 1
    
    return blocks


def _remove_redundant_blocks(blocks, v, k, t, tsub_to_idx):
    """Remove blocks that don't uniquely cover any t-subset."""
    if not blocks:
        return blocks
    
    all_tsubs, _, num_tsubs = _precompute(v, k, t)
    
    # For each t-subset, track which blocks cover it
    tsub_blocks = [[] for _ in range(num_tsubs)]
    for bi, block in enumerate(blocks):
        for ts in itertools.combinations(block, t):
            idx = tsub_to_idx.get(ts)
            if idx is not None:
                tsub_blocks[idx].append(bi)
    
    # A block is redundant if no t-subset is covered only by it
    keep = [True] * len(blocks)
    for bi in range(len(blocks)):
        unique = False
        # Check if this block is the only cover for any t-subset
        # This is expensive; approximate by checking a sample
        for ts in itertools.combinations(blocks[bi], t):
            idx = tsub_to_idx.get(ts)
            if idx is not None and len(tsub_blocks[idx]) == 1:
                unique = True
                break
        if not unique:
            keep[bi] = False
    
    return [b for i, b in enumerate(blocks) if keep[i]]


def _simulated_annealing(blocks, v, k, t, time_limit=85.0):
    """Optimize the covering using simulated annealing with incremental updates."""
    all_tsubs, tsub_to_idx, num_tsubs = _precompute(v, k, t)
    
    # Initialize coverage counts
    cov_count = [0] * num_tsubs
    for block in blocks:
        for ts in itertools.combinations(block, t):
            idx = tsub_to_idx.get(ts)
            if idx is not None:
                cov_count[idx] += 1
    
    num_blocks = len(blocks)
    if num_blocks == 0:
        return blocks
    
    best_blocks = list(blocks)
    best_num_blocks = num_blocks
    best_cov_count = list(cov_count)
    
    # SA parameters
    start_time = time.time()
    deadline = start_time + time_limit
    T = 5.0  # initial temperature
    cooling = 0.999
    min_T = 0.01
    
    current_blocks = list(blocks)
    current_cov = list(cov_count)
    current_num = num_blocks
    
    while time.time() < deadline and T > min_T:
        # Choose a random move type
        move_type = random.random()
        
        if move_type < 0.3 and current_num > 1:
            # Try to remove a block
            bi = random.randint(0, current_num - 1)
            # Check if removing this block would leave any t-subset uncovered
            would_uncover = False
            for ts in itertools.combinations(current_blocks[bi], t):
                idx = tsub_to_idx.get(ts)
                if idx is not None and current_cov[idx] == 1:
                    would_uncover = True
                    break
            if not would_uncover:
                # Remove block
                current_blocks.pop(bi)
                for ts in itertools.combinations(current_blocks[bi] if bi < len(current_blocks) else [], t):
                    pass  # already removed
                # Actually we need to recompute cov for the removed block
                # Let's redo this properly
                # Restore and do it cleanly
                current_blocks.insert(bi, current_blocks[bi])  # no-op, just for clarity
                # Remove properly
                removed_block = current_blocks.pop(bi)
                for ts in itertools.combinations(removed_block, t):
                    idx = tsub_to_idx.get(ts)
                    if idx is not None:
                        current_cov[idx] -= 1
                current_num -= 1
                # Accept (always beneficial)
                if current_num < best_num_blocks:
                    best_num_blocks = current_num
                    best_blocks = list(current_blocks)
                    best_cov_count = list(current_cov)
            else:
                # Reject
                pass
        
        elif move_type < 0.7 and current_num > 0:
            # Try to modify a block: replace one point
            bi = random.randint(0, current_num - 1)
            old_block = current_blocks[bi]
            # Pick a random point to replace
            pi = random.randint(0, len(old_block) - 1)
            old_point = old_block[pi]
            # Pick a new point
            candidates = [x for x in range(v) if x not in old_block]
            if candidates:
                new_point = random.choice(candidates)
                new_block = old_block[:pi] + [new_point] + old_block[pi+1:]
                new_block.sort()
                
                # Compute delta in coverage
                # Old block covers certain t-subsets, new block covers different ones
                old_covered = set()
                for ts in itertools.combinations(old_block, t):
                    idx = tsub_to_idx.get(ts)
                    if idx is not None:
                        old_covered.add(idx)
                
                new_covered = set()
                for ts in itertools.combinations(new_block, t):
                    idx = tsub_to_idx.get(ts)
                    if idx is not None:
                        new_covered.add(idx)
                
                # Check if any t-subset would become uncovered
                would_uncover = False
                for idx in old_covered - new_covered:
                    if current_cov[idx] == 1:
                        would_uncover = True
                        break
                
                if not would_uncover:
                    # Apply the change
                    current_blocks[bi] = new_block
                    for idx in old_covered - new_covered:
                        current_cov[idx] -= 1
                    for idx in new_covered - old_covered:
                        current_cov[idx] += 1
                    # No change in number of blocks, but might enable future removals
                else:
                    # Reject
                    pass
        
        else:
            # Try to add a new block (only if we think it helps)
            # This is less common; we want to minimize blocks
            # Skip for now
            pass
        
        T *= cooling
    
    # Final repair: ensure all t-subsets are covered
    # If not, add blocks greedily for uncovered t-subsets
    uncovered = [i for i in range(num_tsubs) if best_cov_count[i] == 0]
    if uncovered:
        for idx in uncovered:
            ts = all_tsubs[idx]
            block = list(ts)
            while len(block) < k:
                best_point = None
                best_gain = -1
                for x in range(v):
                    if x in block:
                        continue
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        ts2 = tuple(sorted(rest + (x,)))
                        idx2 = tsub_to_idx.get(ts2)
                        if idx2 is not None and best_cov_count[idx2] == 0:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                if best_point is not None:
                    block.append(best_point)
                else:
                    # Add any unused point
                    for x in range(v):
                        if x not in block:
                            block.append(x)
                            break
            block.sort()
            best_blocks.append(block)
            for ts2 in itertools.combinations(block, t):
                idx2 = tsub_to_idx.get(ts2)
                if idx2 is not None:
                    best_cov_count[idx2] += 1
    
    return best_blocks


def construct(v, k, t):
    """Construct a covering design with as few k-blocks as possible."""
    start_time = time.time()
    total_deadline = start_time + 145.0  # 150s limit, 5s safety margin
    
    # Phase 1: Greedy initial cover (30s budget)
    greedy_deadline = min(start_time + 30.0, total_deadline)
    blocks = _greedy_cover(v, k, t, time_limit=greedy_deadline - start_time)
    
    if time.time() > total_deadline:
        return blocks
    
    # Phase 2: Remove redundant blocks
    all_tsubs, tsub_to_idx, num_tsubs = _precompute(v, k, t)
    blocks = _remove_redundant_blocks(blocks, v, k, t, tsub_to_idx)
    
    if time.time() > total_deadline:
        return blocks
    
    # Phase 3: Simulated annealing optimization (remaining time)
    sa_deadline = total_deadline
    sa_time = max(1.0, sa_deadline - time.time())
    blocks = _simulated_annealing(blocks, v, k, t, time_limit=sa_time)
    
    # Final validation and repair if needed
    if time.time() < total_deadline - 2.0:
        cov_count = [0] * num_tsubs
        for block in blocks:
            for ts in itertools.combinations(block, t):
                idx = tsub_to_idx.get(ts)
                if idx is not None:
                    cov_count[idx] += 1
        uncovered = [i for i in range(num_tsubs) if cov_count[i] == 0]
        if uncovered:
            for idx in uncovered:
                ts = all_tsubs[idx]
                block = list(ts)
                while len(block) < k:
                    best_point = None
                    best_gain = -1
                    for x in range(v):
                        if x in block:
                            continue
                        gain = 0
                        for rest in itertools.combinations(block, t - 1):
                            ts2 = tuple(sorted(rest + (x,)))
                            idx2 = tsub_to_idx.get(ts2)
                            if idx2 is not None and cov_count[idx2] == 0:
                                gain += 1
                        if gain > best_gain:
                            best_gain = gain
                            best_point = x
                    if best_point is not None:
                        block.append(best_point)
                    else:
                        for x in range(v):
                            if x not in block:
                                block.append(x)
                                break
                block.sort()
                blocks.append(block)
                for ts2 in itertools.combinations(block, t):
                    idx2 = tsub_to_idx.get(ts2)
                    if idx2 is not None:
                        cov_count[idx2] += 1
    
    return blocks
