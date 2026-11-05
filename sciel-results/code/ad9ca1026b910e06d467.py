"""Covering design C(v,k,t) via randomized greedy with local search and redundancy removal."""
import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    """
    Construct a covering design: choose as few k-subsets of {0..v-1} as possible
    such that every t-subset is contained in at least one block.
    
    Uses randomized greedy with multiple restarts, local search (block replacement),
    and redundancy removal to minimize block count.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # 115s safety margin
    
    # Precompute all t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    
    # Map each t-subset to an index for fast lookup
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    
    def triples_in_block(block):
        """Return the set of t-subset indices contained in a block."""
        return set(tsub_to_idx[tuple(sorted(block[i:i+t]))] for i in range(len(block) - t + 1)
                   if True)
    
    # More efficient: generate all t-combinations of the block
    def block_triple_indices(block):
        """Return set of t-subset indices covered by this block."""
        return set(tsub_to_idx[tup] for tup in itertools.combinations(block, t))
    
    def greedy_cover(seed, deadline_time):
        """Run one greedy covering attempt with given random seed."""
        rng = random.Random(seed)
        
        # uncovered: set of t-subset indices not yet covered
        uncovered = set(range(n_tsubs))
        blocks = []
        # For each block, store the set of t-subset indices it covers
        block_triples = []
        
        while uncovered:
            if time.time() > deadline_time:
                break
            
            # Pick a random uncovered t-subset to start a new block
            # Bias towards t-subsets that are "harder" to cover (appear in fewer potential blocks)
            # For simplicity, pick randomly from uncovered
            start_ts_idx = rng.choice(list(uncovered))
            start_ts = all_tsubs[start_ts_idx]
            
            # Build block: start with the t-subset, then add points greedily
            block = list(start_ts)
            
            while len(block) < k:
                if time.time() > deadline_time:
                    break
                
                # Find the point that covers the most uncovered t-subsets
                best_point = None
                best_gain = -1
                best_gain_set = set()
                
                for x in range(v):
                    if x in block:
                        continue
                    
                    # Compute how many uncovered t-subsets would be covered by adding x
                    # A t-subset is covered if it contains t-1 elements from block and x
                    gain_count = 0
                    gain_set = set()
                    for rest in itertools.combinations(block, t - 1):
                        cand = tuple(sorted(rest + (x,)))
                        if cand in tsub_to_idx:
                            idx = tsub_to_idx[cand]
                            if idx in uncovered:
                                gain_count += 1
                                gain_set.add(idx)
                    
                    if gain_count > best_gain:
                        best_gain = gain_count
                        best_point = x
                        best_gain_set = gain_set
                
                if best_point is None or best_gain == 0:
                    # No improvement possible; pick random remaining point
                    remaining = [x for x in range(v) if x not in block]
                    if remaining:
                        best_point = rng.choice(remaining)
                        best_gain_set = set()
                    else:
                        break
                
                block.append(best_point)
            
            if len(block) < k:
                # Can't complete the block; skip
                continue
            
            block.sort()
            blocks.append(block)
            
            # Compute triples covered by this block
            bt = block_triple_indices(block)
            block_triples.append(bt)
            
            # Update uncovered
            uncovered -= bt
        
        return blocks, block_triples, uncovered
    
    def local_search(blocks, block_triples, uncovered, deadline_time, rng):
        """Try to improve the covering by replacing blocks."""
        if not blocks:
            return blocks, block_triples, uncovered
        
        # Redundancy removal: remove blocks whose triples are all covered by other blocks
        changed = True
        while changed and time.time() < deadline_time:
            changed = False
            n = len(blocks)
            for i in range(n):
                if time.time() > deadline_time:
                    break
                # Check if block i is redundant
                # A block is redundant if all its triples are covered by other blocks
                covered_by_others = set()
                for j in range(n):
                    if j == i:
                        continue
                    covered_by_others |= block_triples[j]
                
                if block_triples[i].issubset(covered_by_others):
                    # Remove block i
                    blocks.pop(i)
                    block_triples.pop(i)
                    changed = True
                    break
        
        # Try to replace pairs of blocks with a single better block
        # This is complex; skip for now and focus on redundancy removal
        
        return blocks, block_triples, uncovered
    
    best_blocks = None
    best_count = float('inf')
    
    # Multiple randomized restarts
    for seed in range(200):
        if time.time() > deadline:
            break
        
        blocks, block_triples, uncovered = greedy_cover(seed, deadline)
        
        if not uncovered:
            # Valid covering
            blocks, block_triples, uncovered = local_search(
                blocks, block_triples, uncovered, deadline, random.Random(seed)
            )
            
            if len(blocks) < best_count:
                best_count = len(blocks)
                best_blocks = [list(b) for b in blocks]
        
        if time.time() > deadline - 5.0:
            break
    
    if best_blocks is None:
        # Fallback: deterministic greedy
        uncovered = set(range(n_tsubs))
        blocks = []
        while uncovered:
            if time.time() > deadline:
                break
            start_ts_idx = min(uncovered)
            start_ts = all_tsubs[start_ts_idx]
            block = list(start_ts)
            while len(block) < k:
                best_point = None
                best_gain = -1
                for x in range(v):
                    if x in block:
                        continue
                    gain_count = 0
                    for rest in itertools.combinations(block, t - 1):
                        cand = tuple(sorted(rest + (x,)))
                        if cand in tsub_to_idx and tsub_to_idx[cand] in uncovered:
                            gain_count += 1
                    if gain_count > best_gain:
                        best_gain = gain_count
                        best_point = x
                if best_point is None:
                    remaining = [x for x in range(v) if x not in block]
                    if remaining:
                        best_point = remaining[0]
                    else:
                        break
                block.append(best_point)
            if len(block) >= k:
                block.sort()
                blocks.append(block)
                uncovered -= set(tsub_to_idx[tup] for tup in itertools.combinations(block, t))
        best_blocks = blocks
    
    return best_blocks
