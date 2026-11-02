"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
import sys
from collections import defaultdict


def _precompute_tsubset_index(v, t):
    """Precompute all t-subsets and map to indices for fast lookup."""
    all_tsubsets = list(itertools.combinations(range(v), t))
    n_tsubsets = len(all_tsubsets)
    sub_to_idx = {sub: i for i, sub in enumerate(all_tsubsets)}
    return all_tsubsets, n_tsubsets, sub_to_idx


def _block_tsubsets(block, t):
    """Yield t-subsets of a block."""
    return list(itertools.combinations(block, t))


def _greedy_cover(v, k, t, deadline, rng, all_tsubsets, n_tsubsets, sub_to_idx):
    """
    Greedy covering algorithm.
    Returns (blocks, uncovered_set) where uncovered_set is a set of t-subset indices not yet covered.
    """
    uncovered = set(range(n_tsubsets))
    blocks = []
    
    # Precompute for each point x, the list of t-subset indices that contain x
    point_to_subsets = [[] for _ in range(v)]
    for idx, sub in enumerate(all_tsubsets):
        for x in sub:
            point_to_subsets[x].append(idx)
    
    while uncovered:
        if time.time() > deadline:
            break
        
        # Pick a random uncovered t-subset to start the block
        start_idx = rng.choice(list(uncovered))
        start_sub = all_tsubsets[start_idx]
        block = list(start_sub)
        block_set = set(block)
        
        # Greedily add points to maximize coverage of uncovered t-subsets
        while len(block) < k:
            if time.time() > deadline:
                break
            
            best_gain = -1
            best_points = []
            for x in range(v):
                if x in block_set:
                    continue
                
                # Compute gain: how many uncovered t-subsets would be covered if we add x
                # A t-subset S is covered by block ∪ {x} if S ⊆ block ∪ {x} and |S| = t
                # So S must contain x and (t-1) elements from block
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    sub = tuple(sorted(rest + (x,)))
                    idx = sub_to_idx.get(sub)
                    if idx is not None and idx in uncovered:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_points = [x]
                elif gain == best_gain and gain > 0:
                    best_points.append(x)
            
            if not best_points:
                break
            
            # Pick randomly among best points for diversity
            chosen = rng.choice(best_points)
            block.append(chosen)
            block_set.add(chosen)
        
        block.sort()
        blocks.append(tuple(block))
        
        # Remove covered t-subsets from uncovered
        for sub in itertools.combinations(block, t):
            idx = sub_to_idx.get(sub)
            if idx is not None:
                uncovered.discard(idx)
    
    return blocks, uncovered


def _remove_redundant_blocks(blocks, v, t, deadline, all_tsubsets, n_tsubsets, sub_to_idx):
    """
    Remove redundant blocks: a block is redundant if all its t-subsets are covered by other blocks.
    Returns a new list of blocks with redundant ones removed.
    """
    if not blocks:
        return blocks
    
    # coverage[i] = set of block indices that cover t-subset i
    coverage = [set() for _ in range(n_tsubsets)]
    
    for bidx, block in enumerate(blocks):
        for sub in itertools.combinations(block, t):
            idx = sub_to_idx.get(sub)
            if idx is not None:
                coverage[idx].add(bidx)
    
    # A block is redundant if for every t-subset it covers, there's another block covering it
    non_redundant = []
    for bidx, block in enumerate(blocks):
        if time.time() > deadline:
            break
        redundant = True
        for sub in itertools.combinations(block, t):
            idx = sub_to_idx.get(sub)
            if idx is not None:
                if len(coverage[idx]) == 1:
                    # Only this block covers this t-subset
                    redundant = False
                    break
        if not redundant:
            non_redundant.append(block)
    
    return non_redundant


def _build_repair_block(uncovered_subs, v, k, t, rng, sub_to_idx):
    """
    Build a single block that covers as many of the given uncovered t-subsets as possible.
    Returns a tuple block (sorted list of k points).
    """
    if not uncovered_subs:
        return tuple(sorted(range(k)))
    
    # Start from a random uncovered subset
    start_sub = rng.choice(uncovered_subs)
    block = list(start_sub)
    block_set = set(block)
    
    # Greedily extend to size k, maximizing coverage of uncovered_subs
    while len(block) < k:
        best_gain = -1
        best_points = []
        for x in range(v):
            if x in block_set:
                continue
            gain = 0
            for rest in itertools.combinations(block, t - 1):
                sub = tuple(sorted(rest + (x,)))
                if sub in uncovered_subs:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_points = [x]
            elif gain == best_gain and gain > 0:
                best_points.append(x)
        if not best_points:
            # No point helps; just fill randomly
            remaining = [x for x in range(v) if x not in block_set]
            if remaining:
                chosen = rng.choice(remaining)
                block.append(chosen)
                block_set.add(chosen)
            else:
                break
        else:
            chosen = rng.choice(best_points)
            block.append(chosen)
            block_set.add(chosen)
    
    block.sort()
    return tuple(block)


def _branch_and_bound_minimize(blocks, v, k, t, deadline, rng, all_tsubsets, n_tsubsets, sub_to_idx):
    """
    Cutting-plane / branch-and-bound style heuristic to minimize block count.
    Iteratively tries to drop a block; if that leaves uncovered t-subsets, try to repair
    with a single new block (or a small number). Accept the change only if total count decreases
    or stays the same with fewer 'unique' t-subsets covered (tie-breaker).
    """
    if not blocks:
        return blocks
    
    # Build coverage map: for each t-subset index, set of block indices covering it
    coverage = [set() for _ in range(n_tsubsets)]
    for bidx, block in enumerate(blocks):
        for sub in itertools.combinations(block, t):
            idx = sub_to_idx.get(sub)
            if idx is not None:
                coverage[idx].add(bidx)
    
    current_blocks = list(blocks)
    current_count = len(current_blocks)
    
    # Iteratively try to drop blocks (in random order), repairing if needed
    improved = True
    while improved:
        if time.time() > deadline:
            break
        improved = False
        
        # Rebuild coverage for current_blocks
        coverage = [set() for _ in range(n_tsubsets)]
        for bidx, block in enumerate(current_blocks):
            for sub in itertools.combinations(block, t):
                idx = sub_to_idx.get(sub)
                if idx is not None:
                    coverage[idx].add(bidx)
        
        # Try each block in random order
        order = list(range(len(current_blocks)))
        rng.shuffle(order)
        
        for bidx in order:
            if time.time() > deadline:
                break
            if bidx >= len(current_blocks):
                continue
            
            # Check if dropping this block leaves any uncovered t-subset
            block = current_blocks[bidx]
            would_uncover = []
            for sub in itertools.combinations(block, t):
                idx = sub_to_idx.get(sub)
                if idx is None:
                    continue
                # Count how many blocks cover this sub
                cnt = len(coverage[idx])
                if cnt <= 1:
                    # Only this block covers it
                    would_uncover.append(idx)
            
            if not would_uncover:
                # Safe to drop
                current_blocks.pop(bidx)
                improved = True
                continue
            
            # Try to repair: build one block that covers as many of would_uncover as possible
            uncovered_set = set(would_uncover)
            # Convert indices to actual subsets for the repair builder
            unc_subs = [all_tsubsets[i] for i in uncovered_set]
            repair_block = _build_repair_block(unc_subs, v, k, t, rng, sub_to_idx)
            
            # Check how many of the uncovered subsets the repair block actually covers
            covered_by_repair = set()
            for sub in itertools.combinations(repair_block, t):
                idx = sub_to_idx.get(sub)
                if idx is not None and idx in uncovered_set:
                    covered_by_repair.add(idx)
            
            remaining_uncovered = uncovered_set - covered_by_repair
            
            # Accept if we reduce total count (drop 1, add 1 => same) AND we actually cover all,
            # or if we reduce count by 1 net (drop 1, add 0) - but we always add 1, so net is 0.
            # Only accept if remaining_uncovered is empty (net 0, same count) -- no benefit.
            # Better: accept if remaining_uncovered is empty AND the repair block is different from
            # some existing block (so we can potentially merge later). For now, only accept
            # if it reduces count: we need remaining_uncovered empty and we DON'T add the repair
            # block (i.e., the drop was safe) -- but we already handled that.
            # 
            # Alternative: try to find if the repair block is already in current_blocks.
            # If yes, we can drop without adding (net -1).
            repair_in_current = repair_block in current_blocks
            if repair_in_current and not remaining_uncovered:
                # Drop bidx, don't add repair (it's already there)
                current_blocks.pop(bidx)
                improved = True
                continue
            
            # If remaining_uncovered is empty, dropping + adding repair keeps count same.
            # Skip (no improvement).
            # If remaining_uncovered is not empty, we'd need more blocks; skip.
    
    return current_blocks


def _local_search_improve(blocks, v, k, t, deadline, rng, all_tsubsets, n_tsubsets, sub_to_idx):
    """
    Try to improve the covering by removing blocks that become redundant
    after other blocks are removed, and by trying to replace blocks with better ones.
    """
    if not blocks:
        return blocks
    
    # First pass: remove redundant blocks iteratively
    improved = True
    while improved:
        if time.time() > deadline:
            break
        improved = False
        new_blocks = []
        for block in blocks:
            # Check if this block is still needed
            needed = False
            for sub in itertools.combinations(block, t):
                idx = sub_to_idx.get(sub)
                if idx is None:
                    continue
                # Check if any other block in blocks (excluding this one) covers this sub
                covered_by_other = False
                for other in blocks:
                    if other == block:
                        continue
                    if sub in other:
                        covered_by_other = True
                        break
                if not covered_by_other:
                    needed = True
                    break
            if needed:
                new_blocks.append(block)
            else:
                improved = True
        
        blocks = new_blocks
    
    return blocks


def construct(v, k, t):
    start_time = time.time()
    # Use a safety margin: stop at 110s to leave buffer for cleanup
    deadline = start_time + 110
    
    # Precompute
    all_tsubsets, n_tsubsets, sub_to_idx = _precompute_tsubset_index(v, t)
    
    # Try multiple random restarts with different seeds
    best_blocks = None
    best_count = float('inf')
    
    # First, do a deterministic greedy run
    rng = random.Random(42)
    blocks, uncovered = _greedy_cover(v, k, t, deadline, rng, all_tsubsets, n_tsubsets, sub_to_idx)
    if len(uncovered) == 0:
        best_blocks = blocks
        best_count = len(blocks)
    
    # Try more random restarts if time permits
    if time.time() < deadline - 20:
        for seed in [1, 2, 3, 4, 5]:
            if time.time() > deadline - 10:
                break
            rng = random.Random(seed)
            blocks, uncovered = _greedy_cover(v, k, t, deadline, rng, all_tsubsets, n_tsubsets, sub_to_idx)
            if len(uncovered) == 0 and len(blocks) < best_count:
                best_blocks = blocks
                best_count = len(blocks)
    
    # Remove redundant blocks
    if best_blocks is not None and time.time() < deadline - 5:
        best_blocks = _remove_redundant_blocks(best_blocks, v, t, deadline, all_tsubsets, n_tsubsets, sub_to_idx)
    
    # Branch-and-bound / cutting-plane minimization
    if best_blocks is not None and time.time() < deadline - 3:
        best_blocks = _branch_and_bound_minimize(
            best_blocks, v, k, t, deadline, rng, all_tsubsets, n_tsubsets, sub_to_idx
        )
    
    if best_blocks is None:
        # Fallback: return an empty list or a trivial solution
        return []
    
    return [list(b) for b in best_blocks]
# EVOLVE-BLOCK-END
