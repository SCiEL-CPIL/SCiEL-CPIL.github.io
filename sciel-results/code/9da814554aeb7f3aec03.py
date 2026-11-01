"""Portfolio search for covering design C(24,7,3)."""
import itertools
import random
import time
import sys

# EVOLVE-BLOCK-START
import numpy as np

def _greedy_covering(v, k, t, rng, time_limit, mode='basic', diversity_bonus=0.0):
    """
    Greedy covering with various scoring modes.
    mode='basic': pure gain
    mode='pair': gain + diversity bonus based on pair rarity
    mode='adaptive': gain with adaptive weight on diversity
    """
    deadline = time.time() + time_limit
    
    # All t-sets
    all_tsets = list(itertools.combinations(range(v), t))
    uncovered = set(all_tsets)
    blocks = []
    
    # Precompute t-sets containing each (t-1)-subset
    # For each (t-1)-tuple S, list of t-sets that contain S
    tset_by_tm1 = {}
    for ts in all_tsets:
        for i in range(t):
            tm1 = ts[:i] + ts[i+1:]
            if tm1 not in tset_by_tm1:
                tset_by_tm1[tm1] = []
            tset_by_tm1[tm1].append(ts)
    
    # For each pair (i,j), track how many blocks contain both
    pair_block_count = {}
    
    def compute_gain(x, block, uncovered):
        """Number of uncovered t-sets that contain x and (t-1) elements of block."""
        cnt = 0
        for S in itertools.combinations(block, t-1):
            if x not in S:
                tset = tuple(sorted(S + (x,)))
                if tset in uncovered:
                    cnt += 1
        return cnt
    
    def compute_diversity_bonus(x, block, pair_block_count):
        """Bonus based on how rare the pairs (x, y) are for y in block."""
        bonus = 0.0
        for y in block:
            p = (min(x, y), max(x, y))
            cnt = pair_block_count.get(p, 0)
            bonus += 1.0 / (1.0 + cnt)
        return bonus
    
    while uncovered and time.time() < deadline:
        # Start with a random uncovered t-set
        # For diversity, sometimes pick the one with most uncovered t-sets
        first_uncovered = min(uncovered)
        block = list(first_uncovered)
        
        while len(block) < k:
            candidates = [x for x in range(v) if x not in block]
            if not candidates:
                break
            
            best_x = None
            best_score = -1
            
            for x in candidates:
                gain = compute_gain(x, block, uncovered)
                
                if mode == 'basic':
                    score = gain
                elif mode == 'pair':
                    bonus = compute_diversity_bonus(x, block, pair_block_count)
                    score = gain + diversity_bonus * bonus
                elif mode == 'adaptive':
                    bonus = compute_diversity_bonus(x, block, pair_block_count)
                    # Adaptive: weight diversity more when gain is low
                    adaptive_w = max(0.1, 1.0 - gain / max(1, len(uncovered)))
                    score = gain + adaptive_w * bonus
                else:
                    score = gain
                
                # Add small random tie-breaker
                score += rng.random() * 0.01
                
                if score > best_score:
                    best_score = score
                    best_x = x
            
            if best_x is None:
                break
            block.append(best_x)
        
        if len(block) < k:
            break
        
        block.sort()
        blocks.append(tuple(block))
        
        # Update pair counts
        for p in itertools.combinations(block, 2):
            pair_block_count[p] = pair_block_count.get(p, 0) + 1
        
        # Remove covered t-sets
        covered = set(itertools.combinations(block, t))
        uncovered -= covered
    
    return [list(b) for b in blocks]

def _local_search(blocks, v, k, t, rng, time_limit):
    """
    Local search: try to improve by removing redundant blocks and swapping.
    """
    deadline = time.time() + time_limit
    blocks = [list(b) for b in blocks]
    
    # Precompute all t-sets
    all_tsets = list(itertools.combinations(range(v), t))
    tset_to_blocks = {ts: [] for ts in all_tsets}
    
    def build_indices(blocks):
        tset_to_blocks = {ts: [] for ts in all_tsets}
        for i, b in enumerate(blocks):
            bset = set(b)
            for ts in itertools.combinations(b, t):
                tset_to_blocks[ts].append(i)
        return tset_to_blocks
    
    tset_to_blocks = build_indices(blocks)
    
    def is_valid_covering(blocks, v, k, t):
        """Check if every t-subset is covered by at least one block."""
        for ts in itertools.combinations(range(v), t):
            if len(tset_to_blocks.get(ts, [])) == 0:
                return False
        return True
    
    def find_redundant(blocks, tset_to_blocks):
        """Find blocks that are redundant (removing them doesn't uncover any t-set)."""
        redundant = []
        for i in range(len(blocks)):
            # Check if any t-set is covered ONLY by block i
            only_covered = False
            b = blocks[i]
            for ts in itertools.combinations(b, t):
                covering = tset_to_blocks[ts]
                if covering == [i]:
                    only_covered = True
                    break
            if not only_covered:
                redundant.append(i)
        return redundant
    
    def try_remove_redundant(blocks, tset_to_blocks):
        """Try to remove redundant blocks. Returns (new_blocks, new_tset_to_blocks, removed)."""
        removed = 0
        changed = True
        while changed and time.time() < deadline:
            changed = False
            redundant = find_redundant(blocks, tset_to_blocks)
            if redundant:
                # Remove the first redundant block
                idx = redundant[0]
                b = blocks[idx]
                # Remove from tset_to_blocks
                for ts in itertools.combinations(b, t):
                    if idx in tset_to_blocks[ts]:
                        tset_to_blocks[ts].remove(idx)
                # Remove block
                blocks.pop(idx)
                # Reindex
                for i in range(len(blocks)):
                    for ts in itertools.combinations(blocks[i], t):
                        # This is inefficient, but let's rebuild
                        pass
                # Rebuild indices
                tset_to_blocks = build_indices(blocks)
                removed += 1
                changed = True
        return blocks, tset_to_blocks, removed
    
    def try_swap(blocks, tset_to_blocks, rng):
        """Try to swap a block with a better one. Returns (new_blocks, new_tset_to_blocks, improved)."""
        n = len(blocks)
        if n == 0:
            return blocks, tset_to_blocks, False
        
        # Pick a random block to potentially swap
        idx = rng.randint(0, n - 1)
        old_block = blocks[idx]
        old_set = set(old_block)
        
        # Find t-sets that are ONLY covered by this block
        critical_tsets = []
        for ts in itertools.combinations(old_block, t):
            covering = tset_to_blocks[ts]
            if covering == [idx]:
                critical_tsets.append(ts)
        
        if not critical_tsets:
            # This block is redundant, remove it
            # Remove from tset_to_blocks
            for ts in itertools.combinations(old_block, t):
                if idx in tset_to_blocks[ts]:
                    tset_to_blocks[ts].remove(idx)
            blocks.pop(idx)
            tset_to_blocks = build_indices(blocks)
            return blocks, tset_to_blocks, True
        
        # Try to find a replacement that covers all critical_tsets
        # Sample some random blocks
        best_new_block = None
        best_extra = -1
        
        for _ in range(20):
            if time.time() >= deadline:
                break
            cand = sorted(rng.sample(range(v), k))
            cand_set = set(cand)
            
            # Check if cand covers all critical_tsets
            covers_all = True
            for ts in critical_tsets:
                if not set(ts).issubset(cand_set):
                    covers_all = False
                    break
            if not covers_all:
                continue
            
            # Count how many additional t-sets this block covers
            extra = 0
            for ts in itertools.combinations(cand, t):
                covering = tset_to_blocks.get(ts, [])
                # Count how many blocks cover ts, excluding the old one
                covering_without_old = [i for i in covering if i != idx]
                if len(covering_without_old) == 0:
                    extra += 1
            
            if extra > best_extra:
                best_extra = extra
                best_new_block = cand
        
        if best_new_block is not None and best_extra > 0:
            # Perform swap
            # Remove old block from indices
            for ts in itertools.combinations(old_block, t):
                if idx in tset_to_blocks[ts]:
                    tset_to_blocks[ts].remove(idx)
            # Add new block
            blocks[idx] = best_new_block
            for ts in itertools.combinations(best_new_block, t):
                tset_to_blocks.setdefault(ts, []).append(idx)
            # Rebuild to clean up indices
            tset_to_blocks = build_indices(blocks)
            return blocks, tset_to_blocks, True
        
        return blocks, tset_to_blocks, False
    
    # Main loop
    while time.time() < deadline:
        # Phase 1: Remove redundant blocks
        blocks, tset_to_blocks, removed = try_remove_redundant(blocks, tset_to_blocks)
        
        # Phase 2: Try swaps
        blocks, tset_to_blocks, improved = try_swap(blocks, tset_to_blocks, rng)
        
        if not improved and removed == 0:
            # Try a few more random swaps before giving up
            for _ in range(5):
                if time.time() >= deadline:
                    break
                blocks, tset_to_blocks, improved = try_swap(blocks, tset_to_blocks, rng)
                if improved:
                    break
            if not improved:
                break
    
    return blocks

def _is_valid_covering(blocks, v, k, t):
    """Check if the covering is valid."""
    # Build a set of all covered t-sets
    covered = set()
    for b in blocks:
        bset = set(b)
        for ts in itertools.combinations(b, t):
            if set(ts).issubset(bset):
                covered.add(tuple(ts))
    
    # Check all t-sets are covered
    for ts in itertools.combinations(range(v), t):
        if tuple(ts) not in covered:
            return False
    return True

def construct(v, k, t):
    start_time = time.time()
    # Reserve 15s safety margin for the 150s kill
    total_budget = 135.0
    
    best_blocks = None
    best_len = float('inf')
    
    # Run multiple greedy variants with different modes and seeds
    modes = ['basic', 'pair', 'adaptive']
    diversity_values = [0.3, 0.5, 0.8]
    num_runs = 3
    
    # Allocate time
    greedy_time = total_budget * 0.50
    ls_time = total_budget * 0.40
    margin = total_budget * 0.10
    
    total_greedy_configs = len(modes) * len(diversity_values) * num_runs
    per_greedy_time = greedy_time / total_greedy_configs
    
    config_idx = 0
    for mode in modes:
        for div in diversity_values:
            for run in range(num_runs):
                config_idx += 1
                if time.time() - start_time > total_budget - ls_time - margin:
                    break
                
                rng = random.Random(42 + config_idx * 137)
                blocks = _greedy_covering(v, k, t, rng, per_greedy_time, mode=mode, diversity_bonus=div)
                
                if len(blocks) < best_len:
                    best_len = len(blocks)
                    best_blocks = blocks
    
    # If no good result, do a simple greedy
    if best_blocks is None or best_len > 80:
        rng = random.Random(0)
        blocks = _greedy_covering(v, k, t, rng, 10.0, mode='basic')
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
    
    if best_blocks is None:
        # Fallback
        blocks = _greedy_covering(v, k, t, random.Random(1), 5.0, mode='basic')
        best_blocks = blocks
        best_len = len(blocks)
    
    # Phase 2: Local search refinement
    remaining = total_budget - (time.time() - start_time) - margin
    if remaining > 5.0 and best_blocks is not None:
        rng = random.Random(12345)
        refined = _local_search(best_blocks, v, k, t, rng, remaining)
        if len(refined) < best_len:
            best_blocks = refined
            best_len = len(refined)
    
    # Final safety check
    if time.time() - start_time > total_budget - 5.0:
        return best_blocks
    
    return best_blocks
# EVOLVE-BLOCK-END
