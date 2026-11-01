"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
from collections import defaultdict


def _greedy_cover(v, k, t, rng, deadline):
    """Greedy covering with randomized block selection and fast gain computation."""
    all_tsubs = set(itertools.combinations(range(v), t))
    uncovered = set(all_tsubs)
    blocks = []
    
    while uncovered:
        if time.time() > deadline:
            return None  # timeout
        
        # Pick a random uncovered t-subset to start a new block
        start_tsub = rng.choice(list(uncovered))
        block = list(start_tsub)
        block_set = set(block)
        
        while len(block) < k:
            # Find point that covers most uncovered t-subsets
            best_gain = -1
            best_point = -1
            
            for x in range(v):
                if x in block_set:
                    continue
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    if tuple(sorted(rest + (x,))) in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
            
            if best_point == -1:
                # No point improves; pick random
                candidates = [x for x in range(v) if x not in block_set]
                best_point = rng.choice(candidates)
            
            block.append(best_point)
            block_set.add(best_point)
        
        block.sort()
        blocks.append(block)
        uncovered.difference_update(itertools.combinations(block, t))
    
    return blocks


def _remove_redundant_blocks(blocks, v, t):
    """Remove blocks that don't cover any unique t-subset, with iterative passes."""
    if not blocks:
        return blocks
    
    current_blocks = list(blocks)
    
    for _ in range(3):  # Multiple passes to catch cascading redundancies
        if not current_blocks:
            break
        
        cover_count = defaultdict(int)
        block_tsubs = []
        
        for block in current_blocks:
            tsubs = list(itertools.combinations(block, t))
            block_tsubs.append(tsubs)
            for ts in tsubs:
                cover_count[ts] += 1
        
        kept = []
        for i, block in enumerate(current_blocks):
            all_redundant = True
            for ts in block_tsubs[i]:
                if cover_count[ts] == 1:
                    all_redundant = False
                    break
            if not all_redundant:
                kept.append(block)
        
        if len(kept) == len(current_blocks):
            break
        current_blocks = kept
    
    return current_blocks


def _local_search_polish(blocks, v, k, t, rng, deadline):
    """Try to reduce block count by removing redundant blocks after point swaps."""
    if not blocks:
        return blocks
    
    best_blocks = [list(b) for b in blocks]
    best_count = len(best_blocks)
    
    # Build coverage structure
    def build_cover_counts(blist):
        cc = defaultdict(int)
        for b in blist:
            for ts in itertools.combinations(b, t):
                cc[ts] += 1
        return cc
    
    # Try removing each block and see if we can fix coverage by modifying another
    for attempt in range(50):
        if time.time() > deadline:
            break
        
        cc = build_cover_counts(best_blocks)
        
        # Find a block that might be removable (has few unique t-subsets)
        removable_candidates = []
        for i, b in enumerate(best_blocks):
            unique_count = sum(1 for ts in itertools.combinations(b, t) if cc[ts] == 1)
            if unique_count <= 2:  # Low unique coverage
                removable_candidates.append((unique_count, i))
        
        if not removable_candidates:
            break
        
        removable_candidates.sort()
        _, rem_idx = removable_candidates[0]
        
        # Try to remove this block
        test_blocks = [list(b) for j, b in enumerate(best_blocks) if j != rem_idx]
        
        # Check if still valid
        all_tsubs = set(itertools.combinations(range(v), t))
        covered = set()
        for b in test_blocks:
            covered.update(itertools.combinations(b, t))
        
        if all_tsubs.issubset(covered):
            best_blocks = test_blocks
            best_count = len(best_blocks)
            continue
        
        # Find uncovered t-subsets
        uncovered = all_tsubs - covered
        
        # Try to fix by modifying a random other block
        fixed = False
        for _ in range(20):
            if time.time() > deadline:
                break
            
            # Pick a random block to modify
            mod_idx = rng.randrange(len(test_blocks))
            orig_block = list(test_blocks[mod_idx])
            
            # Try swapping one point
            for pos in range(k):
                for new_pt in range(v):
                    if new_pt in orig_block:
                        continue
                    
                    candidate = list(orig_block)
                    candidate[pos] = new_pt
                    candidate.sort()
                    
                    # Check if this helps cover some uncovered t-subsets
                    new_tsubs = set(itertools.combinations(candidate, t))
                    old_tsubs = set(itertools.combinations(orig_block, t))
                    net_gain = len(new_tsubs & uncovered) - len((old_tsubs - new_tsubs) & covered)
                    
                    if net_gain > 0:
                        test_blocks[mod_idx] = candidate
                        # Recheck validity
                        covered2 = set()
                        for b in test_blocks:
                            covered2.update(itertools.combinations(b, t))
                        if all_tsubs.issubset(covered2):
                            best_blocks = [list(b) for b in test_blocks]
                            best_count = len(best_blocks)
                            fixed = True
                            break
                
                if fixed:
                    break
            
            if fixed:
                break
        
        if not fixed:
            # Try a different removable candidate
            pass
    
    return best_blocks


def _verify_covering(blocks, v, t):
    """Verify every t-subset is covered by at least one block."""
    covered = set()
    for block in blocks:
        covered.update(itertools.combinations(block, t))
    all_tsubs = set(itertools.combinations(range(v), t))
    return all_tsubs.issubset(covered)


def construct(v, k, t):
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    best_blocks = None
    best_count = float('inf')
    
    # Multiple randomized restarts with different seeds
    max_restarts = 15
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        rng = random.Random(42 + restart * 137)
        
        blocks = _greedy_cover(v, k, t, rng, deadline)
        if blocks is None:
            continue
        
        # Remove redundant blocks
        blocks = _remove_redundant_blocks(blocks, v, t)
        
        # Local search polish
        if time.time() < deadline - 5:  # Leave time for polish
            polish_rng = random.Random(42 + restart * 137 + 999)
            blocks = _local_search_polish(blocks, v, k, t, polish_rng, deadline - 2)
        
        # Verify validity
        if _verify_covering(blocks, v, t):
            if len(blocks) < best_count:
                best_count = len(blocks)
                best_blocks = [list(b) for b in blocks]
        
        # Early exit if we hit a good solution
        if best_count <= 76:
            break
    
    if best_blocks is None:
        # Fallback: deterministic greedy
        uncovered = set(itertools.combinations(range(v), t))
        blocks = []
        while uncovered:
            block = list(min(uncovered))
            block_set = set(block)
            while len(block) < k:
                def gain(x):
                    return sum(1 for rest in itertools.combinations(block, t - 1)
                               if tuple(sorted(rest + (x,))) in uncovered)
                best_pt = max((x for x in range(v) if x not in block_set), key=gain)
                block.append(best_pt)
                block_set.add(best_pt)
            block.sort()
            blocks.append(block)
            uncovered.difference_update(itertools.combinations(block, t))
        best_blocks = blocks
    
    return best_blocks
# EVOLVE-BLOCK-END
