import itertools
import time
import random
from collections import defaultdict

_DEADLINE_S = 145.0  # Safety margin from 150s limit
_START_TIME = time.time()


def _time_left():
    return max(0.0, _DEADLINE_S - (time.time() - _START_TIME))


def _greedy_construct(v, k, t):
    """
    Construct a covering using greedy strategy.
    """
    all_tsubs = list(itertools.combinations(range(v), t))
    uncovered = set(all_tsubs)
    blocks = []
    
    # Precompute t-subsets for each possible k-subset? No, too many.
    # Instead, for a target t-subset, find best k-block containing it.
    
    # Point frequency to guide heuristic
    point_freq = [0] * v
    
    while uncovered:
        if _time_left() < 5.0:
            break
        
        # Select target: uncovered t-subset with minimum sum of point frequencies
        # This tends to pick "harder" points
        best_target = min(uncovered, key=lambda tsub: sum(point_freq[p] for p in tsub))
        
        target_set = set(best_target)
        remaining_points = [p for p in range(v) if p not in target_set]
        
        best_block = None
        best_gain = -1
        
        # Find k-block containing best_target that covers most uncovered t-subsets
        for comb in itertools.combinations(remaining_points, k - t):
            block = tuple(sorted(list(best_target) + list(comb)))
            
            # Count gain
            gain = 0
            for tsub in itertools.combinations(block, t):
                if tsub in uncovered:
                    gain += 1
            
            if gain > best_gain:
                best_gain = gain
                best_block = block
        
        if best_block is None:
            break
        
        blocks.append(list(best_block))
        
        # Update uncovered
        for tsub in itertools.combinations(best_block, t):
            uncovered.discard(tsub)
            
        # Update point frequencies
        for p in best_block:
            point_freq[p] += 1

    return blocks


def _remove_redundant_blocks(blocks, v, k, t):
    """
    Remove blocks that don't cover any unique t-subset.
    Returns new list of blocks and a mapping of tsub to count.
    """
    if not blocks:
        return blocks, defaultdict(int)
    
    # Count coverage for each t-subset
    tsub_count = defaultdict(int)
    for block in blocks:
        for tsub in itertools.combinations(block, t):
            tsub_count[tsub] += 1
    
    new_blocks = []
    for block in blocks:
        # Check if block is redundant
        redundant = True
        for tsub in itertools.combinations(block, t):
            if tsub_count[tsub] == 1:
                redundant = False
                break
        
        if not redundant:
            new_blocks.append(block)
            # Decrement counts for this block's tsubs
            for tsub in itertools.combinations(block, t):
                tsub_count[tsub] -= 1
        # If redundant, we skip it, so counts for its tsubs remain as is 
        # (they are covered by other blocks)
        
    return new_blocks, tsub_count


def _local_search(blocks, v, k, t):
    """
    Local search to minimize number of blocks.
    Strategy:
    1. Maintain coverage counts.
    2. Iteratively remove redundant blocks.
    3. Try to replace a block B with a block B' that covers all unique subsets of B.
       This is valid if the union of unique subsets of B has size <= k.
       If B' is found and is different from B, and it allows further reductions, accept.
       To ensure progress, we accept if B' covers all unique subsets of B.
       Then we check for new redundancies.
    """
    if not blocks:
        return blocks
    
    # Initialize coverage counts
    tsub_count = defaultdict(int)
    for block in blocks:
        for tsub in itertools.combinations(block, t):
            tsub_count[tsub] += 1
    
    # Precompute t-subsets for each block for speed
    block_tsubs = [list(itertools.combinations(block, t)) for block in blocks]
    
    # Helper to get unique tsubs for a block index
    def get_unique_tsubs(idx):
        return [tsub for tsub in block_tsubs[idx] if tsub_count[tsub] == 1]
    
    # Precompute all possible k-subsets? No.
    # We will generate candidates on the fly.
    
    while _time_left() > 2.0:
        # Phase 1: Remove redundant blocks
        changed = True
        while changed:
            changed = False
            for i in range(len(blocks) - 1, -1, -1):
                unique_tsubs = get_unique_tsubs(i)
                if not unique_tsubs:
                    # Block i is redundant
                    # Remove it
                    for tsub in block_tsubs[i]:
                        tsub_count[tsub] -= 1
                    blocks.pop(i)
                    block_tsubs.pop(i)
                    changed = True
                    break
        
        if not blocks:
            break
            
        # Phase 2: Try replacements
        # Shuffle indices to avoid bias
        indices = list(range(len(blocks)))
        random.shuffle(indices)
        
        replaced = False
        for i in indices:
            if _time_left() < 2.0:
                break
                
            unique_tsubs = get_unique_tsubs(i)
            if not unique_tsubs:
                # Should have been removed, but maybe counts changed?
                # Re-check
                if not [tsub for tsub in block_tsubs[i] if tsub_count[tsub] == 1]:
                    continue
                else:
                    # It has unique tsubs
                    pass
            
            # Find all points that MUST be in the new block
            must_include = set()
            for tsub in unique_tsubs:
                must_include.update(tsub)
            
            if len(must_include) > k:
                continue # Cannot replace directly
            
            # Try to find a replacement block B'
            # B' must contain must_include.
            # B' should be different from blocks[i].
            # Ideally, B' covers more "flexible" subsets to allow future removals.
            
            current_block_set = set(blocks[i])
            remaining_points = [p for p in range(v) if p not in must_include]
            need_more = k - len(must_include)
            
            best_candidate = None
            # We want a candidate that covers the unique tsubs (guaranteed by construction)
            # and ideally maximizes the number of t-subsets it covers that are currently 
            # covered by exactly 1 block (other than this one) or are uncovered?
            # Actually, if we replace B with B', the unique tsubs of B are still covered.
            # The other tsubs of B lose one cover. If they had count 1, they become uncovered -> INVALID.
            # So, we MUST ensure that any tsub in B that is NOT in unique_tsubs has count >= 2.
            # If a tsub in B has count 1 and is not in unique_tsubs, that's a contradiction 
            # because unique_tsubs is defined as ALL tsubs with count 1.
            # So, all other tsubs in B have count >= 2.
            # When we remove B, their count decreases by 1. They become count >= 1.
            # When we add B', we add 1 to their count (if they are in B').
            # So validity is maintained as long as B' covers unique_tsubs.
            
            # To make the replacement useful, we want B' to be "better".
            # A good heuristic: B' should cover t-subsets that are currently "tight" (count 1) 
            # from OTHER blocks, making those other blocks potentially redundant?
            # Or simply, B' should be different to allow further search.
            
            # Let's just try to find ANY valid replacement different from current.
            # If we find one, we swap, and then the redundancy removal phase might find new removals.
            
            found_candidate = False
            # Limit search for candidates to avoid timeout
            # If need_more is large, C(v - len(must), need_more) can be huge.
            # For v=19, k=7, t=4.
            # max must_include size is 4 (one tsub) or more.
            # If must_include is size 4, need_more = 3. C(15, 3) = 455. Manageable.
            # If must_include is size 7, need_more = 0. 1 candidate.
            
            for comb in itertools.combinations(remaining_points, need_more):
                new_block = tuple(sorted(list(must_include) + list(comb)))
                if new_block == tuple(blocks[i]):
                    continue
                
                # Check if this new block is already in the list?
                # If it is, we are effectively moving the block to a different index?
                # No, we are replacing block i with a block that might already exist.
                # If it exists, we would have two identical blocks.
                # We should avoid adding duplicates if possible, or handle them.
                # If new_block is already in blocks, then adding it is redundant.
                # But we are REPLACING block i.
                # If new_block is already in blocks (at index j != i), then after replacement,
                # we have two identical blocks. This is not ideal but valid.
                # However, it doesn't help reduce count.
                
                # Heuristic: Prefer candidates that cover more t-subsets with count == 1 
                # (from other blocks) to enable future removals?
                # Or prefer candidates that are "dense" in unique tsubs of other blocks?
                
                # Let's just pick the first valid candidate to keep it simple and fast.
                # The redundancy removal step will clean up.
                found_candidate = True
                best_candidate = new_block
                break
            
            if found_candidate:
                # Perform replacement
                old_block = blocks[i]
                old_tsubs = block_tsubs[i]
                
                # Remove old block from counts
                for tsub in old_tsubs:
                    tsub_count[tsub] -= 1
                
                # Add new block to counts
                new_tsubs = list(itertools.combinations(best_candidate, t))
                for tsub in new_tsubs:
                    tsub_count[tsub] += 1
                
                # Update structures
                blocks[i] = list(best_candidate)
                block_tsubs[i] = new_tsubs
                
                replaced = True
                # Break to restart the loop (re-evaluate redundancies)
                break
        
        if not replaced:
            # No replacements found in this pass.
            # Try a more aggressive move: Randomly swap a block with a random valid block
            # that covers its unique tsubs?
            # If we are stuck, we might need to perturb.
            # Perturbation: Add a random block, then remove redundancies.
            # This might increase block count temporarily but help escape local min.
            # However, we want to MINIMIZE blocks. Adding a block is bad unless it allows
            # removal of multiple others.
            
            # Alternative: Try to replace a block with a random candidate, even if it doesn't
            # seem "better" immediately, hoping to break symmetry.
            # But we must maintain validity.
            
            # If no replacement found, we are likely in a local minimum.
            # We can try to "shake" by randomly replacing a block with a random valid block
            # that covers its unique tsubs, even if it's not the "best" one.
            # We already did that by picking the first candidate.
            
            # If we still can't find a candidate, it means for every block, the set of unique
            # tsubs has size > k, or the only candidate is the block itself.
            # If unique tsubs size > k, the block is "rigid" and cannot be replaced.
            
            # To escape, we might need to accept a temporary increase in blocks?
            # No, the evaluator wants minimum blocks.
            
            # Let's try one more thing: Randomly select a block, and try to replace it
            # with a block that covers its unique tsubs, but allow the candidate to be
            # chosen from a wider set (all valid completions) and pick randomly.
            
            if random.random() < 0.1:
                # Perturbation: Pick a random block, find all valid replacements, pick one randomly.
                idx = random.randint(0, len(blocks)-1)
                unique_tsubs = get_unique_tsubs(idx)
                must_include = set()
                for tsub in unique_tsubs:
                    must_include.update(tsub)
                
                if len(must_include) <= k:
                    remaining_points = [p for p in range(v) if p not in must_include]
                    need_more = k - len(must_include)
                    candidates = []
                    for comb in itertools.combinations(remaining_points, need_more):
                        new_block = tuple(sorted(list(must_include) + list(comb)))
                        if new_block != tuple(blocks[idx]):
                            candidates.append(new_block)
                    
                    if candidates:
                        new_block = random.choice(candidates)
                        old_block = blocks[idx]
                        old_tsubs = block_tsubs[idx]
                        
                        for tsub in old_tsubs:
                            tsub_count[tsub] -= 1
                            
                        new_tsubs = list(itertools.combinations(new_block, t))
                        for tsub in new_tsubs:
                            tsub_count[tsub] += 1
                            
                        blocks[idx] = list(new_block)
                        block_tsubs[idx] = new_tsubs
                        replaced = True
                        break

    return blocks


def construct(v, k, t):
    """
    Construct a covering design C(v, k, t).
    Returns a list of blocks, each a list of k distinct integers in 0..v-1.
    """
    global _START_TIME
    _START_TIME = time.time()
    
    # Phase 1: Greedy construction
    blocks = _greedy_construct(v, k, t)
    
    if _time_left() < 10.0:
        return blocks
    
    # Phase 2: Initial redundancy removal
    blocks, _ = _remove_redundant_blocks(blocks, v, k, t)
    
    if _time_left() < 10.0:
        return blocks
    
    # Phase 3: Local search
    blocks = _local_search(blocks, v, k, t)
    
    # Final redundancy check
    blocks, _ = _remove_redundant_blocks(blocks, v, k, t)
    
    return blocks
