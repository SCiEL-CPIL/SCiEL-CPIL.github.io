import itertools
import random
import numpy as np
from collections import defaultdict
import time

def construct(v, k, t):
    """Construct a covering design C(v,k,t) using VND heuristic.
    
    Args:
        v: number of points
        k: block size
        t: subset size to cover
    
    Returns:
        List of blocks (each a list of k distinct integers in 0..v-1)
    """
    # Precompute all t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    num_tsubs = len(all_tsubs)
    
    # Map t-subset to index
    tsub_to_idx = {tuple(cs): i for i, cs in enumerate(all_tsubs)}
    
    # Precompute for each point, the list of t-subset indices containing it
    point_tsubs = [[] for _ in range(v)]
    for idx, cs in enumerate(all_tsubs):
        for p in cs:
            point_tsubs[p].append(idx)
            
    # Precompute bitmasks for all t-subsets
    tsub_masks = [0] * num_tsubs
    for idx, cs in enumerate(all_tsubs):
        mask = 0
        for p in cs:
            mask |= (1 << p)
        tsub_masks[idx] = mask

    # Greedy Construction
    uncovered = np.ones(num_tsubs, dtype=bool)
    blocks = []
    
    while uncovered.any():
        # Find first uncovered
        idx = int(np.argmax(uncovered))
        if not uncovered[idx]:
            break
        target = all_tsubs[idx]
        block = list(target)
        block_set = set(block)
        block_mask = 0
        for p in block:
            block_mask |= (1 << p)
        
        while len(block) < k:
            gains = np.zeros(v, dtype=np.int32)
            
            # Iterate through points not in block
            # For each candidate p, check t-subsets containing p
            for p in range(v):
                if p in block_set:
                    continue
                for ts_idx in point_tsubs[p]:
                    if not uncovered[ts_idx]:
                        continue
                    # Check if ts_idx is covered by adding p to block
                    # ts_idx is covered if (tsub_masks[ts_idx] & ~block_mask) has only bit p set?
                    # No, ts_idx is covered by block U {p} if tsub_masks[ts_idx] is subset of (block_mask | (1<<p))
                    # Which is equivalent to (tsub_masks[ts_idx] & ~(block_mask | (1<<p))) == 0
                    # Since p is in ts_idx, we know bit p is in tsub_masks[ts_idx].
                    # So we just need to check if the rest of ts_idx is in block.
                    # (tsub_masks[ts_idx] & ~(block_mask | (1<<p))) == 0
                    if (tsub_masks[ts_idx] & ~(block_mask | (1 << p))) == 0:
                        gains[p] += 1
            
            best_p = -1
            best_gain = -1
            for p in range(v):
                if p not in block_set:
                    if gains[p] > best_gain:
                        best_gain = gains[p]
                        best_p = p
            
            if best_p == -1 or best_gain == 0:
                for p in range(v):
                    if p not in block_set:
                        best_p = p
                        break
                if best_p == -1:
                    break
            
            block.append(best_p)
            block_set.add(best_p)
            block_mask |= (1 << best_p)
            
            # Update uncovered mask for t-subsets covered by the new point
            # We can do this at the end of the block construction to save time?
            # But we need `uncovered` to be accurate for the next point selection.
            # However, `uncovered` is global. If we don't update it, the next point selection
            # will see t-subsets that are already covered by the current block as "uncovered".
            # The gain calculation `gains[p]` checks `if not uncovered[ts_idx]: continue`.
            # So if we don't update, we might count gains for already covered t-subsets.
            # But the condition `(tsub_masks[ts_idx] & ~(block_mask | (1 << p))) == 0` ensures
            # that the t-subset is covered by the NEW block (current partial + p).
            # If it was already covered by the previous partial block, it would satisfy the condition
            # for the previous point too.
            # Wait, if a t-subset is covered by the current partial block, it should not be in `uncovered`.
            # So we MUST update `uncovered` after each point addition to keep it accurate.
            
            # Update uncovered
            for ts_idx in point_tsubs[best_p]:
                if (tsub_masks[ts_idx] & ~block_mask) == 0:
                    uncovered[ts_idx] = False

        block.sort()
        blocks.append(block)
        
        # Final update of uncovered for the whole block (redundant but safe)
        for ts_idx in range(num_tsubs):
            if (tsub_masks[ts_idx] & ~block_mask) == 0:
                uncovered[ts_idx] = False

    # VND Phase
    cov_count = np.zeros(num_tsubs, dtype=np.int32)
    for block in blocks:
        block_mask = 0
        for p in block:
            block_mask |= (1 << p)
        for ts_idx in range(num_tsubs):
            if (tsub_masks[ts_idx] & ~block_mask) == 0:
                cov_count[ts_idx] += 1

    def try_replace_block(i):
        old_block = blocks[i]
        old_mask = 0
        for p in old_block:
            old_mask |= (1 << p)
        
        # Find t-subsets uniquely covered by old_block
        must_cover = []
        for ts_idx in range(num_tsubs):
            if cov_count[ts_idx] == 1:
                if (tsub_masks[ts_idx] & ~old_mask) == 0:
                    must_cover.append(ts_idx)
        
        union_points = set()
        for ts_idx in must_cover:
            union_points.update(all_tsubs[ts_idx])
            
        if len(union_points) > k:
            return False
        
        new_block = list(union_points)
        new_block_set = set(new_block)
        new_mask = 0
        for p in new_block:
            new_mask |= (1 << p)
        
        while len(new_block) < k:
            best_p = -1
            best_score = -1
            for p in range(v):
                if p in new_block_set:
                    continue
                score = 0
                for ts_idx in point_tsubs[p]:
                    # Check if ts_idx is covered by adding p
                    if (tsub_masks[ts_idx] & ~(new_mask | (1 << p))) == 0:
                        # It is covered
                        # Weight based on cov_count
                        if cov_count[ts_idx] == 0:
                            score += 10
                        elif cov_count[ts_idx] == 1:
                            score += 5
                        else:
                            score += 1
                if score > best_score:
                    best_score = score
                    best_p = p
            
            if best_p == -1:
                for p in range(v):
                    if p not in new_block_set:
                        best_p = p
                        break
                if best_p == -1:
                    break
            
            new_block.append(best_p)
            new_block_set.add(best_p)
            new_mask |= (1 << best_p)
            
            # Update local view of coverage? 
            # We don't have a local uncovered array here.
            # The score calculation uses `cov_count` which is global.
            # This is fine, we are trying to improve global coverage.
        
        if len(new_block) < k:
            return False
        
        new_block.sort()
        
        # Verify must_cover
        for ts_idx in must_cover:
            if (tsub_masks[ts_idx] & ~new_mask) != 0:
                return False
        
        # Check validity: no t-subset covered by old_block with cov_count 1 should be lost
        for ts_idx in range(num_tsubs):
            if (tsub_masks[ts_idx] & ~old_mask) == 0:
                if cov_count[ts_idx] == 1 and (tsub_masks[ts_idx] & ~new_mask) != 0:
                    return False
        
        # Calculate improvement
        old_cov_count = 0
        for ts_idx in range(num_tsubs):
            if (tsub_masks[ts_idx] & ~old_mask) == 0:
                old_cov_count += 1
                
        new_cov_count = 0
        for ts_idx in range(num_tsubs):
            if (tsub_masks[ts_idx] & ~new_mask) == 0:
                new_cov_count += 1
        
        if new_cov_count >= old_cov_count:
            # Update cov_count
            for ts_idx in range(num_tsubs):
                if (tsub_masks[ts_idx] & ~old_mask) == 0:
                    cov_count[ts_idx] -= 1
                if (tsub_masks[ts_idx] & ~new_mask) == 0:
                    cov_count[ts_idx] += 1
            blocks[i] = new_block
            return True
        return False

    improved = True
    while improved:
        improved = False
        for i in range(len(blocks)):
            if try_replace_block(i):
                improved = True

    return blocks
