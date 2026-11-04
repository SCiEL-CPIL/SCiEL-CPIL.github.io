"""Covering design C(v,k,t) solver using efficient greedy with time management."""
import time
import itertools
import numpy as np

def construct(v, k, t):
    deadline = time.time() + 110  # 110s safety margin within 120s
    
    # Precompute all t-subsets and assign indices
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_index = {s: i for i, s in enumerate(all_tsubs)}
    
    # uncovered: 1 if t-subset not yet covered
    uncovered = np.ones(n_tsubs, dtype=np.int32)
    n_uncovered = n_tsubs
    
    blocks = []
    
    while n_uncovered > 0:
        if time.time() > deadline:
            break
        
        # Start a new block
        # Pick the first uncovered t-subset to anchor
        # Find an uncovered t-subset
        unc_idx = np.where(uncovered == 1)[0]
        if len(unc_idx) == 0:
            break
        anchor = list(all_tsubs[unc_idx[0]])
        block = list(anchor)  # start with t points
        
        # Now add points until block has k points
        while len(block) < k:
            if time.time() > deadline:
                break
            
            b = len(block)
            block_set = set(block)
            
            if b < t - 1:
                # Can't cover any new t-subsets yet; pick point that appears in most uncovered t-subs
                # that contain all of block
                # Actually just pick a point not in block that maximizes potential
                # Simple: pick the point that appears in most uncovered t-subsets containing block
                best_x = None
                best_count = -1
                for x in range(v):
                    if x in block_set:
                        continue
                    # Count uncovered t-subsets that contain block and x
                    # t-subset S contains block (size b) and x: S = block ∪ {x} ∪ (t-b-1 more from remaining)
                    # But we just want a heuristic. Count uncovered t-subs S where block ⊆ S and x ∈ S
                    cnt = 0
                    for i in unc_idx:
                        s = all_tsubs[i]
                        if all(p in s for p in block) and x in s:
                            cnt += 1
                    if cnt > best_count:
                        best_count = cnt
                        best_x = x
                if best_x is None:
                    best_x = next(x for x in range(v) if x not in block_set)
                block.append(best_x)
            else:
                # b >= t-1: adding x covers t-subsets S ⊆ block∪{x} with x∈S
                # These are: choose t-1 from block, add x. Count = C(b, t-1)
                # Among these, count how many are uncovered
                best_x = None
                best_gain = -1
                for x in range(v):
                    if x in block_set:
                        continue
                    gain = 0
                    # t-subsets = (t-1 elements from block) + x
                    for comb in itertools.combinations(block, t - 1):
                        s = tuple(sorted(comb + (x,)))
                        idx = tsub_index.get(s)
                        if idx is not None and uncovered[idx] == 1:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                if best_x is None:
                    best_x = next(x for x in range(v) if x not in block_set)
                block.append(best_x)
        
        if len(block) < k:
            break
        
        block.sort()
        blocks.append(block)
        
        # Update uncovered
        for s in itertools.combinations(block, t):
            idx = tsub_index[s]
            if uncovered[idx] == 1:
                uncovered[idx] = 0
                n_uncovered -= 1
    
    return blocks
