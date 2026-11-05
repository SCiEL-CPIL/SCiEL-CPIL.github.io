"""Direct construction of the Witt design S(5,6,12) with 132 blocks.

We construct the S(5,6,12) Steiner system deterministically using backtracking
with strong symmetry breaking. The design is unique up to isomorphism, so any
valid 132-block packing IS the Witt design.
"""

import time
from itertools import combinations


def construct():
    """Construct a packing of 6-subsets of {0,...,11} with no repeated 5-subset.
    
    Returns the 132-block Witt design S(5,6,12), maximizing the number of blocks.
    """
    points = list(range(12))
    all_5subs = list(combinations(points, 5))
    all_6subs = list(combinations(points, 6))
    
    # Map each 5-subset to the 6-subsets containing it
    five_to_six = {}
    for s in all_5subs:
        s_set = set(s)
        five_to_six[s] = [b for b in all_6subs if s_set <= set(b)]
    
    # Map each 6-subset to its 5-subsets
    six_to_fives = {}
    for b in all_6subs:
        six_to_fives[b] = list(combinations(b, 5))
    
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    # Symmetry breaking: fix first block as (0,1,2,3,4,5)
    first_block = (0, 1, 2, 3, 4, 5)
    covered = set(six_to_fives[first_block])
    used_blocks = {first_block}
    blocks = [first_block]
    
    # Track uncovered 5-subsets for efficiency
    remaining_5 = [s for s in all_5subs if s not in covered]
    
    def backtrack():
        nonlocal covered, used_blocks, blocks, remaining_5
        
        if time.time() > deadline:
            return False
        
        if len(blocks) == 132:
            return len(covered) == 792
        
        # Find most constrained uncovered 5-subset (MRV heuristic)
        best_idx = -1
        best_cands = None
        best_count = float('inf')
        
        for i, s in enumerate(remaining_5):
            if s in covered:
                continue
            cands = [b for b in five_to_six[s] if b not in used_blocks]
            cnt = len(cands)
            if cnt == 0:
                return False
            if cnt < best_count:
                best_count = cnt
                best_idx = i
                best_cands = cands
                if cnt == 1:
                    break
        
        if best_idx == -1:
            return len(covered) == 792
        
        s = remaining_5[best_idx]
        
        for b in best_cands:
            b_fives = six_to_fives[b]
            # Check for conflicts
            conflict = False
            for bs in b_fives:
                if bs in covered:
                    conflict = True
                    break
            if conflict:
                continue
            
            # Place block
            for bs in b_fives:
                covered.add(bs)
            used_blocks.add(b)
            blocks.append(b)
            
            if backtrack():
                return True
            
            # Undo
            for bs in b_fives:
                covered.discard(bs)
            used_blocks.discard(b)
            blocks.pop()
            
            if time.time() > deadline:
                return False
        
        return False
    
    if backtrack() and len(blocks) == 132 and len(covered) == 792:
        return [list(b) for b in blocks]
    
    # Fallback: return what we have
    return [list(b) for b in blocks]


if __name__ == "__main__":
    import time
    start = time.time()
    res = construct()
    elapsed = time.time() - start
    print(f"Blocks: {len(res)}, Time: {elapsed:.3f}s")
    
    seen = set()
    valid = True
    block_set = set()
    for b in res:
        b_tuple = tuple(sorted(b))
        if b_tuple in block_set:
            valid = False
            break
        block_set.add(b_tuple)
        for s in combinations(b, 5):
            if s in seen:
                valid = False
                break
            seen.add(s)
        if not valid:
            break
    print(f"Valid: {valid}")
    print(f"5-subsets covered: {len(seen)} / 792")
    print(f"Blocks: {len(res)} / 132")
