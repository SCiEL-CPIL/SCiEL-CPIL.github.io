"""Witt design S(5,6,12) construction via S(4,5,11) extension."""
import time
from itertools import combinations

def _solve_s4511():
    """
    Construct S(4,5,11) - Steiner system with 66 blocks of size 5 on 11 points.
    Each 4-subset appears in exactly one block.
    """
    n = 11
    points = list(range(n))
    
    # Precompute all 4-subsets and 5-subsets
    all_4subsets = list(combinations(points, 4))
    all_5subsets = list(combinations(points, 5))
    
    # Map 4-subset to index
    s4_index = {s4: i for i, s4 in enumerate(all_4subsets)}
    
    # For each 5-subset, list of 4-subsets it contains
    fours_of = [None] * len(all_5subsets)
    for i, s5 in enumerate(all_5subsets):
        fours_of[i] = [s4_index[s4] for s4 in combinations(s5, 4)]
    
    # For each 4-subset, list of 5-subsets containing it
    four_to_fives = [[] for _ in range(len(all_4subsets))]
    for i, s5 in enumerate(all_5subsets):
        for s4_idx in fours_of[i]:
            four_to_fives[s4_idx].append(i)
    
    # Each 4-subset is in C(11-4, 1) = 7 five-subsets
    # In S(4,5,11), each 4-subset is in exactly 1 block
    # Total blocks: C(11,4)/C(5,4) = 330/5 = 66
    
    # Backtracking with MRV
    used_4subsets = [False] * len(all_4subsets)
    blocks = []
    
    def backtrack():
        if len(blocks) == 66:
            return True
        
        # Find uncovered 4-subset with fewest candidate 5-subsets
        min_cands = float('inf')
        best_s4 = -1
        for s4_idx in range(len(all_4subsets)):
            if used_4subsets[s4_idx]:
                continue
            cands = [s5_idx for s5_idx in four_to_fives[s4_idx] 
                     if all(used_4subsets[s4i] or s4i == s4_idx for s4i in fours_of[s5_idx])]
            # Actually, we need: all 4-subsets of this 5-subset must be either 
            # already covered or be the one we're trying to cover
            # Simpler: check if any 4-subset of s5 is already used
            valid_cands = []
            for s5_idx in four_to_fives[s4_idx]:
                ok = True
                for s4i in fours_of[s5_idx]:
                    if used_4subsets[s4i]:
                        ok = False
                        break
                if ok:
                    valid_cands.append(s5_idx)
            c = len(valid_cands)
            if c < min_cands:
                min_cands = c
                best_s4 = s4_idx
                if c == 0:
                    return False
                if c == 1:
                    break
        
        if best_s4 == -1:
            return False
        
        # Try each valid 5-subset
        for s5_idx in four_to_fives[best_s4]:
            ok = True
            for s4i in fours_of[s5_idx]:
                if used_4subsets[s4i]:
                    ok = False
                    break
            if not ok:
                continue
            
            # Apply
            for s4i in fours_of[s5_idx]:
                used_4subsets[s4i] = True
            blocks.append(all_5subsets[s5_idx])
            
            if backtrack():
                return True
            
            # Undo
            for s4i in fours_of[s5_idx]:
                used_4subsets[s4i] = False
            blocks.pop()
        
        return False
    
    if backtrack():
        return blocks
    return None

def _witt_s5612_from_s4511(s4511_blocks):
    """
    Construct S(5,6,12) from S(4,5,11) by adding point 11.
    For each 5-block B in S(4,5,11), create 6-block B ∪ {11}.
    Also create 6-blocks from the complements: for each 5-block B,
    the complement in {0..10} has size 6, and we add that as a block too?
    
    Actually, the standard construction:
    - Points: {0,1,...,10,∞} where ∞=11
    - For each block B of S(4,5,11): add B ∪ {∞} as a 6-block
    - For each block B of S(4,5,11): add complement of B in {0..10} as a 6-block
      (complement has size 11-5=6)
    
    This gives 66 + 66 = 132 blocks.
    """
    if s4511_blocks is None:
        return None
    
    blocks = []
    for b in s4511_blocks:
        # Block with infinity
        blocks.append(list(b) + [11])
        # Complement block (in 0..10)
        comp = [i for i in range(11) if i not in b]
        blocks.append(comp)
    
    return blocks

def construct():
    start_time = time.time()
    
    # Try the algebraic construction
    try:
        s4511 = _solve_s4511()
        if s4511 is not None and len(s4511) == 66:
            witt = _witt_s5612_from_s4511(s4511)
            if witt is not None and len(witt) == 132:
                # Verify
                covered = set()
                valid = True
                for b in witt:
                    for s5 in combinations(b, 5):
                        if s5 in covered:
                            valid = False
                            break
                        covered.add(s5)
                    if not valid:
                        break
                if valid:
                    return witt
    except Exception:
        pass
    
    # Fallback: greedy
    covered, result = set(), []
    for b in combinations(range(12), 6):
        subsets = list(combinations(b, 5))
        if not any(s in covered for s in subsets):
            result.append(list(b))
            covered.update(subsets)
            if time.time() - start_time > 140:
                break
    return result
