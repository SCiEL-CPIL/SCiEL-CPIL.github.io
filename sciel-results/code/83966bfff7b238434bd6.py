import time
import itertools
from collections import defaultdict

def construct():
    """
    Returns the 132 blocks of the Steiner System S(5, 6, 12).
    
    Uses a backtracking search with strong symmetry breaking and MRV heuristic.
    The optimum is 132 blocks, attained exactly by the Witt design S(5,6,12).
    """
    
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    points = range(12)
    
    # Precompute all 5-subsets and map to IDs
    all_5subsets = list(itertools.combinations(points, 5))
    five_to_id = {sub: i for i, sub in enumerate(all_5subsets)}
    n_fives = len(all_5subsets)  # 792
    
    # Precompute all 6-subsets and their constituent 5-subsets
    all_6subsets = list(itertools.combinations(points, 6))
    n_sixes = len(all_6subsets)  # 924
    
    six_to_fives = []
    for block in all_6subsets:
        fives = []
        for k in range(6):
            sub = block[:k] + block[k+1:]
            fives.append(five_to_id[sub])
        six_to_fives.append(fives)
        
    # For each 5-subset, list of 6-subset indices that contain it
    five_to_six = defaultdict(list)
    for i, fives in enumerate(six_to_fives):
        for fid in fives:
            five_to_six[fid].append(i)

    # Symmetry breaking: fix first block to {0, 1, 2, 3, 4, 5}
    first_block = (0, 1, 2, 3, 4, 5)
    first_idx = all_6subsets.index(first_block)
    
    # State
    used_fives = set(six_to_fives[first_idx])
    blocks_found = [first_block]
    
    # Precompute for each 6-subset a bitmask or set of its 5-subsets for fast validity check
    # We'll use a list of frozensets for faster membership testing
    six_to_fives_set = [frozenset(fives) for fives in six_to_fives]
    
    def is_valid_block(idx):
        """Check if block idx is valid (all its 5-subsets are unused)."""
        for f in six_to_fives_set[idx]:
            if f in used_fives:
                return False
        return True
    
    def backtrack():
        if time.time() > deadline:
            return False
            
        if len(blocks_found) == 132:
            return True
            
        # MRV: find the 5-subset with fewest valid blocks
        min_count = float('inf')
        best_fid = -1
        
        for fid in range(n_fives):
            if fid in used_fives:
                continue
                
            count = 0
            for idx in five_to_six[fid]:
                if is_valid_block(idx):
                    count += 1
                    
            if count == 0:
                return False  # Dead end
                
            if count < min_count:
                min_count = count
                best_fid = fid
                if count == 1:
                    break  # Best possible
            
        if best_fid == -1:
            # All 5-subsets used, must have 132 blocks
            return len(blocks_found) == 132
            
        # Try all valid blocks containing best_fid
        candidates = five_to_six[best_fid]
        
        for idx in candidates:
            if not is_valid_block(idx):
                continue
                
            # Apply
            blocks_found.append(all_6subsets[idx])
            for f in six_to_fives_set[idx]:
                used_fives.add(f)
                
            if backtrack():
                return True
                
            # Undo
            blocks_found.pop()
            for f in six_to_fives_set[idx]:
                used_fives.discard(f)
                
        return False

    if backtrack():
        return [list(b) for b in blocks_found]
        
    # Fallback: return what we have
    return [list(b) for b in blocks_found]

if __name__ == '__main__':
    res = construct()
    print(f"Number of blocks: {len(res)}")
    if len(res) == 132:
        print("Success: Found 132 blocks")
    else:
        print("Warning: Did not find full solution")
