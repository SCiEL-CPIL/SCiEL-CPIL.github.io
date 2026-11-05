import itertools
import time
import sys

def construct():
    """
    Constructs the Steiner System S(5,6,12) (Witt Design) on 12 points.
    Returns a list of 132 distinct 6-subsets of {0, ..., 11}.
    """
    
    # Increase recursion limit to be safe
    sys.setrecursionlimit(10000)
    
    start_time = time.time()
    DEADLINE = 100.0 # 100s safety margin

    def find_s4511():
        points = list(range(11))
        all_4sets = list(itertools.combinations(points, 4))
        N_4SETS = len(all_4sets)
        
        # Map 4-set to index
        four_set_index = {s: i for i, s in enumerate(all_4sets)}
        
        # Generate all 5-sets
        all_5sets = list(itertools.combinations(points, 5))
        
        # Pre-compute bitmask for each 5-set
        # Bit i is set if the 5-set contains the 4-set at index i
        five_masks = {}
        for b5 in all_5sets:
            mask = 0
            for s4 in itertools.combinations(b5, 4):
                idx = four_set_index[s4]
                mask |= (1 << idx)
            five_masks[b5] = mask
            
        # Map 4-set to list of 5-sets containing it
        four_to_fives = {s: [] for s in all_4sets}
        for b5 in all_5sets:
            for s4 in itertools.combinations(b5, 4):
                four_to_fives[s4].append(b5)
        
        covered_mask = 0
        blocks = []
        
        def backtrack():
            nonlocal covered_mask
            if time.time() - start_time > DEADLINE:
                return False # Timeout
            
            # Find the first uncovered 4-set
            target_idx = -1
            # We can optimize this search by maintaining a list of uncovered indices,
            # but 330 is small. A simple loop is fine.
            for i in range(N_4SETS):
                if not (covered_mask & (1 << i)):
                    target_idx = i
                    break
            
            if target_idx == -1:
                return True
            
            target_4set = all_4sets[target_idx]
            candidates = four_to_fives[target_4set]
            
            for b5 in candidates:
                req_mask = five_masks[b5]
                
                # Check if any required 4-set is already covered
                if covered_mask & req_mask:
                    continue
                
                # Apply choice
                covered_mask |= req_mask
                blocks.append(b5)
                
                if backtrack():
                    return True
                
                # Undo choice
                blocks.pop()
                covered_mask &= ~req_mask
            
            return False
        
        # Fix first block to reduce symmetry
        # We must ensure the first block's mask is set
        first_block = (0, 1, 2, 3, 4)
        first_mask = five_masks[first_block]
        covered_mask |= first_mask
        blocks.append(first_block)
        
        if backtrack():
            return blocks
        else:
            return None

    s4511_blocks = find_s4511()
    
    if s4511_blocks is None:
        return []

    w12_blocks = []
    all_points_11 = set(range(11))
    
    for b in s4511_blocks:
        w12_blocks.append(tuple(sorted(list(b) + [11])))
        comp = tuple(sorted(all_points_11 - set(b)))
        w12_blocks.append(comp)
    
    return [list(b) for b in w12_blocks]

if __name__ == "__main__":
    blocks = construct()
    print(f"Constructed {len(blocks)} blocks")
