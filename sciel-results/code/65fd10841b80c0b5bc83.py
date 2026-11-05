import itertools
import time

def construct():
    """
    Constructs the Steiner System S(5,6,12) (Witt Design) on 12 points.
    Returns a list of 132 distinct 6-subsets of {0, ..., 11}.
    """
    
    # Step 1: Construct S(4,5,11)
    # We need a collection of 5-subsets of {0..10} such that every 4-subset is in exactly one block.
    # Total 4-subsets: C(11, 4) = 330.
    # Each 5-block covers 5 4-subsets.
    # Number of blocks: 330 / 5 = 66.
    
    def find_s4511():
        points = list(range(11))
        all_4sets = list(itertools.combinations(points, 4))
        
        # Map 4-set to index for fast lookup
        four_set_index = {s: i for i, s in enumerate(all_4sets)}
        
        # Pre-compute the 4-subsets of any 5-set and their indices
        all_5sets = list(itertools.combinations(points, 5))
        five_to_four_indices = {}
        for b5 in all_5sets:
            subs = list(itertools.combinations(b5, 4))
            indices = [four_set_index[s] for s in subs]
            five_to_four_indices[b5] = indices
        
        # Backtracking state
        covered_4sets = [False] * 330
        blocks = []
        
        # Recursive helper
        def backtrack():
            # Find the first uncovered 4-set
            target_idx = -1
            for i in range(330):
                if not covered_4sets[i]:
                    target_idx = i
                    break
            
            if target_idx == -1:
                return True
            
            target_4set = all_4sets[target_idx]
            
            # Find all 5-sets that contain this 4-set
            # We can pre-compute a map: 4-set -> list of 5-sets
            # But doing it on the fly with a pre-computed list of candidates is faster.
            
            # Optimization: Pre-compute candidates for each 4-set
            # candidates_for_4set[target_idx] = list of b5 tuples
            
            # For now, filter all_5sets (56 of them)
            # This is small enough.
            for b5 in all_5sets:
                if target_4set[0] in b5 and target_4set[1] in b5 and target_4set[2] in b5 and target_4set[3] in b5:
                    sub_indices = five_to_four_indices[b5]
                    valid = True
                    for idx in sub_indices:
                        if covered_4sets[idx]:
                            valid = False
                            break
                    
                    if not valid:
                        continue
                    
                    # Place the block
                    for idx in sub_indices:
                        covered_4sets[idx] = True
                    blocks.append(b5)
                    
                    if backtrack():
                        return True
                    
                    # Undo
                    blocks.pop()
                    for idx in sub_indices:
                        covered_4sets[idx] = False
            
            return False
        
        # Initialize with the first block fixed
        first_block = (0, 1, 2, 3, 4)
        first_indices = five_to_four_indices[first_block]
        for idx in first_indices:
            covered_4sets[idx] = True
        blocks.append(first_block)
        
        if backtrack():
            return blocks
        else:
            return None

    s4511_blocks = find_s4511()
    
    if s4511_blocks is None:
        # Fallback: This should not happen.
        # If it does, return a valid smaller packing? No, return empty.
        return []

    # Step 2: Construct S(5,6,12) from S(4,5,11)
    # Points 0..10 in S(4,5,11), plus point 11.
    # For each block B in S(4,5,11):
    # 1. B + {11}
    # 2. (0..10 \ B)
    
    w12_blocks = []
    all_points_11 = set(range(11))
    
    for b in s4511_blocks:
        # b is a tuple of 5 elements from 0..10
        # Block 1: b plus 11
        w12_blocks.append(tuple(sorted(list(b) + [11])))
        
        # Block 2: complement of b in 0..10
        comp = tuple(sorted(all_points_11 - set(b)))
        w12_blocks.append(comp)
    
    # The result should be 132 blocks
    # Convert to list of lists as per interface
    return [list(b) for b in w12_blocks]

if __name__ == "__main__":
    # Quick self-test
    blocks = construct()
    print(f"Constructed {len(blocks)} blocks")
    # Verify validity
    if len(blocks) == 132:
        seen_5sets = set()
        valid = True
        for b in blocks:
            if len(set(b)) != 6:
                valid = False
                break
            for s in itertools.combinations(b, 5):
                if s in seen_5sets:
                    valid = False
                    break
                seen_5sets.add(s)
            if not valid:
                break
        print(f"Valid: {valid}, Unique 5-sets: {len(seen_5sets)}")
    else:
        print("Invalid count")
