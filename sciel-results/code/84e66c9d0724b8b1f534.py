import itertools
import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # Step 1: Try to find S(4,5,11)
    # S(4,5,11) has 66 blocks on 11 points.
    # Each 4-subset is in exactly one block.
    # Total 4-subsets: C(11,4) = 330.
    # Each block covers C(5,4) = 5 4-subsets.
    # 330 / 5 = 66 blocks.
    
    def find_s4511():
        # Randomized greedy
        # We want to pack 5-sets such that no 4-set is repeated.
        # There are 330 4-sets.
        # We pick a 5-set, add it, remove its 5 4-sets from the "available" pool.
        # If we can pick 66 blocks, we succeeded.
        
        # To make it fast, we can precompute all 5-sets and their 4-subsets.
        points = range(11)
        all_5sets = list(itertools.combinations(points, 5))
        # Map 4-set to a canonical form for quick lookup
        # We will maintain a set of covered 4-sets.
        
        # Strategy: 
        # 1. Shuffle 5-sets.
        # 2. Iterate, if all 4-subsets of the 5-set are uncovered, add it.
        # 3. If we reach 66, return the blocks.
        
        # This simple greedy might fail if we pick a "bad" block early.
        # But for S(4,5,11), the structure is rigid.
        # Let's try multiple restarts.
        
        for _ in range(1000): # Max 1000 restarts
            if time.time() > deadline:
                return None
            random.shuffle(all_5sets)
            covered_4sets = set()
            blocks = []
            for b in all_5sets:
                subsets = list(itertools.combinations(b, 4))
                if not any(s in covered_4sets for s in subsets):
                    blocks.append(b)
                    covered_4sets.update(subsets)
                    if len(blocks) == 66:
                        return blocks
            # If we finished the loop without reaching 66, restart
        return None

    s4511_blocks = find_s4511()
    
    if s4511_blocks:
        # Construct S(5,6,12) from S(4,5,11)
        # Points 0..10, plus 11.
        # For each B in S(4,5,11):
        # 1. B + {11}
        # 2. (0..10 \ B)
        
        w12_blocks = []
        for b in s4511_blocks:
            # b is a tuple of 5 elements from 0..10
            w12_blocks.append(tuple(sorted(list(b) + [11])))
            # Complement in 0..10
            comp = tuple(sorted(set(range(11)) - set(b)))
            w12_blocks.append(comp)
        
        # Verify validity and count
        # Should be 132 blocks
        if len(w12_blocks) == 132:
            # Optional: Verify no 5-subset is repeated
            # This is guaranteed by the construction if S(4,5,11) is valid.
            return [list(b) for b in w12_blocks]
            
    # Fallback: Direct greedy for S(5,6,12)
    # This is the current behavior, but with random restarts.
    
    def find_s5612_greedy():
        points = range(12)
        all_6sets = list(itertools.combinations(points, 6))
        # Total 6-sets: C(12,6) = 924
        # Each covers C(6,5) = 6 5-sets.
        # Total 5-sets: C(12,5) = 792.
        # Max blocks: 792 / 6 = 132.
        
        best_blocks = []
        
        for _ in range(100): # 100 restarts
            if time.time() > deadline:
                break
            random.shuffle(all_6sets)
            covered_5sets = set()
            blocks = []
            for b in all_6sets:
                subsets = list(itertools.combinations(b, 5))
                if not any(s in covered_5sets for s in subsets):
                    blocks.append(b)
                    covered_5sets.update(subsets)
            if len(blocks) > len(best_blocks):
                best_blocks = blocks
                if len(best_blocks) == 132:
                    break
        return best_blocks

    fallback_blocks = find_s5612_greedy()
    return [list(b) for b in fallback_blocks]
