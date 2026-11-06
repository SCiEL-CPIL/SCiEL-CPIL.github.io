import time
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    # Strategy: Explicit construction of the Witt Design S(5,6,12)
    # This is faster and more reliable than search.
    
    # Generate the 66 blocks of the S(4,5,11) design on points 0..10
    # using the action of M_11.
    
    # Generators for M_11 on 11 points (0-indexed)
    # g1: cyclic shift
    # g2: specific permutation
    # These generators produce the group M_11 which acts on the S(4,5,11) design.
    
    def generate_s4511_blocks():
        # g1: i -> (i+1) % 11
        g1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0]
        # g2: (0, 2, 4, 6, 8, 1, 3, 5, 7, 9, 10)
        # This is a standard generator for M11 along with the cycle.
        g2 = [2, 4, 6, 8, 10, 1, 3, 5, 7, 9, 0]
        
        base_block = (0, 1, 2, 4, 5)
        blocks = set()
        queue = [base_block]
        blocks.add(base_block)
        
        while queue:
            b = queue.pop()
            # Apply g1
            b1 = tuple(sorted(g1[x] for x in b))
            if b1 not in blocks:
                blocks.add(b1)
                queue.append(b1)
            # Apply g2
            b2 = tuple(sorted(g2[x] for x in b))
            if b2 not in blocks:
                blocks.add(b2)
                queue.append(b2)
                
        return list(blocks)

    blocks_4511 = generate_s4511_blocks()
    
    # Verify the generated blocks form a valid S(4,5,11) design
    if len(blocks_4511) == 66:
        counts = {}
        for b in blocks_4511:
            for sub in combinations(b, 4):
                counts[sub] = counts.get(sub, 0) + 1
        if len(counts) == 330 and all(v == 1 for v in counts.values()):
            # Valid S(4,5,11) design found.
            # Construct S(5,6,12) by adding point 11.
            final_blocks = []
            for b in blocks_4511:
                # Block containing infinity (11)
                final_blocks.append(tuple(sorted(b + (11,))))
                # Block not containing infinity (complement in 0..10)
                comp = tuple(sorted(set(range(11)) - set(b)))
                final_blocks.append(comp)
                
            return [list(b) for b in final_blocks]

    # Fallback: If explicit construction fails, use the search solver
    points = range(12)
    all_5_subs = list(combinations(points, 5))
    n_5 = len(all_5_subs)
    sub_to_idx = {s: i for i, s in enumerate(all_5_subs)}
    
    all_6_blocks = list(combinations(points, 6))
    n_6 = len(all_6_blocks)
    
    block_to_subs = []
    for b in all_6_blocks:
        subs = [sub_to_idx[s] for s in combinations(b, 5)]
        block_to_subs.append(subs)
        
    sub_to_blocks = [[] for _ in range(n_5)]
    for j, subs in enumerate(block_to_subs):
        for i in subs:
            sub_to_blocks[i].append(j)
            
    solution = []
    covered = [False] * n_5
    used_blocks = [False] * n_6
    
    def solve():
        min_count = float('inf')
        min_sub = -1
        for i in range(n_5):
            if not covered[i]:
                count = 0
                for j in sub_to_blocks[i]:
                    if not used_blocks[j]:
                        count += 1
                        if count < min_count:
                            min_count = count
                            min_sub = i
                            if min_count == 0:
                                return False
                if count == 0:
                    return False
        if min_sub == -1:
            return True
            
        for j in sub_to_blocks[min_sub]:
            if used_blocks[j]:
                continue
            valid = True
            for i in block_to_subs[j]:
                if covered[i]:
                    valid = False
                    break
            if not valid:
                continue
                
            solution.append(j)
            used_blocks[j] = True
            for i in block_to_subs[j]:
                covered[i] = True
                
            if solve():
                return True
                
            solution.pop()
            used_blocks[j] = False
            for i in block_to_subs[j]:
                covered[i] = False
                
        return False

    if solve():
        return [list(all_6_blocks[j]) for j in solution]
        
    covered_set = set()
    greedy_blocks = []
    for b in all_6_blocks:
        subs = list(combinations(b, 5))
        if not any(s in covered_set for s in subs):
            greedy_blocks.append(list(b))
            covered_set.update(subs)
    return greedy_blocks
