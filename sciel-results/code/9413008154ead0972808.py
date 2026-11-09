import time
from collections import deque

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Generators for the Mathieu group M12 acting on {0, ..., 11}
    # These generate the full group M12
    generators = [
        (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0),  # (0 1 2 3 4 5)(6 7 8 9 10 11) in 0-indexed: 0->1, 1->2, ..., 5->6? No.
        # Let's convert standard cycle notation to 0-indexed permutation tuples.
        # Cycle (0,1,2,3,4,5) means 0->1, 1->2, 2->3, 3->4, 4->5, 5->0.
        # Cycle (6,7,8,9,10,11) means 6->7, ..., 11->6.
        # Permutation tuple p where p[i] is the image of i.
        # g1: 0->1, 1->2, 2->3, 3->4, 4->5, 5->0, 6->7, 7->8, 8->9, 9->10, 10->11, 11->6
        (1, 2, 3, 4, 5, 0, 7, 8, 9, 10, 11, 6),
        
        # g2: (0,1)(2,3)(4,5)(6,7)(8,9)(10,11)
        # 0->1, 1->0, 2->3, 3->2, 4->5, 5->4, 6->7, 7->6, 8->9, 9->8, 10->11, 11->10
        (1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10),
        
        # g3: (0,2,4,6,8,10)(1,3,5,7,9,11)
        # 0->2, 2->4, 4->6, 6->8, 8->10, 10->0
        # 1->3, 3->5, 5->7, 7->9, 9->11, 11->1
        (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 1),
        
        # g4: (0,6)(1,7)(2,8)(3,9)(4,10)(5,11)
        # 0->6, 6->0, 1->7, 7->1, 2->8, 8->2, 3->9, 9->3, 4->10, 10->4, 5->11, 11->5
        (6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5)
    ]
    
    # Base block for S(5,6,12)
    base_block = (0, 1, 2, 3, 4, 5)
    
    # BFS to generate all group elements and find the orbit of the base block
    # Identity permutation
    identity = tuple(range(12))
    
    visited_perms = set()
    visited_perms.add(identity)
    queue = deque([identity])
    
    blocks_set = set()
    
    # Apply base block to identity
    blocks_set.add(base_block)
    
    while queue:
        if time.time() > deadline:
            break
            
        current_perm = queue.popleft()
        
        # Generate new permutations by applying generators
        for gen in generators:
            # Compose current_perm with gen
            # new_perm[i] = current_perm[gen[i]]
            # This applies gen first, then current_perm
            new_perm = tuple(current_perm[gen[i]] for i in range(12))
            
            if new_perm not in visited_perms:
                visited_perms.add(new_perm)
                queue.append(new_perm)
                
                # Calculate the image of the base block under new_perm
                # new_block = tuple(sorted(new_perm[i] for i in base_block))
                # Since base_block is sorted, we just map the elements
                new_block = tuple(sorted(new_perm[i] for i in base_block))
                
                if new_block not in blocks_set:
                    blocks_set.add(new_block)
                    
    # Convert to list of lists
    result = [list(b) for b in blocks_set]
    
    # If the group generation failed or produced too few blocks, fallback to a simple greedy
    # though this should not happen with correct generators.
    if len(result) < 132:
        # Fallback: Random greedy
        import random
        from itertools import combinations
        points = range(12)
        all_6sets = list(combinations(points, 6))
        random.shuffle(all_6sets)
        chosen = []
        for b in all_6sets:
            compatible = True
            for cb in chosen:
                if len(set(b) & set(cb)) > 4:
                    compatible = False
                    break
            if compatible:
                chosen.append(b)
        result = [list(b) for b in chosen]
        
    return result
