import time
from itertools import combinations

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0  # 10 second margin
    
    # Method 1: Algebraic Construction via M12 Group Orbit
    # Generators for M12 on points 0..11
    # g1 = (0,1,2,3,4,5)(6,7,8,9,10,11)
    g1 = (1, 2, 3, 4, 5, 0, 7, 8, 9, 10, 11, 6)
    # g2 = (0,6,10)(1,7,11)(2,8,9)
    g2 = (6, 7, 8, 3, 4, 5, 10, 11, 9, 2, 0, 1)
    # g3 = (0,1)(2,3)(4,5)(6,7)(8,9)(10,11)
    g3 = (1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10)
    
    generators = [g1, g2, g3]
    identity = tuple(range(12))
    
    # BFS to generate the group
    group_elements = set([identity])
    queue = [identity]
    
    def compose(p, q):
        # p o q: apply q then p
        # p[i] is the image of i under p
        # q[i] is the image of i under q
        # (p o q)[i] = p[q[i]]
        return tuple(p[q[i]] for i in range(12))
        
    while queue:
        if time.time() > deadline:
            break
        g = queue.pop(0)
        for h in generators:
            gh = compose(g, h)
            if gh not in group_elements:
                group_elements.add(gh)
                queue.append(gh)
            hg = compose(h, g)
            if hg not in group_elements:
                group_elements.add(hg)
                queue.append(hg)
                
    # Generate orbit of base block (0,1,2,3,4,5)
    base_block = (0, 1, 2, 3, 4, 5)
    blocks = set()
    for g in group_elements:
        # Apply g to base_block
        # g maps i -> g[i]
        # We want the set { g[i] for i in base_block }
        mapped = tuple(sorted(g[i] for i in base_block))
        blocks.add(mapped)
        
    # Check if we got the full Steiner system
    if len(blocks) == 132:
        # Verify validity (optional but good for safety)
        # Each 5-subset must appear exactly once
        five_subsets = set()
        valid = True
        for b in blocks:
            for s in combinations(b, 5):
                if s in five_subsets:
                    valid = False
                    break
                five_subsets.add(s)
            if not valid:
                break
        if valid:
            return [list(b) for b in blocks]
            
    # Fallback: Greedy Lexicographic (same as seed)
    # This ensures we return *something* valid, even if not optimal
    covered, greedy_blocks = set(), []
    for b in combinations(range(12), 6):
        if time.time() > deadline:
            break
        subsets = list(combinations(b, 5))
        if not any(s in covered for s in subsets):
            greedy_blocks.append(list(b))
            covered.update(subsets)
            
    return greedy_blocks

