import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110
    
    # Handle general case if not 24,7,3
    if not (v == 24 and k == 7 and t == 3):
        uncovered = set(itertools.combinations(range(v), t))
        blocks = []
        while uncovered:
            block = list(min(uncovered))
            while len(block) < k:
                def gain(x):
                    return sum(1 for rest in itertools.combinations(block, t - 1)
                               if tuple(sorted(rest + (x,))) in uncovered)
                block.append(max((x for x in range(v) if x not in block), key=gain))
            block.sort()
            blocks.append(block)
            uncovered.difference_update(itertools.combinations(block, t))
        return blocks

    # Specific optimization for 24,7,3
    all_3_subsets = list(itertools.combinations(range(v), 3))
    subset_to_id = {s: i for i, s in enumerate(all_3_subsets)}
    n_subsets = len(all_3_subsets)
    
    def get_id_3(a, b, c):
        if a > b: a, b = b, a
        if b > c: b, c = c, b
        if a > b: a, b = b, a
        return subset_to_id[(a, b, c)]

    best_blocks = None
    best_len = float('inf')
    
    restarts = 0
    while time.time() < deadline:
        restarts += 1
        random.seed(restarts)
        
        uncovered = set(range(n_subsets))
        blocks = []
        
        while uncovered:
            # Pick a random uncovered subset
            # Using list() conversion is O(N), but N decreases.
            # For speed, we can maintain a list of uncovered indices and pick random index, then swap/pop?
            # But `uncovered` is a set.
            # Let's just use random.choice(list(uncovered))
            target_id = random.choice(list(uncovered))
            p1, p2, p3 = all_3_subsets[target_id]
            
            fixed = (p1, p2, p3)
            available = [x for x in range(v) if x not in fixed]
            
            best_gain = -1
            best_add = None
            
            # Pre-calculate pairs of fixed points
            # (p1,p2), (p1,p3), (p2,p3)
            
            for add in itertools.combinations(available, 4):
                gain = 1 # The fixed triple itself
                
                # 2 from fixed, 1 from add
                a1, a2, a3, a4 = add
                if get_id_3(p1, p2, a1) in uncovered: gain += 1
                if get_id_3(p1, p2, a2) in uncovered: gain += 1
                if get_id_3(p1, p2, a3) in uncovered: gain += 1
                if get_id_3(p1, p2, a4) in uncovered: gain += 1
                
                if get_id_3(p1, p3, a1) in uncovered: gain += 1
                if get_id_3(p1, p3, a2) in uncovered: gain += 1
                if get_id_3(p1, p3, a3) in uncovered: gain += 1
                if get_id_3(p1, p3, a4) in uncovered: gain += 1
                
                if get_id_3(p2, p3, a1) in uncovered: gain += 1
                if get_id_3(p2, p3, a2) in uncovered: gain += 1
                if get_id_3(p2, p3, a3) in uncovered: gain += 1
                if get_id_3(p2, p3, a4) in uncovered: gain += 1

                # 1 from fixed, 2 from add
                # Pairs in add: (a1,a2), (a1,a3), (a1,a4), (a2,a3), (a2,a4), (a3,a4)
                if get_id_3(p1, a1, a2) in uncovered: gain += 1
                if get_id_3(p1, a1, a3) in uncovered: gain += 1
                if get_id_3(p1, a1, a4) in uncovered: gain += 1
                if get_id_3(p1, a2, a3) in uncovered: gain += 1
                if get_id_3(p1, a2, a4) in uncovered: gain += 1
                if get_id_3(p1, a3, a4) in uncovered: gain += 1
                
                if get_id_3(p2, a1, a2) in uncovered: gain += 1
                if get_id_3(p2, a1, a3) in uncovered: gain += 1
                if get_id_3(p2, a1, a4) in uncovered: gain += 1
                if get_id_3(p2, a2, a3) in uncovered: gain += 1
                if get_id_3(p2, a2, a4) in uncovered: gain += 1
                if get_id_3(p2, a3, a4) in uncovered: gain += 1
                
                if get_id_3(p3, a1, a2) in uncovered: gain += 1
                if get_id_3(p3, a1, a3) in uncovered: gain += 1
                if get_id_3(p3, a1, a4) in uncovered: gain += 1
                if get_id_3(p3, a2, a3) in uncovered: gain += 1
                if get_id_3(p3, a2, a4) in uncovered: gain += 1
                if get_id_3(p3, a3, a4) in uncovered: gain += 1

                # 3 from add
                if get_id_3(a1, a2, a3) in uncovered: gain += 1
                if get_id_3(a1, a2, a4) in uncovered: gain += 1
                if get_id_3(a1, a3, a4) in uncovered: gain += 1
                if get_id_3(a2, a3, a4) in uncovered: gain += 1

                if gain > best_gain:
                    best_gain = gain
                    best_add = add
            
            block = list(fixed) + list(best_add)
            block.sort()
            blocks.append(block)
            
            # Update uncovered
            for sub in itertools.combinations(block, 3):
                sid = subset_to_id[sub]
                uncovered.discard(sid)
                
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
            
        if best_len <= 76:
             break

    if best_blocks is None:
        return []
        
    return best_blocks
