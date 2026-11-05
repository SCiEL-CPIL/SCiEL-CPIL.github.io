import random
import time

def construct():
    start_time = time.time()
    deadline = 145.0 # 5s margin
    
    best_blocks = []
    best_count = 0
    
    # Try standard greedy first
    blocks = _greedy_lexicographic()
    if len(blocks) > best_count:
        best_count = len(blocks)
        best_blocks = blocks
        
    # Try randomized greedy
    for i in range(500):
        if time.time() > deadline:
            break
        blocks = _greedy_random()
        if len(blocks) > best_count:
            best_count = len(blocks)
            best_blocks = blocks
            if best_count == 132:
                return best_blocks
                
    return best_blocks

def _greedy_lexicographic():
    from itertools import combinations
    covered, blocks = set(), []
    for b in combinations(range(12), 6):
        subsets = list(combinations(b, 5))
        if not any(s in covered for s in subsets):
            blocks.append(list(b))
            covered.update(subsets)
    return blocks

def _greedy_random():
    from itertools import combinations
    import random
    
    candidates = list(combinations(range(12), 6))
    random.shuffle(candidates)
    
    covered, blocks = set(), []
    for b in candidates:
        subsets = list(combinations(b, 5))
        if not any(s in covered for s in subsets):
            blocks.append(list(b))
            covered.update(subsets)
    return blocks
