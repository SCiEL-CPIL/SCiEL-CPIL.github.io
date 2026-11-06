import time
from itertools import combinations

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 130  # 20s margin for safety
    
    # Precompute all 6-subsets and their 5-subset members
    all_blocks = list(combinations(range(12), 6))
    block_5subsets = [list(combinations(b, 5)) for b in all_blocks]
    
    # 1. Greedy Lexicographic Packing
    covered = set()
    blocks = []
    
    for i, b in enumerate(all_blocks):
        if time.time() > deadline:
            break
        subsets = block_5subsets[i]
        conflict = False
        for s in subsets:
            if s in covered:
                conflict = True
                break
        if not conflict:
            blocks.append(list(b))
            for s in subsets:
                covered.add(s)
            
            if len(blocks) == 132:
                return blocks

    # 2. If greedy didn't reach 132, try randomized greedy with different orderings
    if len(blocks) < 132:
        import random
        best_count = len(blocks)
        
        for trial in range(200):
            if time.time() > deadline:
                break
            random.seed(trial)
            covered_f = set()
            blocks_f = []
            
            # Create a shuffled ordering of block indices
            indices = list(range(len(all_blocks)))
            random.shuffle(indices)
            
            for idx in indices:
                b = all_blocks[idx]
                subsets = block_5subsets[idx]
                conflict = False
                for s in subsets:
                    if s in covered_f:
                        conflict = True
                        break
                if not conflict:
                    blocks_f.append(list(b))
                    for s in subsets:
                        covered_f.add(s)
                    if len(blocks_f) == 132:
                        return blocks_f
            
            if len(blocks_f) > best_count:
                best_count = len(blocks_f)
                blocks = blocks_f
                if best_count == 132:
                    return blocks
                
    return blocks
