import time
import random
from itertools import combinations

def construct():
    # Pre-compute all 6-subsets and their 5-subsets
    all_blocks = list(combinations(range(12), 6))
    block_5sets = []
    for b in all_blocks:
        block_5sets.append(tuple(combinations(b, 5)))
    
    best_blocks = []
    start_time = time.time()
    deadline = start_time + 100  # Safety margin
    
    # Try multiple random restarts
    for restart in range(2000):
        if time.time() > deadline:
            break
            
        # Shuffle candidates
        indices = list(range(len(all_blocks)))
        random.shuffle(indices)
        
        covered = set()
        current_blocks = []
        
        for idx in indices:
            b = all_blocks[idx]
            # Check if any 5-subset is already covered
            conflict = False
            for s in block_5sets[idx]:
                if s in covered:
                    conflict = True
                    break
            
            if not conflict:
                current_blocks.append(b)
                for s in block_5sets[idx]:
                    covered.add(s)
                
                # Early exit if we hit the theoretical max
                if len(current_blocks) == 132:
                    break
        
        if len(current_blocks) > len(best_blocks):
            best_blocks = current_blocks
            if len(best_blocks) == 132:
                break
                
    return [list(b) for b in best_blocks]
