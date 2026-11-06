import time
import random
from itertools import combinations

def _solve_s4511(deadline):
    # ... (Keep existing code) ...
    pass

def construct():
    # ... (Keep existing code) ...
    pass

def _construct_witt_fallback():
    # ... (Keep existing code) ...
    return _randomized_greedy_packing()

def _randomized_greedy_packing():
    """Randomized greedy construction with restarts."""
    points = list(range(12))
    best_blocks = []
    start_time = time.time()
    time_limit = 5.0  # 5 seconds for fallback
    
    for _ in range(100):  # Max 100 restarts
        if time.time() - start_time > time_limit:
            break
            
        # Shuffle points to break symmetry
        shuffled_points = points[:]
        random.shuffle(shuffled_points)
        
        # Map shuffled to original for output later? 
        # No, we just need the count/validity. 
        # Actually, we need to return valid blocks on original points.
        # So we must track the mapping.
        
        # Simpler: Just shuffle the order of candidate 6-subsets.
        all_6subs = list(combinations(points, 6))
        random.shuffle(all_6subs)
        
        covered_5 = set()
        blocks = []
        
        for block in all_6subs:
            valid = True
            new_5s = []
            for sub in combinations(block, 5):
                if sub in covered_5:
                    valid = False
                    break
                new_5s.append(sub)
            
            if valid:
                blocks.append(list(block))
                for sub in new_5s:
                    covered_5.add(sub)
        
        if len(blocks) > len(best_blocks):
            best_blocks = blocks
        
        # Early exit if we hit optimum
        if len(best_blocks) == 132:
            break

    return best_blocks
