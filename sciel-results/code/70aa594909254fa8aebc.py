# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def is_2_colorable(n, edges_iter):
    """
    Returns True if the hypergraph is 2-colorable.
    Returns False if it is NOT 2-colorable.
    """
    num_masks = 1 << n
    for mask in range(num_masks):
        valid = True
        for e in edges_iter:
            c0 = (mask >> e[0]) & 1
            is_mono = True
            for v in e[1:]:
                if ((mask >> v) & 1) != c0:
                    is_mono = False
                    break
            if is_mono:
                valid = False
                break
        if valid:
            return True
    return False

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n = 9
    all_edges_tuples = [tuple(e) for e in combinations(range(n), 5)]
    
    best_edges = [list(e) for e in all_edges_tuples]
    best_count = len(best_edges)
    
    random.seed(42)
    
    for i in range(100):
        if time.time() > deadline:
            break
            
        trial_order = all_edges_tuples[:]
        random.shuffle(trial_order)
        
        active = set(all_edges_tuples)
        
        for e in trial_order:
            if e not in active:
                continue
                
            active.remove(e)
            
            if not is_2_colorable(n, active):
                pass
            else:
                active.add(e)
                
        if len(active) < best_count:
            best_count = len(active)
            best_edges = [list(t) for t in active]
            
            if best_count <= 51:
                break

    return {'n': n, 'edges': best_edges}
# EVOLVE-BLOCK-END
