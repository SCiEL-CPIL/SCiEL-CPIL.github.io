import time
import random
from itertools import combinations

# Precompute all 4-subsets of [1, 16]
_ALL_SUBSETS = [tuple(c) for c in combinations(range(1, 17), 4)]
_N_SUBSETS = len(_ALL_SUBSETS)  # 1820

def _to_mask(subset):
    m = 0
    for x in subset:
        m |= (1 << (x - 1))
    return m

_MASKS = [_to_mask(s) for s in _ALL_SUBSETS]

# Precompute pairwise intersection masks
_INTER = [[0] * _N_SUBSETS for _ in range(_N_SUBSETS)]
for i in range(_N_SUBSETS):
    mi = _MASKS[i]
    row = _INTER[i]
    for j in range(i + 1, _N_SUBSETS):
        v = mi & _MASKS[j]
        row[j] = v
        _INTER[j][i] = v

def _is_conflict(c, chosen, inter_matrix):
    """Check if adding c to chosen creates a sunflower."""
    m = len(chosen)
    for i in range(m):
        a = chosen[i]
        inter_ac = inter_matrix[a][c]
        for j in range(i + 1, m):
            b = chosen[j]
            if inter_matrix[a][b] == inter_ac and inter_matrix[b][c] == inter_ac:
                return True
    return False

def _greedy_init(rng, deadline):
    """Initial greedy solution."""
    order = list(range(_N_SUBSETS))
    rng.shuffle(order)
    chosen = []
    t_start = time.time()
    for c in order:
        if time.time() - t_start > deadline:
            break
        if not _is_conflict(c, chosen, _INTER):
            chosen.append(c)
    return chosen

def _local_search(chosen, rng, deadline):
    """
    Local search: try to add sets, if stuck, remove one and try again.
    """
    if not chosen:
        # Ensure we start with at least one set if empty
        chosen = [rng.randrange(_N_SUBSETS)]
        
    chosen_set = set(chosen)
    remaining = [i for i in range(_N_SUBSETS) if i not in chosen_set]
    rng.shuffle(remaining)
    
    t_start = time.time()
    while time.time() - t_start < deadline:
        improved = False
        added = False
        # Iterate through remaining
        # To avoid O(N^2) overhead of popping from middle of list repeatedly in a tight loop,
        # we can just scan. But popping is O(N). 
        # Since N=1820, it's manageable.
        
        for idx in range(len(remaining)):
            c = remaining[idx]
            if not _is_conflict(c, chosen, _INTER):
                chosen.append(c)
                chosen_set.add(c)
                remaining.pop(idx)
                improved = True
                added = True
                break
        
        if not added:
            # Stuck. Try to remove a random set from chosen and see if we can add a new one
            if len(chosen) > 10: # Lower threshold to allow more exploration
                # Pick a random set to remove
                remove_idx = rng.randrange(len(chosen))
                removed_set = chosen[remove_idx]
                chosen.pop(remove_idx)
                chosen_set.remove(removed_set)
                
                # Try to add a set from remaining
                # We can try the same remaining list
                for idx in range(len(remaining)):
                    c = remaining[idx]
                    if not _is_conflict(c, chosen, _INTER):
                        chosen.append(c)
                        chosen_set.add(c)
                        remaining.pop(idx)
                        improved = True
                        break
                        
    return chosen

def construct():
    deadline = 85.0
    t0 = time.time()
    
    # Initialize with a valid single-element family to satisfy "1..250 entries"
    # The first subset is [1,2,3,4]
    best_size = 1
    best_result = [list(_ALL_SUBSETS[0])]
    
    rng = random.Random(42)
    
    # Run multiple iterations
    while time.time() - t0 < deadline - 5.0:
        # 1. Greedy Init
        # Give greedy a small budget, e.g., 1 sec
        g_deadline = min(time.time() + 1.0, deadline - 5.0)
        if g_deadline <= time.time():
            break
        init_solution = _greedy_init(rng, g_deadline)
        
        if not init_solution:
            continue
            
        # 2. Local Search
        # Give local search the rest of the time minus safety margin
        ls_deadline = deadline - 5.0
        if ls_deadline < time.time():
            break
        improved_solution = _local_search(init_solution, rng, ls_deadline)
        
        if not improved_solution:
            continue

        size = len(improved_solution)
        if size > best_size:
            best_size = size
            best_result = [list(_ALL_SUBSETS[i]) for i in improved_solution]
            
    return best_result
