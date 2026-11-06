import time
import random
from itertools import combinations

# Global data
ALL_SETS = []
ALL_SETS_BITS = []
INTER = []
GROUPS = []
CONFLICT_SETS = []
N_SETS = 0

def init_data():
    global ALL_SETS, ALL_SETS_BITS, INTER, GROUPS, CONFLICT_SETS, N_SETS
    
    # 1. Generate all 4-subsets of [1..16]
    for row in combinations(range(1, 17), 4):
        mask = 0
        for x in row:
            mask |= (1 << (x - 1))
        ALL_SETS.append(row)
        ALL_SETS_BITS.append(mask)
    
    N_SETS = len(ALL_SETS_BITS)
    
    # 2. Precompute intersections
    # INTER[i][j] is the bitmask of intersection of set i and set j
    INTER = [[0] * N_SETS for _ in range(N_SETS)]
    for i in range(N_SETS):
        bi = ALL_SETS_BITS[i]
        for j in range(N_SETS):
            bj = ALL_SETS_BITS[j]
            INTER[i][j] = bi & bj
            
    # 3. Build groups[i]: dict from intersection mask to set of indices
    # groups[i][mask] = set of j such that INTER[i][j] == mask
    GROUPS = [{} for _ in range(N_SETS)]
    for i in range(N_SETS):
        d = {}
        for j in range(N_SETS):
            m = INTER[i][j]
            if m not in d:
                d[m] = set()
            d[m].add(j)
        GROUPS[i] = d
        
    # 4. Precompute conflict_sets[i][k]
    # conflict_sets[i][k] = set of j such that INTER[i][j] == INTER[i][k] AND INTER[k][j] == INTER[i][k]
    # We exclude i and k from this set because the sunflower sets must be distinct.
    CONFLICT_SETS = [[None] * N_SETS for _ in range(N_SETS)]
    for i in range(N_SETS):
        for k in range(N_SETS):
            if i == k:
                CONFLICT_SETS[i][k] = set()
                continue
            s = INTER[i][k]
            # Get sets of indices having intersection s with i and with k
            set_i_s = GROUPS[i].get(s, set())
            set_k_s = GROUPS[k].get(s, set())
            
            # Intersection of these two sets
            common = set_i_s & set_k_s
            
            # Remove i and k
            common.discard(i)
            common.discard(k)
            
            CONFLICT_SETS[i][k] = common

def greedy_once(random_order):
    family_set = set()
    family_list = []
    
    for idx in random_order:
        can_add = True
        # Check against all existing members
        for i in family_list:
            # If there is any j in family that forms a sunflower with i and idx
            # j must be in CONFLICT_SETS[i][idx]
            # Since we removed i and idx from CONFLICT_SETS[i][idx], we just check intersection with family_set
            if CONFLICT_SETS[i][idx] & family_set:
                can_add = False
                break
        
        if can_add:
            family_set.add(idx)
            family_list.append(idx)
            
    return family_list

def construct():
    start_time = time.time()
    deadline = start_time + 50.0 # 50s margin
    
    # Initialize global data
    if N_SETS == 0:
        init_data()
        
    best_family = []
    
    # Multi-start greedy
    num_restarts = 0
    while time.time() < deadline:
        random_order = list(range(N_SETS))
        random.shuffle(random_order)
        
        family = greedy_once(random_order)
        num_restarts += 1
        
        if len(family) > len(best_family):
            best_family = family
            
        # Optional: If we found a very large family, we might stop or continue
        # Max possible is 1820, but likely much smaller.
        # If we hit a known upper bound or just run out of time, stop.
        
    # Convert indices back to sorted lists of elements
    result = [sorted(ALL_SETS[i]) for i in best_family]
    
    return result

# Initialize data at module load to avoid overhead in construct if called multiple times?
# But construct is the entry point. It's better to init inside or once.
# Since the evaluator calls construct(), we should ensure init happens once.
# Using a flag or checking N_SETS is sufficient.
# However, init_data takes a bit of time. Let's do it in construct if needed.

# To be safe and fast, let's initialize on first call.
_initialized = False

def construct():
    global _initialized, N_SETS
    if not _initialized:
        init_data()
        _initialized = True
        
    start_time = time.time()
    deadline = start_time + 55.0 # 55s margin out of 60s
    
    best_family = []
    
    # Multi-start greedy
    while time.time() < deadline:
        random_order = list(range(N_SETS))
        random.shuffle(random_order)
        
        family = greedy_once(random_order)
        
        if len(family) > len(best_family):
            best_family = family
            
    # Convert indices back to sorted lists of elements
    result = [sorted(ALL_SETS[i]) for i in best_family]
    
    return result
