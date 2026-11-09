import time
import random
import itertools
import sys
from functools import lru_cache

# Precomputation
N_POINTS = 16
ALL_TRIPLES = list(itertools.combinations(range(N_POINTS), 3))
TRIPLE_INDEX = {t: i for i, t in enumerate(ALL_TRIPLES)}
N_TRIPLES = len(ALL_TRIPLES)

# Generate all K(2,2,2) configurations
CONFIG_TRIPLES = []
TRIPLE_CONFIGS = [[] for _ in range(N_TRIPLES)]

def precompute():
    global CONFIG_TRIPLES, TRIPLE_CONFIGS
    CONFIG_TRIPLES = []
    TRIPLE_CONFIGS = [[] for _ in range(N_TRIPLES)]
    
    six_subsets = list(itertools.combinations(range(N_POINTS), 6))
    
    for pts in six_subsets:
        p0 = pts[0]
        rest = pts[1:]
        for i in range(5):
            p1 = rest[i]
            remaining = [x for x in rest if x != p1]
            r0, r1, r2, r3 = remaining
            pairings = [
                [(p0, p1), (r0, r1), (r2, r3)],
                [(p0, p1), (r0, r2), (r1, r3)],
                [(p0, p1), (r0, r3), (r1, r2)]
            ]
            for pairs in pairings:
                a, a_p = pairs[0]
                b, b_p = pairs[1]
                c, c_p = pairs[2]
                
                transversals = [
                    (a, b, c),
                    (a, b, c_p),
                    (a, b_p, c),
                    (a, b_p, c_p),
                    (a_p, b, c),
                    (a_p, b, c_p),
                    (a_p, b_p, c),
                    (a_p, b_p, c_p)
                ]
                
                idxs = []
                for t in transversals:
                    idx = TRIPLE_INDEX.get(tuple(sorted(t)))
                    if idx is None:
                        continue
                    idxs.append(idx)
                
                config_id = len(CONFIG_TRIPLES)
                CONFIG_TRIPLES.append(idxs)
                for idx in idxs:
                    TRIPLE_CONFIGS[idx].append(config_id)

precompute()
N_CONFIGS = len(CONFIG_TRIPLES)

# Build conflict graph: for each triple, which other triples conflict with it
# Two triples conflict if they share a K(2,2,2) configuration
CONFLICTS = [[] for _ in range(N_TRIPLES)]
for cfg_id, cfg_triples in enumerate(CONFIG_TRIPLES):
    for i in range(len(cfg_triples)):
        for j in range(i+1, len(cfg_triples)):
            t1, t2 = cfg_triples[i], cfg_triples[j]
            if t2 not in CONFLICTS[t1]:
                CONFLICTS[t1].append(t2)
            if t1 not in CONFLICTS[t2]:
                CONFLICTS[t2].append(t1)

# Sort conflicts for faster lookup
for i in range(N_TRIPLES):
    CONFLICTS[i].sort()

# Precompute for each config, the list of its 8 triple indices (already done in CONFIG_TRIPLES)
# For fast validity check, we need to know for a given triple, which configs it's in (TRIPLE_CONFIGS)
# And for a config, how many of its 8 triples are currently in the set (config_counts)

def is_valid_addition(triple_idx, config_counts):
    for cfg_id in TRIPLE_CONFIGS[triple_idx]:
        if config_counts[cfg_id] == 7:
            return False
    return True

def update_counts(triple_idx, delta, config_counts):
    for cfg_id in TRIPLE_CONFIGS[triple_idx]:
        config_counts[cfg_id] += delta

def greedy_construct_with_order(order, time_limit, start_time):
    """
    Greedily add triples in the given order.
    Returns (set_of_triples, config_counts)
    """
    current_set = set()
    config_counts = [0] * N_CONFIGS
    
    for t_idx in order:
        if time.time() - start_time > time_limit:
            break
        if is_valid_addition(t_idx, config_counts):
            current_set.add(t_idx)
            update_counts(t_idx, 1, config_counts)
    
    return current_set, config_counts

def greedy_construct(num_restarts=100, time_limit=None):
    best_set = set()
    best_size = 0
    start_time = time.time()
    
    for r in range(num_restarts):
        if time_limit and (time.time() - start_time) > time_limit:
            break
            
        order = list(range(N_TRIPLES))
        random.shuffle(order)
        
        current_set, _ = greedy_construct_with_order(order, time_limit, start_time)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set.copy()
            
    return best_set, best_size

def compute_upper_bound(selected_set, available_triples, config_counts):
    """
    Compute an upper bound on the number of additional triples we can add.
    Uses a greedy coloring / vertex cover heuristic.
    """
    if not available_triples:
        return 0
    
    # Greedy independent set: repeatedly pick a vertex with minimum degree in remaining
    # This gives a lower bound on the independence number, which is an upper bound on what we can add
    # Actually, we want an upper bound on the size of the maximum independent set in the remaining graph
    
    # Simple approach: use the fact that in a graph, alpha(G) <= n * (1 - d_avg/(2*n)) or similar
    # Better: use a greedy packing bound
    
    # For each available triple, check if it can be added
    # Upper bound: number of triples that are individually addable + some slack
    
    addable_count = 0
    for t in available_triples:
        if is_valid_addition(t, config_counts):
            addable_count += 1
    
    # A crude upper bound: we can add at most addable_count triples, but they may conflict with each other
    # Use a simple bound: alpha <= n / (1 + d_avg/n) where d_avg is average degree
    # Or just return addable_count as a loose upper bound
    
    return addable_count

def branch_and_bound(initial_set, time_limit, best_size_initial):
    """
    Branch and bound search for maximum independent set.
    """
    start_time = time.time()
    current_set = initial_set.copy()
    config_counts = [0] * N_CONFIGS
    
    for t_idx in current_set:
        update_counts(t_idx, 1, config_counts)
        
    best_set = current_set.copy()
    best_size = len(current_set)
    
    # Get list of triples not in current set
    not_in_set = [t for t in range(N_TRIPLES) if t not in current_set]
    
    # Sort by some heuristic (e.g., number of conflicts)
    not_in_set.sort(key=lambda t: len(CONFLICTS[t]), reverse=True)
    
    def bb_search(selected, available, config_counts, current_size):
        nonlocal best_set, best_size
        
        if time.time() - start_time > time_limit:
            return
        
        # Upper bound: current_size + len(available)
        if current_size + len(available) <= best_size:
            return
        
        # If no more available, update best
        if not available:
            if current_size > best_size:
                best_size = current_size
                best_set = selected.copy()
            return
        
        # Choose a vertex to branch on
        # Heuristic: pick the vertex with highest degree in the available subgraph
        # Or just pick the first one for simplicity
        
        v = available[0]
        remaining = available[1:]
        
        # Branch 1: Include v
        if is_valid_addition(v, config_counts):
            new_selected = selected.copy()
            new_selected.add(v)
            new_config_counts = config_counts.copy()
            update_counts(v, 1, new_config_counts)
            
            # Filter remaining: remove v and its conflicts
            new_available = [t for t in remaining if t not in CONFLICTS[v]]
            
            bb_search(new_selected, new_available, new_config_counts, current_size + 1)
        
        # Branch 2: Exclude v
        bb_search(selected, remaining, config_counts, current_size)
    
    bb_search(current_set, not_in_set, config_counts, len(current_set))
    
    return best_set, best_size

def construct():
    start_time = time.time()
    total_time_limit = 145.0 # Leave 5s margin for 150s limit
    
    # Phase 1: Greedy to get a good starting point
    greedy_time = min(20.0, total_time_limit * 0.15)
    best_set, best_size = greedy_construct(num_restarts=300, time_limit=greedy_time)
    
    current_time = time.time() - start_time
    if current_time > total_time_limit:
        return [list(ALL_TRIPLES[i]) for i in best_set]
    
    # Phase 2: Branch and bound
    remaining_time = total_time_limit - current_time - 5.0  # Leave 5s for final conversion
    if remaining_time > 10:  # Only do B&B if we have enough time
        best_set, best_size = branch_and_bound(best_set, remaining_time, best_size)
    
    # Convert to list of lists of points
    result = [list(ALL_TRIPLES[i]) for i in best_set]
    
    return result
