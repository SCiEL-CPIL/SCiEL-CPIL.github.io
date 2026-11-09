"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools

# EVOLVE-BLOCK-START
import sys
import numpy as np

N_POINTS = 16
ALL_TRIPLES = list(itertools.combinations(range(N_POINTS), 3))
TRIPLE_INDEX = {t: i for i, t in enumerate(ALL_TRIPLES)}
N_TRIPLES = len(ALL_TRIPLES)

# Precompute all K(2,2,2) configurations and their constituent triples
CONFIG_TRIPLES = []  # List of lists of 8 triple indices
TRIPLE_CONFIGS = [[] for _ in range(N_TRIPLES)]  # For each triple, list of config indices it belongs to

def _precompute_configs():
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
                    if idx is not None:
                        idxs.append(idx)
                
                config_id = len(CONFIG_TRIPLES)
                CONFIG_TRIPLES.append(idxs)
                for idx in idxs:
                    TRIPLE_CONFIGS[idx].append(config_id)

_precompute_configs()

def _is_valid_addition(triple_idx, in_set, config_counts):
    """Check if adding triple_idx to in_set is valid."""
    if triple_idx in in_set:
        return False
    for cfg_id in TRIPLE_CONFIGS[triple_idx]:
        if config_counts[cfg_id] == 7:
            return False
    return True

def _update_counts(triple_idx, delta, config_counts):
    """Update config counts for adding (delta=1) or removing (delta=-1) triple_idx."""
    for cfg_id in TRIPLE_CONFIGS[triple_idx]:
        config_counts[cfg_id] += delta

def _greedy_construct(deadline, seed_offset=0):
    """Greedy construction with randomization."""
    order = list(range(N_TRIPLES))
    random.shuffle(order)
    
    current_set = set()
    config_counts = [0] * len(CONFIG_TRIPLES)
    
    for t_idx in order:
        if time.time() > deadline:
            break
        if _is_valid_addition(t_idx, current_set, config_counts):
            current_set.add(t_idx)
            _update_counts(t_idx, 1, config_counts)
    
    return current_set, config_counts

def _local_search(initial_set, deadline, max_swaps_per_phase=30):
    """Local search: try to add more triples and perform swaps."""
    current_set = set(initial_set)
    config_counts = [0] * len(CONFIG_TRIPLES)
    
    for t_idx in current_set:
        _update_counts(t_idx, 1, config_counts)
    
    best_set = current_set.copy()
    best_size = len(current_set)
    
    while time.time() < deadline:
        improved = False
        
        # Phase 1: Try to add any valid triple
        candidates = [t for t in range(N_TRIPLES) if t not in current_set]
        random.shuffle(candidates)
        
        for t_idx in candidates:
            if time.time() > deadline:
                break
            if _is_valid_addition(t_idx, current_set, config_counts):
                current_set.add(t_idx)
                _update_counts(t_idx, 1, config_counts)
                improved = True
        
        if improved:
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = current_set.copy()
            continue
        
        # Phase 2: No single addition possible. Try swaps
        swapped = False
        for _ in range(max_swaps_per_phase):
            if time.time() > deadline:
                break
            
            if not current_set:
                break
            
            # Try different swap strategies
            strategy = random.random()
            
            if strategy < 0.4:
                # 1-for-1 swap
                t_out = random.choice(list(current_set))
                current_set.remove(t_out)
                _update_counts(t_out, -1, config_counts)
                
                candidates = [t for t in range(N_TRIPLES) if t not in current_set and t != t_out]
                random.shuffle(candidates)
                
                added = False
                for t_idx in candidates:
                    if time.time() > deadline:
                        break
                    if _is_valid_addition(t_idx, current_set, config_counts):
                        current_set.add(t_idx)
                        _update_counts(t_idx, 1, config_counts)
                        added = True
                        break
                
                if added:
                    swapped = True
                    if len(current_set) > best_size:
                        best_size = len(current_set)
                        best_set = current_set.copy()
                else:
                    current_set.add(t_out)
                    _update_counts(t_out, 1, config_counts)
                    
            elif strategy < 0.7:
                # 1-for-2 swap: remove 1, try to add 2
                t_out = random.choice(list(current_set))
                current_set.remove(t_out)
                _update_counts(t_out, -1, config_counts)
                
                candidates = [t for t in range(N_TRIPLES) if t not in current_set and t != t_out]
                random.shuffle(candidates)
                
                added_count = 0
                added_triples = []
                for t_idx in candidates:
                    if time.time() > deadline:
                        break
                    if added_count >= 2:
                        break
                    if _is_valid_addition(t_idx, current_set, config_counts):
                        current_set.add(t_idx)
                        _update_counts(t_idx, 1, config_counts)
                        added_triples.append(t_idx)
                        added_count += 1
                
                if added_count >= 2:
                    swapped = True
                    if len(current_set) > best_size:
                        best_size = len(current_set)
                        best_set = current_set.copy()
                else:
                    for t_idx in added_triples:
                        current_set.remove(t_idx)
                        _update_counts(t_idx, -1, config_counts)
                    current_set.add(t_out)
                    _update_counts(t_out, 1, config_counts)
                    
            else:
                # 2-for-1 swap: remove 2, try to add 1
                if len(current_set) < 2:
                    continue
                    
                t_out1, t_out2 = random.sample(list(current_set), 2)
                current_set.remove(t_out1)
                current_set.remove(t_out2)
                _update_counts(t_out1, -1, config_counts)
                _update_counts(t_out2, -1, config_counts)
                
                candidates = [t for t in range(N_TRIPLES) if t not in current_set and t != t_out1 and t != t_out2]
                random.shuffle(candidates)
                
                added = False
                for t_idx in candidates:
                    if time.time() > deadline:
                        break
                    if _is_valid_addition(t_idx, current_set, config_counts):
                        current_set.add(t_idx)
                        _update_counts(t_idx, 1, config_counts)
                        added = True
                        break
                
                if added:
                    swapped = True
                    if len(current_set) > best_size:
                        best_size = len(current_set)
                        best_set = current_set.copy()
                else:
                    current_set.add(t_out1)
                    current_set.add(t_out2)
                    _update_counts(t_out1, 1, config_counts)
                    _update_counts(t_out2, 1, config_counts)
            
            if swapped:
                break
        
        if not swapped:
            # No improvement possible with simple swaps
            break
    
    return best_set

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    best_result = set()
    best_size = 0
    
    # Phase 1: Multiple greedy restarts to find a good initial solution
    num_restarts = 20
    
    for i in range(num_restarts):
        if time.time() > deadline:
            break
        
        random.seed(i * 7919 + 13)
        current_set, _ = _greedy_construct(deadline)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_result = current_set.copy()
    
    # Phase 2: Local search with simulated annealing on the best result
    if time.time() < deadline:
        best_result = _local_search(best_result, deadline)
    
    return [list(ALL_TRIPLES[t]) for t in sorted(best_result)]
# EVOLVE-BLOCK-END
