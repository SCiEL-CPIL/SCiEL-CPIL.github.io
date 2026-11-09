"""Combined program: precomputed K(2,2,2) configs + greedy + 2-for-2 exchange search."""
import time
import random
import itertools
import heapq

N_POINTS = 16
ALL_TRIPLES = list(itertools.combinations(range(N_POINTS), 3))
TRIPLE_INDEX = {t: i for i, t in enumerate(ALL_TRIPLES)}
N_TRIPLES = len(ALL_TRIPLES)

# Precompute all K(2,2,2) configurations
CONFIG_TRIPLES = []
TRIPLE_CONFIGS = [[] for _ in range(N_TRIPLES)]

def _precompute():
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
                [(p0, p1), (r0, r3), (r1, r2)],
            ]
            for pairs in pairings:
                a, a_p = pairs[0]
                b, b_p = pairs[1]
                c, c_p = pairs[2]
                transversals = [
                    (a, b, c), (a, b, c_p), (a, b_p, c), (a, b_p, c_p),
                    (a_p, b, c), (a_p, b, c_p), (a_p, b_p, c), (a_p, b_p, c_p),
                ]
                idxs = []
                for t in transversals:
                    idx = TRIPLE_INDEX.get(tuple(sorted(t)))
                    if idx is not None:
                        idxs.append(idx)
                cfg_id = len(CONFIG_TRIPLES)
                CONFIG_TRIPLES.append(idxs)
                for idx in idxs:
                    TRIPLE_CONFIGS[idx].append(cfg_id)

_precompute()
N_CONFIGS = len(CONFIG_TRIPLES)

def _is_valid_addition(triple_idx, in_set, config_counts):
    if triple_idx in in_set:
        return False
    for cfg_id in TRIPLE_CONFIGS[triple_idx]:
        if config_counts[cfg_id] == 7:
            return False
    return True

def _update_counts(triple_idx, delta, config_counts):
    for cfg_id in TRIPLE_CONFIGS[triple_idx]:
        config_counts[cfg_id] += delta

def _init_counts(current_set):
    config_counts = [0] * N_CONFIGS
    for t_idx in current_set:
        _update_counts(t_idx, 1, config_counts)
    return config_counts

def _greedy_construct(seed, deadline):
    random.seed(seed)
    order = list(range(N_TRIPLES))
    random.shuffle(order)
    current_set = set()
    config_counts = [0] * N_CONFIGS
    for t_idx in order:
        if time.time() > deadline:
            break
        if _is_valid_addition(t_idx, current_set, config_counts):
            current_set.add(t_idx)
            _update_counts(t_idx, 1, config_counts)
    return current_set, config_counts

def _local_search_add(current_set, config_counts, deadline):
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for t_idx in range(N_TRIPLES):
            if t_idx in current_set:
                continue
            if _is_valid_addition(t_idx, current_set, config_counts):
                current_set.add(t_idx)
                _update_counts(t_idx, 1, config_counts)
                improved = True
    return current_set, config_counts

def _exchange_search(current_set, config_counts, deadline, max_rounds=50):
    """
    2-for-2 exchange local search: repeatedly find and apply the best 
    (remove 2, add 2) move that increases the set size or breaks a deadlock.
    Also does 1-for-1 swaps when no 2-for-2 improves size.
    """
    best_size = len(current_set)
    best_set = current_set.copy()
    best_counts = config_counts.copy()
    
    # Precompute: for each triple not in set, check if it's addable
    # For 2-for-2: pick two to remove, then try to add two new ones
    
    # Strategy: iterate through candidate removal pairs (limited), 
    # and for each, try to find two addable triples.
    
    not_in_set = [t for t in range(N_TRIPLES) if t not in current_set]
    
    for rnd in range(max_rounds):
        if time.time() > deadline:
            break
        
        improved = False
        
        # First try: pure additions
        for t_idx in not_in_set:
            if time.time() > deadline:
                break
            if _is_valid_addition(t_idx, current_set, config_counts):
                current_set.add(t_idx)
                _update_counts(t_idx, 1, config_counts)
                not_in_set.remove(t_idx)
                improved = True
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = current_set.copy()
                    best_counts = config_counts.copy()
        
        if not improved:
            # Try 1-for-1 swaps: remove one, add one different
            swap_improved = False
            in_list = list(current_set)
            random.shuffle(in_list)
            for t_out in in_list:
                if time.time() > deadline:
                    break
                current_set.remove(t_out)
                _update_counts(t_out, -1, config_counts)
                
                # Try to add a better triple
                added = False
                for t_in in not_in_set:
                    if _is_valid_addition(t_in, current_set, config_counts):
                        current_set.add(t_in)
                        _update_counts(t_in, 1, config_counts)
                        not_in_set.remove(t_in)
                        added = True
                        break
                
                if not added:
                    # Revert
                    current_set.add(t_out)
                    _update_counts(t_out, 1, config_counts)
                else:
                    swap_improved = True
                    if len(current_set) > best_size:
                        best_size = len(current_set)
                        best_set = current_set.copy()
                        best_counts = config_counts.copy()
                    break  # restart outer loop after a swap
                
                # If we broke out, go back to top
                if swap_improved:
                    break
            
            if swap_improved:
                # Refresh not_in_set
                not_in_set = [t for t in range(N_TRIPLES) if t not in current_set]
                continue
            
            # Try 2-for-2 exchanges
            # Pick two random triples to remove, then try to add two
            two_improved = False
            in_list = list(current_set)
            random.shuffle(in_list)
            
            # Try a limited number of 2-for-2 attempts
            n_attempts = min(200, len(in_list) * (len(in_list) - 1) // 2)
            for attempt in range(n_attempts):
                if time.time() > deadline:
                    break
                if len(in_list) < 2:
                    break
                
                # Pick two distinct indices
                i = random.randrange(len(in_list))
                j = random.randrange(len(in_list) - 1)
                if j >= i:
                    j += 1
                t_out1 = in_list[i]
                t_out2 = in_list[j]
                
                current_set.remove(t_out1)
                current_set.remove(t_out2)
                _update_counts(t_out1, -1, config_counts)
                _update_counts(t_out2, -1, config_counts)
                
                # Try to add two new triples
                added_count = 0
                added_triples = []
                for t_in in not_in_set:
                    if added_count >= 2:
                        break
                    if _is_valid_addition(t_in, current_set, config_counts):
                        current_set.add(t_in)
                        _update_counts(t_in, 1, config_counts)
                        not_in_set.remove(t_in)
                        added_triples.append(t_in)
                        added_count += 1
                
                if added_count == 2:
                    # Net zero change in size, but may unlock further additions
                    two_improved = True
                    # Try immediate additions
                    for t_in in not_in_set:
                        if _is_valid_addition(t_in, current_set, config_counts):
                            current_set.add(t_in)
                            _update_counts(t_in, 1, config_counts)
                            not_in_set.remove(t_in)
                            if len(current_set) > best_size:
                                best_size = len(current_set)
                                best_set = current_set.copy()
                                best_counts = config_counts.copy()
                            break
                    break
                else:
                    # Revert
                    current_set.add(t_out1)
                    current_set.add(t_out2)
                    _update_counts(t_out1, 1, config_counts)
                    _update_counts(t_out2, 1, config_counts)
            
            if two_improved:
                not_in_set = [t for t in range(N_TRIPLES) if t not in current_set]
                continue
            
            # No improvement this round
            break
    
    return best_set, best_counts

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill
    
    best_result = set()
    best_size = 0
    best_counts = [0] * N_CONFIGS
    
    # Multiple greedy restarts with local search + exchange search
    num_greedy = 8
    for i in range(num_greedy):
        if time.time() > deadline:
            break
        seed = i * 7919 + 104729
        current_set, config_counts = _greedy_construct(seed, deadline)
        current_set, config_counts = _local_search_add(current_set, config_counts, deadline)
        
        if time.time() < deadline - 5:
            current_set, config_counts = _exchange_search(current_set, config_counts, deadline)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_result = current_set.copy()
            best_counts = config_counts.copy()
    
    # Final exchange search on best result with remaining time
    if time.time() < deadline - 2 and best_result:
        best_result, best_counts = _exchange_search(best_result, best_counts, deadline)
    
    # Final local search pass
    if time.time() < deadline:
        best_result, best_counts = _local_search_add(best_result, best_counts, deadline)
    
    return [list(ALL_TRIPLES[i]) for i in best_result]
