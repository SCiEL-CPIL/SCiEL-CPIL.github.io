"""Combined program: precomputed K(2,2,2) configs + greedy + simulated annealing."""
import time
import random
import itertools

N_POINTS = 16
ALL_TRIPLES = list(itertools.combinations(range(N_POINTS), 3))
TRIPLE_INDEX = {t: i for i, t in enumerate(ALL_TRIPLES)}
N_TRIPLES = len(ALL_TRIPLES)

# Precompute all K(2,2,2) configurations
# A K(2,2,2) is defined by 3 disjoint pairs; 8 transversals must all be present.
CONFIG_TRIPLES = []   # list of lists of 8 triple indices
TRIPLE_CONFIGS = [[] for _ in range(N_TRIPLES)]  # for each triple, list of config ids it belongs to

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
    """Check if adding triple_idx keeps all configs with < 8 present triples."""
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
    """Random-order greedy construction."""
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
    """Repeatedly try to add any valid triple until none can be added."""
    improved = True
    while improved and time.time() < deadline:
        improved = False
        candidates = [t for t in range(N_TRIPLES) if t not in current_set]
        random.shuffle(candidates)
        for t_idx in candidates:
            if time.time() > deadline:
                return current_set, config_counts
            if _is_valid_addition(t_idx, current_set, config_counts):
                current_set.add(t_idx)
                _update_counts(t_idx, 1, config_counts)
                improved = True
    return current_set, config_counts

def _simulated_annealing(current_set, config_counts, deadline, start_temp=5.0, end_temp=0.01):
    """Simulated annealing with swap moves to escape local optima."""
    best_set = current_set.copy()
    best_size = len(current_set)
    best_counts = config_counts.copy()
    
    # First, fill with local search additions
    current_set, config_counts = _local_search_add(current_set, config_counts, deadline)
    if len(current_set) > best_size:
        best_size = len(current_set)
        best_set = current_set.copy()
        best_counts = config_counts.copy()
    
    temp = start_temp
    cooling = 0.995
    min_temp = end_temp
    min_temp = 0.01
    
    while time.time() < deadline and temp > min_temp:
        # Pick a random triple to remove
        if not current_set:
            break
        t_out = random.choice(list(current_set))
        current_set.remove(t_out)
        _update_counts(t_out, -1, config_counts)
        
        # Try to add a random triple (possibly the same one back)
        t_in = random.randrange(N_TRIPLES)
        
        if t_in not in current_set and _is_valid_addition(t_in, current_set, config_counts):
            current_set.add(t_in)
            _update_counts(t_in, 1, config_counts)
            # Swap accepted (size unchanged, but different composition)
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = current_set.copy()
                best_counts = config_counts.copy()
        else:
            # Revert
            if t_in in current_set:
                current_set.remove(t_in)
                _update_counts(t_in, -1, config_counts)
            current_set.add(t_out)
            _update_counts(t_out, 1, config_counts)
        
        # After a swap, try to add more
        current_set, config_counts = _local_search_add(current_set, config_counts, deadline)
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set.copy()
            best_counts = config_counts.copy()
        
        temp *= cooling
    
    return best_set, best_counts

def construct():
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin before 150s kill
    
    best_result = set()
    best_size = 0
    best_counts = [0] * N_CONFIGS
    
    # Multiple greedy restarts with local search
    num_greedy = 15
    for i in range(num_greedy):
        if time.time() > deadline:
            break
        seed = i * 7919 + 104729
        current_set, config_counts = _greedy_construct(seed, deadline)
        current_set, config_counts = _local_search_add(current_set, config_counts, deadline)
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_result = current_set.copy()
            best_counts = config_counts.copy()
    
    # Simulated annealing on the best result
    if time.time() < deadline:
        sa_deadline = deadline - 5  # leave a bit of buffer
        best_result, best_counts = _simulated_annealing(best_result, best_counts, sa_deadline)
    
    # Final local search pass
    if time.time() < deadline:
        best_result, best_counts = _local_search_add(best_result, best_counts, deadline)
    
    # Convert to list of triples
    return [list(ALL_TRIPLES[i]) for i in best_result]
