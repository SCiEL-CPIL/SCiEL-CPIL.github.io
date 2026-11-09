import time
import random
import itertools

N = 18
MIN_DISTANCE = 6
DEADLINE = 85  # seconds, safety margin before 90s kill

def _generate_offsets(min_dist):
    """Generate all bitmasks with popcount < min_dist."""
    offsets = []
    for k in range(min_dist):
        for combo in itertools.combinations(range(N), k):
            mask = 0
            for i in combo:
                mask |= (1 << i)
            offsets.append(mask)
    return offsets

def _greedy_set(deadline, offsets, strategy='lowest', seed=0):
    """Greedy construction using set-based approach."""
    random.seed(seed)
    available = set(range(2**N))
    selected = []
    
    start_time = time.time()
    
    while available:
        if time.time() > deadline:
            break
        
        if strategy == 'lowest':
            w = min(available)
        elif strategy == 'highest':
            w = max(available)
        elif strategy == 'random':
            w = random.choice(list(available))
        else:
            w = next(iter(available))
        
        selected.append(w)
        
        # Remove all words within distance < MIN_DISTANCE from w
        to_remove = set()
        for offset in offsets:
            neighbor = w ^ offset
            if neighbor in available:
                to_remove.add(neighbor)
        available -= to_remove
    
    return selected

def _local_improve_set(code, offsets, deadline):
    """Try to add more words using set-based approach."""
    if not code:
        return code
    
    code_set = set(code)
    improved = list(code)
    improved_set = set(code)
    
    # Pre-compute all words not in code
    all_words = set(range(2**N))
    candidates = all_words - improved_set
    
    # Sort candidates by weight for better packing
    sorted_candidates = sorted(candidates, key=lambda x: (x.bit_count(), x))
    
    for w in sorted_candidates:
        if time.time() > deadline:
            break
        
        can_add = True
        for s in improved_set:
            if (w ^ s).bit_count() < MIN_DISTANCE:
                can_add = False
                break
        
        if can_add:
            improved.append(w)
            improved_set.add(w)
    
    return improved

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    offsets = _generate_offsets(MIN_DISTANCE)
    
    best_code = []
    best_len = 0
    
    strategies = ['lowest', 'highest', 'random', 'random', 'random', 'random', 'random', 'random']
    seeds = [0, 1, 42, 123, 999, 7, 13, 2024]
    
    for i in range(len(strategies)):
        if time.time() > deadline - 3:
            break
        
        code = _greedy_set(deadline - 2, offsets, strategy=strategies[i], seed=seeds[i])
        
        if len(code) > best_len:
            best_len = len(code)
            best_code = code
    
    if best_code and time.time() < deadline - 3:
        improved = _local_improve_set(best_code, offsets, deadline - 2)
        if len(improved) > best_len:
            best_code = improved
    
    return best_code
