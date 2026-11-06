"""
Snake-in-the-box in Q_9 using a combined greedy and local-search strategy.
1. Fast greedy initialization with randomized parameters.
2. Local search: repeatedly try to extend ends.
3. Randomized restarts to escape local optima.
"""
import time
import random

N = 9
FULL_MASK = (1 << N) - 1
TIME_LIMIT = 115  # Safety margin before 120s kill

def _hamming(x, y):
    """Hamming distance between x and y."""
    return (x ^ y).bit_count()

def _greedy_snake(start=0, bit_order=None):
    """
    Greedy snake: always extend with the lowest bit flip that keeps the path induced.
    """
    if bit_order is None:
        bit_order = list(range(N))
    
    path = [start]
    used = {start}
    
    while True:
        cur = path[-1]
        extended = False
        for b in bit_order:
            v = cur ^ (1 << b)
            if v in used:
                continue
            # Check if v is adjacent to any non-consecutive vertex
            ok = True
            for p in path[:-1]:
                if _hamming(v, p) == 1:
                    ok = False
                    break
            if ok:
                path.append(v)
                used.add(v)
                extended = True
                break
        if not extended:
            break
    
    return path

def _local_search_extend(path, used, time_budget=0.5):
    """
    Try to extend the path at either end.
    """
    start_time = time.time()
    improved = True
    while improved and time.time() - start_time < time_budget:
        improved = False
        # Try tail
        cur = path[-1]
        for b in range(N):
            v = cur ^ (1 << b)
            if v in used:
                continue
            ok = True
            for p in path[:-1]:
                if _hamming(v, p) == 1:
                    ok = False
                    break
            if ok:
                path.append(v)
                used.add(v)
                improved = True
                break
        
        if improved:
            continue
            
        # Try head
        cur = path[0]
        for b in range(N):
            v = cur ^ (1 << b)
            if v in used:
                continue
            ok = True
            for p in path[1:]:
                if _hamming(v, p) == 1:
                    ok = False
                    break
            if ok:
                path.insert(0, v)
                used.add(v)
                improved = True
                break
                
    return path

def construct():
    start_time = time.time()
    best_path = []
    
    # Initial greedy
    p = _greedy_snake(0)
    best_path = p
    
    # Randomized restarts
    while time.time() - start_time < TIME_LIMIT:
        # Random start vertex
        start_v = random.randint(0, FULL_MASK)
        # Random bit order
        bits = list(range(N))
        random.shuffle(bits)
        
        p = _greedy_snake(start_v, bits)
        
        # Local search to extend ends
        used = set(p)
        p = _local_search_extend(p, used, time_budget=0.1)
        
        if len(p) > len(best_path):
            best_path = p
            # If we find a very good one, we might want to spend more time on it?
            # But for now, just keep it.
            
        # Check time
        if time.time() - start_time > TIME_LIMIT - 5:
            break

    return best_path
