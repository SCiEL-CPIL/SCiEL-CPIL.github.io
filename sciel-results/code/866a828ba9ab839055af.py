import time
import random
import sys

_N = 192
_MAX = 48
_START_TIME = time.time()
_DEADLINE = _START_TIME + 85.0  # Safety margin

def _time_left():
    return _DEADLINE - time.time()

# Precompute divisibility masks for sums
# Max sum is 192*3 = 576
_MAX_SUM = 3 * _N
# mask_div[d] is a bytearray of size _MAX_SUM+1 where mask_div[d][s] = 1 if d divides s
_div_masks = [None] * (_N + 1)
for d in range(1, _N + 1):
    _div_masks[d] = bytearray(_MAX_SUM + 1)
    for s in range(d, _MAX_SUM + 1, d):
        _div_masks[d][s] = 1

def _can_add(S, x):
    """
    Check if x can be added to set S.
    S is a set of ints.
    Condition 1: x does not divide sum of any 3 distinct elements in S.
    Condition 2: For each a in S, a does not divide sum of x and any 2 distinct other elements in S \ {a}.
    """
    if not S:
        return True
    
    S_list = list(S)
    n = len(S_list)
    
    # Condition 1: x does not divide sum of any 3 in S
    mask_x = _div_masks[x]
    
    # Iterate all triples in S
    # Optimization: Early exit
    for i in range(n):
        b = S_list[i]
        for j in range(i+1, n):
            c = S_list[j]
            bc = b + c
            # Check if there exists d in S_list[j+1:] such that (bc + d) % x == 0
            # i.e., mask_x[bc + d] == 1
            for k in range(j+1, n):
                d = S_list[k]
                if mask_x[bc + d]:
                    return False
    
    # Condition 2: For each a in S, a does not divide (x + b + c) for any b,c in S\{a}
    for idx_a, a in enumerate(S_list):
        mask_a = _div_masks[a]
        # We need to check pairs (b, c) from S \ {a}
        # Sum = x + b + c
        # Check mask_a[x + b + c]
        for i in range(n):
            b = S_list[i]
            if b == a: continue
            for j in range(i+1, n):
                c = S_list[j]
                if c == a: continue
                if mask_a[x + b + c]:
                    return False
    return True

def _greedy(order):
    S = set()
    for x in order:
        if x in S:
            continue
        if len(S) >= _MAX:
            break
        if _can_add(S, x):
            S.add(x)
    return S

def _local_search(S, time_budget=10.0):
    S = set(S)
    start = time.time()
    best = set(S)
    best_len = len(S)
    
    while time.time() - start < time_budget and _time_left() > 5:
        improved = False
        
        # 1. Try to add any missing element
        if len(S) < _MAX:
            candidates = [v for v in range(1, _N + 1) if v not in S]
            random.shuffle(candidates)
            for x in candidates:
                if _can_add(S, x):
                    S.add(x)
                    improved = True
                    if len(S) > best_len:
                        best_len = len(S)
                        best = set(S)
                    break # Restart loop to check further additions
        
        # 2. Try 1-out-1-in swap
        if not improved and len(S) > 0:
            elems = list(S)
            random.shuffle(elems)
            candidates = [v for v in range(1, _N + 1) if v not in S]
            random.shuffle(candidates)
            
            # Limit iterations to avoid long stalls
            for a in elems[:20]:
                S_temp = S - {a}
                for x in candidates[:50]:
                    if _can_add(S_temp, x):
                        # Valid swap
                        S = S_temp
                        S.add(x)
                        improved = True
                        if len(S) > best_len:
                            best_len = len(S)
                            best = set(S)
                        break
                if improved:
                    break
        
        # 3. Try 1-out-2-in (Net +1)
        if not improved and len(S) >= 1 and len(S) < _MAX:
            elems = list(S)
            random.shuffle(elems)
            candidates = [v for v in range(1, _N + 1) if v not in S]
            random.shuffle(candidates)
            
            for a in elems[:10]:
                S_temp = S - {a}
                # Try to add two
                found = False
                for x in candidates[:30]:
                    if _can_add(S_temp, x):
                        S_temp2 = S_temp | {x}
                        for y in candidates[:30]:
                            if y == x: continue
                            if _can_add(S_temp2, y):
                                S = S_temp2 | {y}
                                improved = True
                                found = True
                                if len(S) > best_len:
                                    best_len = len(S)
                                    best = set(S)
                                break
                        if found:
                            break
                if found:
                    break
        
        if not improved:
            # Perturb: Remove 2 random elements to escape local optimum
            if len(S) > 2:
                to_remove = random.sample(list(S), 2)
                S -= set(to_remove)
            else:
                # Stuck, try random restart of local search part?
                # Just break and let outer loop handle
                break
                
    return best

def _band_optimization():
    """
    Structural idea from inspiration: Use value bands to guide the search.
    Larger values are less likely to divide sums, so prioritize them.
    """
    best = set()
    best_len = 0
    
    # Try various band configurations
    configs = [
        [(100, 192)],
        [(80, 192)],
        [(60, 192)],
        [(50, 192)],
        [(40, 192)],
        [(100, 192), (20, 39)],
        [(100, 192), (30, 49)],
        [(90, 192), (20, 50)],
        [(80, 192), (15, 40)],
        [(70, 192), (10, 35)],
        [(120, 192), (40, 70)],
        [(110, 192), (30, 60)],
        [(100, 192), (25, 50)],
        [(95, 192), (20, 45)],
        [(90, 192), (15, 40)],
    ]
    
    for config in configs:
        if _time_left() < 5:
            break
        for _ in range(5):
            S = _model_parameterize(config)
            if len(S) > best_len:
                best_len = len(S)
                best = set(S)
    
    return best

def _model_parameterize(band_widths):
    """
    Parameterize the set by value bands.
    band_widths: list of (start, end) ranges to sample from.
    Returns a candidate set.
    """
    S = set()
    for start, end in band_widths:
        # Pick values from [start, end]
        candidates = list(range(start, min(end, _N) + 1))
        random.shuffle(candidates)
        for x in candidates:
            if len(S) >= _MAX:
                break
            if x not in S and _can_add(S, x):
                S.add(x)
    return S

def construct():
    random.seed(42)
    best_set = set()
    
    # Phase 1: Band optimization (model-and-optimize)
    if _time_left() > 30:
        S = _band_optimization()
        if len(S) > len(best_set):
            best_set = S
    
    # Phase 2: Greedy with different orders
    orders = [
        list(range(_N, 0, -1)),
        list(range(1, _N + 1)),
        # Biased towards high
        list(range(100, _N + 1)) + list(range(1, 100)),
        list(range(150, _N + 1)) + list(range(1, 150)),
    ]
    
    for order in orders:
        if _time_left() < 30:
            break
        S = _greedy(order)
        if len(S) > len(best_set):
            best_set = S
            
    # Phase 3: Randomized Greedy
    for _ in range(10):
        if _time_left() < 20:
            break
        # Weighted order
        weights = {v: random.random() + v/100.0 for v in range(1, _N+1)}
        order = sorted(range(1, _N+1), key=lambda v: -weights[v])
        S = _greedy(order)
        if len(S) > len(best_set):
            best_set = S

    # Phase 4: Local Search
    if _time_left() > 10:
        budget = min(30.0, _time_left() - 5)
        S = _local_search(best_set, time_budget=budget)
        if len(S) > len(best_set):
            best_set = S
            
    # Phase 5: Final Polish - Try to add any remaining elements one by one
    if len(best_set) < _MAX:
        remaining = [v for v in range(1, _N + 1) if v not in best_set]
        random.shuffle(remaining)
        for x in remaining:
            if len(best_set) >= _MAX:
                break
            if _can_add(best_set, x):
                best_set.add(x)

    return list(best_set)
