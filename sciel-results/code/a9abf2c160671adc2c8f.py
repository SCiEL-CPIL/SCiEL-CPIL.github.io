# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations
from collections import defaultdict

_N = 192
_MAX = 48
_DEADLINE = time.time() + 85.0  # Safety margin before 90s kill

def _is_valid_addition(S_set, x):
    """Check if adding x to valid set S_set keeps it valid. S_set is a set of ints."""
    S_list = list(S_set)
    n = len(S_list)
    
    # Check 1: x does not divide sum of any 3 in S_set
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                s = S_list[i] + S_list[j] + S_list[k]
                if s % x == 0:
                    return False
    
    # Check 2: for each a in S_set, a does not divide sum of x + 2 others from S_set
    for i, a in enumerate(S_list):
        for j in range(n):
            if j == i:
                continue
            for k in range(j + 1, n):
                if k == i:
                    continue
                s = x + S_list[j] + S_list[k]
                if s % a == 0:
                    return False
    
    return True

def _greedy_optimized(order):
    """Greedy with given order."""
    S = set()
    for x in order:
        if x in S:
            continue
        if _is_valid_addition(S, x):
            S.add(x)
            if len(S) >= _MAX:
                break
    return S

def _local_search_model(S_init, time_budget=20.0):
    """Local search: try swaps, insertions, and 1-out-2-in moves."""
    S = set(S_init)
    start_time = time.time()
    best = set(S)
    best_len = len(S)
    all_vals = list(range(1, _N + 1))
    
    while time.time() - start_time < time_budget:
        improved = False
        
        # Try insertions
        if len(S) < _MAX:
            candidates = [v for v in all_vals if v not in S]
            random.shuffle(candidates)
            for x in candidates[:50]:
                if _is_valid_addition(S, x):
                    S.add(x)
                    improved = True
                    if len(S) > best_len:
                        best_len = len(S)
                        best = set(S)
                    break
        
        # Try 1-out-2-in moves (Grow set by 1)
        if len(S) < _MAX and len(S) >= 1:
            to_remove = list(S)
            random.shuffle(to_remove)
            candidates = [v for v in all_vals if v not in S]
            candidates.sort(reverse=True)
            
            found_move = False
            for a in to_remove[:10]:
                if found_move:
                    break
                S_temp = S - {a}
                for i, x in enumerate(candidates[:40]):
                    if found_move:
                        break
                    if not _is_valid_addition(S_temp, x):
                        continue
                    S_temp_x = S_temp | {x}
                    for y in candidates[i+1:i+40]:
                        if y == x:
                            continue
                        if _is_valid_addition(S_temp_x, y):
                            S = S_temp_x | {y}
                            improved = True
                            found_move = True
                            if len(S) > best_len:
                                best_len = len(S)
                                best = set(S)
                            break
        
        # Try swaps (1-out-1-in)
        if not improved and len(S) >= 1:
            to_remove = list(S)
            random.shuffle(to_remove)
            candidates = [v for v in all_vals if v not in S]
            random.shuffle(candidates)
            
            for a in to_remove[:30]:
                if improved:
                    break
                for x in candidates[:30]:
                    S_new = S - {a}
                    if x not in S_new and _is_valid_addition(S_new, x):
                        S = S_new
                        S.add(x)
                        improved = True
                        if len(S) > best_len:
                            best_len = len(S)
                            best = set(S)
                        break
        
        if not improved:
            # Perturb: remove 2 random elements to escape local optima
            if len(S) > 2:
                to_remove = random.sample(list(S), 2)
                S -= set(to_remove)
    
    return best

def _band_optimization():
    """Optimize band parameters."""
    best = set()
    best_len = 0
    
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
        [(100, 192), (1, 30)],
        [(110, 192), (1, 40)],
        [(90, 192), (1, 50)],
    ]
    
    for config in configs:
        for _ in range(10):
            S = set()
            for start, end in config:
                candidates = list(range(start, min(end, _N) + 1))
                random.shuffle(candidates)
                for x in candidates:
                    if len(S) >= _MAX:
                        break
                    if x not in S and _is_valid_addition(S, x):
                        S.add(x)
            if len(S) > best_len:
                best_len = len(S)
                best = set(S)
    
    return best

def construct():
    """Main entry point: find maximum valid set."""
    deadline = _DEADLINE
    random.seed(42)  # Deterministic
    
    # Phase 1: Band optimization
    best = _band_optimization()
    
    # Phase 2: Greedy with various orders
    if time.time() < deadline - 30:
        orders = [
            range(_N, 0, -1),
            range(1, _N + 1),
            sorted(range(1, _N + 1), key=lambda x: -x),
            list(range(50, _N + 1)) + list(range(1, 50)),
            list(range(80, _N + 1)) + list(range(1, 80)),
            list(range(100, _N + 1)) + list(range(1, 100)),
            list(range(120, _N + 1)) + list(range(1, 120)),
        ]
        for order in orders:
            if time.time() > deadline - 30:
                break
            S = _greedy_optimized(order)
            if len(S) > len(best):
                best = S
    
    # Phase 3: Multi-start randomized greedy
    if time.time() < deadline - 20:
        for _ in range(40):
            if time.time() > deadline - 20:
                break
            order = list(range(1, _N + 1))
            weights = [random.random() * 50 + v for v in order]
            order.sort(key=lambda v: -weights[v - 1])
            S = _greedy_optimized(order)
            if len(S) > len(best):
                best = S
    
    # Phase 4: Local search from best
    if time.time() < deadline - 10:
        budget = min(25.0, deadline - time.time() - 5)
        S = _local_search_model(best, time_budget=budget)
        if len(S) > len(best):
            best = S
    
    # Phase 5: Another local search round if time permits
    if time.time() < deadline - 5:
        budget = min(8.0, deadline - time.time() - 2)
        S = _local_search_model(best, time_budget=budget)
        if len(S) > len(best):
            best = S
    
    return list(best)
# EVOLVE-BLOCK-END
