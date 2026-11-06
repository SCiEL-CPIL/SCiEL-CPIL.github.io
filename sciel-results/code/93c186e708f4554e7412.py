import time
import random
from collections import defaultdict

def _check_add(S, x):
    """S is a set of ints. Check if x can be added to S."""
    S_list = list(S)
    n = len(S_list)
    
    # Condition 1: for x, no 3 distinct in S sum to 0 mod x
    if n >= 3:
        if x == 1:
            return False
        rem_to_idx = defaultdict(list)
        for i, v in enumerate(S_list):
            rem_to_idx[v % x].append(i)
        for i in range(n):
            vi = S_list[i]
            for j in range(i + 1, n):
                target = (-(vi + S_list[j])) % x
                idxs = rem_to_idx.get(target)
                if idxs:
                    for k in idxs:
                        if k != i and k != j:
                            return False
    
    # Condition 2: for each a in S, new triples involving x
    for i in range(n):
        a = S_list[i]
        if a == 1:
            if n >= 3:
                return False
            continue
        rem_to_idx = defaultdict(list)
        for j in range(n):
            if j != i:
                rem_to_idx[S_list[j] % a].append(j)
        for j in range(n):
            if j == i:
                continue
            vj = S_list[j]
            target = (-x - vj) % a
            idxs = rem_to_idx.get(target)
            if idxs:
                for k in idxs:
                    if k != j:
                        return False
    
    return True

def _is_valid(S):
    """Check if set S is valid."""
    S_list = list(S)
    n = len(S_list)
    for i, a in enumerate(S_list):
        others = [S_list[j] for j in range(n) if j != i]
        if len(others) < 3:
            continue
        if a == 1:
            return False
        for b_idx in range(len(others)):
            for c_idx in range(b_idx + 1, len(others)):
                for d_idx in range(c_idx + 1, len(others)):
                    if (others[b_idx] + others[c_idx] + others[d_idx]) % a == 0:
                        return False
    return True

def _greedy(order):
    """Greedy build using given order of candidates."""
    S = set()
    for x in order:
        if len(S) >= 48:
            break
        if _check_add(S, x):
            S.add(x)
    return S

def _backtracking_extend(S, deadline, max_depth=6):
    """Try to extend S by exploring alternative additions with backtracking.
    Returns a set (possibly S itself if no improvement found)."""
    best = set(S)
    unselected = sorted(set(range(1, 193)) - S, reverse=True)
    
    def _bt(current_S, depth, candidates):
        nonlocal best
        if time.time() > deadline:
            return
        if depth >= max_depth:
            if len(current_S) > len(best):
                best = set(current_S)
            return
        
        # Try each candidate
        for idx, x in enumerate(candidates):
            if time.time() > deadline:
                return
            if len(current_S) >= 48:
                if len(current_S) > len(best):
                    best = set(current_S)
                return
            if _check_add(current_S, x):
                current_S.add(x)
                _bt(current_S, depth + 1, candidates[idx+1:])
                current_S.discard(x)
            if len(current_S) > len(best):
                best = set(current_S)
    
    _bt(set(S), 0, unselected)
    return best

def _local_search(S, deadline):
    """Local search: try additions and swaps to improve set S."""
    unselected = set(range(1, 193)) - S
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try to add any unselected element
        for x in sorted(unselected, reverse=True):
            if len(S) >= 48:
                break
            if _check_add(S, x):
                S.add(x)
                unselected.discard(x)
                improved = True
        
        if len(S) >= 48:
            break
        
        # Try single swaps: remove a, try to add one or more
        for a in sorted(S, reverse=True):
            if time.time() >= deadline:
                break
            S.remove(a)
            unselected.add(a)
            added = []
            for x in sorted(unselected, reverse=True):
                if len(S) >= 48:
                    break
                if _check_add(S, x):
                    S.add(x)
                    unselected.discard(x)
                    added.append(x)
            if len(added) >= 2:
                improved = True
            elif len(added) == 0:
                S.add(a)
                unselected.discard(a)
            else:
                can_add_more = False
                for x in sorted(unselected, reverse=True):
                    if len(S) >= 48:
                        break
                    if _check_add(S, x):
                        can_add_more = True
                        break
                if not can_add_more:
                    S.discard(added[0])
                    unselected.add(added[0])
                    S.add(a)
                    unselected.discard(a)
                else:
                    improved = True
                    for x in sorted(unselected, reverse=True):
                        if len(S) >= 48:
                            break
                        if _check_add(S, x):
                            S.add(x)
                            unselected.discard(x)
    
    return S

def _pair_swap_search(S, deadline):
    """Try removing two elements and adding up to two others."""
    unselected = set(range(1, 193)) - S
    S_list = sorted(S, reverse=True)
    n = len(S_list)
    
    for i in range(n):
        if time.time() >= deadline:
            break
        for j in range(i + 1, n):
            if time.time() >= deadline:
                break
            a, b = S_list[i], S_list[j]
            S.remove(a)
            S.remove(b)
            unselected.add(a)
            unselected.add(b)
            
            added = []
            for x in sorted(unselected, reverse=True):
                if len(S) >= 48:
                    break
                if _check_add(S, x):
                    S.add(x)
                    unselected.discard(x)
                    added.append(x)
            
            if len(added) >= 3:
                return True
            elif len(added) == 0:
                S.add(a)
                S.add(b)
                unselected.discard(a)
                unselected.discard(b)
            else:
                can_add_more = False
                for x in sorted(unselected, reverse=True):
                    if len(S) >= 48:
                        break
                    if _check_add(S, x):
                        can_add_more = True
                        break
                if not can_add_more:
                    for x in added:
                        S.discard(x)
                        unselected.add(x)
                    S.add(a)
                    S.add(b)
                    unselected.discard(a)
                    unselected.discard(b)
                else:
                    for x in sorted(unselected, reverse=True):
                        if len(S) >= 48:
                            break
                        if _check_add(S, x):
                            S.add(x)
                            unselected.discard(x)
                    return True
    
    return False

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    best = set()
    
    random.seed(42)
    
    # Descending order greedy
    S = _greedy(range(192, 0, -1))
    if len(S) > len(best):
        best = S
    
    if time.time() > deadline:
        return sorted(best)
    
    # Ascending order greedy
    S = _greedy(range(1, 193))
    if len(S) > len(best):
        best = S
    
    if time.time() > deadline:
        return sorted(best)
    
    # Several random orders
    for _ in range(30):
        if time.time() > deadline:
            break
        order = list(range(1, 193))
        random.shuffle(order)
        S = _greedy(order)
        if len(S) > len(best):
            best = S
    
    # Backtracking extension on best
    if time.time() < deadline:
        bt_deadline = min(deadline, time.time() + 15)
        S_bt = _backtracking_extend(set(best), bt_deadline, max_depth=5)
        if len(S_bt) > len(best):
            best = S_bt
    
    if time.time() > deadline:
        return sorted(best)
    
    # Local search on best
    S = set(best)
    S = _local_search(S, deadline)
    if len(S) > len(best):
        best = S
    
    if time.time() > deadline:
        return sorted(best)
    
    # Pair swap search on best
    S = set(best)
    if _pair_swap_search(S, deadline):
        S = _local_search(S, deadline)
        if len(S) > len(best):
            best = S
    
    if time.time() > deadline:
        return sorted(best)
    
    # More random restarts with local search and backtracking
    for _ in range(60):
        if time.time() > deadline:
            break
        order = list(range(1, 193))
        random.shuffle(order)
        S = _greedy(order)
        
        # Backtracking extension
        if time.time() < deadline:
            bt_deadline = min(deadline, time.time() + 5)
            S_bt = _backtracking_extend(set(S), bt_deadline, max_depth=4)
            if len(S_bt) > len(S):
                S = S_bt
        
        # Local search
        S = _local_search(S, deadline)
        
        if len(S) < 48 and time.time() < deadline:
            if _pair_swap_search(S, deadline):
                S = _local_search(S, deadline)
        
        if len(S) > len(best):
            best = S
    
    # Final local search on best
    S = set(best)
    S = _local_search(S, deadline)
    if len(S) > len(best):
        best = S
    
    return sorted(best)
