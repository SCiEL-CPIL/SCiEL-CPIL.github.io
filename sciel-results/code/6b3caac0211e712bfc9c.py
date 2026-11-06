import time
import random
from collections import defaultdict

def _check_add(S, x):
    """S is a set of ints. Check if x can be added to S.
    Faster: precompute residues once per modulus."""
    S_list = list(S)
    n = len(S_list)
    
    # Condition 1: for x, no 3 distinct in S sum to 0 mod x
    if n >= 3:
        if x == 1:
            return False
        # Build residue -> list of indices
        rem_to_idx = defaultdict(list)
        for i, v in enumerate(S_list):
            rem_to_idx[v % x].append(i)
        # Check all pairs
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
        # Build residue -> list of indices for S \ {a}
        rem_to_idx = defaultdict(list)
        for j in range(n):
            if j != i:
                rem_to_idx[S_list[j] % a].append(j)
        # Check pairs (b, c) in S\{a} with b + c + x ≡ 0 (mod a)
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
        # Check all 3-subsets of S \ {a}
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

def _local_search(S, deadline):
    """Local search: try additions and swaps to improve set S."""
    unselected = set(range(1, 193)) - S
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try to add any unselected element
        for x in list(unselected):
            if len(S) >= 48:
                break
            if _check_add(S, x):
                S.add(x)
                unselected.discard(x)
                improved = True
        
        if len(S) >= 48:
            break
        
        # Try single swaps: remove a, try to add one or more
        for a in list(S):
            if time.time() >= deadline:
                break
            S.remove(a)
            unselected.add(a)
            added = []
            for x in list(unselected):
                if len(S) >= 48:
                    break
                if _check_add(S, x):
                    S.add(x)
                    unselected.discard(x)
                    added.append(x)
            if len(added) >= 2:
                improved = True
            elif len(added) == 0:
                # Revert
                S.add(a)
                unselected.discard(a)
            else:
                # One added: check if we can add more after this swap
                can_add_more = False
                for x in list(unselected):
                    if len(S) >= 48:
                        break
                    if _check_add(S, x):
                        can_add_more = True
                        break
                if not can_add_more:
                    # Revert the swap
                    S.discard(added[0])
                    unselected.add(added[0])
                    S.add(a)
                    unselected.discard(a)
                else:
                    improved = True
                    # Continue adding
                    for x in list(unselected):
                        if len(S) >= 48:
                            break
                        if _check_add(S, x):
                            S.add(x)
                            unselected.discard(x)
    
    return S

def _pair_swap_search(S, deadline):
    """Try removing two elements and adding up to two others."""
    unselected = set(range(1, 193)) - S
    S_list = list(S)
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
            for x in list(unselected):
                if len(S) >= 48:
                    break
                if _check_add(S, x):
                    S.add(x)
                    unselected.discard(x)
                    added.append(x)
            
            if len(added) >= 3:
                # Net gain, keep
                return True
            elif len(added) == 0:
                # Revert
                S.add(a)
                S.add(b)
                unselected.discard(a)
                unselected.discard(b)
            else:
                # Check if we can add more
                can_add_more = False
                for x in list(unselected):
                    if len(S) >= 48:
                        break
                    if _check_add(S, x):
                        can_add_more = True
                        break
                if not can_add_more:
                    # Revert
                    for x in added:
                        S.discard(x)
                        unselected.add(x)
                    S.add(a)
                    S.add(b)
                    unselected.discard(a)
                    unselected.discard(b)
                else:
                    # Keep and continue adding
                    for x in list(unselected):
                        if len(S) >= 48:
                            break
                        if _check_add(S, x):
                            S.add(x)
                            unselected.discard(x)
                    return True
    
    return False

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5 second safety margin
    
    best = set()
    
    # Multiple greedy initializations
    random.seed(42)
    
    # Descending order
    S = _greedy(range(192, 0, -1))
    if len(S) > len(best):
        best = S
    
    if time.time() > deadline:
        return sorted(best)
    
    # Ascending order
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
    
    # More random restarts with local search
    for _ in range(80):
        if time.time() > deadline:
            break
        order = list(range(1, 193))
        random.shuffle(order)
        S = _greedy(order)
        
        # Local search
        S = _local_search(S, deadline)
        
        # Pair swap
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
