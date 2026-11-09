"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def _is_valid_set(S):
    """Check if set S has no forbidden triple."""
    S_set = set(S)
    S_list = sorted(S)
    for i in range(len(S_list)):
        a = S_list[i]
        for j in range(i+1, len(S_list)):
            b = S_list[j]
            if (a+b) not in S_set:
                continue
            for k in range(len(S_list)):
                c = S_list[k]
                if c == a or c == b:
                    continue
                if (a+c) in S_set and (b+c) in S_set:
                    return False
    return True

def _find_conflicts(S, n=128):
    """Find all forbidden triples and count conflicts per element."""
    S_set = set(S)
    S_list = sorted(S)
    conflict_count = {}
    
    for i in range(len(S_list)):
        a = S_list[i]
        for j in range(i+1, len(S_list)):
            b = S_list[j]
            s_ab = a + b
            if s_ab not in S_set:
                continue
            for k in range(len(S_list)):
                c = S_list[k]
                if c == a or c == b:
                    continue
                if (a + c) in S_set and (b + c) in S_set:
                    for v in (a, b, c):
                        conflict_count[v] = conflict_count.get(v, 0) + 1
    return conflict_count

def _conflict_removal(n=128, deadline=None):
    """Start with all elements, remove those in most conflicts until valid."""
    S = set(range(1, n+1))
    
    if deadline is None:
        deadline = time.time() + 30
    
    while time.time() < deadline:
        conflict_count = _find_conflicts(S, n)
        if not conflict_count:
            break
        # Remove element with highest conflict count
        max_elem = max(conflict_count, key=conflict_count.get)
        S.discard(max_elem)
    
    return S

def _greedy_add(S, n=128, deadline=None):
    """Try to add elements back to S greedily."""
    S = set(S)
    if deadline is None:
        deadline = time.time() + 10
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for x in range(1, n+1):
            if x in S:
                continue
            # Check if adding x creates a forbidden triple
            N_x = [a for a in S if a + x <= n and (a + x) in S]
            valid = True
            for i in range(len(N_x)):
                a = N_x[i]
                for j in range(i+1, len(N_x)):
                    b = N_x[j]
                    if a + b in S:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                S.add(x)
                improved = True
                break
    return S

def _local_search_swap(S, n=128, deadline=None):
    """Try swaps: remove one, add another, to improve size."""
    S = set(S)
    if deadline is None:
        deadline = time.time() + 10
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        S_list = list(S)
        random.shuffle(S_list)
        for a in S_list:
            S_without_a = S - {a}
            for x in range(1, n+1):
                if x in S_without_a:
                    continue
                N_x = [b for b in S_without_a if b + x <= n and (b + x) in S_without_a]
                valid = True
                for i in range(len(N_x)):
                    b = N_x[i]
                    for j in range(i+1, len(N_x)):
                        c = N_x[j]
                        if b + c in S_without_a:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    S = S_without_a | {x}
                    improved = True
                    break
            if improved:
                break
    return S

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s limit
    
    # Phase 1: Conflict removal to get a valid set
    S = _conflict_removal(128, deadline=min(deadline, start_time + 30))
    
    # Phase 2: Greedy add to recover elements
    if time.time() < deadline:
        S = _greedy_add(S, 128, deadline=min(deadline, time.time() + 20))
    
    # Phase 3: Local search with swaps
    if time.time() < deadline:
        S = _local_search_swap(S, 128, deadline=min(deadline, time.time() + 20))
    
    # Verify validity
    S_final = list(S)
    if not _is_valid_set(S_final):
        # Fallback to known valid set
        S_final = list(range(65, 129))
    
    # Ensure all are Python ints in 1..128
    S_final = [int(x) for x in S_final if 1 <= x <= 128]
    
    return S_final
# EVOLVE-BLOCK-END
