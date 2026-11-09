"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def _is_valid_set(S):
    """Check if set S has no forbidden triple."""
    S_set = set(S)
    for i in range(len(S)):
        for j in range(i+1, len(S)):
            for k in range(j+1, len(S)):
                a, b, c = S[i], S[j], S[k]
                if (a+b) in S_set and (a+c) in S_set and (b+c) in S_set:
                    return False
    return True

def _greedy_construct(n=128, order=None):
    """Greedy construction: try to add each element, skip if it creates a forbidden triple."""
    if order is None:
        order = list(range(1, n+1))
    
    S = set()
    for x in order:
        # Check if adding x creates a forbidden triple
        can_add = True
        for i, a in enumerate(S):
            if a == x:
                continue
            for b in list(S)[i+1:]:
                if b == x:
                    continue
                # Triple {x, a, b}: forbidden if a+b in S, a+x in S, b+x in S
                if (a + b) in S and (a + x) in S and (b + x) in S:
                    can_add = False
                    break
            if not can_add:
                break
        if can_add:
            S.add(x)
    return S

def _local_search_improve(S, n=128, max_iter=5000):
    """Try to add elements not in S, or swap elements to improve size."""
    S = set(S)
    improved = True
    while improved:
        improved = False
        # Try adding each missing element
        for x in range(1, n+1):
            if x in S:
                continue
            can_add = True
            S_list = list(S)
            for i in range(len(S_list)):
                a = S_list[i]
                for j in range(i+1, len(S_list)):
                    b = S_list[j]
                    if (a + b) in S and (a + x) in S and (b + x) in S:
                        can_add = False
                        break
                if not can_add:
                    break
            if can_add:
                S.add(x)
                improved = True
                break  # Restart after adding
        if improved:
            continue
        # Try swapping: remove one element, add another
        S_list = list(S)
        for i, a in enumerate(S_list):
            for x in range(1, n+1):
                if x in S or x == a:
                    continue
                S_new = S - {a} | {x}
                # Quick check: does S_new have a forbidden triple?
                # Only need to check triples involving x
                valid = True
                for j in range(len(S_new)):
                    if S_new[j] == x:
                        continue
                    for k in range(j+1, len(S_new)):
                        if S_new[k] == x:
                            continue
                        b, c = S_new[j], S_new[k]
                        if b == x or c == x:
                            continue
                        if (b + c) in S_new and (b + x) in S_new and (c + x) in S_new:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    S = S_new
                    improved = True
                    break
            if improved:
                break
    return S

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    
    # Try multiple greedy orders
    best_set = set(range(65, 129))  # Upper half baseline, size 64
    
    # Greedy in increasing order
    if time.time() < deadline:
        S_inc = _greedy_construct(128, order=list(range(1, 129)))
        if len(S_inc) > len(best_set):
            best_set = S_inc
    
    # Greedy in decreasing order
    if time.time() < deadline:
        S_dec = _greedy_construct(128, order=list(range(128, 0, -1)))
        if len(S_dec) > len(best_set):
            best_set = S_dec
    
    # Try to improve best_set with local search
    if time.time() < deadline:
        S_improved = _local_search_improve(best_set, 128, max_iter=2000)
        if len(S_improved) > len(best_set):
            best_set = S_improved
    
    # Verify validity
    S_final = list(best_set)
    if not _is_valid_set(S_final):
        # Fallback to known valid set
        S_final = list(range(65, 129))
    
    # Ensure all are Python ints in 1..128
    S_final = [int(x) for x in S_final if 1 <= x <= 128]
    
    return S_final
# EVOLVE-BLOCK-END
