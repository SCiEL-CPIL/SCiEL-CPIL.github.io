import time
from collections import defaultdict

def _valid(S):
    """Full validity check: for all a<b<=c in S, a does NOT divide b+c."""
    lst = sorted(S)
    n = len(lst)
    for i in range(n):
        a = lst[i]
        for j in range(i+1, n):
            b = lst[j]
            for k in range(j, n):
                c = lst[k]
                if (b + c) % a == 0:
                    return False
    return True

def _is_valid_fast(S_set, S_list):
    """Fast validity check using k-based enumeration."""
    m = len(S_list)
    for i in range(m):
        a = S_list[i]
        for j in range(i+1, m):
            b = S_list[j]
            # Need c >= b, a | (b+c), c in S
            # c = k*a - b for k >= 3 (since b > a, c >= b means k*a >= 2b+a, k >= ceil((2b+a)/a))
            k_min = max(3, (2*b + a - 1) // a)
            k_max = min(512 // a, (256 + b) // a)
            for k in range(k_min, k_max + 1):
                c = k * a - b
                if c >= b and c in S_set:
                    return False
    return True

def _count_violations(S_set, S_list):
    """Count violations per element."""
    viol_count = defaultdict(int)
    m = len(S_list)
    for i in range(m):
        a = S_list[i]
        for j in range(i+1, m):
            b = S_list[j]
            k_min = max(3, (2*b + a - 1) // a)
            k_max = min(512 // a, (256 + b) // a)
            for k in range(k_min, k_max + 1):
                c = k * a - b
                if c >= b and c in S_set:
                    viol_count[a] += 1
                    viol_count[b] += 1
                    viol_count[c] += 1
    return viol_count

def _greedy_ap_free(lo, hi, deadline):
    """Greedy 3-term-AP-free set from hi down to lo."""
    chosen = []
    sset = set()
    for x in range(hi, lo - 1, -1):
        if time.time() > deadline:
            break
        ok = True
        for b in chosen:
            c = 2 * b - x
            if c in sset:
                ok = False
                break
            for c in chosen:
                x1 = 2 * x - c
                if x1 in sset:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            chosen.append(x)
            sset.add(x)
    return chosen

def _repair_and_extend(S_initial, deadline):
    """Repair a nearly-valid set by removing worst violators, then extend by adding elements."""
    S_set = set(S_initial)
    S_list = sorted(S_initial)
    
    # Phase 1: Remove elements causing most violations until valid
    max_removals = 30
    removals = 0
    while removals < max_removals:
        if time.time() > deadline:
            break
        if _is_valid_fast(S_set, S_list):
            break
        viol_count = _count_violations(S_set, S_list)
        if not viol_count:
            break
        worst = max(viol_count, key=viol_count.get)
        S_list.remove(worst)
        S_set.discard(worst)
        removals += 1
    
    # Phase 2: Try to add back removed elements and new elements
    removed = set(S_initial) - S_set
    all_candidates = list(removed) + [x for x in range(1, 257) if x not in S_set]
    all_candidates.sort(reverse=True)
    
    for a in all_candidates:
        if time.time() > deadline:
            break
        if a in S_set:
            continue
        S_list_test = S_list + [a]
        S_list_test.sort()
        S_set_test = S_set | {a}
        if _is_valid_fast(S_set_test, S_list_test):
            S_list = S_list_test
            S_set = S_set_test
    
    return S_list, S_set

def construct():
    start = time.time()
    deadline = start + 85.0
    n = 256
    
    best_S = None
    best_size = 0
    
    # Try multiple lower bounds for the AP-free upper set
    for L in [95, 100, 105, 110, 115, 120, 125, 130, 135, 140]:
        if time.time() > deadline:
            break
        
        upper = _greedy_ap_free(L, n, deadline)
        if not upper:
            continue
        
        # Start with just the upper set, try to add small elements
        S_initial = list(upper)
        
        # Try adding small elements greedily
        S_set = set(S_initial)
        S_list = sorted(S_initial)
        
        # Try to add small elements one by one (from largest to smallest below L)
        small_candidates = list(range(L-1, 0, -1))
        for s in small_candidates:
            if time.time() > deadline:
                break
            if s in S_set:
                continue
            S_list_test = S_list + [s]
            S_list_test.sort()
            S_set_test = S_set | {s}
            if _is_valid_fast(S_set_test, S_list_test):
                S_list = S_list_test
                S_set = S_set_test
        
        # Now repair and extend
        S_list, S_set = _repair_and_extend(S_list, deadline)
        
        if len(S_set) > best_size:
            best_size = len(S_set)
            best_S = list(S_set)
    
    # Also try starting from a broader initial set
    if time.time() < deadline - 5:
        # Start with [129, 256] and repair
        S_initial = list(range(129, 257))
        S_list, S_set = _repair_and_extend(S_initial, deadline)
        if len(S_set) > best_size:
            best_size = len(S_set)
            best_S = list(S_set)
    
    if best_S is None:
        # Fallback: guaranteed valid set
        best_S = list(range(171, n+1))
    
    # Final validation
    if not _valid(best_S):
        # Fallback to safe set
        best_S = list(range(171, n+1))
    
    return best_S
