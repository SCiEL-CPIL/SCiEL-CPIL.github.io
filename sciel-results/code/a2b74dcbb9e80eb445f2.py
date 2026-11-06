"""Program for reciprocal weight of three-progression-free sets on [1,256]."""
import time
from fractions import Fraction

# EVOLVE-BLOCK-START

def _build_ap_index(N):
    """Precompute all 3-APs and index by element."""
    aps_containing = {i: [] for i in range(1, N + 1)}
    for b in range(2, N):
        min_a = max(1, 2 * b - N)
        for a in range(min_a, b):
            c = 2 * b - a
            if c <= N:
                ap = (a, b, c)
                aps_containing[a].append(ap)
                aps_containing[b].append(ap)
                aps_containing[c].append(ap)
    return aps_containing

def _can_add(aps_containing, S, y):
    """Check if adding y to set S creates a 3-AP."""
    for ap in aps_containing[y]:
        a, b, c = ap
        if y == a:
            if b in S and c in S:
                return False
        elif y == b:
            if a in S and c in S:
                return False
        else:  # y == c
            if a in S and b in S:
                return False
    return True

def _can_swap(aps_containing, S, x, y):
    """Check if replacing x with y in set S keeps it 3-AP-free."""
    for ap in aps_containing[y]:
        a, b, c = ap
        if y == a:
            if (b in S and b != x) and (c in S and c != x):
                return False
        elif y == b:
            if (a in S and a != x) and (c in S and c != x):
                return False
        else:  # y == c
            if (a in S and a != x) and (b in S and b != x):
                return False
    return True

def _greedy_ascending(N, aps_containing, start_from=1):
    """Greedy: add numbers from start_from upward if no 3-AP is formed."""
    S = set()
    for x in range(start_from, N + 1):
        if _can_add(aps_containing, S, x):
            S.add(x)
    return S

def _greedy_descending(N, aps_containing, start_from=N):
    """Greedy: add numbers from start_from downward if no 3-AP is formed."""
    S = set()
    for x in range(start_from, 0, -1):
        if _can_add(aps_containing, S, x):
            S.add(x)
    return S

def _local_search(N, aps_containing, initial_S, deadline):
    """Local search: try to add missing small numbers, then swap large for small."""
    S = set(initial_S)
    
    # Phase 1: Try to add any missing number (smallest first for max gain)
    while time.time() < deadline:
        added = False
        for y in range(1, N + 1):
            if y not in S and _can_add(aps_containing, S, y):
                S.add(y)
                added = True
                break
        if not added:
            break
    
    # Phase 2: Try swaps - replace large x with smaller y
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for x in sorted(S, reverse=True):
            if time.time() > deadline:
                break
            for y in range(1, x):
                if y in S:
                    continue
                if _can_swap(aps_containing, S, x, y):
                    S.discard(x)
                    S.add(y)
                    improved = True
                    break
            if improved:
                break
    
    # Phase 3: Try to add again after swaps
    while time.time() < deadline:
        added = False
        for y in range(1, N + 1):
            if y not in S and _can_add(aps_containing, S, y):
                S.add(y)
                added = True
                break
        if not added:
            break
    
    return S

def _score(S):
    """Compute exact reciprocal sum."""
    total = Fraction(0, 1)
    for x in S:
        total += Fraction(1, x)
    return total

def construct():
    start_time = time.time()
    deadline = start_time + 75  # Safety margin before 90s kill
    
    N = 256
    aps_containing = _build_ap_index(N)
    
    best_S = None
    best_score = Fraction(0, 1)
    
    # Strategy 1: Greedy ascending (primary - small numbers have high reciprocal weight)
    S1 = _greedy_ascending(N, aps_containing)
    score1 = _score(S1)
    best_S = S1
    best_score = score1
    
    # Local search on S1
    S1_ls = _local_search(N, aps_containing, S1, deadline)
    score1_ls = _score(S1_ls)
    if score1_ls > best_score:
        best_score = score1_ls
        best_S = S1_ls
    
    # Strategy 2: Greedy descending
    if time.time() < deadline:
        S2 = _greedy_descending(N, aps_containing)
        score2 = _score(S2)
        if score2 > best_score:
            best_score = score2
            best_S = S2
        
        S2_ls = _local_search(N, aps_containing, S2, deadline)
        score2_ls = _score(S2_ls)
        if score2_ls > best_score:
            best_score = score2_ls
            best_S = S2_ls
    
    # Strategy 3: Try different starting points for greedy ascending
    if time.time() < deadline:
        for start in range(1, min(N, 20) + 1):
            if time.time() > deadline:
                break
            S3 = _greedy_ascending(N, aps_containing, start_from=start)
            score3 = _score(S3)
            if score3 > best_score:
                best_score = score3
                best_S = S3
            
            S3_ls = _local_search(N, aps_containing, S3, deadline)
            score3_ls = _score(S3_ls)
            if score3_ls > best_score:
                best_score = score3_ls
                best_S = S3_ls
    
    # Strategy 4: Randomized restarts
    import random
    if time.time() < deadline:
        for trial in range(10):
            if time.time() > deadline:
                break
            # Random subset of large numbers to start, then fill greedily
            S4 = set()
            # Pick a few random large numbers as seeds
            for _ in range(random.randint(0, 3)):
                x = random.randint(N // 2, N)
                if _can_add(aps_containing, S4, x):
                    S4.add(x)
            
            # Fill greedily ascending
            for x in range(1, N + 1):
                if x not in S4 and _can_add(aps_containing, S4, x):
                    S4.add(x)
            
            score4 = _score(S4)
            if score4 > best_score:
                best_score = score4
                best_S = S4
            
            S4_ls = _local_search(N, aps_containing, S4, deadline)
            score4_ls = _score(S4_ls)
            if score4_ls > best_score:
                best_score = score4_ls
                best_S = S4_ls
    
    # Strategy 5: Aggressive swap-based improvement on best
    if time.time() < deadline and best_S:
        # Try multiple passes of swaps
        S5 = set(best_S)
        for _ in range(3):
            if time.time() > deadline:
                break
            improved = True
            while improved and time.time() < deadline:
                improved = False
                for x in sorted(S5, reverse=True):
                    if time.time() > deadline:
                        break
                    for y in range(1, x):
                        if y in S5:
                            continue
                        if _can_swap(aps_containing, S5, x, y):
                            S5.discard(x)
                            S5.add(y)
                            improved = True
                            break
                    if improved:
                        break
            
            # Try to add after swaps
            while time.time() < deadline:
                added = False
                for y in range(1, N + 1):
                    if y not in S5 and _can_add(aps_containing, S5, y):
                        S5.add(y)
                        added = True
                        break
                if not added:
                    break
        
        score5 = _score(S5)
        if score5 > best_score:
            best_score = score5
            best_S = S5
    
    if not best_S:
        return [1]
    
    return sorted(best_S)

# EVOLVE-BLOCK-END
