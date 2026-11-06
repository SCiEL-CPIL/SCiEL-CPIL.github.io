"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _can_add_to(S, x, S_set):
    """Check if x can be added to S. S is list, S_set is set."""
    if x in S_set:
        return False
    n = len(S)
    # Condition 1: for all triples from S, (b+c+d) % x != 0
    if n >= 3:
        for i in range(n):
            si = S[i]
            for j in range(i+1, n):
                sj = S[j]
                for k in range(j+1, n):
                    if (si + sj + S[k]) % x == 0:
                        return False
    # Condition 2: for all a in S, for all pairs c,d from S\{a}, (x+c+d) % a != 0
    for i, a in enumerate(S):
        for j in range(n):
            if j == i:
                continue
            c = S[j]
            for k in range(j+1, n):
                if k == i:
                    continue
                d = S[k]
                if (x + c + d) % a == 0:
                    return False
    return True

def _greedy_build(order, deadline):
    S = []
    S_set = set()
    for x in order:
        if time.time() > deadline:
            break
        if x in S_set:
            continue
        if _can_add_to(S, x, S_set):
            S.append(x)
            S_set.add(x)
    return S

def _local_search(S, deadline, max_steps=20000):
    S = list(S)
    S_set = set(S)
    all_nums = set(range(1, 193))
    available = list(all_nums - S_set)
    best = list(S)
    best_size = len(S)
    stuck = 0
    
    for _ in range(max_steps):
        if time.time() > deadline:
            break
        
        made_progress = False
        
        # Try additions
        if available:
            for x in random.sample(available, min(len(available), 20)):
                if time.time() > deadline:
                    break
                if _can_add_to(S, x, S_set):
                    S.append(x)
                    S_set.add(x)
                    available.remove(x)
                    made_progress = True
                    if len(S) > best_size:
                        best = list(S)
                        best_size = len(S)
                    break
        
        # Try swaps if no addition worked
        if not made_progress and S:
            for _ in range(3):
                if time.time() > deadline:
                    break
                if not S:
                    break
                rem_idx = random.randint(0, len(S)-1)
                rem = S[rem_idx]
                S2 = S[:rem_idx] + S[rem_idx+1:]
                S2_set = set(S2)
                
                for x in random.sample(available, min(len(available), 15)):
                    if time.time() > deadline:
                        break
                    if _can_add_to(S2, x, S2_set):
                        S = S2 + [x]
                        S_set = S2_set | {x}
                        # Update available
                        if rem in available:
                            available.remove(rem)
                        else:
                            available.append(rem)
                        if x in available:
                            available.remove(x)
                        else:
                            available.append(x)
                        made_progress = True
                        if len(S) > best_size:
                            best = list(S)
                            best_size = len(S)
                        break
                if made_progress:
                    break
        
        if not made_progress:
            stuck += 1
            if stuck > 100:
                break
        else:
            stuck = 0
    
    return best

def _verify(S):
    """Verify S is valid."""
    S_set = set(S)
    for a in S:
        others = [x for x in S if x != a]
        for b, c, d in combinations(others, 3):
            if (b + c + d) % a == 0:
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 78
    
    best = []
    best_size = 0
    random.seed(99999)
    
    # Multiple greedy builds
    orderings = [list(range(192, 0, -1)), list(range(1, 193))]
    for _ in range(40):
        o = list(range(1, 193))
        random.shuffle(o)
        orderings.append(o)
    
    for order in orderings:
        if time.time() > deadline - 20:
            break
        S = _greedy_build(order, deadline)
        if len(S) > best_size:
            best = list(S)
            best_size = len(S)
    
    # Local search on best
    if best and time.time() < deadline - 8:
        result = _local_search(best, deadline, max_steps=40000)
        if len(result) > best_size:
            best = result
            best_size = len(result)
    
    # More local searches
    for _ in range(20):
        if time.time() > deadline - 3:
            break
        o = list(range(1, 193))
        random.shuffle(o)
        S = _greedy_build(o, deadline)
        S = _local_search(S, deadline, max_steps=15000)
        if len(S) > best_size:
            best = list(S)
            best_size = len(S)
    
    # Final verification
    if best and not _verify(best):
        # Fallback to a simple valid set
        best = [192, 191, 190, 189, 188]
    
    return best
# EVOLVE-BLOCK-END
