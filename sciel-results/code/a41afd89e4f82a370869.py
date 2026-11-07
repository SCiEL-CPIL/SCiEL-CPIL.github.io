"""Suite task: large integer sets with no two-other-product divisibility."""
import time
import random
import math

NMAX = 256
UNIVERSE = list(range(2, NMAX + 1))

def _can_add(x, S):
    """
    Check if x can be added to set S.
    S is a list of ints.
    For every three distinct a,b,c in the final set, a does not divide b*c.
    When adding x, we must check all triples involving x:
    - (x, b, c): x | b*c?
    - (b, x, c): b | x*c?
    - (c, x, b): c | x*b?
    where b,c are distinct elements of S.
    """
    n = len(S)
    for i in range(n):
        b = S[i]
        for j in range(n):
            if i == j:
                continue
            c = S[j]
            if (b * c) % x == 0:
                return False
            if (x * c) % b == 0:
                return False
            if (x * b) % c == 0:
                return False
    return True

def _greedy(order, deadline):
    S = []
    S_set = set()
    for x in order:
        if time.time() > deadline:
            break
        if x in S_set:
            continue
        if _can_add(x, S):
            S.append(x)
            S_set.add(x)
    return S

def _greedy_conflict_priority(deadline):
    """
    Greedy that at each step picks the candidate with minimum 'conflict potential'.
    """
    S = []
    S_set = set()
    candidates = list(UNIVERSE)
    
    while candidates and time.time() < deadline:
        best_x = None
        best_score = float('inf')
        
        for x in candidates:
            shared = 0
            for s in S:
                if math.gcd(x, s) > 1:
                    shared += 1
            score = shared
            if score < best_score:
                best_score = score
                best_x = x
            elif score == best_score:
                if x > best_x:
                    best_x = x
        
        if best_x is None:
            break
        
        if _can_add(best_x, S):
            S.append(best_x)
            S_set.add(best_x)
            candidates.remove(best_x)
        else:
            candidates.remove(best_x)
    
    return S

def _local_search_improve(S, deadline, max_iters=500, rng_seed=42):
    S = list(S)
    S_set = set(S)
    best = S[:]
    best_size = len(S)
    rng = random.Random(rng_seed)
    
    candidates = [x for x in range(2, NMAX + 1) if x not in S_set]
    
    for _ in range(max_iters):
        if time.time() > deadline:
            break
            
        progress = False
        
        if candidates:
            n_try = min(30, len(candidates))
            trial_candidates = rng.sample(candidates, n_try)
            for x in trial_candidates:
                if _can_add(x, S):
                    S.append(x)
                    S_set.add(x)
                    candidates.remove(x)
                    if len(S) > best_size:
                        best = S[:]
                        best_size = len(S)
                    progress = True
                    break
            
        if progress:
            continue
            
        if not S:
            break
            
        for _attempt in range(5):
            if time.time() > deadline:
                break
                
            idx_remove = rng.randrange(len(S))
            removed = S[idx_remove]
            
            S_temp = S[:]
            S_temp.pop(idx_remove)
            S_temp_set = S_set - {removed}
            
            if not candidates:
                continue
                
            n_try = min(30, len(candidates))
            trial_candidates = rng.sample(candidates, n_try)
            swapped = False
            for x in trial_candidates:
                if _can_add(x, S_temp):
                    S = S_temp + [x]
                    S_set = S_temp_set | {x}
                    if removed not in candidates:
                        candidates.append(removed)
                    if x in candidates:
                        candidates.remove(x)
                    if len(S) > best_size:
                        best = S[:]
                        best_size = len(S)
                    swapped = True
                    progress = True
                    break
            
            if swapped:
                break
    
    return best

def _two_opt_search(S, deadline, max_iters=300, rng_seed=77777):
    """Try removing pairs and adding pairs to improve."""
    S = list(S)
    S_set = set(S)
    best = S[:]
    best_size = len(S)
    rng = random.Random(rng_seed)
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        if len(S) < 2:
            break
        
        i, j = rng.sample(range(len(S)), 2)
        i, j = min(i, j), max(i, j)
        removed = [S[i], S[j]]
        S_temp = S[:i] + S[i+1:j] + S[j+1:]
        S_temp_set = set(S_temp)
        
        candidates = [x for x in range(2, NMAX + 1) if x not in S_temp_set]
        if not candidates:
            continue
        
        rng.shuffle(candidates)
        found = False
        for x in candidates[:30]:
            if not _can_add(x, S_temp):
                continue
            S_temp2 = S_temp + [x]
            S_temp2_set = S_temp_set | {x}
            for y in candidates[:30]:
                if y == x or y in S_temp2_set:
                    continue
                if _can_add(y, S_temp2):
                    S = S_temp2 + [y]
                    S_set = set(S)
                    if len(S) > best_size:
                        best = S[:]
                        best_size = len(S)
                    found = True
                    break
            if found:
                break
        if not found:
            # restore not needed since we only update S on success
            pass
    
    return best

def _simulated_annealing(S, deadline, max_iters=2000, T_start=100.0, T_end=1.0, rng_seed=55555):
    """Simulated annealing for optimization."""
    S = list(S)
    S_set = set(S)
    best = S[:]
    best_size = len(S)
    rng = random.Random(rng_seed)
    
    current_temp = T_start
    cooling = (T_end / T_start) ** (1.0 / max(1, max_iters))
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        current_temp *= cooling
        
        op = rng.random()
        if op < 0.3 and len(S) < NMAX:
            candidates = [x for x in range(2, NMAX + 1) if x not in S_set]
            if candidates:
                x = rng.choice(candidates)
                if _can_add(x, S):
                    S.append(x)
                    S_set.add(x)
                    if len(S) > best_size:
                        best = S[:]
                        best_size = len(S)
                    continue
        elif op < 0.6 and S:
            i = rng.randrange(len(S))
            removed = S[i]
            S.pop(i)
            S_set.discard(removed)
            if len(S) > best_size:
                best = S[:]
                best_size = len(S)
            # Accept removal with some probability based on temperature
            if rng.random() > math.exp(1.0 / current_temp):
                S.append(removed)
                S_set.add(removed)
        elif S:
            # Swap
            i = rng.randrange(len(S))
            removed = S[i]
            S_temp = S[:]
            S_temp.pop(i)
            S_temp_set = set(S_temp)
            candidates = [x for x in range(2, NMAX + 1) if x not in S_temp_set]
            if candidates:
                x = rng.choice(candidates)
                if _can_add(x, S_temp):
                    S = S_temp + [x]
                    S_set = set(S)
                    if len(S) > best_size:
                        best = S[:]
                        best_size = len(S)
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s kill

    best_S = None
    best_size = 0

    # Phase 1: Diverse Greedy Initializations
    # 1. Largest First
    order_largest = list(range(NMAX, 1, -1))
    S1 = _greedy(order_largest, deadline)
    if S1 and len(S1) > best_size:
        best_S = S1
        best_size = len(S1)

    # 2. Smallest First
    order_smallest = list(range(2, NMAX + 1))
    S2 = _greedy(order_smallest, deadline)
    if S2 and len(S2) > best_size:
        best_S = S2
        best_size = len(S2)

    # 3. Conflict-priority greedy
    if time.time() < deadline - 30:
        S3 = _greedy_conflict_priority(deadline - 30)
        if S3 and len(S3) > best_size:
            best_S = S3
            best_size = len(S3)

    # 4. Randomized Orders
    for seed in [1, 7, 42, 100, 200, 500, 777, 1234, 5678]:
        if time.time() > deadline - 40:
            break
        rng = random.Random(seed)
        order_rand = list(range(2, NMAX + 1))
        rng.shuffle(order_rand)
        Sr = _greedy(order_rand, deadline)
        if Sr and len(Sr) > best_size:
            best_S = Sr
            best_size = len(Sr)

    if not best_S:
        best_S = [2, 3]
        best_size = 2

    # Phase 2: Local Search Improvement
    if time.time() < deadline - 20:
        S_improved = _local_search_improve(best_S, deadline - 20, max_iters=3000, rng_seed=42)
        if S_improved and len(S_improved) > best_size:
            best_S = S_improved
            best_size = len(S_improved)

    # Phase 3: Two-Opt Search
    if time.time() < deadline - 15:
        S_2opt = _two_opt_search(best_S, deadline - 15, max_iters=300, rng_seed=77777)
        if S_2opt and len(S_2opt) > best_size:
            best_S = S_2opt
            best_size = len(S_2opt)

    # Phase 4: Simulated Annealing
    if time.time() < deadline - 10:
        S_sa = _simulated_annealing(best_S, deadline - 10, max_iters=2000, T_start=100.0, T_end=1.0, rng_seed=55555)
        if S_sa and len(S_sa) > best_size:
            best_S = S_sa
            best_size = len(S_sa)

    # Phase 5: Final Local Search Polish
    if time.time() < deadline - 3:
        S_final = _local_search_improve(best_S, deadline - 3, max_iters=1500, rng_seed=999)
        if S_final and len(S_final) > best_size:
            best_S = S_final
            best_size = len(S_final)

    # Phase 6: Another Two-Opt on best
    if time.time() < deadline - 1:
        S_2opt2 = _two_opt_search(best_S, deadline - 1, max_iters=100, rng_seed=12345)
        if S_2opt2 and len(S_2opt2) > best_size:
            best_S = S_2opt2
            best_size = len(S_2opt2)

    return [int(x) for x in best_S]
