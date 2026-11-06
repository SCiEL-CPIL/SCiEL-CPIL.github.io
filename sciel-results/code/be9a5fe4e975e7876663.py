"""
Optimization program for maximizing 3-term APs in a subset of [1,256] with no 4-term AP.
Uses precomputed AP structures with incremental delta tracking for fast search.
"""
import time
import random
import math
import sys

N = 256
MAX_TIME = 82  # seconds, safety margin before 90s kill

# Precompute all 3-term APs: (a, b, c) where b = (a+c)//2, c-a = 2d > 0
_three_aps = []
_three_aps_by_elem = [[] for _ in range(N + 1)]  # for each elem, list of (other_two, ap_index)
for d in range(1, N // 2 + 1):
    for a in range(1, N - 2 * d + 1):
        b = a + d
        c = a + 2 * d
        idx = len(_three_aps)
        _three_aps.append((a, b, c))
        _three_aps_by_elem[a].append((b, c, idx))
        _three_aps_by_elem[b].append((a, c, idx))
        _three_aps_by_elem[c].append((a, b, idx))

# Precompute all 4-term APs: (a, a+d, a+2d, a+3d)
_four_aps = []
_four_aps_by_elem = [[] for _ in range(N + 1)]
for d in range(1, N // 3 + 1):
    for a in range(1, N - 3 * d + 1):
        b = a + d
        c = a + 2 * d
        e = a + 3 * d
        idx = len(_four_aps)
        _four_aps.append((a, b, c, e))
        _four_aps_by_elem[a].append((b, c, e, idx))
        _four_aps_by_elem[b].append((a, c, e, idx))
        _four_aps_by_elem[c].append((a, b, e, idx))
        _four_aps_by_elem[e].append((a, b, c, idx))

def count_3aps(S):
    """Count distinct 3-term arithmetic progressions in set S."""
    S_set = set(S)
    count = 0
    for a, b, c in _three_aps:
        if a in S_set and b in S_set and c in S_set:
            count += 1
    return count

def has_4ap(S):
    """Check if set S contains any nonconstant 4-term arithmetic progression."""
    S_set = set(S)
    for a, b, c, d in _four_aps:
        if a in S_set and b in S_set and c in S_set and d in S_set:
            return True
    return False

def greedy_construct():
    """
    Greedy construction: add elements in order of highest marginal 3-AP gain
    while maintaining no-4-AP constraint.
    """
    S = set()
    S_bool = [False] * (N + 1)
    remaining = set(range(1, N + 1))
    current_3ap_count = 0
    
    while remaining:
        best_gain = -1
        best_elem = None
        for x in remaining:
            # Check 4-AP constraint: adding x creates a 4-AP if there's a 4-AP 
            # where x is the missing element and other 3 are in S
            creates_4ap = False
            for (a, b, c, idx) in _four_aps_by_elem[x]:
                if S_bool[a] and S_bool[b] and S_bool[c]:
                    creates_4ap = True
                    break
            if creates_4ap:
                continue
            
            # Count marginal 3-AP gain: 3-APs that include x and have other 2 in S
            gain = 0
            for (p, q, idx) in _three_aps_by_elem[x]:
                if S_bool[p] and S_bool[q]:
                    gain += 1
            
            if gain > best_gain:
                best_gain = gain
                best_elem = x
        
        if best_elem is None:
            break
        S.add(best_elem)
        S_bool[best_elem] = True
        current_3ap_count += best_gain
        remaining.remove(best_elem)
    
    return S

def local_search_1move(S, max_iters=10000, time_limit=None, start_time=None):
    """
    Local search with single-add and single-remove moves using incremental deltas.
    """
    S = set(S)
    S_bool = [False] * (N + 1)
    for x in S:
        S_bool[x] = True
    current_score = count_3aps(S)
    best_S = set(S)
    best_score = current_score
    n = N
    
    for it in range(max_iters):
        if time_limit is not None and start_time is not None:
            if time.time() - start_time > time_limit:
                break
        
        improved = False
        
        # Try adding elements
        for x in range(1, n + 1):
            if not S_bool[x]:
                # Check 4-AP
                creates_4ap = False
                for (a, b, c, idx) in _four_aps_by_elem[x]:
                    if S_bool[a] and S_bool[b] and S_bool[c]:
                        creates_4ap = True
                        break
                if creates_4ap:
                    continue
                
                # Compute gain
                gain = 0
                for (p, q, idx) in _three_aps_by_elem[x]:
                    if S_bool[p] and S_bool[q]:
                        gain += 1
                
                if gain > 0:
                    S.add(x)
                    S_bool[x] = True
                    current_score += gain
                    improved = True
                    break
        
        if improved:
            if current_score > best_score:
                best_score = current_score
                best_S = set(S)
            continue
        
        # Try removing elements
        for x in list(S):
            # Compute loss from removing x
            loss = 0
            for (p, q, idx) in _three_aps_by_elem[x]:
                if S_bool[p] and S_bool[q]:
                    loss += 1
            
            # Removing x might also free up 4-APs, but we only care about 3-AP count
            # New score = current_score - loss
            new_score = current_score - loss
            if new_score > current_score:
                # This can't happen since loss >= 0, but keep for safety
                S.discard(x)
                S_bool[x] = False
                current_score = new_score
                improved = True
                break
        
        if not improved:
            break
    
    return best_S

def _exp(x):
    """Safe exponent for acceptance probability."""
    if x < -50:
        return 0.0
    if x > 50:
        return 1.0
    return math.exp(x)

def simulated_annealing(S, initial_temp=10.0, cooling=0.995, max_iters=50000, time_limit=None, start_time=None):
    """
    Simulated annealing with add, remove, and swap moves using incremental deltas.
    """
    S = set(S)
    S_bool = [False] * (N + 1)
    for x in S:
        S_bool[x] = True
    current_score = count_3aps(S)
    best_S = set(S)
    best_score = current_score
    T = initial_temp
    n = N
    
    for it in range(max_iters):
        if time_limit is not None and start_time is not None:
            if time.time() - start_time > time_limit:
                break
        
        move_type = random.random()
        moved = False
        
        if move_type < 0.3:
            # Add move
            candidates = [x for x in range(1, n + 1) if not S_bool[x]]
            if candidates:
                x = random.choice(candidates)
                creates_4ap = False
                for (a, b, c, idx) in _four_aps_by_elem[x]:
                    if S_bool[a] and S_bool[b] and S_bool[c]:
                        creates_4ap = True
                        break
                if not creates_4ap:
                    gain = 0
                    for (p, q, idx) in _three_aps_by_elem[x]:
                        if S_bool[p] and S_bool[q]:
                            gain += 1
                    delta = gain
                    if delta > 0 or random.random() < _exp(delta / T):
                        S.add(x)
                        S_bool[x] = True
                        current_score += delta
                        moved = True
        elif move_type < 0.5:
            # Remove move
            if S:
                x = random.choice(list(S))
                loss = 0
                for (p, q, idx) in _three_aps_by_elem[x]:
                    if S_bool[p] and S_bool[q]:
                        loss += 1
                delta = -loss
                if delta > 0 or random.random() < _exp(delta / T):
                    S.discard(x)
                    S_bool[x] = False
                    current_score += delta
                    moved = True
        else:
            # Swap move
            in_S = list(S)
            not_in_S = [x for x in range(1, n + 1) if not S_bool[x]]
            if in_S and not_in_S:
                x = random.choice(in_S)
                y = random.choice(not_in_S)
                
                # Check if swap is valid (no 4-AP)
                # Temporarily remove x, add y, check 4-AP
                S_bool[x] = False
                creates_4ap = False
                for (a, b, c, idx) in _four_aps_by_elem[y]:
                    if S_bool[a] and S_bool[b] and S_bool[c]:
                        creates_4ap = True
                        break
                if not creates_4ap:
                    # Compute delta
                    # Loss from removing x
                    loss_x = 0
                    for (p, q, idx) in _three_aps_by_elem[x]:
                        if S_bool[p] and S_bool[q]:
                            loss_x += 1
                    # Gain from adding y
                    gain_y = 0
                    for (p, q, idx) in _three_aps_by_elem[y]:
                        if S_bool[p] and S_bool[q]:
                            gain_y += 1
                    delta = gain_y - loss_x
                    if delta > 0 or random.random() < _exp(delta / T):
                        S.discard(x)
                        S.add(y)
                        S_bool[y] = True
                        current_score += delta
                        moved = True
                S_bool[x] = True
        
        if current_score > best_score:
            best_score = current_score
            best_S = set(S)
        
        T *= cooling
        if T < 0.01:
            break
    
    return best_S

def construct():
    start_time = time.time()
    
    # Phase 1: Greedy construction
    S = greedy_construct()
    score = count_3aps(S)
    best_S = set(S)
    best_score = score
    
    # Phase 2: Local search refinement
    elapsed = time.time() - start_time
    time_left = MAX_TIME - elapsed
    if time_left > 5:
        S = local_search_1move(S, max_iters=5000, time_limit=time_left - 3, start_time=start_time)
        score = count_3aps(S)
        if score > best_score:
            best_score = score
            best_S = set(S)
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > MAX_TIME * 0.4:
        return sorted(best_S)
    
    # Phase 3: Simulated annealing with restarts
    remaining_time = MAX_TIME - elapsed
    n_restarts = max(1, int(remaining_time / 4))
    
    for r in range(n_restarts):
        if time.time() - start_time > MAX_TIME - 5:
            break
        
        # Perturb the best solution
        perturbed = set(best_S)
        for _ in range(random.randint(3, 8)):
            if random.random() < 0.5:
                candidates = [x for x in range(1, N + 1) if x not in perturbed]
                if candidates:
                    perturbed.add(random.choice(candidates))
            else:
                if perturbed:
                    perturbed.discard(random.choice(list(perturbed)))
        
        # Check validity
        if has_4ap(perturbed):
            for x in list(perturbed):
                if not has_4ap(perturbed - {x}):
                    perturbed.discard(x)
                    break
        
        # Run SA from perturbed state
        sa_time_limit = min(3.0, MAX_TIME - (time.time() - start_time) - 2)
        if sa_time_limit > 0.5:
            sa_S = simulated_annealing(perturbed, initial_temp=5.0, cooling=0.998, 
                                       max_iters=20000, time_limit=sa_time_limit, start_time=start_time)
            sa_score = count_3aps(sa_S)
            
            if sa_score > best_score:
                best_score = sa_score
                best_S = set(sa_S)
            
            # Local search on SA result
            ls_time_limit = min(2.0, MAX_TIME - (time.time() - start_time) - 1)
            if ls_time_limit > 0.3:
                ls_S = local_search_1move(sa_S, max_iters=3000, time_limit=ls_time_limit, start_time=start_time)
                ls_score = count_3aps(ls_S)
                if ls_score > best_score:
                    best_score = ls_score
                    best_S = set(ls_S)
    
    # Final verification
    if has_4ap(best_S):
        # Fallback: remove problematic elements
        for x in list(best_S):
            if not has_4ap(best_S - {x}):
                best_S.discard(x)
                break
    
    return sorted(best_S)

if __name__ == "__main__":
    result = construct()
    print(f"Size: {len(result)}, 3-APs: {count_3aps(result)}")
    print(f"Has 4-AP: {has_4ap(result)}")
    print(f"First 20: {result[:20]}")
