import time
import random

N = 256

def _validate(S):
    """Return True iff S satisfies: for all a<b<=c in S, a does not divide b+c."""
    L = sorted(S)
    m = len(L)
    for i in range(m):
        a = L[i]
        for j in range(i + 1, m):
            b = L[j]
            for k in range(j, m):
                c = L[k]
                if (b + c) % a == 0:
                    return False
    return True

def _can_add(x, S):
    """
    Check if x can be added to S while maintaining validity.
    S is a set of integers.
    x is an integer not in S.
    """
    S_list = list(S)
    
    # Case 1: x is a
    for b in S_list:
        if b <= x:
            continue
        for c in S_list:
            if c < b:
                continue
            if (b + c) % x == 0:
                return False
                
    # Case 2: x is b
    for a in S_list:
        if a >= x:
            continue
        for c in S_list:
            if c < x:
                continue
            if (x + c) % a == 0:
                return False
                
    # Case 3: x is c
    for b in S_list:
        if b > x:
            continue
        for a in S_list:
            if a >= b:
                continue
            if (b + x) % a == 0:
                return False
                
    return True

def _greedy_descending():
    n = N
    S = set()
    for x in range(n, 0, -1):
        if _can_add(x, S):
            S.add(x)
    return S

def _greedy_ascending():
    n = N
    S = set()
    for x in range(1, n + 1):
        if _can_add(x, S):
            S.add(x)
    return S

def _greedy_random(seed):
    random.seed(seed)
    n = N
    S = set()
    candidates = list(range(1, n + 1))
    random.shuffle(candidates)
    for x in candidates:
        if _can_add(x, S):
            S.add(x)
    return S

def _local_search_rw(S, deadline):
    """
    Random Walk Local Search with Add and Swap capabilities.
    Returns the best set found.
    """
    best_set = S.copy()
    best_size = len(S)
    current_set = S.copy()
    stall_count = 0
    max_stall = 500  # Max neutral swaps before trying structural changes or restart
    
    while time.time() < deadline:
        # 1. Try to add any element
        added = False
        # Randomize the order of trying to add to avoid bias
        candidates = [x for x in range(1, N + 1) if x not in current_set]
        random.shuffle(candidates)
        
        for x in candidates:
            if _can_add(x, current_set):
                current_set.add(x)
                added = True
                stall_count = 0
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = current_set.copy()
                break
        
        if added:
            continue
            
        # 2. No add possible. Try neutral swaps (1-1)
        if stall_count < max_stall:
            # Try a random 1-1 swap
            # Pick random y in current_set
            # Pick random x not in current_set
            # Check if current_set - {y} + {x} is valid
            # This is equivalent to: is x addable to current_set - {y}?
            
            # Optimization: Pre-compute list of elements in/out
            in_set = list(current_set)
            out_set = [x for x in range(1, N + 1) if x not in current_set]
            
            if not in_set or not out_set:
                break
                
            y = random.choice(in_set)
            x = random.choice(out_set)
            
            S_temp = current_set - {y}
            if _can_add(x, S_temp):
                # Valid swap
                current_set = S_temp
                current_set.add(x)
                stall_count += 1
                # We don't update best_set here as size is same
                # But we might want to check if this swap enables future adds
                # The loop will naturally try to add next iteration
            else:
                # Swap invalid, try another one?
                # To avoid getting stuck on one bad pair, we just increment stall
                # or try a few more random swaps in this iteration?
                # Let's just count it as a stall step
                stall_count += 1
        else:
            # Too many stalls. Try a 1-2 swap (Remove 1, Add 2)
            # Or revert to best and restart
            improved = False
            for y in list(current_set):
                if time.time() > deadline:
                    break
                S_temp = current_set - {y}
                added_count = 0
                # Try to add 2 elements
                # We can try all pairs? No, too slow.
                # Try to add elements greedily to S_temp
                # But we want to maximize the chance of adding 2.
                # Let's just try to add any 2.
                # If we can add 2, size increases by 1.
                
                # To be faster, we can just check if we can add 1, then add 1 more.
                # But that's what _local_search_add does.
                # Let's use a more targeted swap:
                # Find x1, x2 such that S_temp + {x1, x2} is valid.
                
                # Heuristic: Try to add the smallest valid x1, then smallest valid x2.
                x1_found = None
                x2_found = None
                
                # This is O(N^2) for the outer y loop, and O(N^2) for the inner check?
                # No, _can_add is O(|S|^2).
                # We call _can_add for x1, then for x2.
                
                for x1 in range(1, N + 1):
                    if x1 == y or x1 in current_set:
                        continue
                    if _can_add(x1, S_temp):
                        S_temp1 = S_temp | {x1}
                        for x2 in range(1, N + 1):
                            if x2 == y or x2 in current_set or x2 == x1:
                                continue
                            if _can_add(x2, S_temp1):
                                x1_found = x1
                                x2_found = x2
                                break
                        if x2_found:
                            break
                
                if x1_found and x2_found:
                    current_set = S_temp | {x1_found, x2_found}
                    improved = True
                    stall_count = 0
                    if len(current_set) > best_size:
                        best_size = len(current_set)
                        best_set = current_set.copy()
                    break
            
            if not improved:
                # No 1-2 swap found. Revert to best and try again with different random seed?
                # Or just break?
                # Let's try to revert to best_set and randomize the walk
                current_set = best_set.copy()
                stall_count = 0
                # To avoid infinite loop of reverting, we can just break if we've been stuck for a long time
                # But we have time budget.
                # Let's just break out of the while loop if we can't improve for a while?
                # Actually, reverting and restarting the random walk might find a different path.
                # But if the space is rugged, we might need more aggressive moves.
                # For now, let's just break to save time for other strategies if this fails.
                # Or we can try 2-2 swaps?
                
                # Let's try one 2-2 swap attempt
                # Remove y1, y2. Add x1, x2.
                # This is expensive.
                
                # If we fail here, we likely hit a local minimum for this strategy.
                # We will exit the loop and let the main construct function try other seeds.
                break

    return best_set

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    # Generate initial candidates
    candidates = []
    candidates.append(_greedy_descending())
    candidates.append(_greedy_ascending())
    
    # Random restarts
    for i in range(20):
        candidates.append(_greedy_random(i))
        
    # Pick the best valid candidate
    best_set = set()
    best_size = 0
    for S in candidates:
        if _validate(S):
            if len(S) > best_size:
                best_size = len(S)
                best_set = S.copy()
    
    if not best_set:
        best_set = {N}
        best_size = 1
        
    # Run Random Walk Local Search
    # We can run it multiple times or just once with a long budget?
    # Since it's randomized, running it on the best seed is good.
    # We can also run it on other seeds if time permits.
    
    # To maximize chances, we can run RWLS on the top 3 seeds.
    top_seeds = []
    for S in candidates:
        if _validate(S):
            top_seeds.append(S)
    
    top_seeds.sort(key=len, reverse=True)
    
    # Run RWLS on the best seed
    if top_seeds:
        best_set = _local_search_rw(top_seeds[0], deadline)
        
    # If time remains, try on the second best
    if len(top_seeds) > 1 and time.time() < deadline - 10.0:
        # Check if we can improve
        # We need to compare sizes
        # _local_search_rw returns the best found in that run
        # We should keep the global best
        candidate_best = _local_search_rw(top_seeds[1], deadline)
        if len(candidate_best) > len(best_set):
            best_set = candidate_best
            
    # Final polish: try simple additions
    while time.time() < deadline:
        changed = False
        for x in range(1, N + 1):
            if x in best_set:
                continue
            if _can_add(x, best_set):
                best_set.add(x)
                changed = True
        if not changed:
            break
            
    return sorted(best_set)

if __name__ == "__main__":
    _s = construct()
    assert _validate(set(_s)), "constructed set is invalid"
    print("size", len(_s), "first", _s[:5], "last", _s[-5:])
