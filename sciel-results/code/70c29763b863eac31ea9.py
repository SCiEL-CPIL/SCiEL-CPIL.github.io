import time
import random

def _check_and_add(S, sums, x):
    """
    Returns True if x can be added to S, and updates S and sums if so.
    S is a list of sorted integers.
    sums is a set of all triple sums.
    """
    new_sums = set()
    
    # 1. 3x
    s = 3 * x
    if s in sums:
        return False
    new_sums.add(s)
    
    # 2. 2x + y
    for y in S:
        s = 2 * x + y
        if s in sums or s in new_sums:
            return False
        new_sums.add(s)
        
    # 3. x + y + z
    n = len(S)
    for i in range(n):
        y = S[i]
        for j in range(i, n):
            z = S[j]
            s = x + y + z
            if s in sums or s in new_sums:
                return False
            new_sums.add(s)
            
    # Valid
    S.append(x)
    sums.update(new_sums)
    return True

def _local_search(S, sums, time_limit, start_time):
    """
    Try to improve the set by swapping elements.
    For each element in S, try replacing it with a candidate not in S.
    """
    S = S[:]
    sums = sums.copy()
    S_set = set(S)
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        # Try swapping each element
        for idx in range(len(S)):
            if time.time() - start_time > time_limit:
                break
            old_x = S[idx]
            # Build sums without old_x
            sums_without = set()
            # Recompute sums for S without old_x
            S_without = S[:idx] + S[idx+1:]
            for i in range(len(S_without)):
                for j in range(i, len(S_without)):
                    for k in range(j, len(S_without)):
                        sums_without.add(S_without[i] + S_without[j] + S_without[k])
            
            # Try candidates not in S
            candidates = [x for x in range(1, 4097) if x not in S_set]
            random.shuffle(candidates)
            
            for new_x in candidates:
                if time.time() - start_time > time_limit:
                    break
                # Check if new_x can be added to S_without
                valid = True
                for y in S_without:
                    for z in S_without:
                        if y <= z:
                            if new_x + y + z in sums_without:
                                valid = False
                                break
                    if not valid:
                        break
                    if 2 * new_x + y in sums_without:
                        valid = False
                        break
                if not valid:
                    continue
                if 3 * new_x in sums_without:
                    continue
                
                # Valid swap
                S[idx] = new_x
                S_set.remove(old_x)
                S_set.add(new_x)
                S.sort()
                # Recompute sums
                sums = set()
                for i in range(len(S)):
                    for j in range(i, len(S)):
                        for k in range(j, len(S)):
                            sums.add(S[i] + S[j] + S[k])
                improved = True
                break
    return S

def construct():
    start_time = time.time()
    time_limit = 55.0
    
    best_set = []
    best_len = 0
    
    # Strategy 1: Smallest valid greedy
    S1 = []
    sums1 = set()
    for x in range(1, 4097):
        if time.time() - start_time > time_limit:
            break
        _check_and_add(S1, sums1, x)
    if len(S1) > best_len:
        best_set = S1[:]
        best_len = len(S1)

    # Strategy 2: Largest valid greedy
    S2 = []
    sums2 = set()
    for x in range(4096, 0, -1):
        if time.time() - start_time > time_limit:
            break
        _check_and_add(S2, sums2, x)
    if len(S2) > best_len:
        best_set = S2[:]
        best_len = len(S2)

    # Strategy 3: Multiple randomized greedy with local search
    # Use more restarts with tighter time budget per restart
    max_restarts = 30
    for r in range(max_restarts):
        if time.time() - start_time > time_limit - 5.0:
            break
        
        S3 = []
        sums3 = set()
        candidates = list(range(1, 4097))
        random.shuffle(candidates)
        
        for x in candidates:
            if time.time() - start_time > time_limit - 3.0:
                break
            _check_and_add(S3, sums3, x)
        
        if len(S3) > best_len:
            best_set = S3[:]
            best_len = len(S3)
        
        # Local search on the best set so far
        if len(S3) >= best_len and time.time() - start_time < time_limit - 10.0:
            improved = _local_search(S3, sums3, time_limit - 5.0, start_time)
            if len(improved) > best_len:
                best_set = improved[:]
                best_len = len(improved)

    # Final local search on the best set found
    if len(best_set) > 0 and time.time() - start_time < time_limit - 10.0:
        # Recompute sums for best_set
        best_sums = set()
        for i in range(len(best_set)):
            for j in range(i, len(best_set)):
                for k in range(j, len(best_set)):
                    best_sums.add(best_set[i] + best_set[j] + best_set[k])
        final = _local_search(best_set, best_sums, time_limit - 2.0, start_time)
        if len(final) > best_len:
            best_set = final[:]
            best_len = len(final)

    return best_set
