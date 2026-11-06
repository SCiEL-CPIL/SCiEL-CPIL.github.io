import time

def construct():
    N = 256
    S = set()
    present = [False] * (N + 1)
    
    # Greedy Initialization
    for x in range(1, N + 1):
        conflict = False
        # Check x as smallest
        for d in range(1, (N - x) // 2 + 1):
            if present[x + d] and present[x + 2 * d]:
                conflict = True
                break
        if not conflict:
            # Check x as middle
            for d in range(1, x):
                if present[x - d] and present[x + d]:
                    conflict = True
                    break
        if not conflict:
            # Check x as largest
            for d in range(1, (x // 2) + 1):
                if present[x - 2 * d] and present[x - d]:
                    conflict = True
                    break
        
        if not conflict:
            S.add(x)
            present[x] = True

    # Local Search
    start_time = time.time()
    time_limit = 55.0 # Safe margin
    
    def is_valid_addition(v, present_arr, N):
        # v is smallest
        max_d1 = (N - v) // 2
        for d in range(1, max_d1 + 1):
            if present_arr[v + d] and present_arr[v + 2 * d]:
                return False
        # v is middle
        for d in range(1, v):
            if present_arr[v - d] and present_arr[v + d]:
                return False
        # v is largest
        max_d2 = v // 2
        for d in range(1, max_d2 + 1):
            if present_arr[v - 2 * d] and present_arr[v - d]:
                return False
        return True

    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        # Iterate u from largest to smallest to prioritize replacing high-cost elements
        # We need a snapshot of S to iterate, as we modify it
        S_snapshot = list(S)
        for u in sorted(S_snapshot, reverse=True):
            if (time.time() - start_time) > time_limit:
                break
            
            # Try to swap u with v < u
            # v must not be in S
            for v in range(1, u):
                if present[v]:
                    continue
                
                # Attempt swap
                present[u] = False
                if is_valid_addition(v, present, N):
                    # Swap successful
                    present[v] = True
                    S.remove(u)
                    S.add(v)
                    improved = True
                    # Break out to restart the pass with the new set
                    break
                else:
                    # Swap failed, restore u
                    present[u] = True
            
            if improved:
                break

    return list(S)
