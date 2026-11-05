def construct():
    import time
    import random
    import math

    start_time = time.time()
    deadline = start_time + 85.0
    N = 512

    # Precompute conflict masks
    # conflict_masks[x] is a list of masks. If any mask is a subset of S, x cannot be added.
    conflict_masks = [[] for _ in range(N+1)]
    
    # Iterate all 4-APs
    for a in range(1, N+1):
        for d in range(1, N//3 + 1):
            b = a + d
            c = a + 2*d
            e = a + 3*d
            if e > N: break
            
            # AP: a, b, c, e
            # For each element, the other 3 form a conflict mask
            mask_a = (1 << (b-1)) | (1 << (c-1)) | (1 << (e-1))
            mask_b = (1 << (a-1)) | (1 << (c-1)) | (1 << (e-1))
            mask_c = (1 << (a-1)) | (1 << (b-1)) | (1 << (e-1))
            mask_e = (1 << (a-1)) | (1 << (b-1)) | (1 << (c-1))
            
            conflict_masks[a].append(mask_a)
            conflict_masks[b].append(mask_b)
            conflict_masks[c].append(mask_c)
            conflict_masks[e].append(mask_e)

    def can_add(S, x):
        if (S >> (x-1)) & 1:
            return False
        for mask in conflict_masks[x]:
            if (S & mask) == mask:
                return False
        return True

    # Greedy Construction with Randomization
    def greedy(seed):
        random.seed(seed)
        S = 0
        count = 0
        candidates = list(range(1, N+1))
        random.shuffle(candidates)
        for x in candidates:
            if can_add(S, x):
                S |= (1 << (x-1))
                count += 1
        return S, count

    best_S = 0
    best_count = 0
    
    # Run multiple greedy restarts
    num_greedy = 200
    for i in range(num_greedy):
        if time.time() > deadline - 30.0: break
        S, count = greedy(i)
        if count > best_count:
            best_count = count
            best_S = S

    # Simulated Annealing
    S = best_S
    count = best_count
    
    # Helper to get list of elements in S
    def get_elements(S):
        return [i+1 for i in range(N) if (S >> i) & 1]
    
    S_list = get_elements(S)
    
    T = 10.0
    cooling = 0.999
    min_T = 0.01
    
    all_elements = list(range(1, N+1))
    
    # SA Loop
    while T > min_T and time.time() < deadline - 2.0:
        # Choose move
        r = random.random()
        
        if r < 0.5:
            # Try to add
            x = random.choice(all_elements)
            if can_add(S, x):
                S |= (1 << (x-1))
                S_list.append(x)
                count += 1
                # Accept
            else:
                # Reject
                pass
        elif r < 0.8:
            # Try to remove
            if not S_list:
                continue
            x = random.choice(S_list)
            # Accept with prob exp(-1/T)
            if random.random() < math.exp(-1.0/T):
                S ^= (1 << (x-1))
                S_list.remove(x)
                count -= 1
            # Else reject
        else:
            # Try to swap
            if not S_list:
                continue
            x = random.choice(S_list)
            y = random.choice(all_elements)
            if x == y:
                continue
            
            # Temporarily remove x
            S_temp = S ^ (1 << (x-1))
            if can_add(S_temp, y):
                # Valid swap
                S = S_temp | (1 << (y-1))
                S_list.remove(x)
                S_list.append(y)
                # Count unchanged
                # Accept (Delta 0)
            else:
                # Reject
                pass
                
        # Update best
        if count > best_count:
            best_count = count
            best_S = S
            
        T *= cooling
        
        # Occasionally restart or cool faster if stuck?
        # For now, simple cooling.

    # Final local search: try to add any valid element
    for x in all_elements:
        if time.time() > deadline - 1.0: break
        if can_add(best_S, x):
            best_S |= (1 << (x-1))
            best_count += 1

    result = get_elements(best_S)
    return sorted(result)
