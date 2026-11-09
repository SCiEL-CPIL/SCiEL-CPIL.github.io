import numpy as np
import random
import time

N = 512

def precompute_aps(n):
    aps = []
    for d in range(1, n//3 + 1):
        for a in range(1, n - 3*d + 1):
            b = a + d
            c = b + d
            e = c + d
            if e <= n:
                aps.append((a, b, c, e))
    return aps

APS = precompute_aps(N)
# Convert to 0-indexed for numpy
APS = [tuple(x-1 for x in ap) for ap in APS]

# For each index, list of AP indices it belongs to
APS_CONTAINING = [[] for _ in range(N)]
for i, ap in enumerate(APS):
    for x in ap:
        APS_CONTAINING[x].append(i)

def get_base3_set(n):
    s = set()
    for x in range(1, n+1):
        y = x
        valid = True
        while y > 0:
            if y % 3 == 2:
                valid = False
                break
            y //= 3
        if valid:
            s.add(x-1)
    return s

def greedy_fill_numpy(S, n=512):
    # S is a set of 0-indexed integers
    x = np.zeros(n, dtype=bool)
    for i in S:
        x[i] = True
        
    # Precompute AP counts for current state? 
    # We can just check on the fly.
    
    candidates = list(range(n))
    random.shuffle(candidates)
    
    # To speed up, we can maintain a count of how many elements of each AP are in S.
    ap_counts = np.zeros(len(APS), dtype=np.int32)
    for i in S:
        for ap_idx in APS_CONTAINING[i]:
            ap_counts[ap_idx] += 1
            
    for i in candidates:
        if x[i]:
            continue
        # Check if adding i creates a 4-AP
        # i is in APS_CONTAINING[i]
        can_add = True
        for ap_idx in APS_CONTAINING[i]:
            if ap_counts[ap_idx] >= 3:
                can_add = False
                break
        if can_add:
            x[i] = True
            for ap_idx in APS_CONTAINING[i]:
                ap_counts[ap_idx] += 1
                
    return np.where(x)[0].tolist()

def local_search_numpy(S, n=512, time_limit=0.1):
    # S is a list of 0-indexed integers
    x = np.zeros(n, dtype=bool)
    for i in S:
        x[i] = True
        
    ap_counts = np.zeros(len(APS), dtype=np.int32)
    for i in S:
        for ap_idx in APS_CONTAINING[i]:
            ap_counts[ap_idx] += 1
            
    best_size = len(S)
    best_S = list(S)
    
    start_time = time.time()
    while time.time() - start_time < time_limit:
        # Try to add a random element
        # Or try to swap?
        # Let's try to add random elements first
        progress = False
        for _ in range(100): # Try 100 random additions
            i = random.randint(0, n-1)
            if x[i]:
                continue
            can_add = True
            for ap_idx in APS_CONTAINING[i]:
                if ap_counts[ap_idx] >= 3:
                    can_add = False
                    break
            if can_add:
                x[i] = True
                for ap_idx in APS_CONTAINING[i]:
                    ap_counts[ap_idx] += 1
                progress = True
                if len(S) < best_size: # Update size
                    # We need to track size
                    pass
                
        if not progress:
            # Try to remove a random element and then fill
            # Or just restart with perturbation
            break
            
    return best_S

def construct():
    start_time = time.time()
    deadline = start_time + 55 # 55s limit
    
    # Start with base 3 set
    base_S = get_base3_set(N)
    best_S = greedy_fill_numpy(base_S)
    best_size = len(best_S)
    
    # Iterated Local Search
    while time.time() < deadline:
        # Perturb
        current_S = list(best_S)
        # Remove some random elements
        num_remove = max(1, int(len(current_S) * 0.1))
        to_remove = random.sample(current_S, num_remove)
        current_S = [x for x in current_S if x not in set(to_remove)]
        
        # Greedy Fill
        new_S = greedy_fill_numpy(current_S)
        
        if len(new_S) > best_size:
            best_size = len(new_S)
            best_S = new_S
            
    # Return 1-indexed sorted list
    return sorted([x+1 for x in best_S])
