import time
import random
import numpy as np

N = 64
TOTAL = N * N

# Precompute squared distances using numpy for speed
xs = np.repeat(np.arange(N), N)
ys = np.tile(np.arange(N), N)
dx = xs[:, None] - xs[None, :]
dy = ys[:, None] - ys[None, :]
dist2 = (dx * dx + dy * dy).astype(np.int32)

def construct():
    start_time = time.time()
    deadline = start_time + 295
    
    best_set = []
    best_size = 0
    
    # We will try several random permutations
    # A greedy algorithm with random restarts
    for attempt in range(5000):
        if time.time() > deadline:
            break
            
        # Random permutation of all points
        # Using random.shuffle on a list of 4096 ints is fast
        perm = list(range(TOTAL))
        random.shuffle(perm)
        
        si = []
        ud = [] # list of sets
        
        for p_idx in perm:
            if not si:
                si.append(p_idx)
                ud.append(set())
                continue
            
            # Check distinct distances from p to all in si
            # We need the distances for the next check too, so compute them once.
            # If K is small, manual loop might be faster due to numpy overhead.
            # Let's stick to manual loop but optimized.
            
            valid = True
            dists = []
            for q_idx in si:
                d = dist2[p_idx, q_idx]
                dists.append(d)
            
            # Check distinctness
            if len(set(dists)) != len(dists):
                continue
            
            # Check against existing used distances
            for i in range(len(si)):
                if dists[i] in ud[i]:
                    valid = False
                    break
            
            if valid:
                si.append(p_idx)
                # New set for the new point
                new_ud = set(dists)
                ud.append(new_ud)
                
                # Update existing points' used distance sets
                for i in range(len(si) - 1):
                    ud[i].add(dists[i])
        
        if len(si) > best_size:
            best_size = len(si)
            best_set = [(idx // N, idx % N) for idx in si]
            
            # If we get a very large set, we can try to refine it with local search
            # or just keep searching.
            
    return best_set
