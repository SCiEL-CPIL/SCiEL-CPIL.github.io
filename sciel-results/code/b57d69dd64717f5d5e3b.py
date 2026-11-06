import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55  # safety margin before 60s limit
    
    # Generate all 4-subsets of [1,16] as bitmasks
    all_sets = []
    for combo in combinations(range(1, 17), 4):
        mask = 0
        for x in combo:
            mask |= (1 << x)
        all_sets.append((mask, combo))
    
    N = len(all_sets)
    masks = [m for m, _ in all_sets]
    
    # Precompute pairwise intersection sizes
    # inter[i][j] = popcount(masks[i] & masks[j])
    # Use a 2D list for fast access
    inter = [[0] * N for _ in range(N)]
    for i in range(N):
        mi = masks[i]
        inter[i][i] = 4  # self intersection
        for j in range(i + 1, N):
            s = bin(mi & masks[j]).count('1')
            inter[i][j] = s
            inter[j][i] = s
    
    best_family = []
    best_size = 0
    
    # Try multiple random orderings
    rng = random.Random(42)
    
    for trial in range(200):
        if time.time() > deadline:
            break
        
        # Create a random ordering
        order = list(range(N))
        rng.shuffle(order)
        
        chosen = []
        chosen_masks = []
        chosen_indices = []
        
        # For efficient conflict checking:
        # When considering adding set c:
        # For each pair (a,b) in chosen with inter[a][b] = s:
        #   conflict if inter[a][c] == s AND inter[b][c] == s
        
        # We'll do the O(k^2) check but break early
        for idx in order:
            if time.time() > deadline:
                break
            
            mc = masks[idx]
            k = len(chosen_indices)
            conflict = False
            
            # Check against all pairs in chosen
            for i in range(k):
                ci = chosen_indices[i]
                ic = inter[ci][idx]
                
                for j in range(i + 1, k):
                    cj = chosen_indices[j]
                    ij = inter[ci][cj]
                    jc = inter[cj][idx]
                    
                    if ij == ic == jc:
                        conflict = True
                        break
                if conflict:
                    break
            
            if not conflict:
                chosen.append(all_sets[idx][1])
                chosen_masks.append(mc)
                chosen_indices.append(idx)
        
        if len(chosen) > best_size:
            best_size = len(chosen)
            best_family = chosen[:]
            
            # If we've reached a good size, we can try to stop early
            # but let's continue to see if we can do better
    
    return [sorted(s) for s in best_family]
