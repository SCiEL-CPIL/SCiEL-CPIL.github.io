import time
import random
import numpy as np

N = 128

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    # Precompute all nonconstant 4-term APs
    aps = []
    pos_to_aps = [[] for _ in range(N)]
    
    for d in range(1, N // 3 + 1):
        for a in range(N - 3 * d):
            p = (a, a + d, a + 2*d, a + 3*d)
            aps.append(p)
            for idx in p:
                pos_to_aps[idx].append(len(aps) - 1)
    
    num_aps = len(aps)
    
    def is_bad(ap, colors):
        """Check if AP uses fewer than 3 distinct colors."""
        c0 = colors[ap[0]]
        c1 = colors[ap[1]]
        c2 = colors[ap[2]]
        c3 = colors[ap[3]]
        # Count distinct colors
        distinct = 1
        if c1 != c0:
            distinct += 1
        if c2 != c0 and c2 != c1:
            distinct += 1
        if c3 != c0 and c3 != c1 and c3 != c2:
            distinct += 1
        return distinct < 3
    
    def total_bad(colors):
        """Count total bad APs."""
        bad = 0
        for ap in aps:
            if is_bad(ap, colors):
                bad += 1
        return bad
    
    best_colors = None
    best_bad = float('inf')
    
    for K in [3, 4, 5, 6, 8]:
        if time.time() > deadline - 15:
            break
        
        for restart in range(100):
            if time.time() > deadline - 15:
                break
            
            # Random initial coloring
            colors = np.random.randint(0, K, N)
            bad_count = total_bad(colors)
            
            if bad_count < best_bad:
                best_bad = bad_count
                best_colors = colors.copy()
            
            if bad_count == 0:
                break
            
            # Simulated annealing
            T = 1.0
            cooling = 0.999
            min_T = 0.005
            max_steps = 200000
            
            for step in range(max_steps):
                if time.time() > deadline - 15:
                    break
                
                # Pick a random position
                pos = random.randint(0, N - 1)
                old_c = colors[pos]
                
                # Pick a new color different from old
                new_c = random.randint(0, K - 1)
                if new_c == old_c:
                    continue
                
                # Compute delta in bad count for APs involving pos
                old_local_bad = 0
                for ap_idx in pos_to_aps[pos]:
                    if is_bad(aps[ap_idx], colors):
                        old_local_bad += 1
                
                colors[pos] = new_c
                
                new_local_bad = 0
                for ap_idx in pos_to_aps[pos]:
                    if is_bad(aps[ap_idx], colors):
                        new_local_bad += 1
                
                delta = new_local_bad - old_local_bad
                bad_count += delta
                
                if delta <= 0 or random.random() < np.exp(-delta / T):
                    # Accept
                    if bad_count < best_bad:
                        best_bad = bad_count
                        best_colors = colors.copy()
                        if best_bad == 0:
                            break
                else:
                    # Reject
                    colors[pos] = old_c
                
                T *= cooling
                if T < min_T:
                    break
        
        if best_bad == 0:
            break
    
    if best_colors is not None and best_bad == 0:
        return list(best_colors)
    
    # Fallback: use all distinct colors
    return list(range(N))
