import time
import numpy as np

def construct():
    n = 12
    m = 40
    total_masks = 1 << n
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin

    # Precompute comparability matrix
    # C[i, j] is True if i is a strict subset of j or j is a strict subset of i
    # Using numpy for speed
    i_vals = np.arange(total_masks, dtype=np.int32)[:, None]
    j_vals = np.arange(total_masks, dtype=np.int32)[None, :]
    
    # i is subset of j if (i & j) == i
    # j is subset of i if (i & j) == j
    cond1 = (i_vals & j_vals) == i_vals
    cond2 = (i_vals & j_vals) == j_vals
    comp = cond1 | cond2
    np.fill_diagonal(comp, False)
    
    # comp is a boolean numpy array of shape (4096, 4096)
    
    def calc_initial_score(fam):
        fam_arr = np.array(fam, dtype=np.int32)
        # Score is sum of comp[fam_arr[i], fam_arr[j]] for i < j
        # We can use the degree vector
        d = np.zeros(total_masks, dtype=np.int32)
        for f in fam:
            d += comp[:, f]
        # Sum of degrees is 2 * score
        return int(np.sum(d) / 2)

    def greedy_init():
        fam = []
        in_fam = np.zeros(total_masks, dtype=bool)
        d = np.zeros(total_masks, dtype=np.int32)
        
        # Pick first element randomly to break symmetry
        v = np.random.randint(total_masks)
        fam.append(v)
        in_fam[v] = True
        d += comp[:, v]
        
        for _ in range(m - 1):
            # Mask out elements already in family
            d_masked = d.copy()
            d_masked[in_fam] = -1
            v = int(np.argmax(d_masked))
            
            fam.append(v)
            in_fam[v] = True
            d += comp[:, v]
            
        return fam, d

    def local_search(fam, d):
        fam = list(fam)
        d = d.copy()
        in_fam = np.zeros(total_masks, dtype=bool)
        for f in fam:
            in_fam[f] = True
        
        score = calc_initial_score(fam)
        best_score = score
        best_fam = list(fam)
        
        T = 50.0
        cooling_rate = 0.99
        min_T = 0.1
        
        # Precompute list of masks for random selection
        # We need to pick a random element to remove and a random element to add
        # To speed up, we can just pick random indices
        
        while time.time() < deadline - 2:
            if T < min_T:
                # If temperature is too low, restart with higher T or break
                # Let's try a few more iterations with small T then break
                if np.random.random() < 0.01:
                    break
                T = min_T
            
            # Pick random index in fam to remove
            idx_u = np.random.randint(m)
            u = fam[idx_u]
            
            # Pick random mask v not in fam
            # We can pick from all masks and check, or maintain a list of available masks
            # Given 4096 masks and 40 in fam, probability of collision is low
            # But we can do it efficiently
            v = np.random.randint(total_masks)
            if in_fam[v]:
                # Retry a few times
                for _ in range(10):
                    v = np.random.randint(total_masks)
                    if not in_fam[v]:
                        break
                if in_fam[v]:
                    continue
            
            # Calculate delta
            # d[v] is sum of comp[v, y] for y in fam
            # d[u] is sum of comp[u, y] for y in fam
            # comp[v, u] is 1 if comparable
            c_vu = comp[v, u]
            delta = d[v] - c_vu - d[u]
            
            if delta > 0 or np.random.random() < np.exp(delta / T):
                # Accept move
                fam[idx_u] = v
                
                # Update d
                d += comp[:, v]
                d -= comp[:, u]
                
                # Update in_fam
                in_fam[u] = False
                in_fam[v] = True
                
                score += delta
                if score > best_score:
                    best_score = score
                    best_fam = list(fam)
                    
            T *= cooling_rate
            
        return best_fam, best_score

    # Run greedy init
    fam, d = greedy_init()
    best_fam, best_score = local_search(fam, d)
    
    # Try a few restarts if time permits
    # Since local_search uses time, we can just call it again with a new greedy init
    # But greedy init is fast.
    
    while time.time() < deadline - 5:
        fam, d = greedy_init()
        cur_fam, cur_score = local_search(fam, d)
        if cur_score > best_score:
            best_score = cur_score
            best_fam = cur_fam
            
    result = []
    for mask in best_fam:
        subset = [b for b in range(n) if (mask >> b) & 1]
        result.append(subset)
        
    return result
