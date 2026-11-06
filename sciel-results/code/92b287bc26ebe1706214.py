import random
import time
import numpy as np

def construct():
    N = 128
    # Precompute all 4-term APs
    aps = []
    index_to_aps = [[] for _ in range(N)]
    
    for a in range(1, N + 1):
        for d in range(1, N + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if e > N:
                break
            ap = (a - 1, b - 1, c - 1, e - 1) # 0-indexed
            ap_idx = len(aps)
            aps.append(ap)
            for idx in ap:
                index_to_aps[idx].append(ap_idx)
                
    num_aps = len(aps)
    ap_indices = np.array(aps, dtype=np.int32) # shape (num_aps, 4)
    
    # Precompute multipliers for key calculation
    # Key = c0*K^3 + c1*K^2 + c2*K + c3
    
    def try_solve(K, time_limit=10.0):
        start_time = time.time()
        
        # Precompute bad lookup table for this K
        # bad_lookup[key] = True if the 4 colors use < 3 distinct values
        # Key range: 0 to K^4 - 1
        K4 = K ** 4
        bad_lookup = np.zeros(K4, dtype=bool)
        
        # Iterate through all possible combinations of 4 colors
        # This is fast for small K (e.g., K=4 -> 256, K=5 -> 625)
        for c0 in range(K):
            for c1 in range(K):
                for c2 in range(K):
                    for c3 in range(K):
                        key = c0 * K**3 + c1 * K**2 + c2 * K + c3
                        colors_tuple = (c0, c1, c2, c3)
                        # Count distinct colors
                        if len(set(colors_tuple)) < 3:
                            bad_lookup[key] = True
        
        # We will try a few restarts
        max_restarts = 5
        best_solution = None
        
        for restart in range(max_restarts):
            if time.time() - start_time > time_limit:
                break
                
            # Initialize random coloring
            colors = np.random.randint(0, K, N, dtype=np.int32)
            
            # Calculate initial keys and violations
            # keys[ap_idx] = colors[ap_indices[ap_idx, 0]] * K^3 + ...
            # Vectorized key calculation
            # c0, c1, c2, c3 = ap_indices[:, 0], ap_indices[:, 1], ap_indices[:, 2], ap_indices[:, 3]
            c0 = colors[ap_indices[:, 0]]
            c1 = colors[ap_indices[:, 1]]
            c2 = colors[ap_indices[:, 2]]
            c3 = colors[ap_indices[:, 3]]
            
            keys = (c0 * K**3 + c1 * K**2 + c2 * K + c3).astype(np.int32)
            violations = bad_lookup[keys]
            current_penalty = np.sum(violations)
            
            if current_penalty == 0:
                return colors.tolist()
            
            # Simulated Annealing
            T = 1.0
            cooling = 0.999
            min_T = 0.01
            
            # Precompute multipliers for each position in the AP to update keys quickly
            # For an AP (i0, i1, i2, i3), the contribution of color at i0 is c0 * K^3
            # We need to know the multiplier for each index in each AP.
            # multipliers[ap_idx, pos_in_ap] = K^(3 - pos_in_ap)
            # But we need to map index -> (ap_idx, pos_in_ap, multiplier)
            # Let's precompute this mapping for efficient delta calculation.
            
            # index_to_ap_details[idx] = list of (ap_idx, multiplier)
            index_to_ap_details = [[] for _ in range(N)]
            for ap_idx in range(num_aps):
                for pos in range(4):
                    idx = ap_indices[ap_idx, pos]
                    mult = K ** (3 - pos)
                    index_to_ap_details[idx].append((ap_idx, mult))
            
            # SA Loop
            steps = 0
            while T > min_T:
                if time.time() - start_time > time_limit:
                    break
                
                # Pick random index
                i = np.random.randint(0, N)
                old_color = colors[i]
                
                # Pick random new color
                new_color = np.random.randint(0, K)
                if new_color == old_color:
                    continue
                
                # Calculate delta penalty
                # Only APs involving i change
                delta = 0
                details = index_to_ap_details[i]
                
                # We need to update keys and violations for these APs
                # To do this efficiently, we store the current key and violation state
                # But we can just recompute the key for the affected APs.
                
                old_keys_affected = []
                old_violations_affected = []
                ap_indices_affected = []
                
                for ap_idx, mult in details:
                    old_keys_affected.append(keys[ap_idx])
                    old_violations_affected.append(violations[ap_idx])
                    ap_indices_affected.append(ap_idx)
                
                # Apply new color to calculate new keys
                colors[i] = new_color
                
                new_keys_affected = []
                new_violations_affected = []
                
                for ap_idx, mult in details:
                    # Update key: subtract old_color * mult, add new_color * mult
                    old_key = keys[ap_idx]
                    new_key = old_key - old_color * mult + new_color * mult
                    new_keys_affected.append(new_key)
                    new_violations_affected.append(bad_lookup[new_key])
                    
                    # Update global arrays
                    keys[ap_idx] = new_key
                    violations[ap_idx] = new_violations_affected[-1]
                
                # Calculate penalty change
                old_pen_affected = sum(1 for v in old_violations_affected if v)
                new_pen_affected = sum(1 for v in new_violations_affected if v)
                delta = new_pen_affected - old_pen_affected
                current_penalty += delta
                
                # Acceptance
                if delta < 0 or random.random() < np.exp(-delta / T):
                    # Accept
                    pass
                else:
                    # Reject: Revert
                    colors[i] = old_color
                    for ap_idx, new_key, new_viol in zip(ap_indices_affected, new_keys_affected, new_violations_affected):
                        # Revert to old key and violation
                        # We stored old_keys_affected and old_violations_affected
                        pass # We need to revert properly. 
                    
                    # Let's do revert properly
                    for k, v, ap_idx in zip(old_keys_affected, old_violations_affected, ap_indices_affected):
                        keys[ap_idx] = k
                        violations[ap_idx] = v
                    current_penalty -= delta
                
                T *= cooling
                steps += 1
                
                if current_penalty == 0:
                    return colors.tolist()
                    
        return None

    # Try increasing number of colors
    # We expect 4 or 5 to work.
    for K in [4, 5, 6, 7]:
        sol = try_solve(K, time_limit=15.0)
        if sol is not None:
            return sol
            
    # Fallback: Greedy with backtracking or just a valid but high-color solution
    # The seed logic is a valid greedy that produces a solution, though it might use more colors.
    # Let's implement the seed's greedy logic as a fallback to ensure validity.
    colors = []
    for x in range(1, 129):
        for color in range(128):
            if (all(len({color}|{colors[x-1-j*d] for j in (1,2,3)})>=3 for d in range(1,(x-1)//3+1))
                and all(any(colors[x-1-j*d]!=color for j in (1,2)) for d in range(1,(x-1)//2+1))):
                colors.append(color)
                break
    return colors
