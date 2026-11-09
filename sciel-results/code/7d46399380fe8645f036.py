# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    n = 128
    
    # Precompute all nonconstant 4-term APs
    aps_list = []
    for i in range(n):
        for d in range(1, (n - 1 - i) // 3 + 1):
            j3 = i + 3 * d
            if j3 < n:
                aps_list.append((i, i + d, i + 2 * d, j3))
    
    num_aps = len(aps_list)
    
    # For each position, which AP indices contain it
    pos_to_aps = [[] for _ in range(n)]
    for idx, ap in enumerate(aps_list):
        for pos in ap:
            pos_to_aps[pos].append(idx)
    
    def ap_is_violation(c0, c1, c2, c3):
        """Return 1 if the 4 colors have < 3 distinct values."""
        # All same
        if c0 == c1 == c2 == c3:
            return 1
        # Exactly 2 distinct: check all patterns
        # (a,a,a,b), (a,a,b,b), (a,b,b,b)
        if c0 == c1 == c2 and c3 != c2:
            return 1
        if c0 == c1 and c2 == c3 and c1 != c2:
            return 1
        if c1 == c2 == c3 and c0 != c1:
            return 1
        return 0
    
    def local_search(arr, max_time=2.0, color_range=3):
        """Local search to minimize violations."""
        arr = arr.copy()
        
        # Compute initial violation status
        ap_viol = [0] * num_aps
        current_viol = 0
        for idx in range(num_aps):
            ap = aps_list[idx]
            v = ap_is_violation(arr[ap[0]], arr[ap[1]], arr[ap[2]], arr[ap[3]])
            ap_viol[idx] = v
            current_viol += v
        
        if current_viol == 0:
            return arr, 0
        
        t_start = time.time()
        stall = 0
        
        while time.time() - t_start < max_time and current_viol > 0:
            improved = False
            order = list(range(n))
            random.shuffle(order)
            
            for pos in order:
                if time.time() - t_start > max_time:
                    break
                
                old_color = arr[pos]
                affected = pos_to_aps[pos]
                
                # Save old violation states
                old_ap_viol = [ap_viol[idx] for idx in affected]
                
                for nc in range(color_range):
                    if nc == old_color:
                        continue
                    
                    arr[pos] = nc
                    
                    # Compute new violation states for affected APs
                    new_delta = 0
                    for k, ap_idx in enumerate(affected):
                        ap = aps_list[ap_idx]
                        new_v = ap_is_violation(arr[ap[0]], arr[ap[1]], arr[ap[2]], arr[ap[3]])
                        new_delta += new_v - old_ap_viol[k]
                    
                    if new_delta < 0:
                        # Accept
                        current_viol += new_delta
                        for k, ap_idx in enumerate(affected):
                            ap = aps_list[ap_idx]
                            ap_viol[ap_idx] = ap_is_violation(arr[ap[0]], arr[ap[1]], arr[ap[2]], arr[ap[3]])
                        improved = True
                        break
                    else:
                        # Revert
                        arr[pos] = old_color
                
                if improved:
                    break
            
            if not improved:
                stall += 1
                if stall > 300:
                    break
                # Perturb: change a random position to a random color
                pos = random.randint(0, n - 1)
                old_c = arr[pos]
                arr[pos] = random.randint(0, color_range - 1)
                affected = pos_to_aps[pos]
                for ap_idx in affected:
                    ap = aps_list[ap_idx]
                    new_v = ap_is_violation(arr[ap[0]], arr[ap[1]], arr[ap[2]], arr[ap[3]])
                    current_viol += new_v - ap_viol[ap_idx]
                    ap_viol[ap_idx] = new_v
        
        return arr, current_viol
    
    best_arr = None
    best_viol = float('inf')
    
    random.seed(42)
    
    # Phase 1: Try 3-colorings with many restarts
    for restart in range(1000):
        if time.time() > deadline - 5:
            break
        
        arr = [random.randint(0, 2) for _ in range(n)]
        arr_np = np.array(arr, dtype=np.int32)
        
        arr_ls, viol_ls = local_search(arr_np, max_time=0.5, color_range=3)
        
        if viol_ls < best_viol:
            best_viol = viol_ls
            best_arr = arr_ls.copy()
        
        if best_viol == 0:
            break
    
    if best_viol == 0:
        return best_arr.tolist()
    
    # Phase 2: Try 4-colorings
    if time.time() < deadline - 10:
        for restart in range(500):
            if time.time() > deadline - 5:
                break
            
            arr = [random.randint(0, 3) for _ in range(n)]
            arr_np = np.array(arr, dtype=np.int32)
            
            arr_ls, viol_ls = local_search(arr_np, max_time=0.4, color_range=4)
            
            if viol_ls < best_viol:
                best_viol = viol_ls
                best_arr = arr_ls.copy()
        
        if best_viol == 0:
            return best_arr.tolist()
    
    # Phase 3: Try 5-colorings
    if best_viol > 0 and time.time() < deadline - 10:
        for restart in range(300):
            if time.time() > deadline - 5:
                break
            
            arr = [random.randint(0, 4) for _ in range(n)]
            arr_np = np.array(arr, dtype=np.int32)
            
            arr_ls, viol_ls = local_search(arr_np, max_time=0.3, color_range=5)
            
            if viol_ls < best_viol:
                best_viol = viol_ls
                best_arr = arr_ls.copy()
        
        if best_viol == 0:
            return best_arr.tolist()
    
    # Fallback
    if best_arr is None:
        best_arr = np.zeros(n, dtype=np.int32)
    
    return best_arr.tolist()
# EVOLVE-BLOCK-END
