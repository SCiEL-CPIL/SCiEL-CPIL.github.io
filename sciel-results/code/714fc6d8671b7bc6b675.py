import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    n = 128
    aps = []
    max_d = (n - 1) // 3
    for d in range(1, max_d + 1):
        for a in range(n - 3 * d):
            b = a + d
            c = a + 2 * d
            dd = a + 3 * d
            aps.append((a, b, c, dd))
    
    aps_arr = np.array(aps, dtype=np.int32)
    n_aps = len(aps_arr)
    
    pos_ap_indices = [[] for _ in range(n)]
    for idx, (a, b, c, d) in enumerate(aps):
        pos_ap_indices[a].append(idx)
        pos_ap_indices[b].append(idx)
        pos_ap_indices[c].append(idx)
        pos_ap_indices[d].append(idx)
    
    pos_ap_indices_arr = [np.array(info, dtype=np.int32) for info in pos_ap_indices]
    
    # Precompute slots for each pos in each AP? No, too much memory.
    # Just check equality.
    
    def is_bad_ap(c0, c1, c2, c3):
        # Returns True if < 3 distinct colors
        # False if >= 3 distinct colors
        # Distinct colors < 3 means:
        # 1. All same
        # 2. Two same, one other (e.g., A A B C is bad? No, A,A,B has 2 distinct. A,A,B,C has 3 distinct.
        # Wait. "uses AT LEAST three distinct colors".
        # So {A, A, B, C} has 3 distinct colors. Valid.
        # {A, A, B, B} has 2 distinct colors. Invalid.
        # {A, A, A, B} has 2 distinct colors. Invalid.
        # {A, A, A, A} has 1 distinct color. Invalid.
        # So we need to check if the number of unique values in (c0, c1, c2, c3) is < 3.
        
        # Fast check:
        # If c0 != c1 and c0 != c2 and c1 != c2: 3 distinct among first 3. Valid.
        # If c0 != c1 and c0 != c3 and c1 != c3: 3 distinct among 0,1,3. Valid.
        # If c0 != c2 and c0 != c3 and c2 != c3: 3 distinct among 0,2,3. Valid.
        # If c1 != c2 and c1 != c3 and c2 != c3: 3 distinct among 1,2,3. Valid.
        
        if (c0 != c1 and c0 != c2 and c1 != c2): return False
        if (c0 != c1 and c0 != c3 and c1 != c3): return False
        if (c0 != c2 and c0 != c3 and c2 != c3): return False
        if (c1 != c2 and c1 != c3 and c2 != c3): return False
        return True

    def local_search(c, K, max_steps=10000):
        bad_mask = np.zeros(n_aps, dtype=bool)
        for idx in range(n_aps):
            a, b, cpos, d = aps_arr[idx]
            if is_bad_ap(c[a], c[b], c[cpos], c[d]):
                bad_mask[idx] = True
        
        current_bad = int(np.sum(bad_mask))
        
        if current_bad == 0:
            return c, 0
        
        no_improve = 0
        max_no_improve = 500
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
            
            if current_bad == 0:
                break
            
            # Pick a bad AP
            bad_indices = np.where(bad_mask)[0]
            if len(bad_indices) == 0:
                break
            target_idx = bad_indices[random.randint(0, len(bad_indices) - 1)]
            a, b, cpos, d = aps_arr[target_idx]
            pos = [a, b, cpos, d][random.randint(0, 3)]
            
            old_color = c[pos]
            best_new_color = old_color
            best_delta = 0
            
            ap_indices = pos_ap_indices_arr[pos]
            
            for new_color in range(K):
                if new_color == old_color:
                    continue
                
                delta = 0
                for ap_idx in ap_indices:
                    aa, bb, cc, dd = aps_arr[ap_idx]
                    
                    # Determine if this AP is affected
                    # We need to check old vs new bad status
                    was_bad = bad_mask[ap_idx]
                    
                    # Get other 3 colors
                    if aa == pos:
                        o1, o2, o3 = c[bb], c[cc], c[dd]
                        # New tuple: (new_color, o1, o2, o3)
                        is_bad_now = is_bad_ap(new_color, o1, o2, o3)
                    elif bb == pos:
                        o1, o2, o3 = c[aa], c[cc], c[dd]
                        is_bad_now = is_bad_ap(o1, new_color, o2, o3)
                    elif cc == pos:
                        o1, o2, o3 = c[aa], c[bb], c[dd]
                        is_bad_now = is_bad_ap(o1, o2, new_color, o3)
                    else: # dd == pos
                        o1, o2, o3 = c[aa], c[bb], c[cc]
                        is_bad_now = is_bad_ap(o1, o2, o3, new_color)
                    
                    if was_bad and not is_bad_now:
                        delta -= 1
                    elif not was_bad and is_bad_now:
                        delta += 1
                
                if delta < best_delta:
                    best_delta = delta
                    best_new_color = new_color
            
            if best_new_color == old_color:
                no_improve += 1
                if no_improve > max_no_improve:
                    break
                new_color = random.randint(0, K - 1)
                if new_color == old_color:
                    continue
                c[pos] = new_color
                # Update bad mask
                for ap_idx in ap_indices:
                    aa, bb, cc, dd = aps_arr[ap_idx]
                    new_bad = is_bad_ap(c[aa], c[bb], c[cc], c[dd])
                    was_bad = bad_mask[ap_idx]
                    if was_bad and not new_bad:
                        current_bad -= 1
                        bad_mask[ap_idx] = False
                    elif not was_bad and new_bad:
                        current_bad += 1
                        bad_mask[ap_idx] = True
                continue
            
            c[pos] = best_new_color
            for ap_idx in ap_indices:
                aa, bb, cc, dd = aps_arr[ap_idx]
                new_bad = is_bad_ap(c[aa], c[bb], c[cc], c[dd])
                was_bad = bad_mask[ap_idx]
                if was_bad and not new_bad:
                    current_bad -= 1
                    bad_mask[ap_idx] = False
                elif not was_bad and new_bad:
                    current_bad += 1
                    bad_mask[ap_idx] = True
            
            no_improve = 0
        
        return c, current_bad

    best_result = None
    best_bad = float('inf')
    
    for K in [4, 5]:
        if time.time() > deadline:
            break
        for r in range(200):
            if time.time() > deadline:
                break
            c = [random.randint(0, K - 1) for _ in range(n)]
            c, bad_count = local_search(c, K)
            if bad_count == 0:
                return c
            if bad_count < best_bad:
                best_bad = bad_count
                best_result = c[:]
    
    if best_result:
        return best_result
    return [i % 4 for i in range(n)]

