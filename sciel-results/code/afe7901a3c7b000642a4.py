import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s limit to be safe within 60s
    
    N = 256
    # 1. Identify bad vertices (2*v is a perfect square)
    bad_mask = 0
    good_indices = []
    for v in range(1, N + 1):
        if int(math.isqrt(2 * v))**2 == 2 * v:
            bad_mask |= (1 << (v - 1))
        else:
            good_indices.append(v - 1)
            
    M = len(good_indices)
    if M == 0:
        return []
        
    # Initial mask of all good vertices
    current_mask = 0
    for idx in good_indices:
        current_mask |= (1 << idx)
        
    # 2. Build adjacency masks
    adj = [0] * N
    for i in range(N):
        if not (current_mask & (1 << i)):
            continue
        val_i = i + 1
        for j in range(i + 1, N):
            if not (current_mask & (1 << j)):
                continue
            val_j = j + 1
            s = val_i + val_j
            r = int(math.isqrt(s))
            if r * r == s:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                
    best_size = 0
    best_mask = 0

    def popcount(x):
        return x.bit_count()

    def greedy_is(mask):
        if mask == 0:
            return 0, 0
        temp_mask = mask
        set_mask = 0
        count = 0
        while temp_mask:
            min_deg = float('inf')
            min_v = -1
            mm = temp_mask
            while mm:
                v = (mm & -mm).bit_length() - 1
                mm &= mm - 1
                deg = popcount(adj[v] & temp_mask)
                if deg < min_deg:
                    min_deg = deg
                    min_v = v
            if min_v != -1:
                set_mask |= (1 << min_v)
                count += 1
                temp_mask &= ~adj[min_v]
                temp_mask &= ~(1 << min_v)
        return count, set_mask

    def greedy_is_weighted(mask, weight_fn):
        """Greedy with custom weight: pick vertex with min weight, then min degree."""
        if mask == 0:
            return 0, 0
        temp_mask = mask
        set_mask = 0
        count = 0
        while temp_mask:
            best_key = None
            min_v = -1
            mm = temp_mask
            while mm:
                v = (mm & -mm).bit_length() - 1
                mm &= mm - 1
                deg = popcount(adj[v] & temp_mask)
                w = weight_fn(v, deg)
                key = (w, deg, v)
                if best_key is None or key < best_key:
                    best_key = key
                    min_v = v
            if min_v != -1:
                set_mask |= (1 << min_v)
                count += 1
                temp_mask &= ~adj[min_v]
                temp_mask &= ~(1 << min_v)
        return count, set_mask

    def local_search_1_2(start_mask, max_iters=200):
        current_mask_ls = start_mask
        current_size_ls = popcount(start_mask)
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            
            improved = False
            selected_list = []
            tmp = current_mask_ls
            while tmp:
                v = (tmp & -tmp).bit_length() - 1
                tmp &= tmp - 1
                selected_list.append(v)
            
            random.shuffle(selected_list)
            
            for u in selected_list:
                mask_without_u = current_mask_ls & ~(1 << u)
                
                union_neighbors = 0
                tmp = mask_without_u
                while tmp:
                    v = (tmp & -tmp).bit_length() - 1
                    tmp &= tmp - 1
                    union_neighbors |= adj[v]
                
                available = current_mask & ~mask_without_u & ~union_neighbors
                
                avail_list = []
                tmp = available
                while tmp:
                    v = (tmp & -tmp).bit_length() - 1
                    tmp &= tmp - 1
                    avail_list.append(v)
                
                if len(avail_list) < 2:
                    continue
                
                random.shuffle(avail_list)
                
                found_pair = False
                for i in range(len(avail_list)):
                    v = avail_list[i]
                    for j in range(i + 1, len(avail_list)):
                        w = avail_list[j]
                        if not (adj[v] & (1 << w)):
                            new_mask = mask_without_u | (1 << v) | (1 << w)
                            current_mask_ls = new_mask
                            current_size_ls += 1
                            improved = True
                            found_pair = True
                            break
                    if found_pair:
                        break
                
            if not improved:
                break
                
        return current_mask_ls, current_size_ls

    def local_search_2_2(start_mask, max_iters=100):
        """Try to replace two vertices with two others (net zero) to enable future 1-2 moves."""
        current_mask_ls = start_mask
        current_size_ls = popcount(start_mask)
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            
            improved = False
            selected_list = []
            tmp = current_mask_ls
            while tmp:
                v = (tmp & -tmp).bit_length() - 1
                tmp &= tmp - 1
                selected_list.append(v)
            
            if len(selected_list) < 2:
                break
            
            random.shuffle(selected_list)
            
            # Try removing pairs
            for i in range(len(selected_list)):
                for j in range(i + 1, len(selected_list)):
                    u1, u2 = selected_list[i], selected_list[j]
                    mask_without = current_mask_ls & ~(1 << u1) & ~(1 << u2)
                    
                    union_neighbors = 0
                    tmp = mask_without
                    while tmp:
                        v = (tmp & -tmp).bit_length() - 1
                        tmp &= tmp - 1
                        union_neighbors |= adj[v]
                    
                    available = current_mask & ~mask_without & ~union_neighbors
                    avail_list = []
                    tmp = available
                    while tmp:
                        v = (tmp & -tmp).bit_length() - 1
                        tmp &= tmp - 1
                        avail_list.append(v)
                    
                    if len(avail_list) < 2:
                        continue
                    
                    random.shuffle(avail_list)
                    
                    found_pair = False
                    for a in range(len(avail_list)):
                        v = avail_list[a]
                        for b in range(a + 1, len(avail_list)):
                            w = avail_list[b]
                            if not (adj[v] & (1 << w)):
                                new_mask = mask_without | (1 << v) | (1 << w)
                                # Same size, but might enable further improvements
                                current_mask_ls = new_mask
                                improved = True
                                found_pair = True
                                break
                        if found_pair:
                            break
                    if found_pair:
                        break
                if improved:
                    break
            
            if not improved:
                break
                
        return current_mask_ls, current_size_ls

    # Multiple diverse starting points using modular residue seeding
    # Squares mod 4: 0, 1. So a+b is square iff a+b ≡ 0 or 1 mod 4.
    # This means: (0,0)->0, (0,1)->1, (1,0)->1, (1,1)->2. 
    # Conflicts: (0,0), (0,1), (1,0) pairs conflict mod 4.
    # Strategy: try different residue-biased greedy starts.
    
    def residue_class(v):
        return (v + 1) % 4
    
    # Generate diverse starting masks
    start_configs = []
    
    # Config 0: Pure min-degree greedy
    start_configs.append(('min_deg', lambda v, deg: 0))
    
    # Config 1: Prefer residues 1,2 mod 4 (less conflicting)
    def w1(v, deg):
        r = residue_class(v)
        return 0 if r in (1, 2) else 1
    start_configs.append(('res_12', w1))
    
    # Config 2: Prefer residues 2,3 mod 4
    def w2(v, deg):
        r = residue_class(v)
        return 0 if r in (2, 3) else 1
    start_configs.append(('res_23', w2))
    
    # Config 3: Randomized
    def w3(v, deg):
        return random.random()
    start_configs.append(('random', w3))
    
    # Config 4: Prefer higher values (larger numbers have fewer square-sum partners on average)
    def w4(v, deg):
        return -v
    start_configs.append(('high_val', w4))
    
    # Config 5: Mixed - prefer low degree AND favorable residue
    def w5(v, deg):
        r = residue_class(v)
        res_pen = 0 if r in (1, 2) else 0.5
        return res_pen + deg * 0.1
    start_configs.append(('mixed', w5))
    
    # Run each config and keep the best
    for name, wfn in start_configs:
        if time.time() > deadline:
            break
        
        if name == 'min_deg':
            init_size, init_mask = greedy_is(current_mask)
        else:
            init_size, init_mask = greedy_is_weighted(current_mask, wfn)
        
        # Local search 1-2
        improved_mask, improved_size = local_search_1_2(init_mask, max_iters=50)
        if improved_size > best_size:
            best_size = improved_size
            best_mask = improved_mask
        
        # Try 2-2 then 1-2
        m2, s2 = local_search_2_2(improved_mask, max_iters=30)
        m3, s3 = local_search_1_2(m2, max_iters=50)
        if s3 > best_size:
            best_size = s3
            best_mask = m3

    # B&B with better branching: branch on vertex with highest degree
    # but also consider the "constraint" effect
    def solve(mask, current_size, current_set_mask, depth=0):
        nonlocal best_size, best_mask
        if time.time() > deadline:
            return
        
        if mask == 0:
            if current_size > best_size:
                best_size = current_size
                best_mask = current_set_mask
            return
            
        if current_size + popcount(mask) <= best_size:
            return
            
        # Upper bound via greedy on remaining mask
        greedy_ub_size, _ = greedy_is(mask)
        if current_size + greedy_ub_size <= best_size:
            return
            
        # Select vertex with highest degree
        max_deg = -1
        v = -1
        mm = mask
        while mm:
            vv = (mm & -mm).bit_length() - 1
            mm &= mm - 1
            deg = popcount(adj[vv] & mask)
            if deg > max_deg:
                max_deg = deg
                v = vv
                
        if v == -1:
            return

        # Branch 1: Include v
        new_mask_inc = mask & ~adj[v] & ~(1 << v)
        solve(new_mask_inc, current_size + 1, current_set_mask | (1 << v), depth + 1)
        
        # Branch 2: Exclude v
        new_mask_exc = mask & ~(1 << v)
        solve(new_mask_exc, current_size, current_set_mask, depth + 1)

    solve(current_mask, 0, 0)
    
    result = []
    bm = best_mask
    while bm:
        v = (bm & -bm).bit_length() - 1
        bm &= bm - 1
        result.append(v + 1)
        
    return sorted(result)
