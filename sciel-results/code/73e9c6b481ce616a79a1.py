import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n = 16
    adj = [0] * n
    
    def creates_q3(u, v):
        nu = adj[u]
        nv = adj[v]
        
        # Get neighbors of u as a list
        # We need pairs
        # Instead of iterating list, we can iterate bits
        # But we need pairs, so list is easier
        
        n_u_list = []
        mask = nu
        while mask:
            # Get lowest set bit
            lsb = mask & -mask
            idx = (lsb.bit_length() - 1)
            n_u_list.append(idx)
            mask ^= lsb
            
        n_v_list = []
        mask = nv
        while mask:
            lsb = mask & -mask
            idx = (lsb.bit_length() - 1)
            n_v_list.append(idx)
            mask ^= lsb
            
        for i in range(len(n_u_list)):
            for j in range(i + 1, len(n_u_list)):
                x = n_u_list[i]
                y = n_u_list[j]
                
                # x, y not adjacent
                if (adj[x] >> y) & 1:
                    continue
                
                # z in N(v) \cap N(x)
                z_mask = nv & adj[x]
                if z_mask == 0:
                    continue
                
                # w in N(v) \cap N(y)
                w_mask = nv & adj[y]
                if w_mask == 0:
                    continue
                
                # p in N(x) \cap N(y)
                p_mask = adj[x] & adj[y]
                # Exclude u, v, x, y
                # u is not in adj[x] (since u is x's neighbor? No, x is u's neighbor, so u is in adj[x])
                # We need to exclude u, v, x, y from p_mask
                p_mask &= ~(1 << u)
                p_mask &= ~(1 << v)
                p_mask &= ~(1 << x)
                p_mask &= ~(1 << y)
                
                if p_mask == 0:
                    continue
                
                # Iterate z
                z_mask_tmp = z_mask
                while z_mask_tmp:
                    lsb_z = z_mask_tmp & -z_mask_tmp
                    z = (lsb_z.bit_length() - 1)
                    z_mask_tmp ^= lsb_z
                    
                    # Iterate w
                    w_mask_tmp = w_mask
                    while w_mask_tmp:
                        lsb_w = w_mask_tmp & -w_mask_tmp
                        w = (lsb_w.bit_length() - 1)
                        w_mask_tmp ^= lsb_w
                        
                        if z == w:
                            continue
                        if (adj[z] >> w) & 1:
                            continue
                        
                        # Iterate p
                        p_mask_tmp = p_mask
                        while p_mask_tmp:
                            lsb_p = p_mask_tmp & -p_mask_tmp
                            p = (lsb_p.bit_length() - 1)
                            p_mask_tmp ^= lsb_p
                            
                            if p == z or p == w:
                                continue
                            
                            # k in N(z) \cap N(w) \cap N(p)
                            k_mask = adj[z] & adj[w] & adj[p]
                            # Exclude u, v, x, y, z, w, p
                            k_mask &= ~(1 << u)
                            k_mask &= ~(1 << v)
                            k_mask &= ~(1 << x)
                            k_mask &= ~(1 << y)
                            k_mask &= ~(1 << z)
                            k_mask &= ~(1 << w)
                            k_mask &= ~(1 << p)
                            
                            if k_mask != 0:
                                return True
        return False

    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    random.shuffle(all_edges)
    
    for u, v in all_edges:
        if time.time() > deadline - 10.0:
            break
        if not creates_q3(u, v):
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
    current_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if (adj[i] >> j) & 1:
                current_edges.append((i, j))
                
    missing_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if not ((adj[i] >> j) & 1):
                missing_edges.append((i, j))
                
    random.shuffle(current_edges)
    random.shuffle(missing_edges)
    
    max_iterations = 500000
    for _ in range(max_iterations):
        if time.time() > deadline - 5.0:
            break
        if not missing_edges:
            break
        
        add_u, add_v = random.choice(missing_edges)
        
        if not creates_q3(add_u, add_v):
            adj[add_u] |= (1 << add_v)
            adj[add_v] |= (1 << add_u)
            current_edges.append((add_u, add_v))
            missing_edges.remove((add_u, add_v))
        else:
            if not current_edges:
                continue
            rem_idx = random.randint(0, len(current_edges) - 1)
            rem_u, rem_v = current_edges[rem_idx]
            
            adj[rem_u] &= ~(1 << rem_v)
            adj[rem_v] &= ~(1 << rem_u)
            
            if not creates_q3(add_u, add_v):
                adj[add_u] |= (1 << add_v)
                adj[add_v] |= (1 << add_u)
                
                current_edges[rem_idx] = (add_u, add_v)
                missing_edges.remove((add_u, add_v))
                missing_edges.append((rem_u, rem_v))
            else:
                adj[rem_u] |= (1 << rem_v)
                adj[rem_v] |= (1 << rem_u)

    result = []
    for i in range(n):
        for j in range(i + 1, n):
            if (adj[i] >> j) & 1:
                result.append([i, j])
                
    return result
