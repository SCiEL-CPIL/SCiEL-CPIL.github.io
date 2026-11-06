import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    n = 16
    
    def check_q3(adj_list):
        # adj_list: list of 16 ints
        for v in range(n):
            if bin(adj_list[v]).count('1') < 3:
                continue
            neighbors = [u for u in range(n) if (adj_list[v] >> u) & 1]
            L = len(neighbors)
            for i in range(L):
                a = neighbors[i]
                for j in range(i+1, L):
                    b = neighbors[j]
                    if (adj_list[a] >> b) & 1:
                        continue
                    for k in range(j+1, L):
                        c = neighbors[k]
                        if (adj_list[a] >> c) & 1:
                            continue
                        if (adj_list[b] >> c) & 1:
                            continue
                        
                        mask_v = 1 << v
                        mask_a = 1 << a
                        mask_b = 1 << b
                        mask_c = 1 << c
                        forbidden = mask_v | mask_a | mask_b | mask_c
                        
                        xs = (adj_list[a] & adj_list[b]) & ~forbidden
                        ys = (adj_list[a] & adj_list[c]) & ~forbidden
                        zs = (adj_list[b] & adj_list[c]) & ~forbidden
                        
                        if xs == 0 or ys == 0 or zs == 0:
                            continue
                            
                        x_temp = xs
                        while x_temp:
                            x_bit = x_temp & -x_temp
                            x = x_bit.bit_length() - 1
                            x_temp ^= x_bit
                            
                            y_temp = ys
                            while y_temp:
                                y_bit = y_temp & -y_temp
                                y = y_bit.bit_length() - 1
                                y_temp ^= y_bit
                                
                                if x == y:
                                    continue
                                if (adj_list[x] >> y) & 1:
                                    continue
                                
                                z_temp = zs
                                while z_temp:
                                    z_bit = z_temp & -z_temp
                                    z = z_bit.bit_length() - 1
                                    z_temp ^= z_bit
                                    
                                    if z == x or z == y:
                                        continue
                                    if (adj_list[x] >> z) & 1:
                                        continue
                                    if (adj_list[y] >> z) & 1:
                                        continue
                                    
                                    ws = (adj_list[x] & adj_list[y] & adj_list[z])
                                    forbidden2 = forbidden | (1 << x) | (1 << y) | (1 << z)
                                    ws &= ~forbidden2
                                    
                                    if ws:
                                        return True
        return False

    def find_q3_edge(adj_list):
        for v in range(n):
            if bin(adj_list[v]).count('1') < 3:
                continue
            neighbors = [u for u in range(n) if (adj_list[v] >> u) & 1]
            L = len(neighbors)
            for i in range(L):
                a = neighbors[i]
                for j in range(i+1, L):
                    b = neighbors[j]
                    if (adj_list[a] >> b) & 1:
                        continue
                    for k in range(j+1, L):
                        c = neighbors[k]
                        if (adj_list[a] >> c) & 1:
                            continue
                        if (adj_list[b] >> c) & 1:
                            continue
                        
                        mask_v = 1 << v
                        mask_a = 1 << a
                        mask_b = 1 << b
                        mask_c = 1 << c
                        forbidden = mask_v | mask_a | mask_b | mask_c
                        
                        xs = (adj_list[a] & adj_list[b]) & ~forbidden
                        ys = (adj_list[a] & adj_list[c]) & ~forbidden
                        zs = (adj_list[b] & adj_list[c]) & ~forbidden
                        
                        if xs == 0 or ys == 0 or zs == 0:
                            continue
                            
                        x_temp = xs
                        while x_temp:
                            x_bit = x_temp & -x_temp
                            x = x_bit.bit_length() - 1
                            x_temp ^= x_bit
                            
                            y_temp = ys
                            while y_temp:
                                y_bit = y_temp & -y_temp
                                y = y_bit.bit_length() - 1
                                y_temp ^= y_bit
                                
                                if x == y:
                                    continue
                                if (adj_list[x] >> y) & 1:
                                    continue
                                
                                z_temp = zs
                                while z_temp:
                                    z_bit = z_temp & -z_temp
                                    z = z_bit.bit_length() - 1
                                    z_temp ^= z_bit
                                    
                                    if z == x or z == y:
                                        continue
                                    if (adj_list[x] >> z) & 1:
                                        continue
                                    if (adj_list[y] >> z) & 1:
                                        continue
                                    
                                    ws = (adj_list[x] & adj_list[y] & adj_list[z])
                                    forbidden2 = forbidden | (1 << x) | (1 << y) | (1 << z)
                                    ws &= ~forbidden2
                                    
                                    if ws:
                                        return (v, a)
        return None

    best_score = 0
    best_adj = [0] * n
    
    random.seed(42)
    
    for restart in range(200):
        if time.time() > deadline - 2:
            break
        
        # Try different densities
        p = 0.3 + random.random() * 0.3
        
        adj_list = [0] * n
        for u in range(n):
            for v in range(u+1, n):
                if random.random() < p:
                    adj_list[u] |= (1 << v)
                    adj_list[v] |= (1 << u)
        
        # Clean
        steps = 0
        while steps < 500:
            if time.time() > deadline - 2:
                break
            edge_to_remove = find_q3_edge(adj_list)
            if edge_to_remove is None:
                break
            u, v = edge_to_remove
            adj_list[u] &= ~(1 << v)
            adj_list[v] &= ~(1 << u)
            steps += 1
            
        # Add edges greedily
        changed = True
        while changed:
            changed = False
            if time.time() > deadline - 2:
                break
            for u in range(n):
                for v in range(u+1, n):
                    if (adj_list[u] >> v) & 1:
                        continue
                    adj_list[u] |= (1 << v)
                    adj_list[v] |= (1 << u)
                    
                    if check_q3(adj_list):
                        adj_list[u] &= ~(1 << v)
                        adj_list[v] &= ~(1 << u)
                    else:
                        changed = True
        
        # Swap phase
        # Try to swap edges to escape local optima
        swap_attempts = 0
        improved = False
        while swap_attempts < 200:
            if time.time() > deadline - 2:
                break
            
            existing_edges = []
            for u in range(n):
                for v in range(u+1, n):
                    if (adj_list[u] >> v) & 1:
                        existing_edges.append((u, v))
            
            if not existing_edges:
                break
                
            missing_edges = []
            for u in range(n):
                for v in range(u+1, n):
                    if not ((adj_list[u] >> v) & 1):
                        missing_edges.append((u, v))
                        
            if not missing_edges:
                break
                
            e1 = random.choice(existing_edges)
            e2 = random.choice(missing_edges)
            
            u1, v1 = e1
            u2, v2 = e2
            
            adj_list[u1] &= ~(1 << v1)
            adj_list[v1] &= ~(1 << u1)
            adj_list[u2] |= (1 << v2)
            adj_list[v2] |= (1 << u2)
            
            if check_q3(adj_list):
                adj_list[u1] |= (1 << v1)
                adj_list[v1] |= (1 << u1)
                adj_list[u2] &= ~(1 << v2)
                adj_list[v2] &= ~(1 << u2)
            else:
                # Accept swap
                # Try to add more edges
                improved = True
                break
            
            swap_attempts += 1
        
        if improved:
            # Run add edges again
            changed = True
            while changed:
                changed = False
                if time.time() > deadline - 2:
                    break
                for u in range(n):
                    for v in range(u+1, n):
                        if (adj_list[u] >> v) & 1:
                            continue
                        adj_list[u] |= (1 << v)
                        adj_list[v] |= (1 << u)
                        
                        if check_q3(adj_list):
                            adj_list[u] &= ~(1 << v)
                            adj_list[v] &= ~(1 << u)
                        else:
                            changed = True

        current_count = sum(bin(x).count('1') for x in adj_list) // 2
        
        if current_count > best_score:
            best_score = current_count
            best_adj = adj_list[:]
            
    edges = []
    for u in range(n):
        for v in range(u+1, n):
            if (best_adj[u] >> v) & 1:
                edges.append([u, v])
                
    return edges
