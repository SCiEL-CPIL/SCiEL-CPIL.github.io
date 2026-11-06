import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin

    best_n = 39
    best_edges = []
    
    # Target n=46 first, then 45, etc.
    # n=47 is likely impossible due to ex(47, C4) < required edges.
    for target_n in [46, 45, 44, 43, 42, 41, 40, 39]:
        if time.time() > deadline - 10:
            break
            
        success = False
        n = target_n
        min_deg_req = n - 39
        
        # We will try multiple random restarts
        max_restarts = 50
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break
                
            # 1. Initialize with Random Greedy
            adj = [0] * n
            all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
            random.shuffle(all_edges)
            
            for u, v in all_edges:
                # Fast C4 check
                au = adj[u]
                av = adj[v]
                
                if au == 0 or av == 0:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    continue
                
                c4_found = False
                temp = au
                while temp:
                    w = (temp & (-temp)).bit_length() - 1
                    if adj[w] & av:
                        c4_found = True
                        break
                    temp &= temp - 1
                
                if not c4_found:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
            
            degrees = [bin(adj[i]).count('1') for i in range(n)]
            min_deg = min(degrees)
            
            if min_deg >= min_deg_req:
                success = True
                best_n = n
                edges = []
                for i in range(n):
                    for j in range(i+1, n):
                        if (adj[i] >> j) & 1:
                            edges.append([i, j])
                best_edges = edges
                break
            
            # 2. Local Search (Simulated Annealing)
            if time.time() > deadline - 5:
                continue
                
            T = 1.0
            min_T = 0.01
            cooling = 0.99
            max_steps = 5000
            step = 0
            current_min_deg = min(degrees)
            best_min_deg = current_min_deg
            best_adj_copy = [a for a in adj]
            
            while step < max_steps and time.time() < deadline - 5:
                step += 1
                
                # Pick a random edge to remove
                # Find all edges
                # Optimization: Maintain a list of edges? 
                # For n=46, edges ~ 200. List is fine.
                edge_list = []
                for i in range(n):
                    temp = adj[i]
                    while temp:
                        j = (temp & (-temp)).bit_length() - 1
                        if i < j:
                            edge_list.append((i, j))
                        temp &= temp - 1
                
                if not edge_list:
                    break
                
                u_rem, v_rem = random.choice(edge_list)
                
                # Pick a random non-edge to add
                # Generate random pair
                u_add, v_add = random.sample(range(n), 2)
                if u_add > v_add:
                    u_add, v_add = v_add, u_add
                    
                # Check if it's already an edge
                if (adj[u_add] >> v_add) & 1:
                    continue
                
                # Perform Swap: Remove (u_rem, v_rem), Add (u_add, v_add)
                # 1. Remove (u_rem, v_rem)
                adj[u_rem] &= ~(1 << v_rem)
                adj[v_rem] &= ~(1 << u_rem)
                degrees[u_rem] -= 1
                degrees[v_rem] -= 1
                
                # 2. Check if adding (u_add, v_add) creates C4
                au = adj[u_add]
                av = adj[v_add]
                
                c4_found = False
                if au != 0 and av != 0:
                    temp = au
                    while temp:
                        w = (temp & (-temp)).bit_length() - 1
                        if adj[w] & av:
                            c4_found = True
                            break
                        temp &= temp - 1
                
                if not c4_found:
                    # Accept the swap
                    adj[u_add] |= (1 << v_add)
                    adj[v_add] |= (1 << u_add)
                    degrees[u_add] += 1
                    degrees[v_add] += 1
                    
                    new_min_deg = min(degrees)
                    if new_min_deg > current_min_deg:
                        current_min_deg = new_min_deg
                        if current_min_deg > best_min_deg:
                            best_min_deg = current_min_deg
                            best_adj_copy = [a for a in adj]
                            if best_min_deg >= min_deg_req:
                                success = True
                                break
                    elif new_min_deg == current_min_deg:
                        # Accept with probability 1 (or high) to explore
                        pass
                    else:
                        # Worse, accept with prob exp(delta/T)
                        delta = new_min_deg - current_min_deg
                        if random.random() > np.exp(delta / T):
                            # Reject swap
                            # Undo
                            adj[u_add] &= ~(1 << v_add)
                            adj[v_add] &= ~(1 << u_add)
                            degrees[u_add] -= 1
                            degrees[v_add] -= 1
                            adj[u_rem] |= (1 << v_rem)
                            adj[v_rem] |= (1 << u_rem)
                            degrees[u_rem] += 1
                            degrees[v_rem] += 1
                            current_min_deg = min(degrees) # Should be same as before
                        else:
                            current_min_deg = new_min_deg
                else:
                    # C4 found, reject swap (undo removal)
                    adj[u_rem] |= (1 << v_rem)
                    adj[v_rem] |= (1 << u_rem)
                    degrees[u_rem] += 1
                    degrees[v_rem] += 1
                
                T *= cooling
                if T < min_T:
                    T = min_T
            
            if success:
                break
        
        if success:
            break
    
    if best_n == 39:
        return {'n': 39, 'edges': []}
    else:
        return {'n': best_n, 'edges': best_edges}
