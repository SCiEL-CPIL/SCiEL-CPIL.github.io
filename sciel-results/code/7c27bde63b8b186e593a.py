import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = 115.0
    
    # Try n=21 first, then 20
    for n in [21, 20]:
        if time.time() > deadline:
            continue
            
        # Precompute subsets and edge-subset mappings
        subsets_6 = list(itertools.combinations(range(n), 6))
        num_subsets = len(subsets_6)
        
        # subset_masks[i] is the bitmask for subsets_6[i]
        subset_masks = [0] * num_subsets
        for i, s in enumerate(subsets_6):
            m = 0
            for v in s:
                m |= (1 << v)
            subset_masks[i] = m
            
        # edge_subsets[u][v] is list of indices of 6-subsets containing u and v
        # Use a dictionary or 2D list. n <= 22, so 22x22 is fine.
        edge_subsets = [[[] for _ in range(n)] for _ in range(n)]
        for i, s in enumerate(subsets_6):
            for u in s:
                for v in s:
                    if u < v:
                        edge_subsets[u][v].append(i)
                        edge_subsets[v][u].append(i)
                        
        # Initialize random graph
        adj = [0] * n
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
                    
        # Initialize degrees for each subset
        # degs[i] is a list of 6 degrees for the vertices in subsets_6[i]
        degs = [[0]*6 for _ in range(num_subsets)]
        is_reg = [False] * num_subsets
        current_count = 0
        
        for i, s in enumerate(subsets_6):
            m = subset_masks[i]
            ds = []
            for v in s:
                d = (adj[v] & m).bit_count()
                ds.append(d)
            degs[i] = ds
            if len(set(ds)) == 1:
                is_reg[i] = True
                current_count += 1
                
        best_count = current_count
        best_adj = adj[:]
        
        # SA Parameters
        T = 1.0
        cool_rate = 0.999
        min_T = 0.01
        
        while time.time() < deadline - 5 and T > min_T:
            # Pick random edge
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u == v:
                continue
                
            # Flip edge
            # Determine if edge exists
            exists = (adj[u] >> v) & 1
            
            # Update adj
            if exists:
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
            else:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                
            delta = 0
            # Update subsets containing u and v
            for idx in edge_subsets[u][v]:
                s = subsets_6[idx]
                # Find positions of u and v in s
                # Since s is a tuple of 6, we can just check
                # We stored degs[idx] corresponding to s order
                pos_u = -1
                pos_v = -1
                for k, vertex in enumerate(s):
                    if vertex == u:
                        pos_u = k
                    elif vertex == v:
                        pos_v = k
                    
                # Update degrees
                # If edge added, degree increases by 1 for u and v within this subset
                # If edge removed, degree decreases by 1
                change = 1 if not exists else -1
                
                old_deg_u = degs[idx][pos_u]
                old_deg_v = degs[idx][pos_v]
                
                new_deg_u = old_deg_u + change
                new_deg_v = old_deg_v + change
                
                degs[idx][pos_u] = new_deg_u
                degs[idx][pos_v] = new_deg_v
                
                # Check regularity
                # Get all degrees
                d_list = degs[idx]
                was_reg = is_reg[idx]
                
                # Check if all equal
                is_now_reg = True
                first = d_list[0]
                for d in d_list[1:]:
                    if d != first:
                        is_now_reg = False
                        break
                        
                if was_reg and not is_now_reg:
                    delta -= 1
                elif not was_reg and is_now_reg:
                    delta += 1
                # else no change
                
                is_reg[idx] = is_now_reg
                
            current_count += delta
            
            # Acceptance
            if delta < 0 or random.random() < pow(2.718, -delta / T):
                best_count = min(best_count, current_count)
                if current_count == 0:
                    best_adj = adj[:]
                    # Check larger subsets
                    if check_all_subsets(adj, n):
                        return {'n': n, 'edges': get_edges(adj)}
                else:
                    if current_count < best_count:
                        best_count = current_count
                        best_adj = adj[:]
            else:
                # Reject: revert
                if exists:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                else:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                    
                # Revert degs and is_reg
                for idx in edge_subsets[u][v]:
                    s = subsets_6[idx]
                    pos_u = -1
                    pos_v = -1
                    for k, vertex in enumerate(s):
                        if vertex == u:
                            pos_u = k
                        elif vertex == v:
                            pos_v = k
                            
                    change = 1 if not exists else -1
                    # Revert means apply -change
                    degs[idx][pos_u] -= change
                    degs[idx][pos_v] -= change
                    
                    d_list = degs[idx]
                    is_now_reg = True
                    first = d_list[0]
                    for d in d_list[1:]:
                        if d != first:
                            is_now_reg = False
                            break
                    is_reg[idx] = is_now_reg

                current_count -= delta
                # Note: delta was calculated for the forward move.
                # If we revert, the count goes back.
                
            T *= cool_rate

        # If loop finished without finding 0, use best_adj
        if best_count == 0 and check_all_subsets(best_adj, n):
            return {'n': n, 'edges': get_edges(best_adj)}

    # Fallback
    return {'n': 5, 'edges': []}

def get_edges(adj):
    edges = []
    n = len(adj)
    for i in range(n):
        for j in range(i+1, n):
            if (adj[i] >> j) & 1:
                edges.append([i, j])
    return edges

def check_all_subsets(adj, n):
    # Check all subsets of size >= 6
    # This is expensive, so only called when count_6 == 0
    for k in range(6, n+1):
        for s in itertools.combinations(range(n), k):
            mask = 0
            for v in s:
                mask |= (1 << v)
            degs = [(adj[v] & mask).bit_count() for v in s]
            if len(set(degs)) == 1:
                return False
    return True
