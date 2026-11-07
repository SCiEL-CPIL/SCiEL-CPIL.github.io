import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85
    n = 15
    
    # Initialize with K_{7,8}
    # A: 0-6, B: 7-14
    adj = [0] * n
    # Build adjacency masks
    for i in range(7):
        for j in range(7, 15):
            adj[i] |= (1 << j)
            adj[j] |= (1 << i)
    
    def get_degrees(adj):
        return [bin(adj[i]).count('1') for i in range(n)]
        
    def get_deg_masks(deg):
        masks = [0] * (n + 1)
        for i in range(n):
            masks[deg[i]] |= (1 << i)
        return masks

    deg = get_degrees(adj)
    deg_masks = get_deg_masks(deg)
    
    def is_valid_add(u, v, adj, deg, deg_masks):
        # Check if adding (u,v) is valid
        # Nu = N(u) \ {v}
        # Nv = N(v) \ {u}
        Nu = adj[u] & ~(1 << v)
        Nv = adj[v] & ~(1 << u)
        
        # For each degree d, check conflict
        # We iterate over possible degrees present in Nu or Nv
        # Or just iterate 0..n
        for d in range(n + 1):
            A_d = Nu & deg_masks[d]
            B_d = Nv & deg_masks[d]
            if A_d and B_d:
                Union = A_d | B_d
                if bin(Union).count('1') > 1:
                    return False
        return True

    def try_add(u, v, adj, deg, deg_masks):
        if adj[u] & (1 << v):
            return False
        if is_valid_add(u, v, adj, deg, deg_masks):
            # Add edge
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            # Update degrees
            old_deg_u = deg[u]
            old_deg_v = deg[v]
            deg[u] += 1
            deg[v] += 1
            
            # Update deg_masks
            # Remove u from old_deg_u, add to new
            deg_masks[old_deg_u] &= ~(1 << u)
            deg_masks[deg[u]] |= (1 << u)
            
            deg_masks[old_deg_v] &= ~(1 << v)
            deg_masks[deg[v]] |= (1 << v)
            
            return True
        return False

    def count_edges(adj):
        return sum(bin(adj[i]).count('1') for i in range(n)) // 2

    best_adj = adj[:]
    best_deg = deg[:]
    best_masks = deg_masks[:]
    best_count = count_edges(adj)
    
    # Greedy Phase
    # Try to add edges in random order
    random.seed(42)
    
    for trial in range(2000):
        if time.time() > deadline - 5:
            break
            
        cur_adj = best_adj[:]
        cur_deg = best_deg[:]
        cur_masks = best_masks[:]
        
        # Generate list of missing edges
        missing = []
        for i in range(n):
            for j in range(i + 1, n):
                if not (cur_adj[i] & (1 << j)):
                    missing.append((i, j))
        
        random.shuffle(missing)
        
        for u, v in missing:
            if time.time() > deadline - 5:
                break
            if try_add(u, v, cur_adj, cur_deg, cur_masks):
                pass # Added
        
        cur_count = count_edges(cur_adj)
        if cur_count > best_count:
            best_adj = cur_adj[:]
            best_deg = cur_deg[:]
            best_masks = cur_masks[:]
            best_count = cur_count
            
    # ILS Phase: Perturb and Recover
    for trial in range(5000):
        if time.time() > deadline - 5:
            break
            
        cur_adj = best_adj[:]
        cur_deg = best_deg[:]
        cur_masks = best_masks[:]
        
        # Perturb: remove k random edges
        k = random.randint(2, 5)
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if cur_adj[i] & (1 << j):
                    edges.append((i, j))
        
        if len(edges) < k:
            continue
            
        random.shuffle(edges)
        for u, v in edges[:k]:
            # Remove edge
            cur_adj[u] &= ~(1 << v)
            cur_adj[v] &= ~(1 << u)
            
            old_deg_u = cur_deg[u]
            old_deg_v = cur_deg[v]
            cur_deg[u] -= 1
            cur_deg[v] -= 1
            
            cur_masks[old_deg_u] &= ~(1 << u)
            cur_masks[cur_deg[u]] |= (1 << u)
            
            cur_masks[old_deg_v] &= ~(1 << v)
            cur_masks[cur_deg[v]] |= (1 << v)
            
        # Recover: Greedy add
        missing = []
        for i in range(n):
            for j in range(i + 1, n):
                if not (cur_adj[i] & (1 << j)):
                    missing.append((i, j))
        
        random.shuffle(missing)
        for u, v in missing:
            if time.time() > deadline - 5:
                break
            try_add(u, v, cur_adj, cur_deg, cur_masks)
            
        cur_count = count_edges(cur_adj)
        if cur_count > best_count:
            best_adj = cur_adj[:]
            best_deg = cur_deg[:]
            best_masks = cur_masks[:]
            best_count = cur_count

    # Convert to list of lists
    result = []
    for i in range(n):
        for j in range(i + 1, n):
            if best_adj[i] & (1 << j):
                result.append([int(i), int(j)])
                
    return result
