import random
import time
import itertools

def construct():
    start_time = time.time()
    timeout = 115  # Safety margin for 120s limit
    
    n = 20
    # Precompute all triples for full validation
    triples = [(i, j, k) for i in range(n) for j in range(i+1, n) for k in range(j+1, n)]
    
    def count_edges(adj):
        return sum(a.bit_count() for a in adj) // 2

    def is_valid_full(adj):
        for i, j, k in triples:
            if (adj[i] & adj[j] & adj[k]).bit_count() >= 3:
                return False
        return True

    def find_violation_full(adj):
        for i, j, k in triples:
            common = adj[i] & adj[j] & adj[k]
            if common.bit_count() >= 3:
                return (i, j, k), common
        return None, None

    def is_valid_after_add(adj, u, v):
        # Check if adding edge (u,v) creates a K3,3
        # Only triples containing u or v are affected
        # Triples with u: (u, x, y) where x,y != v
        # Common neighbors might gain v if v in N(x) & N(y)
        for x in range(n):
            if x == u or x == v: continue
            for y in range(x+1, n):
                if y == u or y == v: continue
                # Check triple (u, x, y)
                # Current common neighbors: C = N(u) & N(x) & N(y)
                # New common neighbors: C' = C | {v} if v in N(x) & N(y)
                # If v in N(x) & N(y), then count increases by 1.
                # If count was 2, it becomes 3 -> Violation.
                # If count was >= 3, it was already invalid, but we assume current state is valid.
                # So we only need to check if v is a common neighbor of x and y.
                if (adj[x] >> v) & 1 and (adj[y] >> v) & 1:
                    # v is a common neighbor of x and y.
                    # So v is added to the common neighbor set of (u, x, y).
                    # We need to check if the size of N(u) & N(x) & N(y) was 2.
                    # If it was 2, adding v makes it 3.
                    # If it was 0 or 1, adding v makes it 1 or 2.
                    # If it was >= 3, current graph is invalid (should not happen).
                    common_count = (adj[u] & adj[x] & adj[y]).bit_count()
                    if common_count == 2:
                        return False
                # Also, does u get added to common neighbors of (x, y, z)?
                # No, we are iterating over triples containing u.
                # What about triples containing v?
        
        for x in range(n):
            if x == u or x == v: continue
            for y in range(x+1, n):
                if y == u or y == v: continue
                # Check triple (v, x, y)
                # Common neighbors might gain u if u in N(x) & N(y)
                if (adj[x] >> u) & 1 and (adj[y] >> u) & 1:
                    common_count = (adj[v] & adj[x] & adj[y]).bit_count()
                    if common_count == 2:
                        return False
                        
        return True

    def is_valid_after_remove(adj, u, v):
        # Removing an edge cannot create a new K3,3 if the graph was valid.
        # It can only reduce common neighbor counts.
        # So it is always valid.
        return True

    def is_valid_after_swap(adj, u_rem, v_rem, u_add1, v_add1, u_add2, v_add2):
        # Remove (u_rem, v_rem), Add (u_add1, v_add1) and (u_add2, v_add2)
        # This is complex to check incrementally without applying.
        # Simpler to apply, check full, revert.
        # But full check is 1140 iterations.
        # Let's just apply and check full for swaps, as they are less frequent.
        adj[u_rem] &= ~(1 << v_rem)
        adj[v_rem] &= ~(1 << u_rem)
        
        adj[u_add1] |= (1 << v_add1)
        adj[v_add1] |= (1 << u_add1)
        
        adj[u_add2] |= (1 << v_add2)
        adj[v_add2] |= (1 << u_add2)
        
        valid = is_valid_full(adj)
        
        # Revert
        adj[u_rem] |= (1 << v_rem)
        adj[v_rem] |= (1 << u_rem)
        
        adj[u_add1] &= ~(1 << v_add1)
        adj[v_add1] &= ~(1 << u_add1)
        
        adj[u_add2] &= ~(1 << v_add2)
        adj[v_add2] &= ~(1 << u_add2)
        
        return valid

    best_edges = 0
    best_adj = [0] * n
    
    # Multiple restarts
    while time.time() - start_time < timeout:
        # 1. Initialize
        # Try different densities
        p = random.uniform(0.4, 0.7)
        adj = [0] * n
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < p:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        # 2. Cleanse
        # Remove edges until valid
        max_clean_steps = 1000
        steps = 0
        while steps < max_clean_steps:
            viol_triple, viol_common = find_violation_full(adj)
            if viol_triple is None:
                break
            
            u, v, w = viol_triple
            cn_list = [x for x in range(n) if (viol_common >> x) & 1]
            if not cn_list:
                break
                
            # Pick random common neighbor and random vertex from triple
            cn = random.choice(cn_list)
            tv = random.choice([u, v, w])
            
            adj[tv] &= ~(1 << cn)
            adj[cn] &= ~(1 << tv)
            steps += 1
        
        if find_violation_full(adj)[0] is not None:
            continue # Skip if not valid

        current_edges = count_edges(adj)
        
        # 3. Improve
        improved = True
        while improved and time.time() - start_time < timeout:
            improved = False
            
            # Greedy Add
            missing = [(i, j) for i in range(n) for j in range(i+1, n) if not (adj[i] >> j) & 1]
            random.shuffle(missing)
            
            for u, v in missing:
                if time.time() - start_time > timeout:
                    break
                
                if is_valid_after_add(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    current_edges += 1
                    improved = True
            
            # If no improvements in greedy add, try swaps
            if not improved:
                existing = [(i, j) for i in range(n) for j in range(i+1, n) if (adj[i] >> j) & 1]
                if not existing:
                    break
                
                # Try 1-2 swaps (remove 1, add 2)
                swap_improved = False
                # Limit attempts to save time
                swap_attempts = 0
                max_swap_attempts = 200
                
                while not swap_improved and swap_attempts < max_swap_attempts and time.time() - start_time < timeout:
                    swap_attempts += 1
                    # Pick edge to remove
                    u_rem, v_rem = random.choice(existing)
                    
                    # Pick 2 edges to add from missing
                    # We need to pick from the current missing list
                    # Rebuilding missing list is expensive, but n=20 is small
                    missing_now = [(i, j) for i in range(n) for j in range(i+1, n) if not (adj[i] >> j) & 1]
                    if len(missing_now) < 2:
                        break
                    
                    # Filter out edges that share vertices with removed edge? 
                    # Not strictly necessary, but might be better to pick distinct vertices?
                    # No, K3,3 is about structure.
                    
                    # Pick 2 random missing edges
                    e1, e2 = random.sample(missing_now, 2)
                    
                    if is_valid_after_swap(adj, u_rem, v_rem, e1[0], e1[1], e2[0], e2[1]):
                        # Apply
                        adj[u_rem] &= ~(1 << v_rem)
                        adj[v_rem] &= ~(1 << u_rem)
                        
                        adj[e1[0]] |= (1 << e1[1])
                        adj[e1[1]] |= (1 << e1[0])
                        
                        adj[e2[0]] |= (1 << e2[1])
                        adj[e2[1]] |= (1 << e2[0])
                        
                        current_edges += 1
                        swap_improved = True
                        improved = True
                        # Update existing list? 
                        # We break out of swap loop and restart greedy add loop which rebuilds lists.
                        break
                        
                if not swap_improved:
                    # Try 2-3 swaps? Or just break and restart
                    # Let's try a few 2-2 swaps (remove 2, add 2) - no gain in edges, but might escape local optimum?
                    # No, we want to maximize edges.
                    # If stuck, break and restart.
                    break

        if current_edges > best_edges:
            best_edges = current_edges
            best_adj = [a for a in adj]
            
        if time.time() - start_time > timeout - 5:
            break

    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if (best_adj[i] >> j) & 1:
                edges.append([i, j])
                
    return edges
