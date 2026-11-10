import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n = 20
    # adj[i] is a bitmask of neighbors of i
    adj = [0] * n
    
    best_edges = 0
    best_adj = adj.copy()
    
    def popcount(x):
        return bin(x).count('1')
        
    def is_valid_add(u, v):
        if (adj[u] >> v) & 1:
            return False
            
        # Check 1: u and v cannot have >= 3 common neighbors
        common_uv = adj[u] & adj[v]
        if popcount(common_uv) >= 3:
            return False
            
        # Check 2: For any w in N(u), v and w cannot have >= 2 common neighbors
        # Because adding u-v makes u a common neighbor of v and w.
        # So existing common(v, w) must be <= 1.
        # N(u) includes all neighbors. We iterate through bits.
        mask_u = adj[u]
        while mask_u:
            lsb = mask_u & -mask_u
            w = lsb.bit_length() - 1
            mask_u ^= lsb
            if w == v: continue
            common_wv = adj[w] & adj[v]
            if popcount(common_wv) >= 2:
                return False
                
        # Check 3: For any x in N(v), u and x cannot have >= 2 common neighbors
        mask_v = adj[v]
        while mask_v:
            lsb = mask_v & -mask_v
            x = lsb.bit_length() - 1
            mask_v ^= lsb
            if x == u: continue
            common_ux = adj[u] & adj[x]
            if popcount(common_ux) >= 2:
                return False
                
        return True

    def add_edge(u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    def remove_edge(u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    def count_edges():
        return sum(popcount(a) for a in adj) // 2

    # Main search loop
    # We will perform multiple restarts.
    # Each restart:
    # 1. Randomize order of all possible edges
    # 2. Try to add them greedily
    # 3. Try local search: try adding edges that are currently not addable? No, greedy does that.
    #    But greedy might get stuck.
    #    Let's add a "kick" phase: remove a few random edges, then re-greedy.
    
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    for restart in range(10000):
        if time.time() > deadline:
            break
            
        # Reset graph
        adj = [0] * n
        
        # Randomize edge order
        random.shuffle(all_edges)
        
        # Greedy add
        for u, v in all_edges:
            if is_valid_add(u, v):
                add_edge(u, v)
                
        current_edges = count_edges()
        
        if current_edges > best_edges:
            best_edges = current_edges
            best_adj = adj.copy()
            
        # If we have a good graph, try to improve it with local search
        # Local search: try to add any edge. If none, remove random edge and retry.
        # This is essentially a random restart with memory.
        # Let's do a few steps of "kick and fill"
        for _ in range(20):
            if time.time() > deadline:
                break
            
            # Try to add any missing edge
            added_any = False
            # Shuffle edges to break symmetry
            random.shuffle(all_edges)
            for u, v in all_edges:
                if not ((adj[u] >> v) & 1):
                    if is_valid_add(u, v):
                        add_edge(u, v)
                        added_any = True
                        break # Restart loop to re-evaluate all? 
                        # Or just continue?
                        # If we add one, the validity of others changes.
                        # It's safer to restart the scan.
            
            if not added_any:
                # Stuck. Remove a random edge to create space
                if current_edges > 0:
                    # Pick a random existing edge
                    # We can pick random u, then random v in N(u)
                    u = random.randint(0, n-1)
                    if adj[u]:
                        # Get a random bit from adj[u]
                        mask = adj[u]
                        # Convert to list of neighbors
                        neighbors = [i for i in range(n) if (mask >> i) & 1]
                        if neighbors:
                            v = random.choice(neighbors)
                            remove_edge(u, v)
                            current_edges -= 1
                # After removing, try to add again in next iteration of inner loop
            else:
                current_edges += 1
                if current_edges > best_edges:
                    best_edges = current_edges
                    best_adj = adj.copy()

    # Return the best graph found
    edges = []
    for i in range(n):
        mask = best_adj[i]
        while mask:
            lsb = mask & -mask
            j = lsb.bit_length() - 1
            mask ^= lsb
            if i < j:
                edges.append([i, j])
                
    return edges
