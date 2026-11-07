import time

def construct():
    n = 15
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin before 90s kill
    
    # Strategy: Start with K_7,8 (56 edges, valid) and use local search to add edges.
    # Vertices 0..6 form set A (size 7)
    # Vertices 7..14 form set B (size 8)
    
    # Initialize adjacency matrix as bitmasks
    adj = [0] * n
    for u in range(7):
        for v in range(7, 15):
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
    
    def get_degrees(adj):
        return [bin(adj[i]).count('1') for i in range(n)]
    
    def is_valid(adj):
        """
        Check if no simple path s-u-v-t exists where deg(s) == deg(t).
        Equivalent: For every edge (u,v), let N'(u) = N(u) \ {v} and N'(v) = N(v) \ {u}.
        If there exists s in N'(u) and t in N'(v) such that s != t and deg(s) == deg(t),
        then the path s-u-v-t is invalid.
        """
        degrees = get_degrees(adj)
        
        for u in range(n):
            au = adj[u]
            for v in range(u + 1, n):
                if not (au & (1 << v)):
                    continue
                
                # Neighbors of u excluding v
                s_mask = au & ~(1 << v)
                # Neighbors of v excluding u
                t_mask = adj[v] & ~(1 << u)
                
                # We need to check if any s in s_mask and t in t_mask have equal degrees.
                # Optimization: Precompute sets of vertices by degree?
                # Or just iterate. n=15 is small.
                
                s_list = []
                tmp = s_mask
                while tmp:
                    bit = tmp & -tmp
                    s = bit.bit_length() - 1
                    s_list.append(s)
                    tmp ^= bit
                
                t_list = []
                tmp = t_mask
                while tmp:
                    bit = tmp & -tmp
                    t = bit.bit_length() - 1
                    t_list.append(t)
                    tmp ^= bit
                
                # Check pairs
                # To speed up, group t_list by degree
                t_by_deg = {}
                for t in t_list:
                    d = degrees[t]
                    if d not in t_by_deg:
                        t_by_deg[d] = []
                    t_by_deg[d].append(t)
                
                for s in s_list:
                    ds = degrees[s]
                    if ds in t_by_deg:
                        # Check if there is a t != s in t_by_deg[ds]
                        # Since s is in N(u) and t is in N(v), and u-v is an edge.
                        # In a simple path s-u-v-t, s and t must be distinct.
                        # Also s != v and t != u is guaranteed by construction of masks.
                        # But s could equal t? 
                        # If s == t, then s is in N(u) \ {v} AND t is in N(v) \ {u}.
                        # This means s is adjacent to both u and v.
                        # The path would be s-u-v-s. But this is not a simple path 
                        # because s and t are the same vertex. The problem says "simple three-edge path".
                        # A simple path has distinct vertices. So s != t.
                        for t in t_by_deg[ds]:
                            if s != t:
                                return False
        return True

    def count_edges(adj):
        total = 0
        for i in range(n):
            total += bin(adj[i]).count('1')
        return total // 2

    # Verify initial state
    if not is_valid(adj):
        # Fallback to disjoint triangles if something went wrong (should not happen)
        adj = [0] * n
        for i in range(0, n, 3):
            adj[i] |= (1 << (i+1)) | (1 << (i+2))
            adj[i+1] |= (1 << i) | (1 << (i+2))
            adj[i+2] |= (1 << i) | (1 << (i+1))
    
    best_adj = adj[:]
    best_count = count_edges(best_adj)
    
    # Local Search
    # We try to add edges one by one.
    # To be efficient, we can maintain a list of candidate edges.
    # Or just iterate through all possible non-edges and add them if valid.
    # Since we want to maximize, a greedy approach of adding any valid edge is good.
    # We can repeat passes until no more edges can be added.
    
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for u in range(n):
            if time.time() > deadline:
                break
            for v in range(u + 1, n):
                if time.time() > deadline:
                    break
                if not (adj[u] & (1 << v)):
                    # Try adding edge (u, v)
                    new_adj = adj[:]
                    new_adj[u] |= (1 << v)
                    new_adj[v] |= (1 << u)
                    if is_valid(new_adj):
                        adj = new_adj
                        best_adj = adj[:]
                        best_count += 1
                        changed = True
    
    # If we have time, try random swaps to escape local optima?
    # The greedy addition might get stuck in a local optimum where no single edge can be added,
    # but a swap (remove one, add two) might work.
    # However, for n=15, the search space is small. Let's stick to greedy first.
    # If we have significant time left, we can try random restarts or swaps.
    
    if time.time() < deadline - 10:
        import random
        random.seed(42)
        current_adj = best_adj[:]
        current_count = best_count
        
        # Simple random restart / perturbation
        # Try to find a better solution by randomly removing an edge and then greedily adding
        for _ in range(100): # Limit iterations
            if time.time() > deadline:
                break
            
            # Pick a random edge to remove
            edges_to_remove = []
            for i in range(n):
                ai = current_adj[i]
                for j in range(i+1, n):
                    if ai & (1 << j):
                        edges_to_remove.append((i, j))
            
            if not edges_to_remove:
                break
                
            u, v = random.choice(edges_to_remove)
            perturbed_adj = current_adj[:]
            perturbed_adj[u] &= ~(1 << v)
            perturbed_adj[v] &= ~(1 << u)
            
            # Greedily add edges
            changed = True
            while changed and time.time() < deadline:
                changed = False
                for i in range(n):
                    if time.time() > deadline:
                        break
                    for j in range(i + 1, n):
                        if time.time() > deadline:
                            break
                        if not (perturbed_adj[i] & (1 << j)):
                            new_adj = perturbed_adj[:]
                            new_adj[i] |= (1 << j)
                            new_adj[j] |= (1 << i)
                            if is_valid(new_adj):
                                perturbed_adj = new_adj
                                changed = True
            
            if count_edges(perturbed_adj) > best_count:
                best_adj = perturbed_adj[:]
                best_count = count_edges(best_adj)
                current_adj = best_adj[:]

    # Extract edges
    final_edges = []
    for i in range(n):
        ai = best_adj[i]
        for j in range(i + 1, n):
            if ai & (1 << j):
                final_edges.append([i, j])
                
    # Ensure types are Python int
    final_edges = [[int(e[0]), int(e[1])] for e in final_edges]
    
    return final_edges
