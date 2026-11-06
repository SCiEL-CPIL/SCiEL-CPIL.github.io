import random
import time

def construct():
    def find_k4(adj_masks, n):
        """
        Finds a K4 in the graph represented by adj_masks.
        Returns a tuple of 4 vertices (u, v, w, x) or None.
        """
        for u in range(n):
            for v in range(u + 1, n):
                if not (adj_masks[u] & (1 << v)):
                    continue
                common = adj_masks[u] & adj_masks[v]
                # Iterate through common neighbors w
                w_mask = common
                while w_mask:
                    w_bit = w_mask & -w_mask
                    w = w_bit.bit_length() - 1
                    # Check if w has a neighbor x in common, x != u, v
                    # x must be in common and in adj[w]
                    x_mask = common & adj_masks[w]
                    # Mask out u and v
                    x_mask &= ~(1 << u) & ~(1 << v)
                    if x_mask:
                        # Pick the lowest bit as x
                        x_bit = x_mask & -x_mask
                        x = x_bit.bit_length() - 1
                        return (u, v, w, x)
                    w_mask ^= w_bit
        return None

    def find_i6(adj_masks, n):
        """
        Finds an independent set of size 6 in G.
        Equivalent to finding a K6 in the complement graph.
        Returns a tuple of 6 vertices or None.
        """
        # Precompute complement masks for speed? 
        # comp_adj[i] = all_mask ^ adj_masks[i] ^ (1 << i)
        all_mask = (1 << n) - 1
        comp_adj = [0] * n
        for i in range(n):
            comp_adj[i] = all_mask ^ adj_masks[i] ^ (1 << i)
            
        # Search for K6 in comp_adj
        for u in range(n - 5):
            for v in range(u + 1, n - 4):
                if not (comp_adj[u] & (1 << v)):
                    continue
                for w in range(v + 1, n - 3):
                    if not (comp_adj[u] & (1 << w)) or not (comp_adj[v] & (1 << w)):
                        continue
                    for x in range(w + 1, n - 2):
                        if not (comp_adj[u] & (1 << x)) or not (comp_adj[v] & (1 << x)) or not (comp_adj[w] & (1 << x)):
                            continue
                        for y in range(x + 1, n - 1):
                            if not (comp_adj[u] & (1 << y)) or not (comp_adj[v] & (1 << y)) or not (comp_adj[w] & (1 << y)) or not (comp_adj[x] & (1 << y)):
                                continue
                            for z in range(y + 1, n):
                                if not (comp_adj[u] & (1 << z)) or not (comp_adj[v] & (1 << z)) or not (comp_adj[w] & (1 << z)) or not (comp_adj[x] & (1 << z)) or not (comp_adj[y] & (1 << z)):
                                    continue
                                return (u, v, w, x, y, z)
        return None

    def is_valid(adj_masks, n):
        return find_k4(adj_masks, n) is None and find_i6(adj_masks, n) is None

    def flip_edge(adj_masks, n, u, v):
        if u > v:
            u, v = v, u
        adj_masks[u] ^= (1 << v)
        adj_masks[v] ^= (1 << u)

    def edges_from_masks(adj_masks, n):
        edges = []
        for u in range(n):
            mask = adj_masks[u] & ((1 << (u + 1)) - 1) # Only bits < u? No, bits > u
            # Actually, just iterate all and add if u < v
            mask = adj_masks[u]
            while mask:
                v_bit = mask & -mask
                v = v_bit.bit_length() - 1
                if v > u:
                    edges.append([u, v])
                mask ^= v_bit
        return edges

    target_n = 35
    time_limit = 110 # Leave 10s for overhead/return
    
    start_time = time.time()
    
    # Try to find a valid graph for n=35
    # Strategy: Random restarts + Conflict-Directed Local Search
    
    for attempt in range(200): # Max attempts
        if time.time() - start_time > time_limit:
            break
            
        # Initialize random graph
        adj_masks = [0] * target_n
        for i in range(target_n):
            for j in range(i + 1, target_n):
                if random.random() < 0.5:
                    adj_masks[i] |= (1 << j)
                    adj_masks[j] |= (1 << i)
        
        # Local Search
        stuck_counter = 0
        while time.time() - start_time < time_limit:
            # Check for violations
            k4 = find_k4(adj_masks, target_n)
            if k4:
                # Break a K4
                u, v, w, x = k4
                # Pick a random edge from the K4 to flip
                # Edges: (u,v), (u,w), (u,x), (v,w), (v,x), (w,x)
                candidates = [(u,v), (u,w), (u,x), (v,w), (v,x), (w,x)]
                eu, ev = random.choice(candidates)
                flip_edge(adj_masks, target_n, eu, ev)
                stuck_counter = 0
                continue
            
            i6 = find_i6(adj_masks, target_n)
            if i6:
                # Break an I6 by adding an edge
                verts = list(i6)
                # Pick a random pair to connect
                # Generate all pairs
                pairs = []
                for a in range(6):
                    for b in range(a+1, 6):
                        pairs.append((verts[a], verts[b]))
                eu, ev = random.choice(pairs)
                flip_edge(adj_masks, target_n, eu, ev)
                stuck_counter = 0
                continue
            
            # No violations found!
            # Verify just in case
            if is_valid(adj_masks, target_n):
                edges = edges_from_masks(adj_masks, target_n)
                return {'n': target_n, 'edges': edges}
            
            # If we get here, something is wrong or we missed a violation
            # Force a random flip to escape
            u = random.randint(0, target_n - 1)
            v = random.randint(0, target_n - 1)
            if u != v:
                flip_edge(adj_masks, target_n, u, v)
            stuck_counter += 1
            if stuck_counter > 100:
                break # Restart

    # If 35 failed, try 34, 33... down to 17
    for n in range(34, 16, -1):
        if time.time() - start_time > time_limit:
            break
        # Quick check for smaller n
        for attempt in range(50):
            if time.time() - start_time > time_limit:
                break
            adj_masks = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < 0.5:
                        adj_masks[i] |= (1 << j)
                        adj_masks[j] |= (1 << i)
            
            # Simple local search
            while time.time() - start_time < time_limit:
                k4 = find_k4(adj_masks, n)
                if k4:
                    u, v, w, x = k4
                    candidates = [(u,v), (u,w), (u,x), (v,w), (v,x), (w,x)]
                    eu, ev = random.choice(candidates)
                    flip_edge(adj_masks, n, eu, ev)
                    continue
                i6 = find_i6(adj_masks, n)
                if i6:
                    verts = list(i6)
                    pairs = []
                    for a in range(6):
                        for b in range(a+1, 6):
                            pairs.append((verts[a], verts[b]))
                    eu, ev = random.choice(pairs)
                    flip_edge(adj_masks, n, eu, ev)
                    continue
                if is_valid(adj_masks, n):
                    edges = edges_from_masks(adj_masks, n)
                    return {'n': n, 'edges': edges}
                # Random flip
                u = random.randint(0, n - 1)
                v = random.randint(0, n - 1)
                if u != v:
                    flip_edge(adj_masks, n, u, v)
                break # Restart for this n

    # Fallback: Return a known valid small graph (e.g. Paley 17)
    # Paley graph of order 17
    q = 17
    squares = {x * x % q for x in range(1, q)}
    edges = [[i, j] for i in range(q) for j in range(i + 1, q) if (j - i) % q in squares]
    return {'n': q, 'edges': edges}
