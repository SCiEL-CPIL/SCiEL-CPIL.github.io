import time
import random

def construct():
    start_time = time.time()
    deadline = 145

    def elapsed():
        return time.time() - start_time

    def find_c4(adj, n):
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u] & (1 << v):
                    common = adj[u] & adj[v]
                    if common.bit_count() >= 2:
                        tmp = common
                        w = -1
                        x = -1
                        while tmp:
                            lsb = tmp & -tmp
                            idx = lsb.bit_length() - 1
                            if w == -1:
                                w = idx
                            else:
                                x = idx
                                break
                            tmp ^= lsb
                        if w != -1 and x != -1:
                            return (u, v, w, x)
        return None

    def is_c4_free(adj, n):
        return find_c4(adj, n) is None

    def min_deg(adj, n):
        m = n
        for i in range(n):
            d = adj[i].bit_count()
            if d < m:
                m = d
        return m

    def build_edges(adj, n):
        edges = []
        for i in range(n):
            ai = adj[i]
            for j in range(i + 1, n):
                if ai & (1 << j):
                    edges.append([i, j])
        return edges

    def try_n(n):
        target = n - 39
        if target < 1:
            target = 1

        # Start with K_n
        adj = [0] * n
        for i in range(n):
            adj[i] = ((1 << n) - 1) ^ (1 << i)

        # Remove edges to break C4s
        # Use a counter to avoid infinite loops if stuck (shouldn't happen)
        max_iter = 5000
        for _ in range(max_iter):
            if elapsed() > deadline - 1:
                break
            c4 = find_c4(adj, n)
            if c4 is None:
                break
            u, v, w, x = c4
            # Cycle edges: (u,w), (w,v), (v,x), (x,u)
            # Randomly pick one to remove
            # Note: In K_n, (u,v) and (w,x) are also edges, but they are diagonals.
            # Removing a cycle edge is sufficient.
            cycle_edges = [(u, w), (w, v), (v, x), (x, u)]
            a, b = random.choice(cycle_edges)
            adj[a] &= ~(1 << b)
            adj[b] &= ~(1 << a)

        if not is_c4_free(adj, n):
            return None

        # Check min degree
        md = min_deg(adj, n)
        
        # If min degree is too low, try to add edges
        if md < target:
            degs = [adj[i].bit_count() for i in range(n)]
            # Try to add edges to low degree vertices
            for _ in range(200):
                if elapsed() > deadline - 1:
                    break
                min_d = min(degs)
                if min_d >= target:
                    break
                
                # Find a vertex with min degree
                low_verts = [i for i in range(n) if degs[i] == min_d]
                random.shuffle(low_verts)
                
                added = False
                for u in low_verts:
                    if degs[u] >= target:
                        continue
                    
                    # Find non-neighbors
                    non_neighbors = []
                    for v in range(n):
                        if v != u and not (adj[u] & (1 << v)):
                            non_neighbors.append(v)
                            
                    random.shuffle(non_neighbors)
                    
                    for v in non_neighbors:
                        # Check if adding (u, v) creates a C4
                        # This requires |N(u) cap N(v)| <= 1
                        common = adj[u] & adj[v]
                        if common.bit_count() < 2:
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                            degs[u] += 1
                            degs[v] += 1
                            added = True
                            if degs[u] >= target:
                                break
                    if degs[u] >= target:
                        continue
                    if added:
                        break
                if not added:
                    break
            
            # Final validation
            if not is_c4_free(adj, n):
                return None
            md = min_deg(adj, n)
            if md < target:
                return None

        return n, build_edges(adj, n)

    # Try n from 47 down to 40
    for n in range(47, 39, -1):
        if elapsed() > deadline - 5:
            break
        res = try_n(n)
        if res is not None:
            return {'n': res[0], 'edges': res[1]}

    # Fallback
    return {'n': 39, 'edges': []}
