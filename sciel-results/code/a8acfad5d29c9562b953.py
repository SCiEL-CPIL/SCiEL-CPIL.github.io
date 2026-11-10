import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    n = 20
    best_edges = 0
    best_graph = None
    
    # Check if graph is K3,3-free
    # For every pair (u, v), if |N(u) & N(v)| >= 3:
    #   Let C = N(u) & N(v)
    #   If there exists w not in {u, v} such that w is connected to all in C, then K3,3 exists.
    def is_k33_free(G):
        for u in range(n):
            for v in range(u + 1, n):
                common = G[u] & G[v]
                if common.bit_count() >= 3:
                    # Check if any other vertex is connected to all common neighbors
                    # We need to check intersection of N(c) for c in common
                    # If (intersection) contains any w != u, v, then violation
                    inter = (1 << n) - 1
                    mask = common
                    while mask:
                        lsb = mask & -mask
                        c = lsb.bit_length() - 1
                        inter &= G[c]
                        mask -= lsb
                    
                    # Remove u and v from inter
                    inter &= ~(1 << u)
                    inter &= ~(1 << v)
                    
                    if inter != 0:
                        return False
        return True

    # Faster check for adding edge (u, v)
    def can_add(G, u, v):
        # 1. Check if (u, v) itself creates a violation
        common_uv = G[u] & G[v]
        if common_uv.bit_count() >= 3:
            # Check if any w is connected to all in common_uv
            inter = (1 << n) - 1
            mask = common_uv
            while mask:
                lsb = mask & -mask
                c = lsb.bit_length() - 1
                inter &= G[c]
                mask -= lsb
            inter &= ~(1 << u)
            inter &= ~(1 << v)
            if inter != 0:
                return False
        
        # 2. Check impact on neighbors
        # Adding (u, v) adds v to N(u) and u to N(v)
        # This can increase codegree for (u, w) and (v, w)
        
        # For w in N(u):
        #   If v in N(w), then codegree(u, w) increases by 1
        #   If new codegree(u, w) >= 3, we must check K3,3 condition for (u, w)
        
        mask_u = G[u]
        while mask_u:
            lsb = mask_u & -mask_u
            w = lsb.bit_length() - 1
            mask_u -= lsb
            if w == v: continue
            if (G[w] >> v) & 1:
                # v is neighbor of w, so v is new common neighbor of (u, w)
                # New common set: (G[u] & G[w]) | (1 << v)
                # Actually G[u] doesn't have v yet.
                # Current common: G[u] & G[w]
                # New common: (G[u] & G[w]) | (1 << v)
                new_common = (G[u] & G[w]) | (1 << v)
                if new_common.bit_count() >= 3:
                    # Check K3,3 for (u, w)
                    inter = (1 << n) - 1
                    mask = new_common
                    while mask:
                        lsb = mask & -mask
                        c = lsb.bit_length() - 1
                        inter &= G[c]
                        mask -= lsb
                    inter &= ~(1 << u)
                    inter &= ~(1 << w)
                    if inter != 0:
                        return False
        
        # For w in N(v):
        mask_v = G[v]
        while mask_v:
            lsb = mask_v & -mask_v
            w = lsb.bit_length() - 1
            mask_v -= lsb
            if w == u: continue
            if (G[w] >> u) & 1:
                new_common = (G[v] & G[w]) | (1 << u)
                if new_common.bit_count() >= 3:
                    inter = (1 << n) - 1
                    mask = new_common
                    while mask:
                        lsb = mask & -mask
                        c = lsb.bit_length() - 1
                        inter &= G[c]
                        mask -= lsb
                    inter &= ~(1 << v)
                    inter &= ~(1 << w)
                    if inter != 0:
                        return False
        
        return True

    def find_violation(G):
        # Returns (u, v) if K3,3 exists, else None
        # We can just use is_k33_free, but it's slow.
        # Let's find a specific bad triple
        for u in range(n):
            for v in range(u + 1, n):
                common = G[u] & G[v]
                if common.bit_count() >= 3:
                    inter = (1 << n) - 1
                    mask = common
                    while mask:
                        lsb = mask & -mask
                        c = lsb.bit_length() - 1
                        inter &= G[c]
                        mask -= lsb
                    inter &= ~(1 << u)
                    inter &= ~(1 << v)
                    if inter != 0:
                        return (u, v)
        return None

    def repair(G):
        while True:
            pair = find_violation(G)
            if pair is None:
                break
            u, v = pair
            # Remove an edge to break the K3,3
            # Find the common neighbors
            common = G[u] & G[v]
            c_list = []
            mask = common
            while mask:
                lsb = mask & -mask
                c = lsb.bit_length() - 1
                c_list.append(c)
                mask -= lsb
            
            # Find the witness w
            inter = (1 << n) - 1
            mask = common
            while mask:
                lsb = mask & -mask
                c = lsb.bit_length() - 1
                inter &= G[c]
                mask -= lsb
            inter &= ~(1 << u)
            inter &= ~(1 << v)
            
            w_list = []
            mask = inter
            while mask:
                lsb = mask & -mask
                w = lsb.bit_length() - 1
                w_list.append(w)
                mask -= lsb
            
            if not w_list or not c_list:
                break
            
            w = random.choice(w_list)
            c = random.choice(c_list)
            
            # Remove edge (w, c)
            G[w] &= ~(1 << c)
            G[c] &= ~(1 << w)
            
        return G

    def polish(G, time_limit):
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            for u in range(n):
                for v in range(u + 1, n):
                    if not ((G[u] >> v) & 1):
                        if can_add(G, u, v):
                            G[u] |= (1 << v)
                            G[v] |= (1 << u)
                            improved = True
                            if time.time() > time_limit:
                                return G
        return G

    densities = [0.3, 0.4, 0.5, 0.6]
    
    while time.time() < deadline:
        p = random.choice(densities)
        G = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                if random.random() < p:
                    G[u] |= (1 << v)
                    G[v] |= (1 << u)
        
        G = repair(G)
        G = polish(G, deadline)
        
        edge_count = sum(G[u].bit_count() for u in range(n)) // 2
        
        if edge_count > best_edges:
            best_edges = edge_count
            best_graph = G[:]

    if best_graph is None:
        return [[a, b] for a in range(2) for b in range(2, 20)]
        
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if (best_graph[u] >> v) & 1:
                edges.append([u, v])
                
    return edges
