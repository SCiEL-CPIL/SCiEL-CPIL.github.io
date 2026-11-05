import random
import time

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_edges = []
    best_count = 0
    
    # Helper to check if adding edge (u, v) to adj is valid
    # adj is a list of n integers (bitmasks)
    def is_valid_addition(adj, u, v):
        # 1. Same Part Check: 
        # Does there exist a vertex a such that N(u) & N(v) & N(a) has size >= 3?
        # If so, {u,v,a} and those 3 neighbors form K3,3.
        common_uv = adj[u] & adj[v]
        # We need to check if any a has at least 3 common neighbors with u and v.
        # Actually, the 3 neighbors must be distinct from u,v,a.
        # The set of neighbors is S = N(u) & N(v) & N(a).
        # If |S| >= 3, then we have a K3,3 with parts {u,v,a} and S (subset of 3).
        # Note: S does not contain u or v because adj[u] does not have bit u.
        # S might contain a? No, simple graph, no self loops.
        
        # Iterate all a != u, v
        for a in range(n):
            if a == u or a == v:
                continue
            # S = N(u) & N(v) & N(a)
            s = common_uv & adj[a]
            if s & (s - 1):  # popcount >= 2? No, we need >= 3.
                # Check popcount
                if bin(s).count('1') >= 3:
                    return False
        
        # 2. Different Part Check:
        # u in A, v in B.
        # A = {u, a1, a2}, B = {v, b1, b2}.
        # a1, a2 in N(v) \ {u}
        # b1, b2 in N(u) \ {v}
        # a1, a2 must be connected to b1, b2.
        # So N(b1) & N(b2) & N(v) \ {u} must have size >= 2.
        
        ca = adj[v] & ~(1 << u)  # Candidates for A \ {u}
        cb = adj[u] & ~(1 << v)  # Candidates for B \ {v}
        
        # Iterate pairs in cb
        # To optimize, if popcount(cb) < 2, return True (no pair)
        if bin(cb).count('1') < 2:
            return True
            
        # Extract bits for iteration
        # Simple loop over all pairs in cb
        # Get list of bits
        # Or just nested loop
        # Since n=20, we can just loop b1, b2
        for b1 in range(n):
            if not (cb & (1 << b1)):
                continue
            for b2 in range(b1 + 1, n):
                if not (cb & (1 << b2)):
                    continue
                # Check if N(b1) & N(b2) & ca has size >= 2
                s = adj[b1] & adj[b2] & ca
                if bin(s).count('1') >= 2:
                    return False
        
        return True

    def count_edges(adj):
        count = 0
        for i in range(n):
            count += bin(adj[i]).count('1')
        return count // 2

    def run_greedy(seed):
        random.seed(seed)
        adj = [0] * n
        edges = list(range(n))
        # Generate all possible edges
        all_edges = []
        for u in range(n):
            for v in range(u + 1, n):
                all_edges.append((u, v))
        
        random.shuffle(all_edges)
        
        for u, v in all_edges:
            if is_valid_addition(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        
        # Convert to list of lists
        res = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u] & (1 << v):
                    res.append([u, v])
        return res

    # Run multiple restarts
    seed = 42
    while time.time() < deadline:
        edges = run_greedy(seed)
        cnt = len(edges)
        if cnt > best_count:
            best_count = cnt
            best_edges = edges
        seed += 1
        # Optional: local search here?
        # If time allows, try to improve best_edges
        
    return best_edges
