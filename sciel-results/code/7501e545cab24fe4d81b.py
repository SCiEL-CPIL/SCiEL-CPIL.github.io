import random
import time

def construct():
    n = 20
    # 1. Find best circulant graph
    best_circ_edges = []
    best_circ_count = 0
    
    # Differences 1 to 10
    # Bit i corresponds to difference i+1
    # If bit is set, diff is in S.
    # For diff 10, it's symmetric to itself.
    for mask in range(1 << 10):
        S = set()
        for i in range(10):
            if mask & (1 << i):
                S.add(i + 1)
        
        # Build adjacency matrix
        adj = [[0] * n for _ in range(n)]
        for u in range(n):
            for d in S:
                v = (u + d) % n
                if u < v:
                    adj[u][v] = 1
                    adj[v][u] = 1
                elif u > v:
                    # This case is handled by u+(-d) which is u+(n-d)
                    # But since we iterate all u, we just need to ensure we don't double count or miss.
                    # Actually, if we add (u, u+d), we cover all edges.
                    # We just need to build the set of edges.
                    pass
        
        # Re-build edges list properly
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                diff = (v - u) % n
                if diff in S or (n - diff) in S:
                    edges.append([u, v])
                    adj[u][v] = 1
                    adj[v][u] = 1
        
        # Check validity
        if is_valid(adj):
            if len(edges) > best_circ_count:
                best_circ_count = len(edges)
                best_circ_edges = edges.copy()
                best_adj = [row[:] for row in adj]

    # If no circulant graph found (unlikely), start empty
    if best_circ_count == 0:
        best_adj = [[0] * n for _ in range(n)]
        best_circ_edges = []

    # 2. Local Search to improve
    current_adj = [row[:] for row in best_adj]
    current_edges = best_circ_edges.copy()
    
    # We need a fast way to check if adding an edge is valid.
    # We can precompute codeg matrix.
    # codeg[u][v] = number of common neighbors.
    
    def get_codeg(adj):
        codeg = [[0] * n for _ in range(n)]
        for u in range(n):
            for v in range(n):
                if u == v: continue
                cnt = 0
                for w in range(n):
                    if adj[u][w] and adj[v][w]:
                        cnt += 1
                codeg[u][v] = cnt
        return codeg

    codeg = get_codeg(current_adj)
    
    # Helper to check if adding (u,v) is valid
    def can_add(u, v):
        if current_adj[u][v]:
            return False
        # 1. codeg(u, v) must be <= 2
        if codeg[u][v] > 2:
            return False
        # 2. For all x in N(u), codeg(v, x) must be <= 1 (because adding u-v makes u a common neighbor of v and x? No.)
        # Adding (u,v) adds u to N(v) and v to N(u).
        # New common neighbors of (v, x) for x in N(v):
        #   Old common neighbors + u (if u in N(x)).
        #   So if x in N(v) and u in N(x), codeg(v, x) increases by 1.
        #   We need new codeg(v, x) <= 2 => old codeg(v, x) <= 1.
        #   Condition: For all x such that adj[v][x]==1 and adj[u][x]==1, codeg[v][x] <= 1.
        #   Wait, if adj[u][x]==0, u is not a common neighbor.
        #   So we check: for all x, if adj[v][x] and adj[u][x], then codeg[v][x] <= 1.
        #   Similarly for y in N(u): for all y, if adj[u][y] and adj[v][y], then codeg[u][y] <= 1.
        #   This is the same condition.
        
        for x in range(n):
            if x == u or x == v:
                continue
            if current_adj[v][x] and current_adj[u][x]:
                if codeg[v][x] > 1:
                    return False
        return True

    def add_edge(u, v):
        current_adj[u][v] = 1
        current_adj[v][u] = 1
        current_edges.append([u, v])
        # Update codeg
        # Adding (u,v) increases codeg(u, x) for all x in N(v)
        # and codeg(v, x) for all x in N(u)
        # And codeg(u, v) itself? 
        # codeg(u, v) is count of common neighbors. It doesn't change when we add the edge (u,v) itself?
        # Common neighbors of u and v are z such that adj[u][z] and adj[v][z].
        # Adding adj[u][v]=1 does not change adj[u][z] or adj[v][z] for z != u,v.
        # So codeg[u][v] stays the same.
        
        # For any x:
        # If x is in N(v), then v is a common neighbor of u and x?
        #   New common neighbors of (u, x) include v if adj[u][v] (now 1) and adj[x][v] (was 1).
        #   So codeg[u][x] += 1 for all x in N(v).
        # If x is in N(u), then u is a common neighbor of v and x?
        #   So codeg[v][x] += 1 for all x in N(u).
        
        for x in range(n):
            if x == u or x == v:
                continue
            if current_adj[v][x]:
                codeg[u][x] += 1
                codeg[x][u] += 1
            if current_adj[u][x]:
                codeg[v][x] += 1
                codeg[x][v] += 1
        # Note: if x is in both N(u) and N(v), both increments happen.
        # codeg[u][x] and codeg[x][u] are the same.
        # The loop adds 1 if x in N(v) and 1 if x in N(u).
        # So if x in N(u) AND N(v), codeg[u][x] increases by 2?
        # Let's re-verify.
        # codeg[u][x] = |N(u) \cap N(x)|.
        # New N(u) = Old N(u) \cup {v}.
        # New N(x) = Old N(x).
        # New Intersection = (Old N(u) \cup {v}) \cap Old N(x) = (Old N(u) \cap Old N(x)) \cup ({v} \cap Old N(x)).
        # So size increases by 1 if v is in Old N(x), i.e., adj[v][x]==1.
        # So codeg[u][x] increases by 1 if adj[v][x]==1.
        # Similarly codeg[v][x] increases by 1 if adj[u][x]==1.
        # My loop does exactly this.
        
    def remove_edge(u, v):
        current_adj[u][v] = 0
        current_adj[v][u] = 0
        current_edges.remove([u, v])
        # Update codeg
        for x in range(n):
            if x == u or x == v:
                continue
            if current_adj[v][x]:
                codeg[u][x] -= 1
                codeg[x][u] -= 1
            if current_adj[u][x]:
                codeg[v][x] -= 1
                codeg[x][v] -= 1

    # Local Search Loop
    # Try to add edges. If stuck, try to remove one and add another?
    # Simple: Try all missing edges. Add if valid.
    # If none can be added, try a random swap: remove a random edge, try to add 2 edges?
    # Or just restart local search with a different random seed/order?
    
    # To ensure we finish in 120s, we limit the number of iterations.
    start_time = time.time()
    while time.time() - start_time < 110:
        added_any = False
        # Try to add any valid edge
        # Randomize order to avoid bias
        missing = [(u, v) for u in range(n) for v in range(u + 1, n) if not current_adj[u][v]]
        random.shuffle(missing)
        
        for u, v in missing:
            if can_add(u, v):
                add_edge(u, v)
                added_any = True
        
        if not added_any:
            # Stuck. Try to break the deadlock.
            # Remove a random edge that is "critical" or just random.
            if current_edges:
                # Pick a random edge to remove
                idx = random.randint(0, len(current_edges) - 1)
                e = current_edges[idx]
                remove_edge(e[0], e[1])
            else:
                break

    return current_edges

def is_valid(adj):
    n = len(adj)
    for u in range(n):
        for v in range(u + 1, n):
            cnt = 0
            for w in range(n):
                if adj[u][w] and adj[v][w]:
                    cnt += 1
            if cnt > 2:
                return False
    return True
