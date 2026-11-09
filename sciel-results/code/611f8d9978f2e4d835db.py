import time
import random
import sys

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s to be safe
    
    # Fallback: Seed solution (10 disjoint triangles)
    def get_seed():
        edges = []
        for t in range(10):
            a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
            edges += [[a, b], [a, c], [b, c]]
        return {'n': 30, 'edges': edges}
    
    best_result = get_seed()
    best_n = 30
    
    # Helper: Check if adding edge (u, v) creates a C4
    # C4 is u-w-x-v-u. This exists if N(u) and N(v) have a common neighbor w, 
    # and w is connected to x which is connected to v? 
    # Actually, C4 is u-v-x-y-u. 
    # If we add edge (u, v), a C4 is formed if there is a path of length 2 between u and v 
    # that does not use the edge (u, v). 
    # i.e., if |N(u) \cap N(v)| > 0. 
    # Wait, if u and v have a common neighbor w, then u-w-v is a path of length 2. 
    # Adding u-v creates a triangle u-w-v-u. That's a C3, not C4.
    # A C4 involves 4 distinct vertices.
    # If we add edge (u, v), a C4 is formed if there exists a vertex x such that 
    # x is adjacent to some y in N(u) and some z in N(v)? 
    # No. 
    # A C4 containing the new edge (u, v) must be u-v-x-y-u.
    # This requires v-x, x-y, y-u.
    # So x must be in N(v) and y must be in N(u).
    # And x-y must be an edge.
    # So we need to check if there is an edge between N(v) and N(u).
    # Specifically, if there is an edge (x, y) with x in N(v) and y in N(u), 
    # then u-v-x-y-u is a C4 (provided u, v, x, y are distinct).
    # Note: x cannot be u (since u not in N(v) initially, but if u in N(v) we wouldn't add edge).
    # y cannot be v.
    # x cannot be y.
    
    def is_c4_free(adj, u, v):
        # adj is a list of sets
        if v in adj[u]:
            return False # already edge
        # Check for C4: u-v-x-y-u
        # x in N(v), y in N(u), edge (x, y)
        for x in adj[v]:
            for y in adj[u]:
                if x == y:
                    continue
                if y in adj[x]: # edge x-y exists
                    # Ensure distinct vertices: u, v, x, y
                    # u != v (checked)
                    # x != u (x in N(v), u not in N(v) if no edge? No, u might be in N(v) if edge exists, but we check v in adj[u] first)
                    # Actually, if u in N(v), we return False above.
                    # So u not in N(v). Thus x != u.
                    # v not in N(u). Thus y != v.
                    # x != y (checked)
                    # So u, v, x, y are distinct.
                    return False
        return True

    def check_alpha_less_than_11(adj, n):
        # Heuristic: Try to find independent set of size 11
        # If found, return False. If not, return True (optimistic)
        # Simple greedy: pick vertex with min degree, remove it and neighbors.
        # Repeat 11 times. If we can pick 11, alpha >= 11.
        # This is not exact, but fast.
        deg = [len(adj[i]) for i in range(n)]
        vertices = list(range(n))
        removed = [False] * n
        count = 0
        # Greedy independent set
        for _ in range(11):
            # Find vertex with min degree among remaining
            min_deg = float('inf')
            min_v = -1
            for v in vertices:
                if not removed[v]:
                    # Effective degree
                    eff_deg = sum(1 for u in adj[v] if not removed[u])
                    if eff_deg < min_deg:
                        min_deg = eff_deg
                        min_v = v
            if min_v == -1:
                break
            count += 1
            removed[min_v] = True
            for u in adj[min_v]:
                removed[u] = True
            if count >= 11:
                return False # Found independent set of size 11
        return True # Could not find size 11

    # Randomized Search
    # Start with seed
    n = 30
    adj = [set() for _ in range(n)]
    for t in range(10):
        a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
        adj[a].add(b); adj[b].add(a)
        adj[a].add(c); adj[c].add(a)
        adj[b].add(c); adj[c].add(b)
    
    # Try to add vertices
    for attempt in range(500):
        if time.time() > deadline:
            break
        
        # Reset to seed for each attempt? Or continue from best?
        # Let's try to extend the current best.
        # Actually, adding vertices is hard. Let's try to add one vertex at a time.
        
        new_n = n + 1
        new_adj = [set(s) for s in adj]
        new_adj.append(set())
        
        # Try to connect new vertex (new_n - 1) to existing vertices
        # We want to maximize edges to reduce alpha, but keep C4-free.
        # Randomly shuffle existing vertices
        candidates = list(range(n))
        random.shuffle(candidates)
        
        added_edges = 0
        for v in candidates:
            if is_c4_free(new_adj, new_n - 1, v):
                new_adj[new_n - 1].add(v)
                new_adj[v].add(new_n - 1)
                added_edges += 1
        
        # Check alpha
        if check_alpha_less_than_11(new_adj, new_n):
            n = new_n
            adj = new_adj
            if n > best_n:
                best_n = n
                edges = []
                for i in range(n):
                    for j in adj[i]:
                        if i < j:
                            edges.append([i, j])
                best_result = {'n': n, 'edges': edges}
        else:
            # If alpha >= 11, try to add more edges to reduce alpha?
            # Or just discard this vertex.
            pass
            
    return best_result
# EVOLVE-BLOCK-END
