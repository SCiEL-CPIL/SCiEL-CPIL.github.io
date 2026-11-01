import time
import random
import sys

def construct(k, g):
    start_time = time.time()
    deadline = 110  # 10s margin below the 150s hard kill
    
    # Fallback: Lazebnik-Ustimenko-Woldar graph
    def luw_construct(k, g):
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
        else:
            add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0] * (d - 1)
                for c in range(2, d + 1):
                    a, b = rule[c]
                    y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
                yield 1 - side, tuple(y)

        start = (0, (None,) + (0,) * d)
        index, order, edges = {start: 0}, [start], []
        for vertex in order:
            if time.time() > deadline:
                break
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    if k != 4 or g != 10:
        return luw_construct(k, g)

    # For k=4, g=10, the task is the (4,10)-cage.
    # Best known is 320 vertices.
    # We will try to find a graph with n < 320, or return a known 320-vertex graph if found.
    # If we can't find one quickly, we fall back to LUW (which is large, ~2048) or a random search result.
    
    # Fast girth check using BFS up to depth (g-1)//2 = 4
    # If a cycle of length < g exists, BFS from any vertex in the cycle will find a "conflict" 
    # where a node is reached at distance d1 and d2 such that d1+d2+1 < g? 
    # Actually, if we BFS from a vertex s, and we find an edge (u,v) where dist[u] + dist[v] + 1 < g, 
    # it implies a cycle. But the shortest cycle containing s might not be the global girth.
    # However, if the graph has girth >= g, then for every s, the BFS tree up to depth (g-1)//2 is a tree.
    # So if we find any edge (u,v) within depth (g-1)//2 from s such that u and v are not parent-child 
    # and dist[u] + dist[v] + 1 < g, we have a short cycle.
    # Wait, if dist[u] = 2 and dist[v] = 2, and they are connected, cycle is 2+2+1 = 5? No, 5 is odd.
    # In a bipartite graph, cycles are even.
    # The current search uses random bipartite graphs.
    # For a bipartite graph, girth is even.
    # If we check for cycles < 10, we check for 4, 6, 8.
    
    def has_girth_at_least(n, adj, g):
        # g is even for bipartite
        half_g = g // 2
        for s in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[s] = 0
            queue = [s]
            for u in queue:
                if dist[u] >= half_g:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    else:
                        if v != parent[u]:
                            # Found a cycle
                            # The length of the cycle is dist[u] + dist[v] + 1?
                            # This is only true if the paths are disjoint except at s?
                            # No, dist is from s.
                            # If u and v are connected, and both are reached from s,
                            # the cycle is u-v-path_to_s-path_from_s-u.
                            # Length is dist[u] + dist[v] + 1.
                            # This is a valid cycle length.
                            if dist[u] + dist[v] + 1 < g:
                                return False
        return True

    def generate_random_bipartite_4_regular(n):
        if n % 2 != 0:
            return None
        half = n // 2
        # A = 0..half-1, B = half..n-1
        # We want a 4-regular bipartite graph.
        # This is equivalent to 4 perfect matchings between A and B that are edge-disjoint.
        # Or just a random 4-regular bipartite graph.
        
        # Method: Random Configuration Model for bipartite
        # Stubs in A: 4 * half
        # Stubs in B: 4 * half
        # Randomly pair them.
        
        a_stubs = [i for i in range(half) for _ in range(4)]
        b_stubs = [i + half for i in range(half) for _ in range(4)]
        
        for _ in range(50):
            random.shuffle(b_stubs)
            edges = set()
            valid = True
            for a, b in zip(a_stubs, b_stubs):
                if a == b: # Should not happen in bipartite
                    valid = False
                    break
                if (a, b) in edges:
                    valid = False
                    break
                edges.add((a, b))
            if valid:
                return list(edges)
        return None

    def get_adj(edges, n):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        return adj

    # Search for n < 320
    # Start from 244 (lower bound is 243) up to 318.
    # Step size 2 (must be even).
    
    best_n = None
    best_edges = None
    
    # We know 320 is possible. If we find nothing < 320, we should return a 320 graph if we have one.
    # But we don't have a hardcoded 320 graph.
    # So we must either find one or fall back to LUW (which is valid but large).
    # To score better than LUW, we need n < 2048.
    # Random search for n=320 might succeed?
    
    # Let's try a few sizes.
    # The space of 4-regular bipartite graphs is large.
    # Girth 10 is a strong constraint.
    
    # Try n=320 first? Or smaller?
    # Smaller n is better score.
    
    # Let's iterate n from 244 to 318.
    # If time allows, try n=320.
    
    # Optimization: Use a set for edges for fast lookup in local search?
    # But for generation, we just check validity.
    
    # Let's try to improve the random generation or local search.
    # Local search: 2-switches.
    # Pick two edges (u,v) and (x,y) in the bipartite graph.
    # If u, x in A and v, y in B, swap to (u,y) and (x,v).
    # Must ensure (u,y) and (x,v) are not existing edges.
    
    def local_search(n, edges, max_steps=100):
        adj = get_adj(edges, n)
        # Convert to set for fast check?
        # But we need to update adj.
        # Let's use a set of edges for the graph state.
        edge_set = set(edges)
        
        # Precompute neighbors for A and B
        # A nodes: 0..n//2-1
        # B nodes: n//2..n-1
        
        # We need to find short cycles to break.
        # Instead of checking girth every step (expensive), 
        # we can just do random swaps and check girth periodically.
        
        # Actually, checking girth is O(n * E) roughly? 
        # n=300, E=600. BFS is fast.
        # 100 steps * BFS(300) is fast.
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
            
            # Pick random edges
            u, v = random.choice(edges)
            x, y = random.choice(edges)
            
            # Ensure they are distinct and form a valid switch
            # u, x must be in same partition (A), v, y in same (B)
            # Or u, y in A, v, x in B.
            # Since it's bipartite, u is in A iff v is in B.
            # So if u, x are both in A, then v, y are both in B.
            
            # Check partitions
            if (u < n//2) == (x < n//2):
                # u, x in A, v, y in B
                new_edges = [(u, y), (x, v)]
                # Check if new edges exist
                if (u, y) not in edge_set and (x, v) not in edge_set:
                    # Perform swap
                    edge_set.remove((u, v))
                    edge_set.remove((x, y))
                    edge_set.add((u, y))
                    edge_set.add((x, v))
                    
                    edges = list(edge_set)
                    adj = get_adj(edges, n)
                    
                    if has_girth_at_least(n, adj, 10):
                        return edges
            else:
                # u in A, x in B. Then v in B, y in A.
                # Swap (u,v) and (x,y) -> (u,y) and (x,v) is not a valid 2-switch for bipartite?
                # A 2-switch in bipartite graph: (a1, b1) and (a2, b2) -> (a1, b2) and (a2, b1).
                # So we need a1, a2 in A.
                # Here u is in A, x is in B.
                # So we should swap (u, v) and (y, x) where y is in A, x is in B.
                # i.e. edges (u,v) and (y,x).
                # New: (u,x) and (y,v).
                # u in A, x in B. Valid.
                # y in A, v in B. Valid.
                new_edges = [(u, x), (y, v)]
                if (u, x) not in edge_set and (y, v) not in edge_set:
                    edge_set.remove((u, v))
                    edge_set.remove((y, x))
                    edge_set.add((u, x))
                    edge_set.add((y, v))
                    
                    edges = list(edge_set)
                    adj = get_adj(edges, n)
                    
                    if has_girth_at_least(n, adj, 10):
                        return edges
        return None

    # Search loop
    # We prioritize smaller n.
    # If we find a valid graph, return it.
    
    # To maximize chance, we can try multiple restarts for each n.
    
    # Let's start from n=300 down to 244? Or up?
    # Smaller n is harder.
    # Let's try n=320 first to see if we can find one quickly?
    # If we find n=320, score is 320.
    # If we find n=300, score is 300.
    
    # Let's try a range.
    
    # Since we don't have a guaranteed 320 graph, and LUW is 2048,
    # finding ANY graph < 2048 is a huge win.
    # Finding one < 320 is the goal.
    
    # Let's try n from 244 to 318 in steps of 2.
    # And also n=320.
    
    # Time budget: 110s.
    # Let's allocate time per n.
    
    # If we fail to find any, return LUW.
    
    # Note: The LUW graph for k=4, g=10 has d = 5 (g-5).
    # Size is q * (q^d - 1)/(q-1) * 2?
    # For q=4, d=5.
    # Number of points = 4^5 = 1024?
    # No, the construction in the seed returns the connected component of the zero point.
    # For LUW, the graph is regular and connected?
    # If it's the full incidence graph, it's 2 * q^d?
    # Let's assume LUW returns a valid graph.
    
    # Strategy:
    # 1. Try to find a graph with n < 320.
    # 2. If not found, try to find a graph with n = 320.
    # 3. If not found, return LUW.
    
    # Let's iterate n.
    
    # To save time, we can check if n is feasible (Moore bound).
    # Moore bound for (4,10) is 1 + 4*(1 + 3 + 9 + 27) = 1 + 4*40 = 161.
    # So n >= 161.
    # But best known is 320.
    
    # Let's try n = 300, 302, ..., 318.
    # And n = 320.
    
    # We will use a time slice for each n.
    
    # Let's try 20 restarts for each n?
    # Or 1 restart with 500 steps?
    
    # Let's try 5 restarts for each n, with 100 steps each.
    
    # Total n values: (318 - 244)/2 + 1 = 38 values.
    # Plus 320.
    # 39 values * 5 restarts * 100 steps = 19500 steps.
    # Each step is O(1) swap + O(n) check?
    # Check is O(n * deg) = O(n).
    # n=300. 19500 * 300 = 6M ops. Should be fast in Python?
    # 110s is plenty.
    
    # But we should start with smaller n?
    # Or larger n is easier to find?
    # Yes, larger n is easier to satisfy girth constraint.
    # So try n=318 first?
    # If we find n=318, we can return it.
    # But we want the MINIMUM n.
    # So we should try smaller n first?
    # But if smaller n is impossible or very hard, we waste time.
    # Let's try n=300 first?
    
    # Actually, the task says "minimized".
    # So we should search for the smallest possible.
    # But if we don't know it, we search from low to high?
    # Or high to low to find A valid graph, then try to improve?
    
    # Let's try high to low to find a valid graph quickly, then refine?
    # No, just search low to high. If we find one, return it.
    # If we don't find one for n, try n+2.
    
    # But if n=244 is impossible, we waste time.
    # Let's start from n=300.
    # If we find one, return.
    # If not, try 302, etc.
    # If we reach 320 and find one, return 320.
    # If we don't find any, return LUW.
    
    # Let's do that.
    
    for n in range(300, 321, 2):
        if time.time() > deadline - 10:
            break
        
        # Try to find a graph of size n
        found = False
        for restart in range(10):
            if time.time() > deadline - 10:
                break
            
            edges = generate_random_bipartite_4_regular(n)
            if not edges:
                continue
            
            adj = get_adj(edges, n)
            if has_girth_at_least(n, adj, 10):
                return n, edges
            
            # Local search
            result = local_search(n, edges, max_steps=200)
            if result:
                return n, result
        
    # If we didn't find one in [300, 320], try a bit lower?
    # Or just return LUW.
    
    # Let's try a few lower ones if time permits.
    if time.time() < deadline - 20:
        for n in range(244, 300, 2):
            if time.time() > deadline - 10:
                break
            for restart in range(5):
                if time.time() > deadline - 10:
                    break
                edges = generate_random_bipartite_4_regular(n)
                if not edges:
                    continue
                adj = get_adj(edges, n)
                if has_girth_at_least(n, adj, 10):
                    return n, edges
                result = local_search(n, edges, max_steps=100)
                if result:
                    return n, result

    return luw_construct(k, g)
