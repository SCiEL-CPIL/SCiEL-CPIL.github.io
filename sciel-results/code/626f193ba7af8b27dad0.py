# EVOLVE-BLOCK-START
import random
import itertools
from collections import deque
import time

def construct(k, g):
    """
    Construct a k-regular graph with no cycle shorter than g.
    Strategy: 
    1. Try to find a random Cayley graph on S6 or S7 with the required girth.
    2. Fall back to the Lazebnik-Ustimenko-Woldar (L-U-W) construction with minimal parameters.
    """
    
    def moore_bound(k, g):
        if g % 2 == 0:
            return 2 * sum((k-1)**i for i in range(g//2))
        else:
            return 1 + k * sum((k-1)**i for i in range((g-1)//2))
            
    def check_cayley_girth(generators, group_size, girth_req):
        """
        Check if the Cayley graph on the group (represented by generators acting on indices)
        has girth at least girth_req.
        """
        # This function assumes we have a way to multiply elements.
        # For efficiency, we pass the multiplication logic or use a precomputed table.
        pass

    def luw_construct(k, g):
        """Lazebnik-Ustimenko-Woldar graph CD(d, q) with minimal d."""
        q = k
        # Minimal d such that 2d+1 >= g
        # 2d >= g-1 => d >= (g-1)/2
        # d must be integer.
        # If g=9, d >= 4.
        d = (g - 1) // 2
        if d < 3: d = 3
        
        # For q=5, d=4, girth is 9.
        
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add = lambda x, y: x ^ y
            sub = lambda x, y: x ^ y
            mul = lambda x, y: table[x][y]
        else:
            add = lambda x, y: (x + y) % q
            sub = lambda x, y: (x - y) % q
            mul = lambda x, y: x * y % q
        
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
                    if side == 0:
                        y[c] = add(x[c], mul(y[a], x[b]))
                    else:
                        y[c] = sub(x[c], mul(x[a], y[b]))
                yield 1 - side, tuple(y)
        
        start = (0, (None,) + (0,) * d)
        index, order, edges = {start: 0}, [start], []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    # Try Random Cayley on S6
    def try_s6_cayley(k, g, max_trials=5000):
        # Generate S6 elements
        elements = list(itertools.permutations(range(6)))
        n = len(elements)
        idx = {e: i for i, e in enumerate(elements)}
        
        # Precompute multiplication table? 720x720 = 518k entries.
        # Might be slow to build, but fast to query.
        # Or just compute on the fly?
        # On the fly: tuple indexing is fast enough.
        
        def mult(p, q):
            # p * q = p(q(x))
            return tuple(p[q[i]] for i in range(6))
        
        identity = tuple(range(6))
        
        # Find involutions
        involutions = []
        for e in elements:
            if mult(e, e) == identity:
                involutions.append(e)
        
        # Filter to ensure we pick distinct generators?
        # The set of generators should be distinct to avoid loops?
        # If s1 == s2, we have repeated edges?
        # The graph must be simple.
        # If generators are distinct, and no generator is identity, 
        # and no product of two generators is identity (i.e. they are distinct),
        # then edges are simple?
        # Edge (u, u*s). If s1 != s2, u*s1 != u*s2.
        # So distinct generators ensure simple graph.
        
        for _ in range(max_trials):
            # Pick 5 distinct involutions
            if len(involutions) < k:
                continue
            gens = random.sample(involutions, k)
            
            # Check if they generate the whole group
            # BFS from identity
            visited = {identity: 0}
            parent = {identity: None}
            queue = deque([identity])
            
            while queue:
                u = queue.popleft()
                du = visited[u]
                for s in gens:
                    v = mult(u, s)
                    if v not in visited:
                        visited[v] = du + 1
                        parent[v] = u
                        queue.append(v)
                    else:
                        # Check cycle length
                        # Cycle is u -> v (edge) + path v to root + path root to u?
                        # No, the cycle is formed by the tree paths and the edge.
                        # Length = dist[u] + 1 + dist[v]
                        # This is valid if v is not the parent of u and u is not the parent of v?
                        # In undirected graph BFS, we might see back edges.
                        # If v is parent of u, dist[v] = dist[u] - 1.
                        # Length = dist[u] + 1 + dist[u] - 1 = 2*dist[u].
                        # This is the cycle formed by going up and down?
                        # No, that's just the edge traversed back.
                        # We should ignore the edge to the parent.
                        if parent[u] == v:
                            continue
                        # Also, if we encounter a visited node, it might be a cross edge.
                        # The cycle length is dist[u] + 1 + dist[v].
                        # We need this >= g.
                        if dist[u] + 1 + dist[v] < g:
                            return None
            # If we reach here, we need to verify the girth properly.
            # The BFS check above is a heuristic/early exit.
            # We need to check ALL edges.
            
            # Re-run BFS to get distances, then check all edges
            dist = {identity: 0}
            queue = deque([identity])
            while queue:
                u = queue.popleft()
                for s in gens:
                    v = mult(u, s)
                    if v not in dist:
                        dist[v] = dist[u] + 1
                        queue.append(v)
            
            # Check all edges
            valid = True
            for u in elements:
                if u not in dist: continue # Should not happen if generated
                for s in gens:
                    v = mult(u, s)
                    # Edge (u, v)
                    # Cycle length via this edge in BFS tree:
                    # If v is parent of u, skip? No, the cycle is not formed by tree edge.
                    # The cycle is formed by non-tree edges.
                    # In an undirected graph, every edge is either a tree edge or a back edge.
                    # If we use BFS tree, an edge (u,v) is a back edge if dist[u] and dist[v] differ by <= 1?
                    # Actually, for any edge (u,v), the cycle length is dist[u] + 1 + dist[v] ONLY IF the paths to root are disjoint?
                    # No, the cycle is u -> ... -> LCA -> ... -> v -> u.
                    # Length = dist[u] + dist[v] - 2*dist[LCA] + 1.
                    # To be safe, we can just do a full girth check using BFS from every node?
                    # Too slow.
                    
                    # Standard check:
                    # If we find an edge (u,v) such that dist[u] + 1 + dist[v] < g, it's a strong indicator.
                    # But LCA matters.
                    # However, for Cayley graphs, if we BFS from identity, 
                    # the shortest cycle containing the edge (u,v) is indeed bounded by dist[u]+1+dist[v].
                    # If dist[u]+1+dist[v] >= g for all edges, is girth >= g?
                    # Yes, because any cycle C of length L corresponds to a sequence of generators.
                    # The BFS distance is the word length.
                    # This is a known property.
                    
                    if dist[u] + 1 + dist[v] < g:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                # Construct edges
                edges = []
                for u in elements:
                    i = idx[u]
                    for s in gens:
                        v = mult(u, s)
                        j = idx[v]
                        if i < j:
                            edges.append((i, j))
                return n, edges
                
        return None

    # Try S6
    res = try_s6_cayley(k, g)
    if res:
        return res
        
    # Fallback to L-U-W
    return luw_construct(k, g)

# EVOLVE-BLOCK-END
