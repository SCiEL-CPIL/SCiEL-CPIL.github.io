import random
from collections import deque
import itertools

# EVOLVE-BLOCK-START
def construct(k, g):
    if k == 3:
        # Try to find a small Cayley graph on Z_n x Z_m
        n, edges = _search_cayley_cubic(g)
        if n is not None:
            return n, edges
    # Fallback to LUW if Cayley search fails or for other k
    return _luw_construct(k, g)


def _search_cayley_cubic(g):
    """Search for a 3-regular Cayley graph on Z_n x Z_m with girth >= g."""
    # Generators: (1,0), (0,1), (a,b)
    # We iterate over n, m and a, b.
    # To minimize n*m, we check products in increasing order.
    # n, m must be > 1.
    # Max size to try: let's say 2000 vertices to stay within time limits?
    # BFS girth check is fast.
    
    max_vertices = 2000
    best_n = None
    best_edges = None
    
    # Iterate over n, m such that n*m <= max_vertices
    # We can optimize by iterating product P from 512 to max_vertices
    # But simpler: iterate n from 2 to max_vertices, m from 2 to max_vertices//n
    # To ensure we find small ones, we should check smaller products first.
    
    # Let's just iterate n, m up to a reasonable bound, say 100.
    # 100*100 = 10000 vertices. BFS on 10000 vertices: 10000 * 40000 = 4*10^8.
    # This might be too slow in Python for many candidates.
    # Let's limit to n, m <= 50. 50*50=2500.
    # 2500 * 10000 = 2.5*10^7. This is fast.
    
    for n in range(2, 51):
        for m in range(2, 51):
            if n * m > 2000:
                continue
            # Try a few (a,b) pairs
            # a in 0..n-1, b in 0..m-1
            # We can restrict a, b to avoid symmetry?
            # (a,b) and (n-a, m-b) are inverses.
            # We need g3 != 0, g3 != g3^-1 (unless n=2 or m=2, handle carefully)
            # Actually, if g3 = g3^-1, it's an involution. That's fine for undirected graph,
            # as long as it's not 0.
            
            # We should check distinct generators:
            # g1=(1,0), g2=(0,1), g3=(a,b)
            # g1 != g2, g1 != g3, g2 != g3
            # g1 != -g2, g1 != -g3, g2 != -g3
            # -g1 = (n-1, 0), -g2 = (0, m-1), -g3 = (n-a, m-b)
            
            # To save time, we can just pick a few promising (a,b)
            # e.g. a=1, b=1; a=1, b=2; a=2, b=1; a=1, b=3...
            
            candidates = []
            for a in range(n):
                for b in range(m):
                    # Skip identity
                    if a == 0 and b == 0:
                        continue
                    # Check distinctness from g1, g2 and their inverses
                    g3 = (a, b)
                    g1 = (1, 0)
                    g2 = (0, 1)
                    ng1 = (n-1, 0)
                    ng2 = (0, m-1)
                    ng3 = (n-a, m-b)
                    
                    if g3 in [g1, g2, ng1, ng2]:
                        continue
                    # Also g3 should not be inverse of g1 or g2?
                    # If g3 == ng1, then g3 is inverse of g1. 
                    # This creates a double edge between u and u+g1?
                    # Edge (u, u+g1) and (u, u+ng1) = (u, u-g1).
                    # In undirected graph, edge {u, u+g1} is the same as {u+g1, u}.
                    # If we add generator g1 and ng1, we just traverse the same edge.
                    # So degree would be 2, not 3.
                    # So we must ensure g3 is not in {g1, g2, ng1, ng2, ng3(if g3=ng3 then involution)}
                    # If g3 == ng3, it's an involution. That's fine.
                    # But if g3 == ng1, it's redundant with g1.
                    
                    # So check:
                    if g3 == ng1 or g3 == ng2:
                        continue
                    # Also g1 != ng1 (n!=2), g2 != ng2 (m!=2)
                    # If n=2, g1=ng1. Then degree from g1 is 1.
                    # We need degree 3.
                    # If n=2, g1 is involution.
                    # Let's just require n>2 and m>2 for simplicity to avoid involution complications in counting,
                    # or handle them.
                    # For now, skip n=2, m=2.
                    if n == 2 or m == 2:
                        continue
                        
                    # Also ensure g1 != g2, etc. (1,0) != (0,1) always if n,m>1.
                    # g1 != ng2 -> (1,0) != (0, m-1) -> 1!=0 and 0!=m-1 -> m!=1. OK.
                    
                    candidates.append((a, b))
            
            # Shuffle candidates to find good ones early?
            # Or just check a subset.
            # Let's check all candidates for this (n,m)
            for a, b in candidates:
                # Build graph and check girth
                # Vertices: (x,y) for x in 0..n-1, y in 0..m-1
                # Index: x*m + y
                n_verts = n * m
                adj = [[] for _ in range(n_verts)]
                
                g1 = (1, 0)
                g2 = (0, 1)
                g3 = (a, b)
                
                # Add edges for g1, g2, g3
                # Since undirected, add edge for g and 1/g?
                # No, just iterate all vertices and add edge to v+g.
                # To avoid duplicates, we can just build the full adjacency list.
                # For each v, neighbors are v+g1, v+g2, v+g3.
                # This gives degree 3 if g1,g2,g3 distinct and not 0.
                
                for x in range(n):
                    for y in range(m):
                        v = x * m + y
                        # Neighbor 1: (x+1)%n, y
                        u1 = ((x + 1) % n) * m + y
                        # Neighbor 2: x, (y+1)%m
                        u2 = x * m + ((y + 1) % m)
                        # Neighbor 3: (x+a)%n, (y+b)%m
                        u3 = ((x + a) % n) * m + ((y + b) % m)
                        
                        adj[v].append(u1)
                        adj[v].append(u2)
                        adj[v].append(u3)
                
                # Check simplicity: no loops, no multiedges
                # Loops: v == u?
                # Multiedges: u1 == u2 or u1 == u3 or u2 == u3?
                # We already checked g1!=g2, g1!=g3, g2!=g3.
                # But (x+1)%n == x? Only if n=1.
                # So no loops if n,m > 1.
                # Multiedges:
                # u1 == u2 => (x+1)%n == x and y == (y+1)%m => 1=0 mod n, 0=1 mod m. Impossible.
                # u1 == u3 => (x+1)%n == (x+a)%n and y == (y+b)%m => 1=a mod n, 0=b mod m.
                # So if a==1 and b==0, g3==g1. We checked g3!=g1.
                # u2 == u3 => 0=a mod n, 1=b mod m.
                # So if a==0 and b==1, g3==g2. We checked g3!=g2.
                
                # So graph is simple.
                
                # Check Girth
                girth = _min_girth(adj)
                if girth >= g:
                    # Found a valid graph
                    edges = []
                    for v in range(n_verts):
                        for u in adj[v]:
                            if v < u:
                                edges.append((v, u))
                    return n_verts, edges
    return None, None


def _min_girth(adj):
    """Calculate minimum cycle length in an unweighted undirected graph."""
    n = len(adj)
    min_g = float('inf')
    for start in range(n):
        # BFS to find shortest cycle containing start?
        # Or just BFS from start, if we encounter a visited node, 
        # cycle length = dist[current] + dist[visited] + 1.
        # This finds the shortest cycle that passes through start?
        # Actually, standard BFS from start finds shortest path to all nodes.
        # If we see an edge (u,v) where v is visited and v is not parent of u,
        # we have a cycle.
        # To find global min girth, we can do BFS from each node.
        
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                else:
                    # Back edge or cross edge
                    # In undirected BFS, if v is visited and v != parent[u], 
                    # and u != parent[v] (to avoid double counting the tree edge),
                    # then we have a cycle.
                    # Cycle length = dist[u] + dist[v] + 1
                    # But we must ensure it's not the tree edge.
                    if v != parent[u] and u != parent[v]:
                        # This logic is tricky.
                        # Standard method:
                        # When exploring u, for each neighbor v:
                        # if v is unvisited, mark it.
                        # if v is visited and v is not parent of u:
                        #   cycle_len = dist[u] + dist[v] + 1
                        #   update min_g
                        # This works because BFS tree is layered.
                        # However, this might count the same cycle multiple times or 
                        # not find the absolute shortest if the shortest cycle doesn't 
                        # pass through start? 
                        # No, every cycle passes through some start.
                        # But does BFS from start find the shortest cycle *in the graph*?
                        # Yes, the global min girth is the min over all starts of 
                        # the shortest cycle found via BFS from that start?
                        # Actually, a cycle of length L will be found when BFS 
                        # reaches the "closing" edge.
                        # The length calculated is correct.
                        
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < min_g:
                            min_g = cycle_len
    return min_g


def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar explicit construction (fallback, always valid)."""
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
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges
# EVOLVE-BLOCK-END
