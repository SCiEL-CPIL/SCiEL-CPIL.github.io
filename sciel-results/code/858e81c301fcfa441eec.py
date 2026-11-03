import time
import math
import random
from collections import deque

# EVOLVE-BLOCK-START
def construct(k, g):
    """
    Construct a k-regular graph of girth at least g on as few vertices as possible.
    For k=4, g=10, we search for a circulant graph (or similar) with n around the
    known upper bound of 320. We fall back to the Lazebnik-Ustimenko-Woldar (LUW)
    construction if no small graph is found within the time limit.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # 5 second margin
    
    # For k=4, g=10, the Moore bound is 2 * (1 + 3 + 3^2) = 26.
    # The best known upper bound is 320 (Exoo et al.).
    # We search for a 4-regular graph with girth >= 10.
    
    if k != 4 or g != 10:
        # Fallback to LUW for other parameters (less optimized)
        return _luw_construct(k, g)
    
    # Search for a circulant graph C_n(s1, s2) which is 4-regular.
    # Vertices 0..n-1, edges (i, i+s1), (i, i-s1), (i, i+s2), (i, i-s2) mod n.
    # We require n to be such that the graph is simple (no multiple edges, no loops).
    # This requires s1, s2 != 0, s1 != s2, s1 != -s2, and s1, s2 not causing collisions.
    # Specifically, the set {s1, -s1, s2, -s2} mod n must have size 4.
    
    # We know a solution exists with n=320. Let's search around there.
    # We can try to construct one using known methods or just brute force with BFS check.
    # A better approach: use the fact that the (4,10)-cage is a specific graph.
    # One known construction is a covering of a smaller graph or a specific circulant.
    
    # Let's try to find a valid pair (n, s1, s2).
    # We iterate n from 243 (lower bound) to 350.
    # For each n, we iterate s1, s2.
    
    def is_valid_circulant(n, s1, s2):
        # Check simplicity: distinct steps
        steps = [s1 % n, (-s1) % n, s2 % n, (-s2) % n]
        if len(set(steps)) < 4:
            return False
        if 0 in steps:
            return False
        
        # Check girth using BFS on the non-backtracking line graph? 
        # Or just BFS on the graph from each vertex to find shortest cycle?
        # For n=320, BFS from each vertex is O(n * (n+k log n)) which is fast.
        # But doing it for many (n, s1, s2) pairs might be slow.
        # We can prune: if a short cycle is found early, stop.
        
        # Build adjacency
        adj = [[] for _ in range(n)]
        for i in range(n):
            adj[i].append((i + s1) % n)
            adj[i].append((i - s1) % n)
            adj[i].append((i + s2) % n)
            adj[i].append((i - s2) % n)
        
        # Check girth
        # A cycle of length L corresponds to a non-backtracking walk of length L.
        # We can check if there is a cycle of length < g.
        # This is equivalent to checking if the BFS tree from any vertex has an edge
        # connecting two nodes at the same depth or depth differing by 1?
        # Actually, shortest cycle through a vertex v is found by BFS.
        # We need min cycle length in the whole graph.
        
        # Efficient check:
        # For each start vertex s, do BFS.
        # dist[s][v] = shortest distance from s to v.
        # A cycle of length L exists if there is an edge (u, v) such that
        # dist[s][u] + 1 + dist[s][v] == L and the paths don't overlap in a way that
        # makes it a "walk" not a "cycle"? 
        # Actually, the shortest cycle containing s is min_{edge (u,v) not in tree} dist[u] + dist[v] + 1.
        # But we need the global minimum cycle.
        
        # Simpler: BFS from each vertex, record min cycle length found.
        # For n=320, this is very fast.
        
        min_girth = 10**9
        for s in range(n):
            # BFS
            dist = [-1] * n
            dist[s] = 0
            q = deque([s])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        q.append(v)
                    else:
                        # Found a cycle
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < min_girth:
                            min_girth = cycle_len
                            if min_girth < g:
                                return False
            if min_girth < g:
                return False
        
        return min_girth >= g
    
    # Search range
    # We know 320 works. Let's try to find a smaller one or just return 320.
    # To be safe and fast, we can try a few candidates.
    # Known (4,10)-cage is not necessarily circulant, but there are circulant (4,9) or (4,10)?
    # Actually, the (4,10)-cage has 320 vertices. Is it circulant?
    # According to some sources, the (4,10)-cage is NOT circulant. It is a specific graph.
    # However, there might be a circulant graph with girth 10 and n > 320?
    # If we can't find n < 320, we should return a valid n=320 graph if we can construct it.
    # But constructing the exact (4,10)-cage is complex.
    
    # Alternative: Use the LUW graph which is known to be valid and has n=1296.
    # If we can find ANY graph with n < 1296, we improve.
    # Let's search for a circulant graph with n < 1296.
    # We can scan n from 243 to 1295.
    # For each n, we need to find s1, s2.
    # This is a 2D search. To speed up, we can use heuristics.
    
    # Heuristic: s1 and s2 should be "independent".
    # Let's try to find a solution.
    
    best_n = 1296
    best_edges = None
    
    # Try a few specific n values first
    # n = 320 is the target. Let's see if a circulant exists.
    # If not, we might find n=324, 330, etc.
    
    # To avoid timing out, we limit the search.
    # We will try n in a specific range.
    
    # Let's try to construct the (4,10)-cage directly if we can find a definition.
    # The (4,10)-cage is a Cayley graph? 
    # It is a quotient of the universal cover of the 4-regular tree.
    
    # Since constructing the exact cage is hard, let's rely on the search.
    # We'll search n from 243 to 400.
    # For each n, we try random s1, s2 or systematic.
    
    # Systematic search is too slow for all n.
    # Let's try a set of candidate n values.
    
    # Actually, a known 4-regular graph of girth 10 with 320 vertices is the (4,10)-cage.
    # It can be constructed as a voltage graph on a 4-cycle? Or a bipartite graph?
    # The (4,10)-cage is bipartite.
    
    # If we can't find it quickly, we return the LUW graph (n=1296).
    # To beat the seed (1.0), we need n < 1296.
    # Even n=1000 would be an improvement.
    
    # Let's try to find a circulant graph.
    # We'll sample s1, s2 for several n.
    
    candidates_n = []
    # Include 320
    candidates_n.append(320)
    # Include nearby numbers
    for n in range(243, 400):
        candidates_n.append(n)
    
    # Shuffle to find a solution quickly? Or stick to 320 first?
    # Let's try 320 first with a limited search of s1, s2.
    
    def search_circulant(n, time_limit):
        # Try to find s1, s2 for given n
        # s1 in [1, n//2], s2 in [1, n//2]
        # To speed up, we can check constraints:
        # gcd(n, s1) = 1? Not necessary for connectivity if combined with s2.
        # But if gcd(n, s1) > 1 and gcd(n, s2) > 1, it might be disconnected.
        
        # We'll use a random search to find a valid pair quickly.
        # If random search fails, we might do a systematic sweep.
        
        start = time.time()
        # Try random pairs
        attempts = 0
        while time.time() - start < time_limit:
            s1 = random.randint(1, n//2)
            s2 = random.randint(1, n//2)
            if s1 == s2 or s1 + s2 == n or s1 + s2 == 0:
                continue
            if is_valid_circulant(n, s1, s2):
                # Construct edges
                edges = []
                for i in range(n):
                    edges.append((i, (i + s1) % n))
                    edges.append((i, (i - s1) % n))
                    edges.append((i, (i + s2) % n))
                    edges.append((i, (i - s2) % n))
                # Remove duplicates and sort
                # The above list has 2n edges. Each edge is added twice (once from i, once from j)?
                # (i, i+s1) and (i+s1, i) are distinct in the list.
                # We need unique undirected edges.
                edge_set = set()
                for u, v in edges:
                    if u > v:
                        u, v = v, u
                    edge_set.add((u, v))
                return n, list(edge_set)
            attempts += 1
        return None
    
    # Try n=320 first
    result = search_circulant(320, 5.0)
    if result:
        return result
    
    # Try other n
    # We'll iterate n from 243 to 1295, but with a small time budget each.
    # This might be too slow. 
    # Better: try a few specific n values known to be good candidates.
    
    # Let's try n = 320, 324, 330, 336, 342, 348, 354, 360
    for n in [320, 324, 330, 336, 342, 348, 354, 360, 366, 372, 378, 384, 390, 396]:
        if time.time() > deadline - 10:
            break
        result = search_circulant(n, 2.0)
        if result:
            return result
    
    # Fallback to LUW
    return _luw_construct(4, 10)

def _luw_construct(k, g):
    """
    Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k.
    """
    q = k
    # d is the "depth" parameter. Girth is roughly d+5.
    # We want girth >= g. So d >= g-5.
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
        
    # Define the rule for the LUW graph
    # This part is complex to derive from scratch, so we copy from the seed
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
    index = {start: 0}
    order = [start]
    edges = []
    
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
                
    return len(order), edges
# EVOLVE-BLOCK-END
