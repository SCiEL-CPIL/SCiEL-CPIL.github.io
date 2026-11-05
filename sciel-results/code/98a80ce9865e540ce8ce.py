import time
import random
import sys

def construct(k, g):
    """
    Construct a k-regular graph with girth at least g.
    For k=4, g=10, attempts to find a graph with n < seed_n.
    The seed (LUW) for k=4, g=10 likely produces a specific n.
    We use a randomized greedy approach to build the graph layer by layer,
    checking girth constraints using BFS from each vertex.
    """
    
    # Safety deadline
    start_time = time.time()
    deadline = start_time + 110  # 10 seconds margin

    def time_left():
        return deadline - time.time()

    # The evaluator passes k and g. 
    # For this specific task, k=4, g=10.
    # We need to return (n, edges).

    def _luw_fallback(k, g):
        """
        Re-implementing seed logic quickly for fallback.
        """
        q = k
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

    def _is_valid(n, adj, g):
        """
        Check if the graph is simple, k-regular, and has girth >= g.
        """
        # Check simplicity and degree
        for u in range(n):
            if len(adj[u]) != k or len(set(adj[u])) != len(adj[u]):
                return False
            for v in adj[u]:
                if v == u:
                    return False
                if u not in adj[v]:
                    return False
                # Check for duplicate edges (symmetry)
                # adj[u] is a set, so no duplicates within u.
                # Need to ensure no multi-edges globally.
                # Since adj[u] is a set, u-v is unique for u.
                # But we need to ensure that for all u, v, if u in adj[v] then v in adj[u].
        
        # Check girth using BFS from each node up to depth r = (g-1)//2
        # For even g, depth = g/2 - 1.
        # For odd g, depth = (g-1)//2.
        # If we find a cross edge (u, v) where dist(u) + dist(v) + 1 < g, invalid.
        # Actually, if BFS tree up to depth r is a tree, then no cycle shorter than 2r+1.
        # For g=10, r=4. 2*4+1=9. So any cycle detected is < 10.
        # We need to ensure no cycle < g.
        # So if we detect any "cycle" (back edge or cross edge) in BFS up to depth floor((g-1)/2), it implies a cycle <= 2*floor((g-1)/2)+1 < g.
        # For g=10, floor(9/2)=4. Cycle <= 9. Correct.
        
        r = (g - 1) // 2
        
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = [start]
            head = 0
            while head < len(queue):
                u = queue[head]
                head += 1
                if dist[u] >= r:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    else:
                        # Already visited
                        if v == parent[u]:
                            continue
                        # Check cycle length
                        # In a BFS tree, an edge (u,v) where both are visited and v != parent[u] 
                        # implies a cycle. 
                        # The length of the cycle is at most dist[u] + dist[v] + 1.
                        # Since dist[u] <= r and dist[v] <= r (because we only expand up to r, 
                        # but v might be deeper? No, if v is already visited, its dist is fixed.
                        # If dist[v] > r, we wouldn't have expanded it, but it could have been 
                        # visited by a node at depth r? No, we stop expanding at r.
                        # So all visited nodes have dist <= r.
                        # Thus cycle length <= r + r + 1 = 2r + 1 < g.
                        # So any such edge is invalid.
                        return False
        return True

    def _generate_random_4_regular(n):
        """
        Generate a random 4-regular graph on n vertices using configuration model.
        Returns None if invalid (self-loop or multi-edge).
        """
        if n < 5:
            return None
            
        stubs = []
        for i in range(n):
            for _ in range(4):
                stubs.append(i)
                
        random.shuffle(stubs)
        
        adj = [set() for _ in range(n)]
        for i in range(0, len(stubs), 2):
            u = stubs[i]
            v = stubs[i+1]
            if u == v:
                return None
            if v in adj[u]:
                return None
            adj[u].add(v)
            adj[v].add(u)
            
        return [list(adj[i]) for i in range(n)]

    def _construct_cage(k, g, deadline):
        """
        Attempt to construct a small k-regular graph of girth g.
        """
        if k != 4 or g != 10:
            return _luw_fallback(k, g)
            
        # Known lower bound is 243, best known is 320.
        # Seed is 2048.
        # We will try random graphs from n=320 upwards.
        # If we find a valid one, we return it.
        
        # To be safe, start from a slightly higher n if time is tight?
        # No, 320 is the target.
        
        best_n = None
        best_edges = None
        
        # Try sizes from 320 to 400
        for n in range(320, 420, 2):
            if time_left() < 5:
                break
                
            # Number of attempts depends on time left
            # Let's say 20 attempts per size if plenty of time, less if not.
            attempts = 20 if time_left() > 30 else 5
            
            for _ in range(attempts):
                if time_left() < 2:
                    break
                    
                adj = _generate_random_4_regular(n)
                if adj is None:
                    continue
                
                if _is_valid(n, adj, g):
                    # Convert to edge list
                    edges = []
                    for u in range(n):
                        for v in adj[u]:
                            if u < v:
                                edges.append((u, v))
                    best_n = n
                    best_edges = edges
                    break
            
            if best_n is not None:
                break
        
        if best_n is not None:
            return best_n, best_edges
        
        # Fallback to LUW if no random graph found
        return _luw_fallback(k, g)

    return _construct_cage(k, g, deadline)
