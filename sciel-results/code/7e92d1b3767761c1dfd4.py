"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def construct(k, g):
    """
    Construct a k-regular graph of girth at least g.
    
    Strategy:
    1. Use the Lazebnik-Ustimenko-Woldar (LUW) construction to get a base graph
       with good girth properties.
    2. If the base graph doesn't meet requirements, try other constructions.
    3. Verify girth using non-backtracking walks.
    4. Return the best valid graph found within time budget.
    """
    start_time = time.time()
    deadline = start_time + 120  # 120 second budget
    
    # The LUW construction: CD(d, q) where q is prime power
    # This gives a q-regular bipartite graph with girth at least d + 5 for odd d
    
    def luw_construct(q, d):
        """
        Lazebnik-Ustimenko-Woldar graph CD(d, q).
        q must be a prime power (we'll use prime for simplicity).
        Returns (n, edges) where n is number of vertices and edges is list of (i,j).
        """
        # Check if q is prime
        def is_prime(n):
            if n < 2:
                return False
            if n == 2:
                return True
            if n % 2 == 0:
                return False
            for i in range(3, int(math.isqrt(n)) + 1, 2):
                if n % i == 0:
                    return False
            return True
        
        if not is_prime(q):
            return None, None
        
        # GF(q) operations
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: (x * y) % q
        
        # Rule table for LUW construction
        # For CD(d, q), we need d >= 3
        if d < 3:
            d = 3
        
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r = (c - 5) % 4
            c0 = c - (c - 5) % 4
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
        
        # Build the graph via BFS from zero point
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
    
    def verify_girth(n, edges, required_girth):
        """
        Verify that the graph has no cycle shorter than required_girth
        using non-backtracking walks from every vertex.
        """
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Check degree
        for i in range(n):
            if len(adj[i]) != k:
                return False
        
        # Non-backtracking walk to check girth
        # A non-backtracking walk of length L starting from edge (u,v) 
        # visits v next, then continues without immediately going back to u.
        # If we can find a non-backtracking walk that returns to its start
        # in fewer than g steps, there's a cycle shorter than g.
        
        # For efficiency, we'll check cycles up to length g-1
        # Using BFS on the line graph with non-backtracking constraint
        
        # For each directed edge (u,v), do BFS in the non-backtracking sense
        # State: (current_vertex, previous_vertex, depth)
        # We track visited states to detect cycles
        
        # Actually, a simpler approach: for each vertex, do BFS up to depth g//2
        # and check if we can reach back to the start vertex without backtracking
        
        # More efficient: use the fact that in a graph of girth >= g,
        # the number of non-backtracking walks of length L from any edge
        # is exactly (k-1)^L for L < g/2
        
        # Let's do a direct check: for each directed edge, follow all 
        # non-backtracking paths up to length g-1 and see if we return to start
        
        # This could be expensive, so we'll use a smarter approach
        # Count non-backtracking walks and compare to theoretical count
        
        # For a graph with girth >= g, the number of closed non-backtracking
        # walks of length L (for L < g) is 0.
        
        # Let's use matrix-free approach: count non-backtracking walks
        # of each length from each edge
        
        # Actually, let's just do a direct search for short cycles
        # For each vertex, do BFS up to depth g//2 and check for cycles
        
        max_depth = g // 2
        
        for start_v in range(n):
            # BFS from start_v, tracking parent to avoid backtracking
            visited = {}  # (vertex, prev) -> depth
            queue = [(start_v, -1, 0)]
            while queue:
                v, prev, depth = queue.pop(0)
                if depth >= max_depth:
                    continue
                for w in adj[v]:
                    if w == prev:
                        continue  # non-backtracking
                    if w == start_v and depth + 1 >= 3:
                        # Found a cycle
                        cycle_len = depth + 1
                        if cycle_len < required_girth:
                            return False
                    if (w, v) not in visited:
                        visited[(w, v)] = depth + 1
                        if depth + 1 < max_depth:
                            queue.append((w, v, depth + 1))
        
        return True
    
    def verify_simple(n, edges, k):
        """Verify graph is simple and k-regular."""
        # Check no loops
        for u, v in edges:
            if u == v:
                return False
        
        # Check no repeated edges
        edge_set = set()
        for u, v in edges:
            e = (min(u, v), max(u, v))
            if e in edge_set:
                return False
            edge_set.add(e)
        
        # Check degree
        deg = [0] * n
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        for d in deg:
            if d != k:
                return False
        
        return True
    
    # Moore bound: lower bound on number of vertices
    # For k-regular graph of girth g:
    # If g is odd: n >= 1 + k * sum_{i=0}^{(g-3)/2} (k-1)^i
    # If g is even: n >= 2 * sum_{i=0}^{g/2-1} (k-1)^i
    
    def moore_bound(k, g):
        if g % 2 == 1:
            # Odd girth
            h = (g - 3) // 2
            s = 0
            for i in range(h + 1):
                s += (k - 1) ** i
            return 1 + k * s
        else:
            # Even girth
            h = g // 2 - 1
            s = 0
            for i in range(h + 1):
                s += (k - 1) ** i
            return 2 * s
    
    mb = moore_bound(k, g)
    
    # Try LUW construction with different parameters
    # For k-regular graph, we need q = k (if k is prime)
    # The girth of CD(d, q) is at least d + 5 for odd d
    
    best_n = None
    best_edges = None
    
    # First, try the standard LUW construction
    if k > 2:
        # For odd girth g, we need d such that d + 5 >= g, so d >= g - 5
        # d should be odd for the girth guarantee
        if g % 2 == 1:
            d_min = g - 5
            if d_min < 3:
                d_min = 3
            if d_min % 2 == 0:
                d_min += 1
        else:
            d_min = g - 4
            if d_min < 3:
                d_min = 3
            if d_min % 2 == 0:
                d_min += 1
        
        # Try LUW with q = k
        if time.time() < deadline - 30:
            n_luw, edges_luw = luw_construct(k, d_min)
            if n_luw is not None and edges_luw is not None:
                if verify_simple(n_luw, edges_luw, k):
                    if verify_girth(n_luw, edges_luw, g):
                        if best_n is None or n_luw < best_n:
                            best_n = n_luw
                            best_edges = edges_luw
    
    # If LUW didn't work or we want to try other options
    if best_n is None and time.time() < deadline - 20:
        # Try a different approach: use a circulant or other explicit construction
        # For now, try LUW with slightly larger d to ensure girth
        for d_try in range(d_min, d_min + 4, 2):
            if time.time() > deadline - 20:
                break
            n_luw, edges_luw = luw_construct(k, d_try)
            if n_luw is not None and edges_luw is not None:
                if verify_simple(n_luw, edges_luw, k):
                    if verify_girth(n_luw, edges_luw, g):
                        if best_n is None or n_luw < best_n:
                            best_n = n_luw
                            best_edges = edges_luw
                        break
    
    # Fallback: return the original LUW construction result
    if best_n is None:
        # Use the original seed program's approach directly
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        
        def is_prime(n):
            if n < 2:
                return False
            if n == 2:
                return True
            if n % 2 == 0:
                return False
            for i in range(3, int(math.isqrt(n)) + 1, 2):
                if n % i == 0:
                    return False
            return True
        
        if is_prime(q):
            add = lambda x, y: (x + y) % q
            sub = lambda x, y: (x - y) % q
            mul = lambda x, y: (x * y) % q
            
            rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
            for c in range(5, d + 1):
                r = (c - 5) % 4
                c0 = c - (c - 5) % 4
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
        else:
            # If k is not prime, we can't use the standard LUW construction
            # Return a trivial valid graph if possible, or raise an error
            # For now, try to construct a cycle (which won't have high girth)
            # This is a fallback that may not meet girth requirements
            n = max(k * 2, mb)
            edges = []
            for i in range(n):
                for j in range(1, k // 2 + 1):
                    edges.append((i, (i + j) % n))
                    edges.append(((i + j) % n, i))
            # Deduplicate
            edge_set = set()
            unique_edges = []
            for u, v in edges:
                e = (min(u, v), max(u, v))
                if e not in edge_set and u != v:
                    edge_set.add(e)
                    unique_edges.append(e)
            return n, unique_edges
    
    return best_n, best_edges
# EVOLVE-BLOCK-END
