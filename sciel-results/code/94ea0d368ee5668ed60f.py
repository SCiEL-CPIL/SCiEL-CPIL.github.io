"""Rewritten program for the 5-regular, girth-9 cage construction task."""
# EVOLVE-BLOCK-START
import time
import sys

def construct(k, g):
    """
    Construct a k-regular graph with girth >= g on as few vertices as possible.
    Returns (n, edges) where edges is a list of (u, v) tuples.
    """
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s kill

    def elapsed():
        return time.time() - start_time

    def within_deadline():
        return elapsed() < deadline

    def verify_girth(n, adj, k, g):
        """
        Verify girth >= g using BFS in the non-backtracking state space.
        State (prev, curr) at depth d means a non-backtracking walk of d+1 edges
        ending at curr (came from prev).
        A cycle of length L exists if we can reach a state (v, s) where s is the
        start vertex, from initial states (s, w) for w in adj[s].
        The cycle length is (depth + 1) + 1 = depth + 2.
        We need to ensure no cycle of length < g, i.e., depth + 2 < g, i.e., depth < g - 2.
        So we BFS up to depth g - 2 (inclusive) and check if we return to s.
        """
        # For each start vertex s
        for s in range(n):
            if not within_deadline():
                return False
            visited = {}
            queue = []
            for w in adj[s]:
                state = (s, w)
                visited[state] = 0
                queue.append(state)
            
            qi = 0
            while qi < len(queue):
                prev, curr = queue[qi]
                qi += 1
                d = visited[(prev, curr)]
                # If d >= g-1, we don't need to expand (cycle length would be >= g+1 > g)
                if d >= g - 1:
                    continue
                for x in adj[curr]:
                    if x == prev:
                        continue
                    new_state = (curr, x)
                    if new_state in visited:
                        continue
                    new_d = d + 1
                    visited[new_state] = new_d
                    # Check if we returned to s
                    if x == s:
                        cycle_len = new_d + 1  # new_d + 1 edges in the walk, but it's a cycle of length new_d + 1? 
                        # Wait: walk is s -> ... -> curr -> s. Number of edges = new_d + 1.
                        # Actually, the walk from s to s has new_d + 1 edges. That's the cycle length.
                        if cycle_len < g:
                            return False
                    queue.append(new_state)
        return True

    def build_luw_d(d, q):
        """
        Build LUW(d, q) graph.
        Returns (n, edges) or None if construction fails.
        """
        # LUW construction
        # Vertices are (side, x) where side in {0,1}, x = (x_1, ..., x_d) in F_q^d
        # Neighbors defined by rules.
        
        if q == 2:
            add = lambda x, y: x ^ y
            sub = lambda x, y: x ^ y
            mul = lambda x, y: x & y  # field of 2 elements, multiply is AND
        elif q == 3:
            add = lambda x, y: (x + y) % 3
            sub = lambda x, y: (x - y) % 3
            mul = lambda x, y: x * y % 3
        elif q == 4:
            # GF(4): use {0, 1, 2, 3} with 1 as alpha, alpha^2 = alpha + 1
            # Addition: XOR
            # Multiplication:
            # 0*x=0, 1*x=x
            # alpha*(alpha+1) = alpha^2 + alpha = (alpha+1) + alpha = 1
            # alpha*alpha = alpha+1 = 2
            # alpha*1 = alpha = 1
            # Let's use a table
            mul_table = [
                [0, 0, 0, 0],
                [0, 1, 2, 3],
                [0, 2, 3, 1],
                [0, 3, 1, 2]
            ]
            add = lambda x, y: x ^ y
            sub = lambda x, y: x ^ y
            mul = lambda x, y: mul_table[x][y]
        elif q == 5:
            add = lambda x, y: (x + y) % 5
            sub = lambda x, y: (x - y) % 5
            mul = lambda x, y: x * y % 5
        else:
            # General prime field
            add = lambda x, y: (x + y) % q
            sub = lambda x, y: (x - y) % q
            mul = lambda x, y: x * y % q

        # Rules for LUW(d, q)
        # For c = 2, ..., d:
        # r_c = c - 5 if c >= 5, else ...
        # Standard LUW rules:
        # For c=2: a=1, b=1
        # For c=3: a=2, b=1
        # For c=4: a=1, b=2
        # For c>=5: 
        #   r = (c-5) % 4
        #   c0 = c - r
        #   rules = [(1, c0-2), (c0-1, 1), (c0, 1), (1, c0+1)]
        #   a, b = rules[r]
        
        rule = {}
        if d >= 2:
            rule[2] = (1, 1)
        if d >= 3:
            rule[3] = (2, 1)
        if d >= 4:
            rule[4] = (1, 2)
        for c in range(5, d + 1):
            r = (c - 5) % 4
            c0 = c - r
            rules = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)]
            rule[c] = rules[r]

        def get_neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None] + [0] * d  # y[1..d]
                y[1] = first
                for c in range(2, d + 1):
                    a, b = rule[c]
                    if side == 0:
                        y[c] = add(x[c], mul(y[a], x[b]))
                    else:
                        y[c] = sub(x[c], mul(x[a], y[b]))
                yield (1 - side), tuple(y[1:])

        # Build the graph
        start = (0, tuple([0] * d))
        index = {start: 0}
        order = [start]
        edges = []
        
        for vertex in order:
            if not within_deadline():
                return None
            for other in get_neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        
        n = len(order)
        
        # Build adjacency for verification
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        return n, edges, adj

    # Try LUW constructions
    # For k=5, g=9:
    # LUW(d, q) is q-regular (if q is prime power and d >= 2).
    # Wait, LUW(d, q) is q-regular? Let me check.
    # Each vertex has q neighbors (one for each "first" value).
    # So LUW(d, q) is q-regular.
    # For k=5, we need q=5.
    # Girth of LUW(d, q) is at least d+5 (for odd d) or d+4 (for even d)?
    # Actually, the girth is 2*ceil(d/2) + 4? No.
    # Standard result: girth >= d + 5.
    # For g=9, we need d+5 >= 9 => d >= 4.
    # But LUW graphs are bipartite, so girth is even.
    # If g=9 (odd), we need girth >= 10.
    # So d+5 >= 10 => d >= 5.
    
    # Try d=5, q=5 first
    best_result = None
    
    # Try d values from 5 upwards
    for d in range(5, 15):
        if not within_deadline():
            break
        try:
            result = build_luw_d(d, 5)
            if result is None:
                continue
            n, edges, adj = result
            # Verify girth
            if verify_girth(n, adj, 5, g):
                best_result = (n, edges)
                break
        except Exception as e:
            continue
    
    if best_result is not None:
        return best_result
    
    # Fallback: try other constructions or raise error
    # For now, raise an error if no graph found
    raise RuntimeError("No valid graph found within deadline")
# EVOLVE-BLOCK-END
