import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115
    
    # Specialized high-performance path for the specific task (4, 10)
    if k == 4 and g == 10:
        result = try_petersen_lift(4, 10, deadline)
        if result is not None:
            return result
        # Fallback to LUW if lift fails (should not happen)
        return luw_construction(4, 10)
    
    # General fallback for other k, g
    return luw_construction(k, g)

def try_petersen_lift(k, g, deadline):
    """
    Constructs a 4-regular graph on 320 vertices using a voltage lift
    of the Petersen graph over Z_32.
    """
    # Petersen graph edges
    # Outer pentagon: 0-1-2-3-4-0
    # Inner pentagram: 5-7-9-6-8-5
    # Spokes: 0-5, 1-6, 2-7, 3-8, 4-9
    petersen_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    
    n_base = 10
    group_size = 32
    n_lift = n_base * group_size
    
    # Pre-calculate cycles in Petersen of length < 10 to optimize voltage check?
    # Or just build and verify. Building is O(E) = 15 * 32 = 480 edges.
    # Verification is the bottleneck.
    
    max_attempts = 500
    for _ in range(max_attempts):
        if time.time() > deadline - 5:
            break
            
        # Assign random voltages
        voltages = {}
        for u, v in petersen_edges:
            voltages[(u, v)] = random.randint(0, group_size - 1)
        
        # Build lift graph
        edges = []
        for u, v in petersen_edges:
            vol = voltages[(u, v)]
            for x in range(group_size):
                u_id = u * group_size + x
                v_id = v * group_size + (x + vol) % group_size
                edges.append((u_id, v_id))
        
        # Verify girth
        if verify_girth(n_lift, edges, g):
            return n_lift, edges
            
    return None

def verify_girth(n, edges, g):
    """
    Verifies that the graph has girth at least g using non-backtracking walks.
    Returns True if girth >= g, False otherwise.
    """
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    # Check degree
    for i in range(n):
        if len(adj[i]) != 4: # k=4 in this context
            return False
            
    # Non-backtracking walk verification
    # A cycle of length L < g exists iff there is a non-backtracking walk of length L
    # starting and ending at the same vertex.
    # We check for any such walk of length 1 to g-1.
    
    for start in range(n):
        # State: (current_vertex, previous_vertex)
        # We use a set of visited states to avoid re-processing
        visited = set()
        queue = deque()
        
        # Initialize: neighbors of start at depth 1
        for neighbor in adj[start]:
            queue.append((neighbor, start))
            
        depth = 1
        while queue and depth < g:
            new_queue = deque()
            for (v, prev) in queue:
                if v == start:
                    return False  # Found a cycle of length 'depth' which is < g
                if (v, prev) in visited:
                    continue
                visited.add((v, prev))
                for w in adj[v]:
                    if w != prev:
                        new_queue.append((w, v))
            queue = new_queue
            depth += 1
    return True

def luw_construction(k, g):
    """
    Lazebnik-Ustimenko-Woldar graph CD(d, q)
    """
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
