"""Rewritten program: Random Voltage Assignment on the Petersen graph for k=3, g=16."""
import time
import random
import sys
from collections import deque

# EVOLVE-BLOCK-START

def construct(k, g):
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s kill
    
    # For the cage problem n(3,16), we use Random Voltage Assignment on the Petersen graph.
    # The Petersen graph has 10 vertices, is 3-regular, and is the smallest cage for (3,5).
    # We assign random voltages from a group to the edges to lift it into a larger graph.
    
    if k != 3:
        # Fallback for other k values: use LUW construction
        return _luw_construct(k, g, deadline)
    
    # The Petersen graph edges (standard labeling)
    # Vertices 0-4 form a 5-cycle, 5-9 form a star, with connections between them
    petersen_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),  # outer 5-cycle
        (5, 6), (6, 7), (7, 8), (8, 9), (9, 5),  # inner 5-cycle (star)
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)   # spokes
    ]
    
    # We need a group of order q such that q*10 is large enough.
    # The Moore bound for (3,16) is 1 + 3*sum_{i=0}^{7} 2^i = 1 + 3*(2^8-1) = 1 + 3*255 = 766.
    # The known best is 936. We need q*10 >= some lower bound.
    # Let's try various prime orders q and see what girth we get.
    # For a voltage graph, the girth is related to the shortest closed walk in the base graph
    # whose voltage product is the identity.
    
    # We'll use cyclic groups Z_q for prime q.
    # Try primes: 5, 7, 11, 13, 17, 19, 23, 29, 31, ...
    # For q=5: n=50, too small
    # For q=11: n=110, too small
    # For q=19: n=190, too small
    # For q=29: n=290, too small
    # For q=31: n=310, too small
    # For q=37: n=370, too small
    # For q=41: n=410, too small
    # For q=43: n=430, too small
    # For q=47: n=470, too small
    # For q=53: n=530, close to Moore bound
    # For q=59: n=590
    # For q=61: n=610
    # For q=67: n=670
    # For q=71: n=710
    # For q=73: n=730
    # For q=79: n=790
    # For q=83: n=830
    # For q=89: n=890
    # For q=97: n=970
    # For q=101: n=1010
    # For q=103: n=1030
    # For q=107: n=1070
    # For q=109: n=1090
    # For q=113: n=1130
    # For q=127: n=1270
    
    # The key insight: we need the shortest closed walk in the Petersen graph with trivial voltage.
    # The Petersen graph has girth 5, so there are 5-cycles. If a 5-cycle has trivial voltage,
    # we get a 5-cycle in the lift. We need to avoid this.
    
    # Strategy: Try multiple primes q and random voltage assignments, checking the girth.
    # Return the best (smallest n) graph with girth >= g.
    
    primes_to_try = [5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127]
    
    best_n = float('inf')
    best_edges = None
    
    for q in primes_to_try:
        if time.time() > deadline - 10:
            break
        
        # Try multiple random voltage assignments for this q
        max_trials = 200
        for trial in range(max_trials):
            if time.time() > deadline - 5:
                break
            
            # Assign random voltages to each edge
            voltages = {}
            for u, v in petersen_edges:
                voltages[(u, v)] = random.randint(1, q - 1)  # non-identity voltages
                voltages[(v, u)] = q - voltages[(u, v)]  # inverse
            
            # Build the lifted graph
            n = 10 * q
            edges = []
            for (u, v), vol in voltages.items():
                if u > v:  # avoid duplicates
                    continue
                for i in range(q):
                    u_lift = u * q + i
                    v_lift = v * q + ((i + vol) % q)
                    edges.append((u_lift, v_lift))
            
            # Check girth using non-backtracking BFS from each vertex
            # This is expensive, so we do a quick check first
            if _quick_girth_check(n, edges, g):
                # More detailed check
                if _verify_girth_full(n, edges, g):
                    if n < best_n:
                        best_n = n
                        best_edges = edges
                        # If we found a good one, we can try to improve
                        if n <= 936:
                            # Good result, continue searching for better
                            pass
                        else:
                            # Keep searching
                            pass
    
    # If we found a valid graph, return it
    if best_edges is not None and best_n < float('inf'):
        return best_n, best_edges
    
    # Fallback: return LUW construction
    return _luw_construct(k, g, deadline)


def _quick_girth_check(n, edges, target_girth):
    """Quick check: verify degree and do a limited BFS for short cycles."""
    # Build adjacency
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Check degree
    for i in range(n):
        if len(adj[i]) != 3:
            return False
    
    # Check for very short cycles (length 3, 4, 5)
    # For efficiency, only check a sample of vertices
    sample_size = min(n, 100)
    sampled = random.sample(range(n), sample_size)
    
    for start in sampled:
        # BFS up to depth target_girth//2
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        
        while queue:
            u = queue.popleft()
            if dist[u] >= target_girth // 2:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u] and u != parent[v]:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < target_girth:
                        return False
    
    return True


def _verify_girth_full(n, edges, target_girth):
    """Full verification of girth using non-backtracking walks."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Non-backtracking BFS from each vertex
    for start in range(n):
        # State: (vertex, previous_vertex)
        # dist[(u, v)] = length of shortest non-backtracking walk from (start, -1) to (u, v)
        # We want to find if there's a cycle of length < target_girth
        
        # Use a dictionary for efficiency
        visited = {}
        queue = deque()
        
        # Start with all neighbors of start
        for v in adj[start]:
            state = (v, start)
            visited[state] = 1
            queue.append(state)
        
        while queue:
            u, prev = queue.popleft()
            d = visited[(u, prev)]
            
            if d >= target_girth - 1:
                continue
            
            for w in adj[u]:
                if w == prev:
                    continue  # non-backtracking
                
                new_state = (w, u)
                cycle_len = d + 1
                
                if w == start:
                    # Found a cycle
                    if cycle_len < target_girth:
                        return False
                elif new_state not in visited:
                    visited[new_state] = cycle_len
                    queue.append(new_state)
                else:
                    # Already visited, check if this creates a shorter cycle
                    # This is complex; for now, skip
                    pass
    
    return True


def _luw_construct(k, g, deadline):
    """LUW construction fallback."""
    start_time = time.time()
    
    q = k
    d = None
    for candidate_d in range(3, 30):
        if candidate_d % 2 == 1:
            girth_bound = candidate_d + 5
        else:
            girth_bound = candidate_d + 4
        if girth_bound >= g:
            d = candidate_d
            break
    
    if d is None:
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
    # Define field operations
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
    
    # Define the rule for each coordinate
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        options = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)]
        rule[c] = options[r]
    
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
    
    # BFS from the zero vertex to get its connected component
    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    for i, vertex in enumerate(order):
        if time.time() > deadline - 5:
            break
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    
    n = len(order)
    return n, edges

# EVOLVE-BLOCK-END
