"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from collections import deque

# EVOLVE-BLOCK-START

def _moore_bound_lower(k, g):
    """Return a safe lower bound on n(k,g)."""
    if g % 2 == 0:
        return 2 * sum((k-1)**i for i in range(g//2))
    else:
        return 1 + k * sum((k-1)**i for i in range((g-1)//2))

def _verify_girth_fast(n, adj, k, g):
    """
    Fast girth verification using BFS from each vertex.
    Returns True if girth >= g.
    adj is a list of lists of neighbors.
    """
    # Early check for regularity and simplicity
    for i in range(n):
        if len(adj[i]) != k:
            return False
        if len(set(adj[i])) != k:
            return False
        # Check for self-loops
        if i in adj[i]:
            return False

    # For each vertex, do a BFS to find the shortest cycle.
    # We stop BFS when depth reaches g. If we find a cycle of length < g, return False.
    # Optimization: We can use a single array for visited depths and a timestamp to avoid clearing.
    
    # Since n is small (320), and g is 10, a simple BFS is fast enough.
    # We need to check for cycles of length < g.
    # A cycle of length L corresponds to a path of length L-1 from a neighbor of v back to v?
    # No, standard BFS from v finds shortest paths. If we find a neighbor of v at distance d < g-1, 
    # and that neighbor is not the parent, we have a cycle of length d+1.
    
    for v in range(n):
        # dist[u] = shortest distance from v to u
        dist = [-1] * n
        dist[v] = 0
        queue = deque([v])
        
        while queue:
            node = queue.popleft()
            d = dist[node]
            if d >= g - 1:
                continue
                
            for neighbor in adj[node]:
                if dist[neighbor] == -1:
                    dist[neighbor] = d + 1
                    queue.append(neighbor)
                else:
                    # If we see a visited node, it's part of a cycle.
                    # The length of the cycle is dist[node] + dist[neighbor] + 1?
                    # No, that's for undirected graphs if we consider the tree.
                    # Actually, if dist[neighbor] <= d, it's a back edge or cross edge.
                    # In an undirected graph BFS, if we see a visited node that is not the parent,
                    # we have a cycle. The length is dist[node] + 1 + (dist[neighbor] if neighbor is not parent? No).
                    # Let's stick to the non-backtracking walk definition or standard cycle detection.
                    
                    # Simpler: Just check if dist[neighbor] + 1 < g and neighbor is not the parent?
                    # Standard BFS doesn't track parents easily in a flat array without storing them.
                    
                    # Let's use the standard property: 
                    # If we find an edge (node, neighbor) where dist[neighbor] != -1, 
                    # the cycle length is dist[node] + dist[neighbor] + 1? 
                    # This is only true if the LCA is v. 
                    # Actually, any edge connecting two nodes at depths d1, d2 forms a cycle of length d1 + d2 + 1 
                    # only if they are in different branches. If they are in the same branch, it's shorter.
                    
                    # To be safe and correct, we use the non-backtracking walk logic from the original code 
                    # but optimized.
                    pass
        
        # The simple BFS above is tricky for exact girth without parent tracking.
        # Let's revert to the original logic but make it faster by breaking early.
        
        # Re-implementing the specific check for this function to be safe:
        visited = {}
        q = deque()
        for w in adj[v]:
            visited[(w, v)] = 1
            q.append((w, v, 1))
            
        while q:
            node, parent, depth = q.popleft()
            if depth >= g:
                continue
            for neighbor in adj[node]:
                if neighbor == parent:
                    continue
                if neighbor == v:
                    if depth + 1 < g:
                        return False
                    continue
                state = (neighbor, node)
                if state not in visited:
                    visited[state] = depth + 1
                    q.append((neighbor, node, depth + 1))
    return True

def _construct_luw(k, g):
    """Lazebnik-Ustimenko-Woldar construction - the baseline."""
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

def _construct_cage_voltage(k, g, deadline):
    """
    Attempt to construct the (4,10)-cage using voltage graph method.
    Base graph: Bipartite double cover of K5 (10 vertices, 4-regular).
    Voltage group: Z_32.
    Lift has 320 vertices.
    """
    if k != 4 or g != 10:
        return None

    n_base = 10
    n_lift = 320
    group_size = 32
    
    # Base graph edges: (i, 5+j) for i != j, i in 0..4, j in 0..4
    base_edges = []
    for i in range(5):
        for j in range(5):
            if i != j:
                base_edges.append((i, 5 + j))
                
    # Map edge to index for voltage array
    edge_indices = {e: i for i, e in enumerate(base_edges)}
    num_edges = len(base_edges)
    
    def build_adj(voltages):
        """Build adjacency list for the lift graph."""
        adj = [[] for _ in range(n_lift)]
        for idx, (u, v) in enumerate(base_edges):
            w = voltages[idx]
            for x in range(group_size):
                u_id = u * group_size + x
                v_id = v * group_size + (x + w) % group_size
                adj[u_id].append(v_id)
                adj[v_id].append(u_id)
        return adj

    def check_girth(voltages):
        adj = build_adj(voltages)
        return _verify_girth_fast(n_lift, adj, k, g)

    # Local Search
    rng = random.Random(42)
    best_voltages = None
    best_girth_found = 0
    
    # Initialize with random voltages
    current_voltages = [rng.randint(0, group_size - 1) for _ in range(num_edges)]
    
    # Try to improve girth
    # We don't calculate exact girth easily, so we just check if it meets the target.
    # If not, we can try to minimize the number of short cycles.
    # For simplicity, we do a random restart with local search.
    
    max_iterations = 2000
    current_time = time.time()
    
    for _ in range(max_iterations):
        if time.time() > deadline - 5:
            break
        
        if check_girth(current_voltages):
            adj = build_adj(current_voltages)
            edges = []
            for u in range(n_lift):
                for v in adj[u]:
                    if u < v:
                        edges.append((u, v))
            return n_lift, edges
        
        # Pick a random edge and change its voltage
        idx = rng.randint(0, num_edges - 1)
        old_val = current_voltages[idx]
        new_val = rng.randint(0, group_size - 1)
        if new_val == old_val:
            continue
        current_voltages[idx] = new_val
        
        # Accept if it doesn't make things worse? 
        # Since we can't easily measure "worse", we just accept and hope.
        # Or we can keep the best few.
        
        # To be more effective, we can try all possible voltages for the chosen edge 
        # and keep the one that allows the longest BFS depth before hitting a cycle?
        # That's expensive.
        
        # Let's just do random walk for now, but with a bias towards structured patterns.
        # Actually, random walk on 20 edges in Z_32 is very slow to find a specific girth 10 solution.
        
        # Better strategy: Use the known structure if possible, or a smarter heuristic.
        # The (4,10) cage is known to exist. 
        # A common construction uses the incidence graph of a generalized quadrangle or similar.
        # The voltage graph approach with K5 double cover is valid but finding the right voltages is hard.
        
        # Let's try a few specific "good" patterns first before random search.
        # Pattern: voltages[i] = (i * 7) % 32
        # Pattern: voltages[i] = (i * 11) % 32
        
    # If local search fails, try structured patterns
    for seed in range(100):
        if time.time() > deadline - 5:
            break
        voltages = [(i * 7 + seed) % group_size for i in range(num_edges)]
        if check_girth(voltages):
            adj = build_adj(voltages)
            edges = []
            for u in range(n_lift):
                for v in adj[u]:
                    if u < v:
                        edges.append((u, v))
            return n_lift, edges
            
    return None

def construct(k, g):
    """Construct a k-regular graph with girth >= g on as few vertices as possible."""
    deadline = time.time() + 130  # Safety margin before 150s limit
    
    # Try the specific cage construction for (4,10)
    if k == 4 and g == 10:
        result = _construct_cage_voltage(k, g, deadline)
        if result is not None:
            return result
            
    # Fallback to LUW construction
    n, edges = _construct_luw(k, g)
    
    return n, edges

# EVOLVE-BLOCK-END
