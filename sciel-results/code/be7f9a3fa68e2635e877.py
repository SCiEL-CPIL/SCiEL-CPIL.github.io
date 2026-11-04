import time
import random
from collections import deque

def construct(k, g):
    """Construct a 3-regular graph of girth >= g on as few vertices as possible."""
    start_time = time.time()
    deadline = 150.0  # evaluator kills at 150s
    
    # For k=3, g=16: use voltage lift of Petersen graph
    # Petersen graph: 10 vertices, 15 edges, girth 5
    # Lift with group G of size m -> 10*m vertices
    
    if k != 3:
        # Fallback: LUW construction for other k
        return _luw_fallback(k, g, deadline, start_time)
    
    if g < 16:
        # For smaller girth, we can use smaller groups
        pass
    
    # Petersen graph adjacency
    # Vertices: 0-4 (outer pentagon), 5-9 (inner star)
    # Outer: 0-1-2-3-4-0
    # Inner: 5-7-9-6-8-5
    # Spokes: 0-5, 1-6, 2-7, 3-8, 4-9
    pet_adj = [
        [1, 4, 5],   # 0
        [0, 2, 6],   # 1
        [1, 3, 7],   # 2
        [2, 4, 8],   # 3
        [3, 0, 9],   # 4
        [0, 7, 8],   # 5
        [1, 8, 9],   # 6
        [2, 5, 9],   # 7
        [3, 6, 7],   # 8
        [4, 5, 6],   # 9
    ]
    
    # For each undirected edge, we'll assign a voltage to one direction and inverse to the other
    # Edges (undirected):
    edges = [(0,1),(1,2),(2,3),(3,4),(4,0),
             (5,7),(7,9),(9,6),(6,8),(8,5),
             (0,5),(1,6),(2,7),(3,8),(4,9)]
    
    # We'll use cyclic group Z_m
    # Try different m values, find smallest that gives girth >= g
    
    # For a cyclic group, voltage of a closed walk is sum of voltages mod m
    # We need: no closed walk of length < g has sum ≡ 0 mod m
    
    # Enumerate non-backtracking closed walks in Petersen up to length g-1
    # State: (vertex, prev_vertex, length, voltage_sum)
    # Start: for each vertex v, for each neighbor u, walk v->u
    
    # To be efficient, we track for each (v, prev) the set of voltage sums seen at each length
    # Actually, we want to know: is there ANY closed walk of length L < g with trivial voltage?
    
    # Let's do BFS over (current, prev) states, tracking voltage sums
    # For each state, maintain a set of voltage sums modulo m
    # A closed walk of length L starting at v is: v->v1->...->v_{L-1}->v where no backtracking
    # The voltage is sum of edge voltages along the walk
    
    # For a given m, check if girth condition holds:
    # For all closed walks of length 2..g-1, voltage sum != 0 mod m
    
    # Efficient check: for each starting vertex v, do BFS in non-backtracking state space
    # State: (curr, prev), depth, and we track voltage sums
    # At depth L, if curr == v (start) and L >= 2, check if 0 is in the voltage set
    
    def check_girth_cyclic(m, max_depth):
        """Check if no closed walk of length < max_depth has trivial voltage in Z_m."""
        # For each start vertex, BFS non-backtracking
        for start in range(10):
            # State: (curr, prev) -> set of voltage sums at current depth
            # Initialize: for each neighbor of start, (neighbor, start) with voltage = volt[start][neighbor]
            # We need to know the voltage of directed edge (a, b)
            # volt[a][b] = voltage when going from a to b
            
            # Build voltage lookup: volt[u][v] for directed edges
            # We'll assign voltages after choosing m
            
            # Actually, we need to pass in the voltage assignment
            pass
        
        # This function will be called with a specific voltage assignment
        return True  # placeholder
    
    # Better approach: for a given m and random voltage assignment, check girth
    # Let's assign random voltages in Z_m to each directed edge (with volt[b][a] = -volt[a][b] mod m)
    
    def assign_random_voltages(m, seed):
        rng = random.Random(seed)
        volt = [[0]*10 for _ in range(10)]
        for u, v in edges:
            val = rng.randint(1, m-1)  # non-trivial voltage
            volt[u][v] = val % m
            volt[v][u] = (-val) % m
        return volt
    
    def check_girth_voltages(volt, m, max_depth):
        """Check if no closed walk of length < max_depth has trivial voltage."""
        # For each start vertex, do BFS in non-backtracking state space
        # State: (curr, prev) -> set of voltage sums (mod m) reached at current depth
        # We process depth by depth
        
        for start in range(10):
            # Depth 1: from start to each neighbor
            # State at depth 1: (neighbor, start) with voltage = volt[start][neighbor]
            # We need to track: at each depth, for each (curr, prev), what voltage sums are possible
            
            # Use a dict: (curr, prev) -> set of voltage sums
            # Process depth from 1 to max_depth-1
            
            # At depth L, state (curr, prev) means we're at curr, came from prev
            # Next: go to next_curr where next_curr != prev (non-backtracking)
            # New state: (next_curr, curr) with voltage + volt[curr][next_curr]
            
            # A closed walk of length L >= 2 starting at 'start' is detected when:
            # At depth L, curr == start (and prev != start, since non-backtracking means prev is the vertex before start in the walk)
            # Wait: the walk is start -> v1 -> v2 -> ... -> v_{L-1} -> start
            # At depth L, we're at start, came from v_{L-1}
            # State: (start, v_{L-1}) at depth L
            # We need L < max_depth (i.e., L <= max_depth - 1 = g - 1... wait, girth >= g means no cycle of length < g, so check L = 2, 3, ..., g-1)
            
            # Actually girth >= g means shortest cycle has length >= g, so no cycle of length 2, 3, ..., g-1
            # In a simple graph, no 2-cycles. So check L = 3, 4, ..., g-1
            # But non-backtracking closed walk of length L in a simple graph corresponds to a cycle of length L (if L >= 3)
            # Actually a non-backtracking closed walk of length L in a simple graph is a cycle of length L (since no repeated vertices in a non-backtracking walk of length < 2*something... actually no, it could revisit vertices)
            # Hmm, a non-backtracking closed walk can revisit vertices. But if it revisits a vertex, it contains a shorter cycle.
            # For girth checking, it suffices to check: is there a non-backtracking closed walk of length L < g with trivial voltage?
            # If such a walk exists, it contains a cycle of length <= L < g, and that cycle also has trivial voltage (since the voltage of the whole walk is trivial, but the cycle's voltage might not be...).
            # Actually no: the voltage of a closed walk is the sum of voltages. If the walk has trivial voltage but contains a sub-cycle, the sub-cycle's voltage need not be trivial.
            # So I need to check all non-backtracking closed walks, not just simple cycles.
            
            # Let's just do the BFS and check.
            
            # states: dict from (curr, prev) to set of voltage sums
            states = {}
            # Depth 1
            for nb in pet_adj[start]:
                v = volt[start][nb]
                states.setdefault((nb, start), set()).add(v)
            
            # Process depths 2 to max_depth-1
            for depth in range(2, max_depth):
                new_states = {}
                for (curr, prev), vsums in states.items():
                    for next_curr in pet_adj[curr]:
                        if next_curr == prev:
                            continue  # non-backtracking
                        new_v = volt[curr][next_curr]
                        new_vsums = set()
                        for vs in vsums:
                            nv = (vs + new_v) % m
                            new_vsums.add(nv)
                        key = (next_curr, curr)
                        if key in new_states:
                            new_states[key].update(new_vsums)
                        else:
                            new_states[key] = new_vsums
                
                # Check for closed walks: state (start, prev) at this depth
                if (start, None) in new_states:  # placeholder, won't match
                    pass
                # Check: is there a state (start, x) in new_states?
                for (curr, prev), vsums in new_states.items():
                    if curr == start:
                        if 0 in vsums:
                            return False  # found closed walk of length 'depth' with trivial voltage
                states = new_states
        
        return True
    
    # Now search for the smallest m and voltage assignment that works
    # m must be such that 10*m >= Moore bound = 512, so m >= 52
    # But we want smallest n = 10*m, so try m from 52 upward
    
    # Actually, we can try smaller m if the voltage assignment works
    # But n = 10*m, and we want n small. Moore bound is 512, so m >= 52.
    # Let's try m = 52 first, with random seeds
    
    best_result = None
    best_n = float('inf')
    
    # Try m values starting from 52
    for m in range(52, 100):
        if time.time() > deadline - 10:
            break
        
        # Try multiple random seeds
        found = False
        for seed in range(50):
            if time.time() > deadline - 10:
                break
            volt = assign_random_voltages(m, seed)
            if check_girth_voltages(volt, m, g):
                # Found a valid voltage assignment!
                # Build the lifted graph
                n = 10 * m
                edges_lifted = []
                # Vertex (v, t) where v in 0..9, t in 0..m-1
                # Edge between (u, t) and (v, t + volt[u][v] mod m) for each undirected edge (u,v)
                for u, v in edges:
                    for t in range(m):
                        tu = t
                        tv = (t + volt[u][v]) % m
                        a = u * m + tu
                        b = v * m + tv
                        edges_lifted.append((a, b))
                
                n_final = n
                best_result = (n_final, edges_lifted)
                best_n = n_final
                found = True
                break
        
        if found:
            # Verify the result
            if _verify_graph(best_result[0], best_result[1], k, g, deadline, time.time()):
                return best_result
            # If verification fails, continue searching
    
    # If no small m worked, fall back to LUW
    if time.time() > deadline - 5:
        if best_result:
            return best_result
        return _luw_fallback(k, g, deadline, start_time)
    
    # Try larger m values
    for m in range(100, 200):
        if time.time() > deadline - 10:
            break
        for seed in range(20):
            if time.time() > deadline - 10:
                break
            volt = assign_random_voltages(m, seed)
            if check_girth_voltages(volt, m, g):
                n = 10 * m
                edges_lifted = []
                for u, v in edges:
                    for t in range(m):
                        tv = (t + volt[u][v]) % m
                        a = u * m + t
                        b = v * m + tv
                        edges_lifted.append((a, b))
                result = (n, edges_lifted)
                if _verify_graph(n, edges_lifted, k, g, deadline, time.time()):
                    return result
    
    # Fallback
    if best_result:
        return best_result
    return _luw_fallback(k, g, deadline, start_time)


def _verify_graph(n, edges, k, g, deadline, current_time):
    """Verify the graph is k-regular, simple, and has girth >= g."""
    if current_time > deadline - 5:
        return False  # not enough time to verify
    
    # Check regularity
    deg = [0] * n
    edge_set = set()
    for u, v in edges:
        if u == v:
            return False  # loop
        key = (min(u,v), max(u,v))
        if key in edge_set:
            return False  # repeated edge
        edge_set.add(key)
        deg[u] += 1
        deg[v] += 1
    
    for d in deg:
        if d != k:
            return False
    
    # Check girth using non-backtracking BFS from each vertex
    # For each vertex, BFS in (curr, prev) state space, depth up to g-1
    # If we return to start at depth < g, girth violated
    
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    for start in range(n):
        if current_time > deadline - 5:
            return False
        # BFS: state (curr, prev), depth
        # visited: set of (curr, prev) at each depth
        # We need to check if start is reached at depth < g (depth >= 2)
        
        # Use a queue: (curr, prev, depth)
        # visited: set of (curr, prev) to avoid reprocessing
        visited = set()
        queue = deque()
        
        for nb in adj[start]:
            queue.append((nb, start, 1))
            visited.add((nb, start))
        
        while queue:
            curr, prev, depth = queue.popleft()
            if depth >= g:
                continue
            for next_curr in adj[curr]:
                if next_curr == prev:
                    continue
                if next_curr == start and depth + 1 < g:
                    return False  # found cycle of length depth+1 < g
                if (next_curr, curr) not in visited:
                    visited.add((next_curr, curr))
                    queue.append((next_curr, curr, depth + 1))
    
    return True


def _luw_fallback(k, g, deadline, start_time):
    """Fallback LUW construction."""
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
