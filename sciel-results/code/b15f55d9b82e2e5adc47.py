"""Voltage graph construction over K_{5,5} for 5-regular graph of girth >= 9."""
import random
import itertools
from collections import deque

# EVOLVE-BLOCK-START
def _bfs_nonbacktracking_girth(n, adj):
    """Compute girth exactly via BFS over non-backtracking walks from every vertex.
    Returns the length of the shortest cycle, or a large number if none found up to depth limit.
    We use the standard trick: BFS on (vertex, prev) states, but we must be careful.
    Actually, for exact girth we do BFS on the directed line graph (non-backtracking).
    A cycle of length L corresponds to a non-backtracking closed walk of length L.
    The shortest such closed walk gives the girth.
    We do multi-source BFS: for each edge (u,v), start a walk at v having come from u.
    If we reach a state (w, x) where x == start_vertex, we found a cycle.
    """
    # Build adjacency list
    # BFS on states (current, prev) where we came from prev to current
    # For each directed edge (prev -> current), we track the distance (length of walk so far)
    # and the origin (the prev of the first step, i.e., the vertex we started from on the other side)
    # Actually, standard approach: 
    # For each edge e = (u, v), consider the state (v, u) at distance 1.
    # We want to find the shortest cycle: a path v0->v1->...->vk=v0 with no immediate backtracking.
    # This is equivalent to: find the shortest path in the state space where state is (current, prev),
    # starting from all (v, u) for each edge (u,v), and we want to reach a state (u, v) where 
    # the walk started from (v, u) -- i.e., we've closed a loop.
    
    # More efficient: for each vertex v, do BFS on non-backtracking walks starting from v,
    # but that's O(n^2). Instead, use the following:
    # The girth is the minimum over all edges (u,v) of the shortest non-backtracking 
    # closed walk containing that edge. 
    # 
    # Standard method: For each directed edge (u->v), do a BFS in the non-backtracking 
    # state space. The state is (current, prev). Start: (v, u) at distance 1.
    # Transition: from (cur, prev), go to (nxt, cur) for all nxt in adj[cur] with nxt != prev.
    # We detect a cycle when we reach a state (u, v) where the walk started from (v, u)
    # -- but tracking origin is expensive.
    #
    # Alternative: The girth equals the minimum, over all vertices v, of the girth 
    # computed by BFS from v in the non-backtracking sense. But let's use a cleaner method.
    
    # Clean method: For each vertex s, run BFS on states (v, prev) where the walk 
    # starts at s. Actually, let's just do: for each edge (u,v), compute the shortest 
    # non-backtracking path from v back to u that doesn't immediately backtrack, 
    # with the constraint that the first step is from u to v.
    
    # I'll use the approach: girth = min over all (u,v) edges of dist_nonbacktracking(v, u) 
    # where the path starts by going from u to v, and we need to return to u without 
    # immediately backtracking at any step. The total cycle length is dist + 1... 
    # No, let me think again.
    
    # Let me use a well-known algorithm: 
    # For each vertex v, perform BFS on the "non-backtracking" state space.
    # State: (current_vertex, previous_vertex). 
    # Initialize: for each neighbor u of v, state (u, v) at distance 1.
    # From state (cur, prev), for each nxt in adj[cur] with nxt != prev, 
    #   if nxt == v and distance >= 3: found a cycle of length distance + 1? 
    #   No: the walk is v -> ... -> cur -> nxt. If nxt == v, cycle length = distance + 1.
    #   But we need distance >= 2 to avoid the trivial 2-cycle (which doesn't exist in simple graphs).
    #   Actually in a simple graph, the shortest cycle through v has length >= 3.
    #   We stop when we first reach v (other than the start).
    
    # This is O(n * (n + m)) which for n up to ~1000 is fine.
    
    best = 10**9
    for s in range(n):
        # BFS from s: states are (cur, prev), dist
        # Start: for each neighbor u of s, (u, s) at dist 1
        visited = {}  # (cur, prev) -> dist
        queue = deque()
        for u in adj[s]:
            state = (u, s)
            if state not in visited:
                visited[state] = 1
                queue.append(state)
        while queue:
            cur, prev = queue.popleft()
            d = visited[(cur, prev)]
            for nxt in adj[cur]:
                if nxt == prev:
                    continue
                if nxt == s:
                    # Cycle found: s -> ... -> cur -> s, length = d + 1
                    cycle_len = d + 1
                    if cycle_len < best:
                        best = cycle_len
                    # Don't enqueue, we've closed the loop
                    continue
                state = (nxt, cur)
                if state not in visited:
                    visited[state] = d + 1
                    queue.append(state)
    return best

def _voltage_lift_girth(m, voltages, base_adj):
    """Given voltage assignment on K_{5,5} edges, compute girth of the lift.
    base_adj: list of lists, base_adj[i] = list of (j, voltage) for edges from i to j.
    The lift has vertices (v, t) where v in 0..9, t in 0..m-1.
    Edge ((v,t), (w, t+voltage)) for each edge (v,w) with voltage.
    """
    n_base = 10
    n_lift = n_base * m
    
    # Build adjacency for the lift
    adj = [[] for _ in range(n_lift)]
    for u in range(n_base):
        for (v, vol) in base_adj[u]:
            # Edge from u to v with voltage vol
            # In the lift: (u, t) connects to (v, t+vol mod m) for all t
            for t in range(m):
                a = u * m + t
                b = v * m + ((t + vol) % m)
                adj[a].append(b)
                adj[b].append(a)
    
    # Compute girth
    girth = _bfs_nonbacktracking_girth(n_lift, adj)
    return girth

def construct(k, g):
    # We're targeting k=5, g=9. For other values, fall back to a simple construction.
    if k != 5:
        # Fallback: use a simple cycle-based construction or return something valid
        # For now, handle k=5 specifically; for other k, use a crude method
        # This is a placeholder for robustness
        raise ValueError("Only k=5 is supported in this construction")
    
    # Target girth is 9
    target_girth = g  # should be 9
    
    # Search over m and voltage assignments
    # K_{5,5} has 25 edges. We assign voltages in Z_m.
    # By symmetry, we can fix some voltages.
    # Let's try small m values first.
    
    best_n = 10**9
    best_edges = None
    
    for m in range(2, 60):  # Try m from 2 up to 59
        n_total = 10 * m
        if n_total >= best_n:
            break
        
        # We need to find voltage assignments such that girth >= target_girth
        # Strategy: random search over voltage assignments
        # Fix voltage of edge (0,5) to 0 by symmetry (we can shift all voltages)
        # Also fix voltage of edge (0,6) to 0 by automorphism? 
        # Actually, the automorphism group of K_{5,5} allows us to permute the 5 vertices on each side.
        # We can fix: edge (0,5) voltage = 0.
        # And by permuting the right side, fix edge (0,6) voltage = 0? No, that might not be independent.
        
        # Let's just do random search with some structure.
        # There are 25 edges. Let's label them.
        # Left: 0,1,2,3,4; Right: 5,6,7,8,9
        # Edges: (i, j) for i in 0..4, j in 5..9
        
        # For efficiency, fix voltages of a few edges:
        # Fix (0,5) = 0 (global shift symmetry)
        # Fix (0,6) = 0 (permutation of right side fixing 5)
        # Fix (1,5) = 0 (permutation of left side fixing 0)
        # That leaves 22 free variables.
        
        # Actually, let's be more careful. The voltage assignment is defined up to:
        # 1. Adding a constant to all voltages (shift)
        # 2. Permuting left vertices (changes which edges have which voltages)
        # 3. Permuting right vertices
        # 4. For each vertex, adding a constant to all incident edges (gauge transformation)
        #    -- this doesn't change the girth!
        
        # The gauge freedom means we can set one edge incident to each vertex to 0.
        # So we can fix: for each left vertex i, edge (i, 5+i) has voltage 0? 
        # Let's fix: edge (i, 5) has voltage 0 for all i in 0..4. (5 edges fixed)
        # Then we have 20 remaining edges to assign.
        
        # Let's try random assignments for the remaining 20 edges.
        fixed_edges = [(i, 5) for i in range(5)]  # voltage 0
        free_edges = [(i, j) for i in range(5) for j in range(6, 10)]
        
        # Number of random trials
        n_trials = 2000
        found = False
        
        for trial in range(n_trials):
            # Random assignment
            vols = {}
            for (i, j) in fixed_edges:
                vols[(i, j)] = 0
                vols[(j, i)] = 0  # undirected, but we'll store both directions
            for (i, j) in free_edges:
                v = random.randint(0, m - 1)
                vols[(i, j)] = v
                vols[(j, i)] = (-v) % m
            
            # Build base_adj
            base_adj = [[] for _ in range(10)]
            for i in range(5):
                for j in range(5, 10):
                    vol = vols[(i, j)]
                    base_adj[i].append((j, vol))
                    base_adj[j].append((i, vols[(j, i)]))
            
            # Compute girth
            girth = _voltage_lift_girth(m, vols, base_adj)
            
            if girth >= target_girth:
                # Build the lift edges
                n_lift = 10 * m
                edges = []
                for i in range(5):
                    for j in range(5, 10):
                        vol = vols[(i, j)]
                        for t in range(m):
                            a = i * m + t
                            b = j * m + ((t + vol) % m)
                            if a < b:
                                edges.append((a, b))
                            else:
                                edges.append((b, a))
                
                # Deduplicate and verify
                edge_set = set()
                unique_edges = []
                for (a, b) in edges:
                    if a == b:
                        continue
                    key = (min(a, b), max(a, b))
                    if key not in edge_set:
                        edge_set.add(key)
                        unique_edges.append(key)
                
                # Verify degree
                deg = [0] * n_lift
                for (a, b) in unique_edges:
                    deg[a] += 1
                    deg[b] += 1
                if all(d == k for d in deg):
                    best_n = n_lift
                    best_edges = unique_edges
                    found = True
                    break
        
        if found:
            break
    
    if best_edges is None:
        # Fallback: return a valid but possibly larger graph
        # Use the Lazebnik-Ustimenko-Woldar construction from the seed
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
    
    return best_n, best_edges
# EVOLVE-BLOCK-END
