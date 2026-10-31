"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
from collections import deque, defaultdict

def _verify_girth_fast(adj, n, g):
    """Check no cycle shorter than g using non-backtracking BFS from each vertex.
    Returns True if girth >= g."""
    # For each starting vertex, do a BFS tracking (node, prev) states
    # A cycle of length L is found when we revisit a state (node, prev)
    # at depth >= L, but we need the cycle to not backtrack.
    # Simpler: for each edge (u,v), check if there's a path of length g-2 from v to u
    # that doesn't use edge (u,v) in the forward direction.
    # Actually, use the standard method: BFS from each directed edge.
    
    # For girth >= g, we need: for every directed edge (u->v), 
    # the shortest non-backtracking walk from v back to u has length >= g-1.
    # Equivalently, in the non-backtracking graph, dist((u,v), (v,u)) >= g-1.
    
    # Efficient check: for each vertex s, do BFS in the non-backtracking state space.
    # State = (current_node, prev_node). Start from all neighbors of s.
    # If we reach (s, x) where x != prev_start, we found a cycle.
    
    for s in range(n):
        # BFS from s in non-backtracking state space
        # dist[(node, prev)] = distance from s
        visited = {}  # (node, prev) -> dist
        queue = deque()
        for nb in adj[s]:
            state = (nb, s)
            visited[state] = 1
            queue.append(state)
        
        while queue:
            node, prev = queue.popleft()
            dist = visited[(node, prev)]
            if dist >= g:
                break  # No cycle shorter than g found from this direction
            for nb in adj[node]:
                if nb == prev:
                    continue  # non-backtracking
                if nb == s:
                    # Found a cycle of length dist + 1
                    if dist + 1 < g:
                        return False
                else:
                    state2 = (nb, node)
                    if state2 not in visited:
                        visited[state2] = dist + 1
                        queue.append(state2)
    return True

def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar graph construction (fallback)."""
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

def _build_lift(base_adj, base_n, voltages, k):
    """Build a voltage lift of the base graph.
    base_adj: list of lists of neighbors
    voltages: for each edge (u,v) in base (u<v), a permutation of Z_k
    Returns (n, edges) or None if not k-regular or not simple."""
    n = base_n * k
    # For each base vertex u and fiber element a, the lifted vertex is (u, a) -> u*k + a
    # Edge in lift: (u, a) connected to (v, perm_uv(a)) where perm_uv is the voltage on edge (u,v)
    # We need to handle both directions: if base has edge u-v with voltage pi (u->v),
    # then v->u has voltage pi^{-1}.
    
    # Build permutation for each directed edge
    # voltages: dict from (min(u,v), max(u,v)) to a tuple of k values (permutation)
    # For edge (u,v) with u < v, voltages[(u,v)][a] = b means (u,a) connects to (v,b)
    # Then (v,a) connects to (u, pi^{-1}(a))
    
    edge_set = set()
    edges = []
    deg = [0] * n
    
    for u in range(base_n):
        for v in base_adj[u]:
            if u < v:
                perm = voltages.get((u, v))
                if perm is None:
                    return None
                # u->v direction
                for a in range(k):
                    b = perm[a]
                    su = u * k + a
                    sv = v * k + b
                    if su == sv:
                        return None
                    if (su, sv) in edge_set or (sv, su) in edge_set:
                        return None
                    edge_set.add((su, sv))
                    edges.append((su, sv))
                    deg[su] += 1
                    deg[sv] += 1
                # v->u direction (inverse permutation)
                inv_perm = [0] * k
                for a in range(k):
                    inv_perm[perm[a]] = a
                for a in range(k):
                    b = inv_perm[a]
                    sv = v * k + a
                    su = u * k + b
                    if su == sv:
                        return None
                    if (su, sv) in edge_set or (sv, su) in edge_set:
                        return None
                    edge_set.add((su, sv))
                    edges.append((su, sv))
                    deg[su] += 1
                    deg[sv] += 1
            elif v < u:
                perm = voltages.get((v, u))
                if perm is None:
                    return None
                # v->u direction: (v,a) connects to (u, perm(a))
                # We're iterating from u's perspective, so u->v is the inverse
                inv_perm = [0] * k
                for a in range(k):
                    inv_perm[perm[a]] = a
                for a in range(k):
                    b = inv_perm[a]
                    su = u * k + a
                    sv = v * k + b
                    if su == sv:
                        return None
                    if (su, sv) in edge_set or (sv, su) in edge_set:
                        return None
                    edge_set.add((su, sv))
                    edges.append((su, sv))
                    deg[su] += 1
                    deg[sv] += 1
    
    # Check regularity
    for i in range(n):
        if deg[i] != k:
            return None
    
    return n, edges

def _random_perm(k):
    p = list(range(k))
    random.shuffle(p)
    return tuple(p)

def construct(k, g):
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    # First try LUW as baseline
    best_n, best_edges = _luw_construct(k, g)
    
    # Now try voltage lifts on small base graphs
    # The base graph should be k-regular (or we can use a multigraph base)
    # Actually, for a k-fold voltage lift to be k-regular, the base should be k-regular.
    # Let's try small k-regular base graphs.
    
    # For k=3: try base graphs with 3-6 vertices that are 3-regular
    # 3-regular on 4 vertices: K4
    # 3-regular on 6 vertices: K_{3,3}
    # 3-regular on 8 vertices: cube, etc.
    
    base_graphs = []
    
    if k == 3:
        # K4
        base_graphs.append(([1,2,3],[0,2,3],[0,1,3],[0,1,2]))
        # K_{3,3}
        base_graphs.append(([3,4,5],[4,5,3],[5,3,4],[0,1,2],[0,1,2],[0,1,2]))
        # Cube (8 vertices)
        base_graphs.append(([1,2,4],[0,3,5],[0,3,6],[1,2,7],[0,5,6],[1,4,7],[2,4,7],[3,5,6]))
    
    for base_adj in base_graphs:
        base_n = len(base_adj)
        # Generate random voltage assignments
        # Number of edges in base
        base_edges = set()
        for u in range(base_n):
            for v in base_adj[u]:
                if u < v:
                    base_edges.add((u, v))
        
        max_attempts = 200
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            # Assign random permutations to each base edge
            voltages = {}
            for (u, v) in base_edges:
                voltages[(u, v)] = _random_perm(k)
            
            result = _build_lift(base_adj, base_n, voltages, k)
            if result is None:
                continue
            
            n, edges = result
            if n < best_n:
                # Verify girth
                adj = [[] for _ in range(n)]
                for (u, v) in edges:
                    adj[u].append(v)
                    adj[v].append(u)
                
                if _verify_girth_fast(adj, n, g):
                    best_n = n
                    best_edges = edges
    
    return best_n, best_edges
# EVOLVE-BLOCK-END
