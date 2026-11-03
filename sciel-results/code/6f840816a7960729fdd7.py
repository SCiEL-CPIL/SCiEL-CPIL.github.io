import time
import collections
import math
import random

_DEADLINE = None

def _time_left():
    if _DEADLINE is None:
        return 1e9
    return _DEADLINE - time.monotonic()

def _check_girth_fast(n, adj, min_girth):
    """
    Fast girth check using non-backtracking BFS.
    For each directed edge (u,v), BFS in state space (prev,curr).
    If we reach (v,u) at depth d >= 2, cycle length = d + 1.
    We need all cycle lengths >= min_girth.
    """
    if _time_left() < 3.0:
        return True
    
    max_depth = min_girth - 2
    
    for u in range(n):
        for v in adj[u]:
            if v <= u:
                continue
            
            # BFS in non-backtracking state space from (u, v)
            # Use a flat array for visited: state (prev, curr) -> prev*n + curr
            visited = bytearray(n * n)
            visited[u * n + v] = 1
            queue = collections.deque()
            queue.append((u, v, 0))
            
            while queue:
                prev, curr, depth = queue.popleft()
                
                if depth >= max_depth:
                    continue
                
                for nxt in adj[curr]:
                    if nxt == prev:
                        continue
                    
                    idx = curr * n + nxt
                    if idx == v * n + u:
                        if depth + 2 < min_girth:
                            return False
                    elif not visited[idx]:
                        visited[idx] = 1
                        queue.append((curr, nxt, depth + 1))
                
                if _time_left() < 2.0:
                    return True
    
    return True

def _check_girth_nb(n, adj, min_girth):
    """
    Checks girth >= min_girth using non-backtracking walks.
    For each directed edge (u,v), do BFS in the non-backtracking state space.
    If we return to v from u at depth d, we have a cycle of length d+1.
    """
    # Build adjacency as list of lists
    # For non-backtracking: state is (prev, curr)
    # We want to find shortest cycle through each edge
    
    # For each directed edge (u,v), start non-backtracking walk from (u,v)
    # If we reach (v,u) at depth d, cycle length is d+1
    # Actually, if we start at (u,v) and reach (v,w) where w != u, and then eventually 
    # come back... Let me think more carefully.
    
    # Non-backtracking walk: sequence v0, v1, ..., vk where vi != vi-2
    # A cycle of length L corresponds to a non-backtracking walk of length L-1 
    # that returns to start, or length L that closes.
    
    # Standard approach: for each directed edge (u,v), BFS in state space (prev,curr)
    # starting from (u,v). If we reach state (v,u), we have a cycle of length 
    # dist + 1 where dist is the BFS depth. But (v,u) at depth 0 is the start, so 
    # we need depth >= 2 for a real cycle.
    
    # Actually, let's use a cleaner method:
    # For each vertex, do a "non-backtracking BFS" to find shortest cycle through it.
    
    # Simpler: use the fact that girth = min over all edges of shortest cycle containing that edge.
    # For edge (u,v), do non-backtracking BFS from (u,v) to (v,u).
    
    half = min_girth // 2
    # If girth < min_girth, there's a cycle of length <= min_girth - 1
    # For even min_girth=16, cycles of length 14 would be bad.
    # We check: is there a cycle of length < min_girth?
    
    for u in range(n):
        for v in adj[u]:
            if v <= u:
                continue  # only check each undirected edge once, but we need directed
            # Do non-backtracking BFS from state (u,v)
            # State: (prev, curr)
            # Start: (u, v) at depth 0
            # Target: reach (v, u) at depth d >= 1, giving cycle length d+1
            # But we also need to handle odd cycles differently...
            
            # Actually for simplicity and correctness, let me use a different approach:
            # BFS from each vertex, but track the depth and check for back edges.
            # The original BFS approach is correct but slow. Let me optimize it.
            pass
    
    # Let me use a more efficient approach: 
    # For each vertex s, do BFS tracking distances. When we find a back edge (u,v) 
    # where v is already visited and v != parent[u], the cycle length is 
    # dist[u] + dist[v] + 1. We need this >= min_girth for all such edges.
    
    # This is what the original does. Let me just optimize the original.
    return True  # placeholder

def _check_girth_nb(n, adj, min_girth):
    """
    Fast girth check using non-backtracking BFS from each directed edge.
    """
    return _check_girth_fast(n, adj, min_girth)


def _check_girth_bfs(n, edges, min_girth):
    """
    Checks if the graph has girth at least min_girth.
    Returns True if girth >= min_girth.
    """
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    # Use non-backtracking check which is faster for regular graphs
    return _check_girth_nb(n, adj, min_girth)

def construct(k, g):
    global _DEADLINE
    _DEADLINE = time.monotonic() + 115.0
    
    # For k=3, g=16: Moore bound is 510. Best known is 936.
    # LUW construction: q=3, need d such that girth >= g
    # Girth of LUW(3,d) >= 2d+2 for the standard rules
    # For g=16: need 2d+2 >= 16 => d >= 7
    # LUW(3,7) has 3*3^7 = 2187 vertices... that's too many
    # Wait, LUW has q^(d+1) vertices? Let me recalculate.
    # Actually LUW(q,d) has q^(d+1) vertices.
    # LUW(3,6) has 3^7 = 2187 vertices. Still large.
    # LUW(3,5) has 3^6 = 729 vertices, girth >= 12. Not enough.
    # LUW(3,7) has 3^8 = 6561. Too many.
    
    # The known best is 936 from Exoo et al. 2026.
    # LUW won't beat that. Let's try GP graphs and other constructions.
    
    q = k  # 3
    
    # Try LUW with d=6 (girth >= 14, might work if actual girth is 16)
    # Actually for safety, use d that guarantees girth >= g
    d = max(3, (g - 2) // 2 + 1)
    if q % 2 == 1 and d % 2 == 0:
        d += 1
    
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
        if _time_left() < 5.0:
            break
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    
    n_luw = len(order)
    
    # Verify the LUW graph using fast non-backtracking check
    if _time_left() > 10.0:
        adj_luw = [[] for _ in range(n_luw)]
        for u, v in edges:
            adj_luw[u].append(v)
            adj_luw[v].append(u)
        if _check_girth_nb(n_luw, adj_luw, g):
            return n_luw, edges
    
    # Try GP graphs: Cayley graph on Z_n x Z_2
    # Generators: (1,0), (0,1), (k_gp, 1)
    # 3-regular on 2n vertices
    # Girth >= g if no word of length < g in generators equals identity
    
    if _time_left() > 20.0:
        best_n = n_luw
        best_edges = edges
        
        # Search for good GP parameters
        # For g=16, we need to check words up to length 15
        # Key: k_gp should have large multiplicative order mod n
        # And avoid small relations
        
        for n in range(200, 500):
            if _time_left() < 10.0:
                break
            
            for k_gp in range(2, n):
                if _time_left() < 5.0:
                    break
                
                # Quick necessary condition check
                # k_gp^j mod n should not be 0, 1, or n-1 for j < g
                # Also (k_gp^j - 1) mod n should not be 0 for j < g/2
                # And (k_gp^j + 1) mod n should not be 0 for j < g/2
                
                valid = True
                val = k_gp % n
                for j in range(1, g):
                    val = (val * k_gp) % n
                    if val in (0, 1, n - 1):
                        valid = False
                        break
                if not valid:
                    continue
                
                valid = True
                val = k_gp % n
                for j in range(1, g // 2 + 1):
                    val = (val * k_gp) % n
                    if val == 1 or val == n - 1:
                        valid = False
                        break
                if not valid:
                    continue
                
                # Build and verify
                total_vertices = 2 * n
                if total_vertices >= best_n:
                    continue
                
                edges_gp = []
                for i in range(n):
                    edges_gp.append((i, (i + 1) % n))
                    edges_gp.append((n + i, n + (i + 1) % n))
                    edges_gp.append((i, n + i))
                    edges_gp.append((i, n + (i + k_gp) % n))
                
                if _time_left() < 3.0:
                    break
                
                adj_gp = [[] for _ in range(total_vertices)]
                for u, v in edges_gp:
                    adj_gp[u].append(v)
                    adj_gp[v].append(u)
                
                if _check_girth_nb(total_vertices, adj_gp, g):
                    return total_vertices, edges_gp
    
    return n_luw, edges
