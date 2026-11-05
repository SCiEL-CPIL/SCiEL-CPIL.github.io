import time
import random
import sys
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = 145.0  # Leave 5s margin before 150s kill
    
    def luw_construct():
        """LUW (Lubell-Weichsel) construction for k-regular graphs of high girth."""
        q = k  # prime, q = 5
        # For k=5, g=9: need d such that 1 + k*(k-1)^(d-1) >= required
        # girth g means no cycles < g. For LUW, d relates to girth.
        # girth = 2d+1 or 2d depending on construction
        # For g=9 (odd), we want girth >= 9, so d=4 gives girth up to 9
        # Actually for LUW with parameter d, the girth is 2d+1
        # So for g=9, we need 2d+1 >= 9, so d >= 4
        d = (g + 1) // 2  # d=5 for g=9
        
        # But let's use the standard LUW construction
        # For k=5, g=9: q=5, d=4 should give girth 9
        # Let me use d such that the construction gives girth >= g
        # Standard LUW: vertices are (side, x_1, ..., x_d) where side in {0,1}
        # Neighbors: (1-side, y) where y is computed from x
        
        # For k=5 (prime), use arithmetic mod 5
        d = (g - 1) // 2  # d=4 for g=9, gives girth 2d+1=9
        
        rule = {}
        # Rule for LUW construction
        # For c=2: rule[2] = (1,1) meaning y_2 = x_2 + y_1 * x_1
        # For c>=3: rule[c] = (c-2, 1) meaning y_c = x_c + y_{c-2} * x_1
        # But this needs to be adjusted for proper girth
        
        # Standard LUW rule:
        # y_1 = first (free variable)
        # y_2 = x_2 + y_1 * x_1 (for side 0) or x_2 - x_1 * y_1 (for side 1)
        # y_c = x_c + y_{c-1} * x_{c-2} for c >= 3 (side 0)
        # y_c = x_c - x_{c-2} * y_{c-1} for c >= 3 (side 1)
        
        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0] * (d - 1)
                for c in range(2, d + 1):
                    if side == 0:
                        y[c] = (x[c] + y[c-1] * x[c-2]) % q
                    else:
                        y[c] = (x[c] - x[c-2] * y[c-1]) % q
                yield (1 - side, tuple(y))
        
        start = (0, (0,) * d)
        index = {}
        order = []
        edges = []
        
        index[start] = 0
        order.append(start)
        
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        
        return len(order), edges
    
    def verify_girth(n, edges, k, g):
        """Verify graph has girth >= g using non-backtracking walks from every vertex."""
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Check degree
        for i in range(n):
            if len(adj[i]) != k:
                return False
        
        # Check for cycles shorter than g using non-backtracking BFS
        # From each vertex, do BFS avoiding backtracking
        # If we find a cycle of length < g, return False
        
        # For efficiency, use the fact that in a k-regular graph,
        # non-backtracking walks from a vertex: at each step, we can go to any neighbor except the one we came from
        # So the number of non-backtracking walks of length L is k*(k-1)^(L-1)
        # If two such walks end at the same vertex, we have a cycle
        
        # Better: BFS with (vertex, prev_vertex) states
        # dist[(v, p)] = shortest non-backtracking path from start to v that arrived from p
        
        half_g = (g - 1) // 2  # We need to check up to this depth
        # Actually, a cycle of length L < g means there's a non-backtracking walk of length L that returns to start
        # Or more generally, two non-backtracking walks from different starts that meet
        
        # Simpler: for each vertex s, do BFS in the non-backtracking graph
        # State: (current, previous)
        # Start states: (s, -1) for all neighbors... actually start from s with no previous
        # From (v, p), we can go to (w, v) for all w in adj[v] where w != p
        
        # To detect cycles < g: from vertex s, do BFS in non-backtracking graph
        # If we reach s again with path length < g, we have a cycle
        # But we need to be careful: the cycle might not include s
        
        # Better approach: check all pairs of vertices
        # For each vertex s, compute shortest non-backtracking distance to all other vertices
        # If for any edge (u,v), dist_nonbt[s->u] + dist_nonbt[s->v] + 1 < g and the paths are disjoint...
        # This is complex. Let me use a simpler check.
        
        # For a graph to have girth >= g, it must have no cycles of length 3,4,...,g-1
        # We can check by: for each vertex s, do BFS in the "non-backtracking line graph"
        # and check if we can return to s in less than g steps
        
        # Actually, the standard way: a graph has girth >= g iff for every vertex v,
        # the number of non-backtracking walks of length L from v is exactly k*(k-1)^(L-1) for L < g
        # (i.e., no two walks of length < g end at the same vertex)
        
        # Let me do: for each vertex s, BFS in non-backtracking states up to depth half_g
        # and check for collisions
        
        max_depth = (g - 1) // 2  # depth 4 for g=9
        
        for s in range(n):
            # BFS from s in non-backtracking graph
            # State: (vertex, prev_vertex), prev_vertex = -1 for start
            visited = set()
            visited.add((s, -1))
            queue = deque([(s, -1, 0)])
            
            while queue:
                v, p, depth = queue.popleft()
                if depth >= max_depth:
                    continue
                for w in adj[v]:
                    if w == p:
                        continue  # non-backtracking
                    state = (w, v)
                    if state in visited:
                        # Found a cycle of length depth+1
                        if depth + 1 < g:
                            return False
                    else:
                        visited.add(state)
                        queue.append((w, v, depth + 1))
        
        return True
    
    # First try LUW construction
    n_luw, edges_luw = luw_construct()
    
    if time.time() > deadline - 5:
        return n_luw, edges_luw
    
    # Verify LUW graph
    if verify_girth(n_luw, edges_luw, k, g):
        # LUW works, try to improve with random search
        best_n, best_edges = n_luw, edges_luw
    else:
        # LUW failed, need fallback
        best_n, best_edges = None, None
    
    # Try random search for smaller graphs
    # Moore bound for k=5, g=9: 1 + 5*(4^4 - 1)/(4-1) = 1 + 5*255/3 = 1 + 425 = 426
    moore_bound = 1 + k * ((k - 1) ** ((g - 1) // 2) - 1) // (k - 2)
    
    if best_n is None:
        # LUW failed, use it as fallback
        best_n, best_edges = n_luw, edges_luw
    
    # Try to find smaller graphs
    search_start = max(moore_bound, 400)
    search_end = best_n - 1
    
    if search_end >= search_start and time.time() < deadline - 20:
        # Random search
        def random_k_regular(n, k):
            """Generate a random k-regular graph on n vertices."""
            if n * k % 2 != 0:
                return None
            # Configuration model
            half_edges = n * k // 2
            vertices = []
            for i in range(n):
                vertices.extend([i] * k)
            
            random.shuffle(vertices)
            
            adj = [[] for _ in range(n)]
            for i in range(half_edges):
                u = vertices[2*i]
                v = vertices[2*i+1]
                if u == v:
                    return None  # loop
                if v in adj[u]:
                    return None  # multi-edge
                adj[u].append(v)
                adj[v].append(u)
            
            edges = []
            for u in range(n):
                for v in adj[u]:
                    if u < v:
                        edges.append((u, v))
            
            return n, edges
        
        def quick_girth_check(n, edges, k, g):
            """Quick girth check - returns True if girth >= g."""
            adj = [[] for _ in range(n)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
            
            # Check for short cycles using BFS from each vertex
            # For girth >= 9, check cycles of length 3,4,5,6,7,8
            # Use non-backtracking BFS up to depth 4
            
            max_depth = (g - 1) // 2
            
            for s in range(n):
                # BFS in non-backtracking graph
                visited = set()
                visited.add((s, -1))
                queue = deque([(s, -1, 0)])
                
                while queue:
                    v, p, depth = queue.popleft()
                    if depth >= max_depth:
                        continue
                    for w in adj[v]:
                        if w == p:
                            continue
                        state = (w, v)
                        if state in visited:
                            if depth + 1 < g:
                                return False
                        else:
                            visited.add(state)
                            queue.append((w, v, depth + 1))
            
            return True
        
        # Try different n values
        for n in range(search_start, search_end + 1):
            if time.time() > deadline - 15:
                break
            
            if n * k % 2 != 0:
                continue
            
            # Try random graphs
            trials = 100
            for trial in range(trials):
                if time.time() > deadline - 15:
                    break
                
                result = random_k_regular(n, k)
                if result is None:
                    continue
                
                n_test, edges_test = result
                
                if quick_girth_check(n_test, edges_test, k, g):
                    best_n, best_edges = n_test, edges_test
                    break  # Found a valid graph with this n
            
            if best_n == n:
                break  # Found smallest possible
    
    return best_n, best_edges
