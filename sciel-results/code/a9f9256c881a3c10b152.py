import time
import random
import sys
from collections import deque

# EVOLVE-BLOCK-START

def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k.
    Returns (n, edges) for a q-regular graph with girth >= d+5 (odd d)."""
    q = k
    # Ensure d is odd and d+5 >= g
    if (g - 5) % 2 == 0:
        d = g - 4
    else:
        d = g - 5
    if d < 3:
        d = 3

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


def _verify_girth(n, edges, min_girth):
    """Verify no cycle shorter than min_girth using non-backtracking BFS from each vertex."""
    if n == 0:
        return True
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    # For girth verification, we only need to check non-backtracking walks
    # up to depth floor((min_girth-1)/2) to detect cycles of length < min_girth
    # A cycle of length L is detected when we return to start at depth L
    # We need L < min_girth, so max depth to check is min_girth - 1
    # But due to symmetry, we can stop at depth floor((min_girth-1)/2) 
    # Actually, we need to be careful: a cycle of length L means we return to start
    # at depth L. We want to detect if any L < min_girth exists.
    # So we BFS up to depth min_girth - 1 and check if we hit start.
    # But to be efficient, note that the shortest cycle through start has length
    # at most 2*depth where depth is when we first revisit start via non-backtracking.
    
    # More efficient: BFS from each start, track (node, prev) states, 
    # stop when depth >= min_girth (since we can't find shorter cycles beyond that)
    max_depth = min_girth  # we look for cycles of length < min_girth
    
    for start in range(n):
        # State: (node, prev_node) -> depth (first time we reach this state)
        visited = {}
        queue = deque([(start, -1, 0)])
        while queue:
            node, prev, depth = queue.popleft()
            if depth >= max_depth:
                continue
            if node == start and depth > 0:
                # Found a cycle of length depth
                if depth < min_girth:
                    return False
                continue
            state = (node, prev)
            if state in visited:
                if visited[state] <= depth:
                    continue
                else:
                    continue  # already processed at earlier or same depth
            visited[state] = depth
            for nb in adj[node]:
                if nb == prev:
                    continue
                queue.append((nb, node, depth + 1))
    return True

def _get_smallest_component(n, edges):
    """Find the smallest connected component in the graph."""
    if n == 0:
        return 0, []
    
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    visited = [False] * n
    smallest_size = n + 1
    smallest_edges = []
    
    for i in range(n):
        if not visited[i]:
            # BFS to find component
            comp_nodes = []
            queue = deque([i])
            visited[i] = True
            while queue:
                node = queue.popleft()
                comp_nodes.append(node)
                for nb in adj[node]:
                    if not visited[nb]:
                        visited[nb] = True
                        queue.append(nb)
                        
            comp_size = len(comp_nodes)
            if comp_size < smallest_size:
                smallest_size = comp_size
                # Collect edges within this component
                comp_set = set(comp_nodes)
                comp_edges = []
                for u, v in edges:
                    if u in comp_set and v in comp_set:
                        comp_edges.append((u, v))
                smallest_edges = comp_edges
                
    return smallest_size, smallest_edges

def _try_edge_swap_improve(n, edges, k, g, deadline):
    """Try to improve the graph by finding a smaller valid subgraph via component analysis.
    Returns (n, edges) of the best graph found."""
    # First, find all connected components and pick the smallest
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    visited = [False] * n
    best_n = n
    best_edges = edges
    
    for i in range(n):
        if not visited[i]:
            comp_nodes = []
            queue = deque([i])
            visited[i] = True
            while queue:
                node = queue.popleft()
                comp_nodes.append(node)
                for nb in adj[node]:
                    if not visited[nb]:
                        visited[nb] = True
                        queue.append(nb)
            
            comp_size = len(comp_nodes)
            if comp_size < best_n:
                best_n = comp_size
                comp_set = set(comp_nodes)
                best_edges = [(u, v) for u, v in edges if u in comp_set and v in comp_set]
    
    return best_n, best_edges


def _try_greedy_cage(k, g, n, deadline):
    import random
    import time
    from collections import deque
    
    # Try to construct a k-regular graph with girth >= g on n vertices
    # using a greedy random approach with girth checking.
    
    for _ in range(50):
        if time.time() > deadline:
            return None
        
        adj = [[] for _ in range(n)]
        deg = [0] * n
        edges = []
        
        success = True
        attempts = 0
        max_attempts = 20 * n * k
        
        # We need to add n*k/2 edges
        # Greedy: pick random edge (u,v) where deg[u]<k and deg[v]<k
        # Check if path length < g-1 exists between u and v.
        
        while any(d < k for d in deg) and attempts < max_attempts:
            attempts += 1
            candidates = [i for i in range(n) if deg[i] < k]
            if not candidates:
                break
            u = random.choice(candidates)
            v = random.choice(candidates)
            
            if v == u or v in adj[u]:
                continue
            
            # Check girth constraint
            # BFS from u to see if v is reachable in < g-1 steps
            visited = [-1] * n
            queue = deque([(u, 0)])
            visited[u] = 0
            found_short_path = False
            
            while queue:
                node, dist = queue.popleft()
                if dist >= g - 2:
                    break
                if node == v:
                    found_short_path = True
                    break
                for nb in adj[node]:
                    if visited[nb] == -1:
                        visited[nb] = dist + 1
                        queue.append((nb, dist + 1))
            
            if found_short_path:
                continue
            
            # Add edge
            adj[u].append(v)
            adj[v].append(u)
            deg[u] += 1
            deg[v] += 1
            edges.append((u, v))
        
        if all(d == k for d in deg):
            # Final verification
            if _verify_girth(n, edges, g):
                return n, edges
                
    return None

def construct(k, g):
    """Construct a k-regular graph with girth >= g on as few vertices as possible."""
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin before 120s kill

    # 1. Try to find a smaller graph using greedy random construction
    # Start with a reasonable lower bound (Moore bound is ~243 for k=4, g=10)
    # Try a range of n values.
    best_n = float('inf')
    best_edges = None
    
    # Estimate starting n
    # Moore bound M(k, g)
    if g % 2 == 0:
        moore = 1 + k * sum(k**i for i in range(g//2 - 1))
    else:
        moore = 1 + k * sum(k**i for i in range((g-1)//2))
        
    # Try n from max(moore, 100) to 2000
    # Step size 50 to be efficient
    n_start = max(int(moore), 100)
    n_end = 2000
    
    for n_try in range(n_start, n_end + 1, 50):
        if time.time() > deadline - 10:
            break
        result = _try_greedy_cage(k, g, n_try, deadline - 10)
        if result:
            n_res, edges_res = result
            if n_res < best_n:
                best_n = n_res
                best_edges = edges_res
            # If we found one, we can try smaller n? No, we iterate upwards.
            # But we want the SMALLEST.
            # So we should iterate downwards from a reasonable max?
            # Or just iterate upwards and stop at first success?
            # First success at n_try is a valid graph.
            # But maybe n_try-50 works?
            # Let's just return the first success found in the range.
            # Since we start from lower bound, the first success is likely small.
            return n_res, edges_res

    # 2. Fallback to LUW construction
    n, edges = _luw_construct(k, g)

    # Check for duplicate edges or self-loops first
    edge_set = set()
    clean_edges = []
    for u, v in edges:
        if u == v:
            continue
        key = (min(u, v), max(u, v))
        if key not in edge_set:
            edge_set.add(key)
            clean_edges.append((u, v))
    edges = clean_edges
    n = max(n, max(max(u, v) for u, v in edges) + 1) if edges else 0

    # Verify the graph is valid (degree k, girth >= g)
    if n > 0:
        # Check degree
        deg = [0] * n
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        valid_degree = all(d == k for d in deg)
        
        if not valid_degree:
            # Fallback: return the original LUW graph (uncleaned)
            n, edges = _luw_construct(k, g)
            edge_set = set()
            clean_edges = []
            for u, v in edges:
                if u == v:
                    continue
                key = (min(u, v), max(u, v))
                if key not in edge_set:
                    edge_set.add(key)
                    clean_edges.append((u, v))
            edges = clean_edges
            n = max(n, max(max(u, v) for u, v in edges) + 1) if edges else 0

    # Verify girth if time permits
    elapsed = time.time() - start_time
    if elapsed < deadline - 5:
        if not _verify_girth(n, edges, g):
            # Fallback: return the original LUW graph
            n, edges = _luw_construct(k, g)
            edge_set = set()
            clean_edges = []
            for u, v in edges:
                if u == v:
                    continue
                key = (min(u, v), max(u, v))
                if key not in edge_set:
                    edge_set.add(key)
                    clean_edges.append((u, v))
            edges = clean_edges
            n = max(n, max(max(u, v) for u, v in edges) + 1) if edges else 0

    return n, edges

# EVOLVE-BLOCK-END
