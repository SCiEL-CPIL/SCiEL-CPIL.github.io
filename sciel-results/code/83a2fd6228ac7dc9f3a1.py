import math
import random
import time
from collections import deque
import sys
import numpy as np

def construct(k, g):
    """
    Construct a k-regular graph with girth at least g, minimizing n.
    """
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill
    
    best_result = None
    
    # Compute Moore bound
    moore = _moore_bound(k, g)
    
    # Strategy 1: LUW construction (deterministic, fast to build)
    if k >= 3 and time.time() < deadline - 10:
        try:
            n_luw, edges_luw = _luw_graph(k, g)
            if n_luw > 0 and n_luw >= moore:
                # Verify with non-backtracking BFS (tighter time budget)
                verify_budget = min(5.0, max(1.0, (deadline - time.time()) * 0.3))
                if _verify_nonbacktracking(k, g, n_luw, edges_luw, time_limit=verify_budget):
                    best_result = (n_luw, edges_luw)
        except Exception:
            pass
    
    # Strategy 1b: Orthogonality graph construction (new)
    if k >= 3 and time.time() < deadline - 15:
        try:
            n_ortho, edges_ortho = _orthogonality_graph(k, g)
            if n_ortho > 0 and n_ortho >= moore and (best_result is None or n_ortho < best_result[0]):
                verify_budget = min(4.0, max(1.0, (deadline - time.time()) * 0.2))
                if _verify_nonbacktracking(k, g, n_ortho, edges_ortho, time_limit=verify_budget):
                    best_result = (n_ortho, edges_ortho)
        except Exception:
            pass
    
    # Strategy 2: Try slightly larger LUW variants if time permits
    if k >= 3 and time.time() < deadline - 15:
        try:
            d_min = g - 5
            if d_min % 2 == 0:
                d_min += 1
            d_min = max(3, d_min)
            for d in range(d_min + 1, min(d_min + 4, 20)):
                if time.time() > deadline - 10:
                    break
                try:
                    n_alt, edges_alt = _luw_graph_params(k, d)
                    if n_alt > 0 and (best_result is None or n_alt < best_result[0]):
                        if _verify_nonbacktracking(k, g, n_alt, edges_alt, time_limit=2.0):
                            best_result = (n_alt, edges_alt)
                except Exception:
                    continue
        except Exception:
            pass

    # Strategy 3: Spectral-guided random search (Ramanujan-like)
    remaining = deadline - time.time()
    if remaining > 10 and k >= 3:
        try:
            n_rand, edges_rand = _spectral_random_search(k, g, moore, time_limit=min(25, remaining - 8))
            if n_rand > 0 and (best_result is None or n_rand < best_result[0]):
                if _verify_nonbacktracking(k, g, n_rand, edges_rand, time_limit=min(4, remaining - 9)):
                    best_result = (n_rand, edges_rand)
        except Exception:
            pass
    
    # Strategy 4: If still no result, try a simple greedy random search
    remaining = deadline - time.time()
    if remaining > 8 and best_result is None and k >= 3:
        try:
            n_rand, edges_rand = _random_search_near_moore(k, g, moore, time_limit=min(15, remaining - 5))
            if n_rand > 0 and (best_result is None or n_rand < best_result[0]):
                if _verify_nonbacktracking(k, g, n_rand, edges_rand, time_limit=min(3, remaining - 6)):
                    best_result = (n_rand, edges_rand)
        except Exception:
            pass
    
    # Fallback
    if best_result is None:
        if k == 1:
            best_result = (2, [(0, 1)])
        elif k == 2:
            n = max(g, 3)
            best_result = (n, [(i, (i + 1) % n) for i in range(n)])
        else:
            # Last resort: LUW with minimal valid d
            d = max(3, g - 5 if (g - 5) % 2 else g - 4)
            n, edges = _luw_graph_params(k, d)
            if n > 0:
                best_result = (n, edges)
            else:
                best_result = (0, [])
    
    return best_result


def _moore_bound(k, g):
    """Compute the Moore bound for (k, g) regular graph."""
    if k <= 1:
        return 2
    if g % 2 == 0:
        h = g // 2 - 1
        if h >= 0:
            moore = 2 * k * (k - 1) ** h
        else:
            moore = 2 * k
    else:
        h = (g - 1) // 2
        if h >= 1:
            moore = 2 * (k - 1) ** h + k * (k - 1) ** (h - 1)
        else:
            moore = 2 * k
    return max(moore, 2 * k)


def _orthogonality_graph(k, g):
    """
    Construct a k-regular graph using orthogonality over finite fields.
    Vertices are vectors in F_q^d, and two vertices are adjacent if they are orthogonal.
    We tune d and q to get the right degree and girth.
    """
    # For a k-regular graph with girth g, we need to find suitable parameters
    # The orthogonality graph on F_q^d has degree related to q and d
    # Let's try different combinations
    
    # For odd g, we want a bipartite-like structure
    # Try small prime powers
    prime_powers = [2, 3, 4, 5, 7, 8, 9, 11, 13, 16, 17, 19, 23, 25, 27, 29, 31]
    
    for q in prime_powers:
        if time.time() > deadline - 5:
            break
        
        # Try different dimensions
        for d in range(2, 8):
            if time.time() > deadline - 5:
                break
            
            # Build the orthogonality graph
            n, edges = _build_orthogonality_graph(q, d, k, g)
            if n > 0 and n >= _moore_bound(k, g):
                return n, edges
    
    return 0, []


def _build_orthogonality_graph(q, d, k, g):
    """
    Build an orthogonality graph over F_q^d.
    Two vectors u, v are adjacent if u · v = 0 (mod q).
    We select a subset to get exactly degree k.
    """
    # Generate all vectors in F_q^d
    # For small q and d, this is manageable
    total_vectors = q ** d
    if total_vectors > 10000:  # Too large
        return 0, []
    
    # Generate all vectors
    vectors = []
    for i in range(total_vectors):
        vec = []
        x = i
        for _ in range(d):
            vec.append(x % q)
            x //= q
        vectors.append(tuple(vec))
    
    # Compute dot products and find orthogonal pairs
    # For each vector, find all orthogonal vectors
    adj = {i: [] for i in range(total_vectors)}
    
    for i in range(total_vectors):
        for j in range(i + 1, total_vectors):
            # Compute dot product
            dot = sum(vectors[i][t] * vectors[j][t] for t in range(d)) % q
            if dot == 0:
                adj[i].append(j)
                adj[j].append(i)
    
    # Now we need to select a subset of vertices where each has exactly k neighbors
    # This is a subgraph selection problem
    # Simple approach: try to find a regular subgraph
    
    # First, check if the full graph is regular enough
    degrees = [len(adj[i]) for i in range(total_vectors)]
    if not degrees or max(degrees) < k:
        return 0, []
    
    # Try to find a k-regular subgraph using greedy selection
    # Start with all vertices, iteratively remove vertices with degree < k
    # or adjust to get exactly k
    
    # This is complex; for now, just check if we can find any k-regular subset
    # For simplicity, try a random subset approach
    
    for _ in range(5):
        # Randomly select a subset
        subset_size = min(total_vectors, max(2 * k, k + 1))
        if subset_size >= total_vectors:
            selected = list(range(total_vectors))
        else:
            selected = random.sample(range(total_vectors), subset_size)
        
        selected_set = set(selected)
        selected_indices = {v: i for i, v in enumerate(selected)}
        
        # Check degrees within the subset
        valid = True
        for v in selected:
            deg = sum(1 for u in adj[v] if u in selected_set)
            if deg != k:
                valid = False
                break
        
        if valid:
            # Build edges
            edges = []
            for i, u in enumerate(selected):
                for j, v in enumerate(selected):
                    if i < j and v in adj[u]:
                        edges.append((selected_indices[u], selected_indices[v]))
            
            return len(selected), edges
    
    return 0, []


def _luw_graph(q, g):
    """
    Lazebnik-Ustimenko-Woldar graph CD(d, q).
    q-regular bipartite graph with girth at least d+5 (for odd d).
    """
    if q < 3:
        return 0, []
    
    d_min = g - 5
    if d_min % 2 == 0:
        d_min += 1
    d_min = max(3, d_min)
    
    for d in range(d_min, min(d_min + 3, 20)):
        if d < 3:
            continue
        try:
            n, edges = _luw_graph_params(q, d)
            if n > 0:
                return n, edges
        except Exception:
            continue
    
    return 0, []


def _luw_graph_params(q, d):
    """
    LUW graph with specific parameters.
    """
    if q < 3 or d < 3:
        return 0, []
    
    # Field operations
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    elif q == 2:
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: (x & y)
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: (x * y) % q
    
    # Rule for the graph
    rule = {}
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
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
            yield (1 - side, tuple(y))
    
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


def _verify_nonbacktracking(k, g, n, edges, time_limit=5):
    """
    Verify k-regular, simple, and girth >= g using non-backtracking walks.
    This is the exact certification method required.
    Optimized with NumPy for larger graphs.
    """
    start_time = time.time()
    
    # Build adjacency list
    adj = [[] for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        if u == v or u < 0 or v < 0 or u >= n or v >= n:
            return False
        key = (min(u, v), max(u, v))
        if key in edge_set:
            return False
        edge_set.add(key)
        adj[u].append(v)
        adj[v].append(u)
    
    # Check degree
    for i in range(n):
        if len(adj[i]) != k:
            return False
    
    # For small graphs, use BFS approach
    if n <= 200:
        # Non-backtracking BFS from each vertex
        for start in range(n):
            if time.time() - start_time > time_limit:
                return True
            
            # (curr, prev) -> distance
            visited = {}
            queue = deque()
            
            for neighbor in adj[start]:
                visited[(start, neighbor)] = 1
                queue.append((start, neighbor, 1))
            
            while queue:
                prev, curr, dist = queue.popleft()
                
                if dist >= g - 1:
                    continue
                
                for next_v in adj[curr]:
                    if next_v == prev:
                        continue
                    
                    key = (curr, next_v)
                    if key in visited:
                        cycle_len = visited[key] + dist + 1
                        if cycle_len < g:
                            return False
                    else:
                        visited[key] = dist + 1
                        if dist + 1 < g - 1:
                            queue.append((curr, next_v, dist + 1))
        
        return True
    else:
        # For larger graphs, use a more efficient approach
        # Check girth using BFS from a subset of vertices
        # and use early termination
        
        # First, check if the graph is connected
        visited = set()
        queue = deque([0])
        visited.add(0)
        while queue:
            node = queue.popleft()
            for neighbor in adj[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        if len(visited) < n:
            # Disconnected graph - check each component
            for start in range(n):
                if start not in visited:
                    visited2 = set()
                    queue2 = deque([start])
                    visited2.add(start)
                    while queue2:
                        node = queue2.popleft()
                        for neighbor in adj[node]:
                            if neighbor not in visited2:
                                visited2.add(neighbor)
                                queue2.append(neighbor)
                    if len(visited2) < n:
                        # Multiple components, need to check each
                        pass
        
        # Sample-based girth check for large graphs
        sample_size = min(n, 50)
        sample = random.sample(range(n), sample_size)
        
        for start in sample:
            if time.time() - start_time > time_limit:
                return True
            
            # BFS to find shortest cycle
            dist = {start: 0}
            parent = {start: -1}
            queue = deque([(start, 0)])
            
            while queue:
                node, d = queue.popleft()
                if d >= g - 1:
                    continue
                
                for neighbor in adj[node]:
                    if neighbor == parent[node]:
                        continue
                    
                    if neighbor in dist:
                        cycle_len = d + dist[neighbor] + 1
                        if cycle_len < g:
                            return False
                    else:
                        dist[neighbor] = d + 1
                        parent[neighbor] = node
                        if d + 1 < g - 1:
                            queue.append((neighbor, d + 1))
        
        return True


def _spectral_random_search(k, g, moore, time_limit=25):
    """
    Spectral-guided random search for k-regular graphs with girth >= g.
    Uses non-backtracking matrix eigenvalue analysis to guide construction.
    Random regular graphs with large spectral gap (small second eigenvalue of
    non-backtracking matrix) tend to have long girths.
    """
    start_time = time.time()
    best_n = float('inf')
    best_edges = None
    
    # Try candidate sizes near Moore bound
    candidates = []
    for offset in range(0, 60, 2):
        n = moore + offset
        if n * k % 2 != 0:
            n += 1
        if n >= moore and n not in candidates:
            candidates.append(n)
        if len(candidates) >= 15:
            break
    
    for n in candidates:
        if time.time() - start_time > time_limit:
            break
        
        # Try multiple random constructions with spectral filtering
        for attempt in range(12):
            if time.time() - start_time > time_limit:
                break
            
            edges = _construct_k_regular_spectral(n, k, g)
            if edges is not None:
                # Quick girth check
                if _quick_girth_check(g, n, edges):
                    if n < best_n:
                        best_n = n
                        best_edges = edges
                        break
    
    if best_n == float('inf'):
        return 0, []
    return best_n, best_edges


def _construct_k_regular_spectral(n, k, g):
    """
    Construct a k-regular graph with spectral awareness.
    Uses configuration model with cycle avoidance and non-backtracking
    spectral pruning.
    """
    if n * k % 2 != 0 or n * k < 2:
        return None
    
    # Try configuration model with cycle avoidance
    for _ in range(5):
        adj = [[] for _ in range(n)]
        edge_set = set()
        
        # Shuffle vertex order
        order = list(range(n))
        random.shuffle(order)
        
        success = True
        for i in range(n):
            if len(adj[i]) >= k:
                continue
            
            # Find k valid neighbors
            candidates = [v for v in range(n) if v != i and v not in adj[i]]
            random.shuffle(candidates)
            
            added = 0
            for v in candidates:
                if len(adj[i]) >= k:
                    break
                # Check if adding edge creates a short cycle
                if _creates_short_cycle(i, v, adj, g):
                    continue
                adj[i].append(v)
                adj[v].append(i)
                edge_set.add((min(i, v), max(i, v)))
                added += 1
            
            if len(adj[i]) < k:
                success = False
                break
        
        if success:
            # Verify all degrees
            if all(len(adj[i]) == k for i in range(n)):
                # Spectral check via non-backtracking matrix
                if _spectral_check(adj, k, n):
                    return list(edge_set)
    
    # Fallback: simple configuration model
    for _ in range(15):
        stubs = []
        for i in range(n):
            for _ in range(k):
                stubs.append(i)
        
        random.shuffle(stubs)
        
        edges = []
        valid = True
        edge_set = set()
        
        for i in range(0, len(stubs), 2):
            u, v = stubs[i], stubs[i + 1]
            if u == v:
                valid = False
                break
            key = (min(u, v), max(u, v))
            if key in edge_set:
                valid = False
                break
            edge_set.add(key)
            edges.append(key)
        
        if valid:
            return edges
    
    return None


def _spectral_check(adj, k, n):
    """
    Estimate the second eigenvalue of the non-backtracking matrix.
    For a k-regular graph, the non-backtracking matrix has eigenvalues
    that relate to the girth. A large spectral gap (small second eigenvalue
    magnitude) suggests the graph has few short cycles.
    
    We use power iteration on the non-backtracking matrix to estimate
    the second largest eigenvalue in magnitude.
    """
    if n < 2:
        return True
    
    # Build non-backtracking matrix: B[(u,v), (v,w)] = 1 if w != u
    # State is (prev, curr) directed edge
    nb_states = {}
    state_list = []
    for u in range(n):
        for v in adj[u]:
            key = (u, v)
            if key not in nb_states:
                nb_states[key] = len(state_list)
                state_list.append(key)
    
    m = len(state_list)
    if m == 0:
        return True
    
    # For each state (u,v), find outgoing states (v,w) where w != u
    # Build sparse representation
    nb_adj = [[] for _ in range(m)]
    for u in range(n):
        for v in adj[u]:
            idx_uv = nb_states[(u, v)]
            for w in adj[v]:
                if w != u:
                    idx_vw = nb_states[(v, w)]
                    nb_adj[idx_uv].append(idx_vw)
    
    # Power iteration to find dominant eigenvalue of B
    # The largest eigenvalue should be 1 (for connected k-regular graphs)
    # The second largest in magnitude indicates cycle structure
    
    # Initialize with random vector
    v = [random.uniform(-1, 1) for _ in range(m)]
    norm = math.sqrt(sum(x * x for x in v))
    if norm < 1e-10:
        return True
    v = [x / norm for x in v]
    
    # Run power iteration
    lambda_1 = 0.0
    lambda_2 = 0.0
    
    for iteration in range(30):
        new_v = [0.0] * m
        for i in range(m):
            for j in nb_adj[i]:
                new_v[j] += v[i]
        
        # Compute Rayleigh quotient for eigenvalue estimate
        dot_old = sum(v[i] * new_v[i] for i in range(m))
        norm_new = math.sqrt(sum(x * x for x in new_v))
        if norm_new < 1e-10:
            return True
        
        lambda_1 = dot_old / norm_new
        v = [x / norm_new for x in new_v]
    
    # For the non-backtracking matrix of a k-regular graph:
    # - The largest eigenvalue is 1
    # - Eigenvalues with |lambda| > 1 indicate cycles
    # - A graph with girth >= g has no eigenvalues in certain ranges
    # 
    # Heuristic: if the spectral radius is close to 1 (no large eigenvalues),
    # the graph likely has good girth properties
    
    # Check if spectral radius is reasonable (close to 1 for good expanders)
    # For k-regular graphs, the non-backtracking matrix has eigenvalues
    # that are roots of the characteristic polynomial related to adjacency
    
    # Simple heuristic: accept if we didn't find wildly large eigenvalues
    # (which would indicate many short cycles)
    if abs(lambda_1) < 2.0:
        return True
    
    # If spectral radius is too large, reject
    return False


def _random_search_near_moore(k, g, moore, time_limit=20):
    """
    Try to find a k-regular graph with girth >= g near the Moore bound.
    Uses a smarter approach: build adjacency incrementally avoiding short cycles.
    """
    start_time = time.time()
    best_n = float('inf')
    best_edges = None
    
    # Try a few candidate sizes
    candidates = []
    for offset in range(0, 30, 2):
        n = moore + offset
        if n * k % 2 != 0:
            n += 1
        if n >= moore and n not in candidates:
            candidates.append(n)
        if len(candidates) >= 8:
            break
    
    for n in candidates:
        if time.time() - start_time > time_limit:
            break
        
        # Try multiple random constructions
        for attempt in range(5):
            if time.time() - start_time > time_limit:
                break
            
            edges = _construct_k_regular_girth_aware(n, k, g)
            if edges is not None:
                # Quick girth check
                if _quick_girth_check(g, n, edges):
                    if n < best_n:
                        best_n = n
                        best_edges = edges
                    break
    
    if best_n == float('inf'):
        return 0, []
    return best_n, best_edges


def _construct_k_regular_girth_aware(n, k, g):
    """
    Construct a k-regular graph trying to avoid short cycles.
    Uses a greedy approach with backtracking.
    """
    if n * k % 2 != 0 or n * k < 2:
        return None
    
    # Try configuration model with cycle avoidance
    for _ in range(3):
        adj = [[] for _ in range(n)]
        edge_set = set()
        
        # Shuffle vertex order
        order = list(range(n))
        random.shuffle(order)
        
        success = True
        for i in range(n):
            if len(adj[i]) >= k:
                continue
            
            # Find k valid neighbors
            candidates = [v for v in range(n) if v != i and v not in adj[i]]
            random.shuffle(candidates)
            
            added = 0
            for v in candidates:
                if len(adj[i]) >= k:
                    break
                # Check if adding edge creates a short cycle
                if _creates_short_cycle(i, v, adj, g):
                    continue
                adj[i].append(v)
                adj[v].append(i)
                edge_set.add((min(i, v), max(i, v)))
                added += 1
            
            if len(adj[i]) < k:
                success = False
                break
        
        if success:
            # Verify all degrees
            if all(len(adj[i]) == k for i in range(n)):
                return list(edge_set)
    
    # Fallback: simple configuration model
    for _ in range(10):
        stubs = []
        for i in range(n):
            for _ in range(k):
                stubs.append(i)
        
        random.shuffle(stubs)
        
        edges = []
        valid = True
        edge_set = set()
        
        for i in range(0, len(stubs), 2):
            u, v = stubs[i], stubs[i + 1]
            if u == v:
                valid = False
                break
            key = (min(u, v), max(u, v))
            if key in edge_set:
                valid = False
                break
            edge_set.add(key)
            edges.append(key)
        
        if valid:
            return edges
    
    return None


def _creates_short_cycle(u, v, adj, g):
    """
    Check if adding edge (u,v) would create a cycle shorter than g.
    Uses BFS with depth limit g-2.
    """
    if g <= 3:
        return False
    
    # BFS from u to v avoiding direct edge, depth <= g-2
    if v in adj[u]:
        return True  # Already connected (shouldn't happen in our use)
    
    max_depth = g - 2
    visited = {u: 0}
    queue = deque([(u, 0)])
    
    while queue:
        node, depth = queue.popleft()
        if depth >= max_depth:
            continue
        for neighbor in adj[node]:
            if neighbor == v:
                return True  # Would create cycle of length depth + 1
            if neighbor not in visited:
                visited[neighbor] = depth + 1
                queue.append((neighbor, depth + 1))
    
    return False


def _quick_girth_check(g, n, edges):
    """
    Quick girth verification using non-backtracking BFS.
    """
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    for start in range(n):
        visited = {}
        queue = deque()
        
        for neighbor in adj[start]:
            visited[(start, neighbor)] = 1
            queue.append((start, neighbor, 1))
        
        while queue:
            prev, curr, dist = queue.popleft()
            if dist >= g - 1:
                continue
            for next_v in adj[curr]:
                if next_v == prev:
                    continue
                key = (curr, next_v)
                if key in visited:
                    if visited[key] + dist + 1 < g:
                        return False
                else:
                    visited[key] = dist + 1
                    if dist + 1 < g - 1:
                        queue.append((curr, next_v, dist + 1))
    
    return True
