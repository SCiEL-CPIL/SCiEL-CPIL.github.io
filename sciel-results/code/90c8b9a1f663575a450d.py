"""Optimized k-regular girth-g graph construction with fast girth checking and improved search."""
import time
import random
from collections import deque

_DEADLINE = 115  # safety margin before 120s wall-clock kill


def _time_left():
    return _DEADLINE - time.time()


def _moore_lower(k, g):
    """Moore bound: minimum possible n for k-regular graph with girth g."""
    if k <= 2:
        return g
    if g % 2 == 0:
        r = g // 2
        return 2 * ((k - 1) ** r - 1) // (k - 2)
    else:
        r = (g - 1) // 2
        return 1 + k * ((k - 1) ** r - 1) // (k - 2)


def _check_girth_fast(adj, n, g):
    """Check no cycle shorter than g using non-backtracking BFS from each vertex.
    Uses integer-encoded states for speed. Returns True if girth >= g."""
    if n == 0:
        return True
    max_depth = (g - 1) // 2
    stride = n + 1
    
    for start in range(n):
        # Use a list as a boolean array for visited states
        # State encoding: node * stride + (prev+1), where prev=-1 encodes to 0
        total_states = n * stride
        visited = bytearray(total_states)
        queue = deque()
        
        # Start state: node=start, prev=-1 -> encoded as start*stride + 0
        start_state = start * stride
        visited[start_state] = 1
        queue.append(start_state)
        
        depth = 0
        while queue and depth <= max_depth:
            next_count = len(queue)
            for _ in range(next_count):
                state = queue.popleft()
                node = state // stride
                prev_enc = state % stride  # 0 means prev=-1, else prev=prev_enc-1
                
                for nb in adj[node]:
                    # Skip backtracking
                    if prev_enc != 0 and nb == prev_enc - 1:
                        continue
                    
                    new_state = nb * stride + (node + 1)
                    if visited[new_state]:
                        # Found a cycle of length depth+2
                        if depth + 2 < g:
                            return False
                        continue
                    visited[new_state] = 1
                    queue.append(new_state)
            depth += 1
    return True


def _check_degree(adj, n, k):
    """Check every vertex has degree exactly k."""
    for i in range(n):
        if len(adj[i]) != k:
            return False
    return True


def _check_simple(adj, n):
    """Check no loops, no repeated edges."""
    for i in range(n):
        if i in adj[i]:
            return False
        s = set(adj[i])
        if len(s) != len(adj[i]):
            return False
    return True


def _is_valid(adj, n, k, g):
    if n == 0:
        return False
    if not _check_simple(adj, n):
        return False
    if not _check_degree(adj, n, k):
        return False
    if not _check_girth_fast(adj, n, g):
        return False
    return True


def _lazebnik_construct(k, g):
    """LUW graph construction - fast, always valid for prime k or k=4."""
    q = k
    # d such that girth >= d + 5 (for odd d) or d + 4 (for even d)
    if (g - 5) % 2 == 1:
        d = g - 5
    else:
        d = g - 4
    d = max(3, d)
    
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
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges


def _generate_k_regular_girth_aware(n, k, g, max_attempts=20):
    """Generate a random k-regular graph with girth awareness using BFS-based stub pairing."""
    if n < k + 1 or k * n % 2 != 0 or k >= n:
        return None
    
    for attempt in range(max_attempts):
        # Track remaining stubs per vertex
        stubs_per_vertex = [list(range(k)) for _ in range(n)]
        adj = [set() for _ in range(n)]
        
        valid = True
        total_pairs = n * k // 2
        
        for pair_idx in range(total_pairs):
            # Find a vertex with remaining stubs
            candidates = [i for i in range(n) if stubs_per_vertex[i]]
            if not candidates:
                break
            u = random.choice(candidates)
            stubs_per_vertex[u].pop()
            
            # BFS from u to find vertices at distance >= g-1
            target_dist = g - 1
            visited = {u: 0}
            frontier = [u]
            depth = 0
            while frontier and depth < target_dist:
                next_frontier = []
                for node in frontier:
                    for nb in adj[node]:
                        if nb not in visited:
                            visited[nb] = depth + 1
                            next_frontier.append(nb)
                frontier = next_frontier
                depth += 1
            
            # Eligible vertices: not visited (unreachable or at distance >= target_dist)
            eligible_vertices = [i for i in range(n) if i != u and i not in visited and stubs_per_vertex[i]]
            
            if not eligible_vertices:
                # Fallback: try any vertex with remaining stubs
                eligible_vertices = [i for i in range(n) if i != u and stubs_per_vertex[i]]
                if not eligible_vertices:
                    valid = False
                    break
            
            v = random.choice(eligible_vertices)
            stubs_per_vertex[v].pop()
            
            if v in adj[u]:
                valid = False
                break
            
            adj[u].add(v)
            adj[v].add(u)
        
        if valid:
            adj_list = [list(s) for s in adj]
            if _check_girth_fast(adj_list, n, g):
                return adj_list
    return None


def _greedy_search(k, g, time_budget=60.0):
    """Greedy randomized search for a smaller k-regular graph with girth >= g."""
    start_time = time.time()
    best_n = None
    best_edges = None
    
    lower = _moore_lower(k, g)
    max_target = min(lower + 100, 400)  # cap to keep search tractable
    
    for offset in range(0, 150):
        if time.time() - start_time > time_budget:
            break
        target_n = lower + offset
        if target_n < k + 1 or k * target_n % 2 != 0 or target_n > max_target:
            continue
        
        # Try multiple random initializations
        attempts = 5 if target_n <= lower + 30 else 3
        for attempt in range(attempts):
            if time.time() - start_time > time_budget:
                break
            
            adj = _generate_k_regular_girth_aware(target_n, k, g, max_attempts=3)
            if adj is None:
                continue
            
            if _check_girth_fast(adj, target_n, g):
                edges = []
                for i in range(target_n):
                    for j in adj[i]:
                        if i < j:
                            edges.append((i, j))
                if best_n is None or target_n < best_n:
                    best_n = target_n
                    best_edges = edges
                break
    
    if best_n is not None:
        return best_n, best_edges
    return None, None


def construct(k, g):
    """Construct a k-regular graph with girth >= g, minimizing n."""
    start_time = time.time()
    
    # Always get LUW as baseline
    luw_n, luw_edges = _lazebnik_construct(k, g)
    best_n, best_edges = luw_n, luw_edges
    
    # Try greedy search with remaining time
    elapsed = time.time() - start_time
    remaining = _DEADLINE - elapsed - 8  # 8s safety margin
    if remaining > 5:
        greedy_n, greedy_edges = _greedy_search(k, g, time_budget=remaining)
        if greedy_n is not None and greedy_n < best_n:
            best_n, best_edges = greedy_n, greedy_edges
    
    # Final safety check: verify the result is valid
    if time.time() - start_time < _DEADLINE - 3:
        adj_check = [[] for _ in range(best_n)]
        for u, v in best_edges:
            adj_check[u].append(v)
            adj_check[v].append(u)
        if not _is_valid(adj_check, best_n, k, g):
            # Fallback to LUW
            best_n, best_edges = luw_n, luw_edges
    
    return best_n, best_edges
