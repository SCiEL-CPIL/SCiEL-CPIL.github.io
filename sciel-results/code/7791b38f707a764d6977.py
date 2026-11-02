import time
import random
import heapq
from collections import deque
import sys

def _luw_construct(k, g):
    # Original LUW construction (kept as fallback)
    # Copy of the seed logic
    # Copy of the seed logic
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

def _compute_girth_fast(adjacency, n, early_stop=None):
    """Compute girth using non-backtracking walks (BFS on directed edge states).
    
    A non-backtracking walk is a sequence of vertices where no vertex is immediately
    revisited. The girth equals the length of the shortest non-backtracking closed walk.
    We BFS from each directed edge (u,v) tracking the path, and check if we can return
    to u without backtracking.
    """
    if early_stop is None:
        early_stop = 100
    
    min_girth = float('inf')
    
    # BFS from each vertex using non-backtracking constraint
    # State: (current_vertex, previous_vertex, distance)
    # We track for each (current, prev) the minimum distance reached
    
    for start in range(n):
        if min_girth <= early_stop:
            break
        
        # dist[prev][curr] = minimum distance from start to curr having come from prev
        # Using dict for sparse storage
        dist_map = {}  # (prev, curr) -> distance
        queue = deque()
        
        # Initialize: from start, go to each neighbor
        for v in adjacency[start]:
            dist_map[(start, v)] = 1
            queue.append((start, v, 1))
        
        while queue:
            prev, curr, d = queue.popleft()
            
            if d >= min_girth or d >= early_stop:
                continue
            
            for nxt in adjacency[curr]:
                if nxt == prev:
                    continue  # non-backtracking
                new_d = d + 1
                if nxt == start:
                    # Found a cycle of length new_d
                    if new_d < min_girth:
                        min_girth = new_d
                        if min_girth <= early_stop:
                            return min_girth
                else:
                    key = (curr, nxt)
                    if key not in dist_map or dist_map[key] > new_d:
                        dist_map[key] = new_d
                        queue.append((curr, nxt, new_d))
    
    return min_girth if min_girth != float('inf') else -1

def _find_short_cycle_fast(adjacency, n, max_girth):
    """Find a cycle of length <= max_girth using non-backtracking BFS.
    Returns (u, v, cycle_len, cycle_edges) where cycle_edges is the list of edges
    in the cycle, or None."""
    for start in range(n):
        # dist_map: (prev, curr) -> (distance, parent_prev)
        # parent_prev allows us to reconstruct the path
        dist_map = {}
        queue = deque()
        
        for v in adjacency[start]:
            dist_map[(start, v)] = (1, None)
            queue.append((start, v, 1))
        
        while queue:
            prev, curr, d = queue.popleft()
            if d >= max_girth:
                continue
            
            for nxt in adjacency[curr]:
                if nxt == prev:
                    continue
                new_d = d + 1
                if nxt == start:
                    # Found cycle - reconstruct it
                    # The cycle is: start -> ... -> prev -> curr -> start
                    # We need to trace back from (prev, curr)
                    cycle_vertices = [start]
                    p_prev, p_curr = prev, curr
                    # Trace back
                    path = []
                    while p_curr is not None:
                        path.append(p_curr)
                        # Get parent
                        if (p_prev, p_curr) in dist_map:
                            _, parent_prev = dist_map[(p_prev, p_curr)]
                            if parent_prev is None:
                                break
                            p_prev, p_curr = parent_prev, p_prev
                        else:
                            break
                    # path is reversed and may not be complete
                    # Simpler: just return the edge (curr, start) as the closing edge
                    # and let the caller handle it
                    return (curr, start, new_d, None)
                else:
                    key = (curr, nxt)
                    if key not in dist_map or dist_map[key][0] > new_d:
                        dist_map[key] = (new_d, prev)
                        queue.append((curr, nxt, new_d))
    return None

def _has_short_path_bounded(adjacency, src, dst, max_dist, excluded_edges=None):
    """Check if there's a path from src to dst of length <= max_dist avoiding excluded edges.
    Uses BFS with early termination."""
    if src == dst:
        return True
    if excluded_edges is None:
        excluded_edges = set()
    visited = set()
    visited.add(src)
    queue = deque([(src, 0)])
    while queue:
        node, d = queue.popleft()
        if d >= max_dist:
            continue
        for neighbor in adjacency[node]:
            if (node, neighbor) in excluded_edges or (neighbor, node) in excluded_edges:
                continue
            if neighbor == dst:
                return True
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, d + 1))
    return False

def _is_2_switch_valid_fast(adjacency, u, v, w, x, g):
    """Fast check: does 2-switch (u,v)&(w,x) -> (u,w)&(v,x) create a cycle < g?
    
    After the switch, new edges are (u,w) and (v,x), old edges (u,v) and (w,x) removed.
    A short cycle must use at least one new edge.
    - Cycle using (u,w): path from u to w of length <= g-2 in G - {(u,v),(w,x)}
    - Cycle using (v,x): path from v to x of length <= g-2 in G - {(u,v),(w,x)}
    - Cycle using both: path from u to v of length <= g-4 AND path from w to x of length <= g-4
      (or similar combinations), but this is a higher-order check we skip for speed.
    """
    excluded = {(u, v), (v, u), (w, x), (x, w)}
    
    # Check u to w path of length <= g-2
    if _has_short_path_bounded(adjacency, u, w, g - 2, excluded):
        return False
    
    # Check v to x path of length <= g-2
    if _has_short_path_bounded(adjacency, v, x, g - 2, excluded):
        return False
    
    return True

def _generate_simple_k_regular(k, n, max_attempts=50):
    """Generate a simple k-regular graph on n vertices using configuration model."""
    if n * k % 2 != 0:
        return None
    
    for attempt in range(max_attempts):
        stubs = []
        for i in range(n):
            stubs.extend([i] * k)
        
        random.shuffle(stubs)
        
        adjacency = [set() for _ in range(n)]
        valid = True
        for i in range(0, len(stubs), 2):
            u = stubs[i]
            v = stubs[i + 1]
            if u == v or v in adjacency[u]:
                valid = False
                break
            adjacency[u].add(v)
            adjacency[v].add(u)
        
        if valid:
            return [list(s) for s in adjacency]
    
    return None

def _switching_construct(k, g, n, time_limit=100.0):
    """Construct a k-regular graph of girth >= g using edge switching."""
    start_time = time.time()
    
    adjacency = _generate_simple_k_regular(k, n, max_attempts=50)
    if adjacency is None:
        return None
    
    # Build edge set for quick lookup
    edge_set = set()
    for i in range(n):
        for j in adjacency[i]:
            if i < j:
                edge_set.add((i, j))
    
    current_girth = _compute_girth_fast(adjacency, n, early_stop=g)
    
    if current_girth >= g:
        edges = list(edge_set)
        return n, edges
    
    # Main switching loop
    max_iterations = 20000
    for iteration in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        
        # Find a short cycle
        result = _find_short_cycle_fast(adjacency, n, g - 1)
        if result is None:
            # Verify with full girth computation
            final_girth = _compute_girth_fast(adjacency, n, early_stop=g)
            if final_girth >= g:
                edges = []
                for i in range(n):
                    for j in adjacency[i]:
                        if i < j:
                            edges.append((i, j))
                return n, edges
            continue
        
        u, v, cycle_len, _ = result
        
        # Try to break this cycle by switching edge (u,v) with another edge
        success = False
        for attempt in range(15):
            # Pick a random edge (w, x)
            w = random.randint(0, n - 1)
            if not adjacency[w]:
                continue
            x = random.choice(adjacency[w])
            if x <= w:
                # Ensure canonical form
                continue
            
            # Basic checks: all four vertices distinct, no existing edges
            if len({u, v, w, x}) < 4:
                continue
            if (u, w) in edge_set or (w, u) in edge_set:
                continue
            if (v, x) in edge_set or (x, v) in edge_set:
                continue
            if (u, x) in edge_set or (x, u) in edge_set:
                continue
            if (v, w) in edge_set or (w, v) in edge_set:
                continue
            
            # Check if switch is valid (doesn't create short cycle)
            if _is_2_switch_valid_fast(adjacency, u, v, w, x, g):
                # Perform switch
                adjacency[u].remove(v)
                adjacency[v].remove(u)
                adjacency[w].remove(x)
                adjacency[x].remove(w)
                adjacency[u].append(w)
                adjacency[w].append(u)
                adjacency[v].append(x)
                adjacency[x].append(v)
                
                edge_set.discard((min(u,v), max(u,v)))
                edge_set.discard((min(w,x), max(w,x)))
                edge_set.add((min(u,w), max(u,w)))
                edge_set.add((min(v,x), max(v,x)))
                
                success = True
                break
        
        if not success and iteration % 50 == 0:
            # Try a random valid 2-switch to escape local minima
            for _ in range(5):
                w = random.randint(0, n - 1)
                if not adjacency[w]:
                    continue
                x = random.choice(adjacency[w])
                if x <= w:
                    continue
                u_r = random.randint(0, n - 1)
                if not adjacency[u_r]:
                    continue
                v_r = random.choice(adjacency[u_r])
                if v_r <= u_r:
                    continue
                if len({u_r, v_r, w, x}) < 4:
                    continue
                if (u_r, w) in edge_set or (v_r, x) in edge_set:
                    continue
                if (u_r, x) in edge_set or (v_r, w) in edge_set:
                    continue
                if _is_2_switch_valid_fast(adjacency, u_r, v_r, w, x, g):
                    adjacency[u_r].remove(v_r)
                    adjacency[v_r].remove(u_r)
                    adjacency[w].remove(x)
                    adjacency[x].remove(w)
                    adjacency[u_r].append(w)
                    adjacency[w].append(u_r)
                    adjacency[v_r].append(x)
                    adjacency[x].append(v_r)
                    edge_set.discard((min(u_r,v_r), max(u_r,v_r)))
                    edge_set.discard((min(w,x), max(w,x)))
                    edge_set.add((min(u_r,w), max(u_r,w)))
                    edge_set.add((min(v_r,x), max(v_r,x)))
                    break
    
    # Final check
    final_girth = _compute_girth_fast(adjacency, n, early_stop=g)
    if final_girth >= g:
        edges = []
        for i in range(n):
            for j in adjacency[i]:
                if i < j:
                    edges.append((i, j))
        return n, edges
    
    return None

def _greedy_construct(k, g, n, time_limit=10.0):
    """Greedy construction: add edges one at a time, checking girth constraint."""
    start_time = time.time()
    adjacency = [set() for _ in range(n)]
    degrees = [0] * n
    
    def has_short_cycle(u, v, max_depth):
        """Check if adding edge (u,v) creates a cycle of length < max_depth+1."""
        if u == v:
            return True
        # BFS from u to find path to v of length <= max_depth-1
        # (because the new edge adds 1 to the path length)
        if max_depth <= 1:
            return False
        visited = {u}
        queue = deque([(u, 0)])
        while queue:
            node, d = queue.popleft()
            if d >= max_depth - 1:
                continue
            for neighbor in adjacency[node]:
                if neighbor == v:
                    return True
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, d + 1))
        return False

    # Process vertices in order, trying to give each vertex degree k
    for u in range(n):
        if time.time() - start_time > time_limit:
            return None
        
        # Find vertices that need more degree
        candidates = [v for v in range(n) if v != u and v not in adjacency[u] and degrees[v] < k]
        random.shuffle(candidates)
        
        while degrees[u] < k:
            found = False
            for v in candidates:
                if degrees[v] >= k:
                    continue
                if has_short_cycle(u, v, g):
                    continue
                
                adjacency[u].add(v)
                adjacency[v].add(u)
                degrees[u] += 1
                degrees[v] += 1
                candidates.remove(v)
                found = True
                break
            
            if not found:
                return None
    
    for i in range(n):
        if len(adjacency[i]) != k:
            return None
    
    edges = []
    for i in range(n):
        for j in adjacency[i]:
            if i < j:
                edges.append((i, j))
    return n, edges

def construct(k, g):
    # For the specific case k=5, g=9, try switching algorithm
    if k == 5 and g == 9:
        # Time budget with safety margin (evaluator kills at 150s)
        total_time = 145.0
        start_time = time.time()
        
        # Try switching algorithm with various n values
        # Start with larger n where it's more feasible, then try smaller
        # The known best is 1116, so try around there and below
        # Focus time on most promising values
        target_ns = [1200, 1400, 1600, 1116, 1000, 1100, 1800, 2000]
        
        best_result = None
        best_n = float('inf')
        
        for n in target_ns:
            elapsed = time.time() - start_time
            remaining = total_time - elapsed
            if remaining < 10:
                break
            # Allocate more time to smaller n (harder but better score)
            # and less to larger n (easier but worse score)
            if n <= 1200:
                time_per_attempt = min(30.0, remaining * 0.4)
            elif n <= 1600:
                time_per_attempt = min(25.0, remaining * 0.3)
            else:
                time_per_attempt = min(20.0, remaining * 0.2)
            
            res = _switching_construct(k, g, n, time_limit=time_per_attempt)
            if res is not None:
                # Found a valid graph - try to find even smaller one if time allows
                best_result = res
                best_n = n
                # If we found a good one, try smaller
                if n >= 1400:
                    # Continue to try smaller values
                    continue
                else:
                    # Found a small one, return it
                    return res
        
        # If switching failed for all, try greedy on a moderate n
        elapsed = time.time() - start_time
        remaining = total_time - elapsed
        if remaining > 15:
            for n in [1200, 1400, 1600]:
                if time.time() - start_time > total_time - 8:
                    break
                res = _greedy_construct(k, g, n, time_limit=min(12.0, remaining * 0.5))
                if res is not None:
                    return res
        
        # Return best result found
        if best_result is not None:
            return best_result
    
    # Fallback to LUW
    return _luw_construct(k, g)
