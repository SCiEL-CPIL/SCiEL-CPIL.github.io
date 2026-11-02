import time
import random
import heapq
import copy
from collections import deque

def _luw_construct(k, g):
    """LUW construction – returns (n, edges) for a k-regular graph of girth >= g."""
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

def _find_shortest_cycles(adj, n, max_len):
    """Find all cycles of length < max_len using BFS from each vertex."""
    cycles = []
    for start in range(n):
        dist = [-1] * n
        dist[start] = 0
        queue = deque([start])
        while queue:
            u = queue.popleft()
            if dist[u] >= max_len - 1:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    queue.append(v)
                elif dist[u] + dist[v] + 1 < max_len:
                    # Found a cycle of length dist[u] + dist[v] + 1
                    cycles.append((dist[u] + dist[v] + 1, u, v))
    return cycles

def _compute_girth(adj, n, max_girth=None):
    """Compute girth of the graph. If max_girth is given, stop early if found."""
    if max_girth is None:
        max_girth = n
    cycles = _find_shortest_cycles(adj, n, max_girth)
    if not cycles:
        return max_girth
    return min(c[0] for c in cycles)

def _generate_initial_k_regular(k, n, seed=None):
    """Generate a random k-regular simple graph using configuration model."""
    if seed is not None:
        random.seed(seed)
    
    # Try multiple times to get a simple graph
    for attempt in range(100):
        stubs = []
        for i in range(n):
            stubs.extend([i] * k)
        random.shuffle(stubs)
        
        adj = [set() for _ in range(n)]
        valid = True
        for i in range(0, len(stubs), 2):
            u, v = stubs[i], stubs[i + 1]
            if u == v or v in adj[u]:
                valid = False
                break
            adj[u].add(v)
            adj[v].add(u)
        
        if valid and all(len(adj[i]) == k for i in range(n)):
            return adj
    
    # Fallback: use Walecki construction for odd k
    if k % 2 == 1 and n % 2 == 0:
        adj = [set() for _ in range(n)]
        for i in range(n // 2):
            u = i
            v = (i + n // 2) % n
            adj[u].add(v)
            adj[v].add(u)
        # Add remaining edges
        for i in range(n):
            while len(adj[i]) < k:
                # Find a vertex with degree < k that's not already adjacent
                for j in range(n):
                    if i != j and j not in adj[i] and len(adj[j]) < k:
                        adj[i].add(j)
                        adj[j].add(i)
                        break
                else:
                    break
        
        if all(len(adj[i]) == k for i in range(n)):
            return adj
    
    return None

def _switch_edges(adj, n, u1, v1, u2, v2):
    """Try a 2-switch: remove (u1,v1) and (u2,v2), add (u1,u2) and (v1,v2) or (u1,v2) and (v1,u2)."""
    if u1 == v1 or u2 == v2:
        return False, None
    if len({u1, v1, u2, v2}) < 4:
        return False, None
    if v1 in adj[u1] and v2 in adj[u2]:
        # Try pairing 1
        new_edges = [(u1, u2), (v1, v2)]
        valid = True
        for a, b in new_edges:
            if a == b or b in adj[a]:
                valid = False
                break
        if valid:
            # Perform the switch
            adj[u1].remove(v1)
            adj[v1].remove(u1)
            adj[u2].remove(v2)
            adj[v2].remove(u2)
            for a, b in new_edges:
                adj[a].add(b)
                adj[b].add(a)
            return True, new_edges
        
        # Try pairing 2
        new_edges = [(u1, v2), (v1, u2)]
        valid = True
        for a, b in new_edges:
            if a == b or b in adj[a]:
                valid = False
                break
        if valid:
            adj[u1].remove(v1)
            adj[v1].remove(u1)
            adj[u2].remove(v2)
            adj[v2].remove(u2)
            for a, b in new_edges:
                adj[a].add(b)
                adj[b].add(a)
            return True, new_edges
    return False, None

def _switch_to_improve_girth(adj, n, target_girth, max_iterations=1000, timeout=None, start_time=None):
    """
    Use edge switching to improve girth.
    Priority: eliminate short cycles, then improve girth.
    """
    if start_time is None:
        start_time = time.time()
    
    current_girth = _compute_girth(adj, n, target_girth + 1)
    
    if current_girth >= target_girth:
        return adj, current_girth
    
    for iteration in range(max_iterations):
        if timeout is not None and time.time() - start_time > timeout:
            break
        
        # Find shortest cycles
        max_check = min(current_girth, target_girth)
        cycles = _find_shortest_cycles(adj, n, max_check + 1)
        
        if not cycles:
            # No short cycles found, try to improve girth further
            new_girth = _compute_girth(adj, n, current_girth + 1)
            if new_girth >= target_girth:
                return adj, new_girth
            if new_girth > current_girth:
                current_girth = new_girth
            continue
        
        # Pick a cycle to break
        cycle_len, u, v = min(cycles, key=lambda c: c[0])
        
        # Find edges in the cycle path
        # For simplicity, just pick two random non-adjacent edges to switch
        # Get all edges
        all_edges = []
        for i in range(n):
            for j in adj[i]:
                if i < j:
                    all_edges.append((i, j))
        
        if len(all_edges) < 2:
            break
        
        # Try to find two edges that, when switched, break a short cycle
        success = False
        for _ in range(50):
            if timeout is not None and time.time() - start_time > timeout:
                return adj, current_girth
            
            e1, e2 = random.sample(all_edges, 2)
            u1, v1 = e1
            u2, v2 = e2
            
            if len({u1, v1, u2, v2}) < 4:
                continue
            
            # Try the switch
            adj_copy = [set(s) for s in adj]
            ok, _ = _switch_edges(adj_copy, n, u1, v1, u2, v2)
            
            if not ok:
                continue
            
            # Check if girth improved
            new_girth = _compute_girth(adj_copy, n, max(current_girth + 1, target_girth))
            
            if new_girth >= current_girth:
                # Accept the switch
                adj = adj_copy
                if new_girth > current_girth:
                    current_girth = new_girth
                    if current_girth >= target_girth:
                        return adj, current_girth
                success = True
                break
        
        if not success:
            # No improvement after many attempts
            if iteration % 100 == 0:
                current_girth = _compute_girth(adj, n, target_girth + 1)
            continue
    
    return adj, current_girth

def construct(k, g):
    total_deadline = 145.0  # Safety margin before 150s
    start_time = time.time()
    
    if k == 5 and g == 9:
        # Try different n values starting from small
        # Moore bound for (5,9): roughly 2*5^(1+9/2) is an upper bound, but actual min is ~430
        target_ns = [450, 500, 600, 700, 800, 1000]
        
        time_budget = 80.0
        time_per_n = time_budget / len(target_ns)
        
        random.seed(42)
        
        for n in target_ns:
            if time.time() - start_time > 130:
                break
            
            time_left = min(time_per_n, 135.0 - (time.time() - start_time))
            if time_left < 5:
                break
            
            # Generate initial graph
            adj = _generate_initial_k_regular(k, n)
            if adj is None:
                continue
            
            # Try to improve girth
            adj, girth = _switch_to_improve_girth(
                adj, n, g,
                max_iterations=2000,
                timeout=time_left
            )
            
            if girth >= g:
                # Build edge list
                edges = []
                for i in range(n):
                    for j in adj[i]:
                        if i < j:
                            edges.append((i, j))
                
                # Verify
                verify_adj = [set() for _ in range(n)]
                for u, v in edges:
                    verify_adj[u].add(v)
                    verify_adj[v].add(u)
                
                if all(len(verify_adj[i]) == k for i in range(n)):
                    verify_girth = _compute_girth(verify_adj, n, g)
                    if verify_girth >= g:
                        return n, edges
        
        # Fallback to LUW
        if time.time() - start_time < 140:
            return _luw_construct(k, g)
    
    # Generic fallback
    return _luw_construct(k, g)
