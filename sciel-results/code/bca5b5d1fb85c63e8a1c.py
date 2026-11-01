import time
import random
import heapq
import collections

def _luw_construct(k, g):
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

def _random_k_regular(n, k):
    """Generate a random k-regular graph on n vertices using configuration model."""
    if n * k % 2 != 0:
        return None
    if k > n - 1:
        return None
        
    stubs = []
    for i in range(n):
        for _ in range(k):
            stubs.append(i)
            
    random.shuffle(stubs)
    
    adjacency = [set() for _ in range(n)]
    
    for i in range(0, n * k, 2):
        u = stubs[i]
        v = stubs[i + 1]
        if u == v:
            return None # Loop, restart
        if v in adjacency[u]:
            return None # Multi-edge, restart
        adjacency[u].add(v)
        adjacency[v].add(u)
        
    return adjacency

def _find_short_cycle(adjacency, n, target_girth):
    """Find a cycle of length < target_girth. Returns (length, edge) or (inf, None)."""
    min_len = float('inf')
    best_edge = None
    
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = collections.deque([start])
        found = False
        while queue:
            u = queue.popleft()
            for v in adjacency[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u]:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < min_len:
                        min_len = cycle_len
                        best_edge = (u, v)
                        found = True
                        # We can break early if we found a very short cycle
                        if min_len <= 3:
                            return min_len, best_edge
            if found and min_len <= target_girth:
                # We found a cycle shorter than target. We can stop BFS for this start?
                # No, we want the MIN length. But if we just want ANY short cycle to break,
                # we can return as soon as we find one < target.
                # However, breaking a 4-cycle is better than breaking a 7-cycle?
                # Usually yes, it's easier.
                # So let's just return the first one found that is < target.
                # This saves time.
                if min_len < target_girth:
                    return min_len, best_edge
        if min_len < target_girth:
            return min_len, best_edge
            
    return min_len, best_edge

def _greedy_switching_construct(k, g, n, time_limit=20.0):
    start_time = time.time()
    
    for attempt in range(20):
        if time.time() - start_time > time_limit:
            return None
            
        adjacency = _random_k_regular(n, k)
        if adjacency is None:
            continue
            
        # Initial girth check
        girth, edge = _find_short_cycle(adjacency, n, g)
        
        if girth >= g:
            edges = []
            for i in range(n):
                for j in adjacency[i]:
                    if i < j:
                        edges.append((i, j))
            return n, edges
            
        stuck_count = 0
        while time.time() - start_time < time_limit:
            girth, edge = _find_short_cycle(adjacency, n, g)
            
            if girth >= g:
                edges = []
                for i in range(n):
                    for j in adjacency[i]:
                        if i < j:
                            edges.append((i, j))
                return n, edges
                
            if edge is None:
                break
                
            u, v = edge
            
            # Find candidate edges to swap with (u,v)
            # We want to pick (x,y) such that x,y are not u,v
            # and (u,x), (v,y) are not existing edges.
            
            # To make it faster, we don't shuffle all edges every time.
            # We can pick random edges.
            
            success = False
            # Try a few random swaps
            for _ in range(50):
                x = random.randint(0, n - 1)
                y = random.randint(0, n - 1)
                if x == y:
                    continue
                if x == u or x == v or y == u or y == v:
                    continue
                if y not in adjacency[x]:
                    continue # (x,y) is not an edge
                
                # Try swap (u,v) and (x,y) -> (u,x) and (v,y)
                if x in adjacency[u] or y in adjacency[v]:
                    # Try other pairing: (u,y) and (v,x)
                    if y in adjacency[u] or x in adjacency[v]:
                        continue
                    # Check if (u,y) and (v,x) are valid
                    # We need to ensure (u,y) and (v,x) don't exist
                    # They don't exist by the check above.
                    
                    # Perform swap
                    adjacency[u].remove(v)
                    adjacency[v].remove(u)
                    adjacency[x].remove(y)
                    adjacency[y].remove(x)
                    
                    adjacency[u].add(y)
                    adjacency[y].add(u)
                    adjacency[v].add(x)
                    adjacency[x].add(v)
                    
                    new_girth, _ = _find_short_cycle(adjacency, n, g)
                    if new_girth >= girth:
                        girth = new_girth
                        success = True
                        break
                    else:
                        # Reverse
                        adjacency[u].remove(y)
                        adjacency[y].remove(u)
                        adjacency[v].remove(x)
                        adjacency[x].remove(v)
                        
                        adjacency[u].add(v)
                        adjacency[v].add(u)
                        adjacency[x].add(y)
                        adjacency[y].add(x)
                else:
                    # Standard pairing (u,x) and (v,y)
                    adjacency[u].remove(v)
                    adjacency[v].remove(u)
                    adjacency[x].remove(y)
                    adjacency[y].remove(x)
                    
                    adjacency[u].add(x)
                    adjacency[x].add(u)
                    adjacency[v].add(y)
                    adjacency[y].add(v)
                    
                    new_girth, _ = _find_short_cycle(adjacency, n, g)
                    if new_girth >= girth:
                        girth = new_girth
                        success = True
                        break
                    else:
                        # Reverse
                        adjacency[u].remove(x)
                        adjacency[x].remove(u)
                        adjacency[v].remove(y)
                        adjacency[y].remove(v)
                        
                        adjacency[u].add(v)
                        adjacency[v].add(u)
                        adjacency[x].add(y)
                        adjacency[y].add(x)
                        
            if not success:
                stuck_count += 1
                if stuck_count > 50:
                    break
    return None

def construct(k, g):
    if k == 5 and g == 9:
        target_ns = [1116, 1120, 1150, 1200, 1500]
        
        for n in target_ns:
            res = _greedy_switching_construct(k, g, n, time_limit=20.0)
            if res is not None:
                return res
                
    return _luw_construct(k, g)
