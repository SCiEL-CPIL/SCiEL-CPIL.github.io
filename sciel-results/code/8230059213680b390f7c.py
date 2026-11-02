import time
import random
import heapq
from collections import deque

def _compute_girth(adjacency, k, g_max):
    """Compute the girth of the graph. Returns girth or g_max if no cycle shorter than g_max found."""
    n = len(adjacency)
    min_girth = g_max
    
    for start in range(n):
        # BFS with non-backtracking
        # visited[(vertex, prev)] = depth
        visited = {}
        queue = deque()
        
        for neighbor in adjacency[start]:
            visited[(neighbor, start)] = 1
            queue.append((neighbor, start, 1))
        
        while queue:
            vertex, prev, depth = queue.popleft()
            
            # Check if we can close a cycle
            # If vertex is adjacent to start, we have a cycle of length depth + 1
            if start in adjacency[vertex] and depth + 1 < min_girth:
                min_girth = depth + 1
                if min_girth <= g_max:
                    return min_girth
            
            for neighbor in adjacency[vertex]:
                if neighbor == prev:
                    continue
                state = (neighbor, vertex)
                if state not in visited:
                    visited[state] = depth + 1
                    queue.append((neighbor, vertex, depth + 1))
    
    return min_girth

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

def _generate_random_k_regular(k, n):
    """Generate a random k-regular graph on n vertices using configuration model + 2-switches."""
    if k * n % 2 != 0:
        return None
    
    # Configuration model
    stubs = []
    for i in range(n):
        for _ in range(k):
            stubs.append(i)
    
    random.shuffle(stubs)
    
    # Pair up stubs to form edges
    adjacency = [set() for _ in range(n)]
    edges = []
    success = True
    for i in range(0, len(stubs), 2):
        u, v = stubs[i], stubs[i + 1]
        if u == v:
            success = False
            break
        if v in adjacency[u]:
            success = False
            break
        adjacency[u].add(v)
        adjacency[v].add(u)
        edges.append((u, v))
    
    if not success:
        # Try to fix by swapping stubs? Or just restart? 
        # For now, restart is simpler but might be slow if n is large.
        # Let's try a few times.
        for _ in range(10):
            random.shuffle(stubs)
            adjacency = [set() for _ in range(n)]
            edges = []
            success = True
            for i in range(0, len(stubs), 2):
                u, v = stubs[i], stubs[i + 1]
                if u == v or v in adjacency[u]:
                    success = False
                    break
                adjacency[u].add(v)
                adjacency[v].add(u)
                edges.append((u, v))
            if success:
                break
        if not success:
            return None

    # Check degrees
    for i in range(n):
        if len(adjacency[i]) != k:
            return None
    
    return adjacency

def _edge_switching_construct(k, g, n, time_limit=10.0):
    """Construct k-regular graph of girth >= g using edge switching."""
    start_time = time.time()
    
    # Generate initial random k-regular graph
    adjacency = _generate_random_k_regular(k, n)
    if adjacency is None:
        return None
    
    # Compute initial girth
    current_girth = _compute_girth(adjacency, k, g)
    
    # If girth is already >= g, we're done
    if current_girth >= g:
        edges = []
        for i in range(n):
            for j in adjacency[i]:
                if i < j:
                    edges.append((i, j))
        return n, edges
    
    # Prepare edge list for efficient switching
    all_edges = []
    for i in range(n):
        for j in adjacency[i]:
            if i < j:
                all_edges.append((i, j))
    
    m = len(all_edges)
    
    for _ in range(200000):  # Max iterations
        if time.time() - start_time > time_limit:
            break
        
        if m < 2:
            break
        
        # Pick two random edges
        # To avoid shared vertices, we can just pick and check.
        # With n=1000, m=2500, probability of sharing vertex is low.
        e1_idx = random.randint(0, m - 1)
        e2_idx = random.randint(0, m - 1)
        
        if e1_idx == e2_idx:
            continue
        
        a, b = all_edges[e1_idx]
        c, d = all_edges[e2_idx]
        
        # Check that the edges don't share vertices
        if a == c or a == d or b == c or b == d:
            continue
        
        # Try switch: remove (a,b) and (c,d), add (a,c) and (b,d)
        # Check that (a,c) and (b,d) are not already edges
        if c in adjacency[a] or d in adjacency[b]:
            continue
        
        # Perform switch
        adjacency[a].remove(b)
        adjacency[b].remove(a)
        adjacency[c].remove(d)
        adjacency[d].remove(c)
        adjacency[a].add(c)
        adjacency[c].add(a)
        adjacency[b].add(d)
        adjacency[d].add(b)
        
        # Update all_edges (remove old, add new)
        all_edges[e1_idx] = (a, c) if a < c else (c, a)
        all_edges[e2_idx] = (b, d) if b < d else (d, b)
        
        # Compute new girth
        new_girth = _compute_girth(adjacency, k, g)
        
        # Accept if girth improved or maintained
        if new_girth >= current_girth:
            current_girth = new_girth
            if current_girth >= g:
                edges = []
                for i in range(n):
                    for j in adjacency[i]:
                        if i < j:
                            edges.append((i, j))
                return n, edges
        else:
            # Revert switch
            adjacency[a].remove(c)
            adjacency[c].remove(a)
            adjacency[b].remove(d)
            adjacency[d].remove(b)
            adjacency[a].add(b)
            adjacency[b].add(a)
            adjacency[c].add(d)
            adjacency[d].add(c)
            all_edges[e1_idx] = (a, b)
            all_edges[e2_idx] = (c, d)
    
    # Final check
    if current_girth >= g:
        edges = []
        for i in range(n):
            for j in adjacency[i]:
                if i < j:
                    edges.append((i, j))
        return n, edges
    
    return None

def construct(k, g):
    # For the specific case k=5, g=9, use edge-switching approach
    if k == 5 and g == 9:
        start_time = time.time()
        time_limit = 145.0  # Leave some margin for safety
        
        # Try different n values, starting from smaller to larger
        # Must be even since 5*n must be even
        # LUW for (5,9) is typically around 2000+. We try to beat that.
        target_ns = [500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1800, 2000]
        
        for n in target_ns:
            if time.time() - start_time > time_limit:
                break
            
            # Try to construct with edge switching
            remaining_time = time_limit - (time.time() - start_time)
            if remaining_time < 5.0:
                break
            
            result = _edge_switching_construct(k, g, n, time_limit=min(10.0, remaining_time))
            if result is not None:
                return result
    
    # Fallback to LUW
    return _luw_construct(k, g)
