import time
import sys
import random
import numpy as np
from collections import deque, defaultdict

_DEADLINE = 115.0

def construct(k, g):
    t_start = time.time()
    
    # Fallback for general k, g
    if k != 4 or g != 10:
        return _luw(k, g)
    
    # Specific search for (4, 10) cage
    # Moore bound for (4, 10) is 246.
    # Best known is 320.
    
    # 1. Quick circulant search (fast, limited range)
    # We search for a circulant graph C(n, {s1, s2})
    for n in range(246, 330):
        if time.time() - t_start > _DEADLINE - 30:
            break
        max_step = n // 2
        for s1 in range(1, max_step + 1):
            if time.time() - t_start > _DEADLINE - 30:
                break
            for s2 in range(s1 + 1, max_step + 1):
                if (s1 + s2) % n == 0:
                    continue
                if _check_girth_circulant(n, s1, s2, 10):
                    edges = []
                    for u in range(n):
                        v1 = (u + s1) % n
                        v2 = (u + s2) % n
                        if u < v1:
                            edges.append((u, v1))
                        if u < v2:
                            edges.append((u, v2))
                    return n, edges

    # 2. Stochastic Local Search (Simulated Annealing)
    # Try to find a graph with girth >= 10 on n vertices.
    # We start with n=320 (best known) and try to improve if possible, 
    # but primarily we want to find *any* valid graph quickly if the 
    # circulant search fails (which it likely will for n<320).
    # Actually, if circulant fails, we should try to find a graph of size 320.
    # If we find one, we return it.
    # If we have time, we can try smaller n.
    
    # Let's try n=320 first as it's the target.
    # Then if we have time, try n=300, 280, etc.
    
    target_ns = [320]
    # If we have lots of time, try smaller?
    # But 320 is the "best known". Beating it is hard.
    # The task says "Score: n, minimized".
    # So if we find a valid graph with n=320, score is 320.
    # If we find one with n=300, score is 300.
    # The seed program returns LUW result.
    # LUW for k=4, g=10: d=5, q=4.
    # Vertices: 2 * q^d = 2 * 4^5 = 2048.
    # So current score is 2048.
    # Any n < 2048 is an improvement.
    # So finding a valid graph at n=320 gives score 320/2048 = 0.156?
    # Wait, "Score: n, minimized; it is an upper bound...".
    # "search score is seed/value".
    # Seed value is the n returned by seed program.
    # Seed program returns LUW result which is 2048.
    # So if I return 320, score = 2048/320 = 6.4.
    # If I return 2048, score = 1.0.
    # So I MUST find a graph smaller than 2048.
    # n=320 is a huge improvement.
    
    # So the priority is to find a valid graph with n close to 320.
    # If I can't find one, I fall back to LUW (2048).
    
    # Let's run the local search for n=320.
    result = _stochastic_local_search(k, g, 320, t_start)
    if result is not None:
        return result
    
    # If 320 fails, try slightly larger? No, larger is worse score.
    # Try smaller? 300, 280...
    # But girth 10 requires n >= 246.
    # Let's try a few sizes in descending order of likelihood of success
    # (i.e., closer to 320 is more likely to have a solution than 246).
    # But 320 is the "cage". It might be unique or rare.
    # Actually, the cage is the *smallest* graph of girth g.
    # So n=320 is the target.
    # If I can't find it, I should return the best I found.
    # But I must return a *valid* graph.
    # If I can't find a valid graph with n < 2048, I return LUW.
    
    # Let's try a few other sizes if 320 fails.
    # Maybe 324, 328? No, those are larger.
    # Maybe 300?
    # Let's stick to 320 as the primary target for the stochastic search.
    # If it fails, maybe try 320 again with different seeds?
    
    # For now, if 320 fails, try 320 again with a different random seed?
    # Or just fall back to LUW.
    # Given the 120s limit, and the difficulty, let's do a few restarts for n=320.
    
    for i in range(5):
        if time.time() - t_start > _DEADLINE - 5:
            break
        random.seed(12345 + i)
        result = _stochastic_local_search(k, g, 320, t_start)
        if result is not None:
            return result

    # Fallback to LUW
    return _luw(k, g)

def _check_girth_circulant(n, s1, s2, g):
    """
    Check if circulant graph C(n, {s1, s2}) has girth at least g.
    """
    # Expected counts for k=4, g=10 (radius 4)
    # d=0: 1
    # d=1: 4
    # d=2: 12
    # d=3: 36
    # d=4: 108
    expected = [1, 4, 12, 36, 108]
    
    # BFS from 0
    visited = [-1] * n
    dist = [0] * n
    q = deque()
    q.append(0)
    visited[0] = 0
    dist[0] = 0
    
    counts = [0] * 5
    counts[0] = 1
    
    while q:
        u = q.popleft()
        d = dist[u]
        if d >= 4:
            continue
        
        neighbors = [(u + s1) % n, (u - s1) % n, (u + s2) % n, (u - s2) % n]
        
        for v in neighbors:
            if visited[v] == -1:
                visited[v] = d + 1
                dist[v] = d + 1
                q.append(v)
                if d + 1 < 5:
                    counts[d + 1] += 1
            else:
                if dist[v] <= d:
                    cycle_len = d + 1 + dist[v]
                    if cycle_len < g:
                        return False
    
    for d in range(1, 5):
        if counts[d] < expected[d]:
            return False
            
    return True

def _stochastic_local_search(k, g, n, t_start):
    """
    Stochastic local search using 2-switches to maximize girth.
    """
    # Generate random k-regular graph
    adjacency = _generate_random_k_regular(n, k)
    if adjacency is None:
        return None
    
    # Initial girth check
    current_girth = _compute_girth(adjacency, g)
    if current_girth >= g:
        return _graph_to_edges(adjacency)
    
    # Simulated Annealing
    # Objective: minimize number of short cycles (length < g)
    # Or maximize girth.
    # Let's use "penalty" = sum of 1/length for cycles of length < g?
    # Or just count of cycles of length < g.
    # Counting all cycles is expensive.
    # Let's use a proxy: number of vertices at distance < g/2 that have "collisions"?
    # No, let's stick to counting short cycles via BFS if we can do it fast.
    # For n=320, BFS from each vertex is 320 * (4^4) ~ 320 * 256 ~ 80k ops.
    # 320 * 80k = 25M ops. Might be too slow for many iterations.
    
    # Alternative: use the "girth" as the objective.
    # But computing girth is the expensive part.
    
    # Let's try a heuristic:
    # Randomly pick a 2-switch.
    # Check if it reduces the number of short cycles.
    # To check efficiently, we only need to check the local neighborhood of the switch.
    # But a 2-switch can affect cycles globally?
    # No, a 2-switch only changes the edges incident to 4 vertices.
    # Any cycle affected must pass through one of these 4 vertices.
    # So we can re-evaluate the girth by checking cycles through these vertices?
    # That's still complex.
    
    # Let's try a simpler approach:
    # Just run the search for a fixed number of steps, checking girth periodically.
    
    steps = 0
    max_steps = 5000
    temperature = 1.0
    cooling = 0.99
    
    # Pre-compute adjacency list for faster access
    adj_list = [adjacency[i] for i in range(n)]
    
    while steps < max_steps:
        if time.time() - t_start > _DEADLINE - 10:
            break
        
        # Pick a random 2-switch
        # Pick two random edges (u1, v1) and (u2, v2)
        # Such that they are not adjacent and u1!=u2, etc.
        # And the new edges (u1, v2) and (u2, v1) are not already present.
        
        # To pick random edges, maintain a list of edges?
        # Or pick random u, then random neighbor v.
        # Collision check is needed.
        
        u1 = random.randint(0, n - 1)
        v1 = random.choice(adj_list[u1])
        
        # Find another edge
        found = False
        while not found:
            u2 = random.randint(0, n - 1)
            if u2 == u1 or u2 == v1:
                continue
            # Check if u2 is neighbor of u1 or v1?
            # If u2 is neighbor of u1, then (u1, v1) and (u2, v2) share u1? No, u2 is distinct.
            # But if u2 is neighbor of v1, then (u1, v1) and (u2, v2) share v1?
            # We need the edges to be disjoint.
            if u2 in adj_list[u1] or u2 in adj_list[v1]:
                continue
            
            v2 = random.choice(adj_list[u2])
            if v2 == u1 or v2 == v1:
                continue
            if v2 in adj_list[u1] or v2 in adj_list[v1]:
                continue
            
            # Check if new edges exist
            if v2 in adj_list[u1] or v1 in adj_list[u2]:
                continue
            
            found = True
        
        # Apply 2-switch
        # Remove (u1, v1) and (u2, v2)
        adj_list[u1].remove(v1)
        adj_list[v1].remove(u1)
        adj_list[u2].remove(v2)
        adj_list[v2].remove(u2)
        
        # Add (u1, v2) and (u2, v1)
        adj_list[u1].append(v2)
        adj_list[v2].append(u1)
        adj_list[u2].append(v1)
        adj_list[v1].append(u2)
        
        # Update adjacency matrix (for girth check)
        # This is O(n^2) to update matrix?
        # We can keep the matrix and update 4 entries.
        # But we don't have the matrix in this scope?
        # We passed adjacency (matrix) to _stochastic_local_search.
        # Let's update it.
        
        adjacency[u1][v1] = 0
        adjacency[v1][u1] = 0
        adjacency[u2][v2] = 0
        adjacency[v2][u2] = 0
        
        adjacency[u1][v2] = 1
        adjacency[v2][u1] = 1
        adjacency[u2][v1] = 1
        adjacency[v1][u2] = 1
        
        steps += 1
        
        # Check girth periodically
        if steps % 100 == 0:
            current_girth = _compute_girth(adjacency, g)
            if current_girth >= g:
                return _graph_to_edges(adjacency)
        
        # Cooling
        temperature *= cooling
    
    # Final check
    current_girth = _compute_girth(adjacency, g)
    if current_girth >= g:
        return _graph_to_edges(adjacency)
    
    return None

def _generate_random_k_regular(n, k):
    """
    Generate a random k-regular graph on n vertices.
    Returns adjacency matrix (numpy array) or None.
    """
    # Use the configuration model
    # Create a list of 2*E = n*k stubs
    stubs = []
    for i in range(n):
        for _ in range(k):
            stubs.append(i)
    
    random.shuffle(stubs)
    
    # Pair them up
    adjacency = np.zeros((n, n), dtype=int)
    
    for i in range(0, n * k, 2):
        u = stubs[i]
        v = stubs[i + 1]
        if u == v:
            # Self-loop, invalid for simple graph
            # Try to fix? Or restart?
            # For simplicity, restart if self-loop.
            # But this is slow.
            # Let's just return None and let the caller retry?
            # Or use a different generation method.
            # Let's use the "switching" method to fix self-loops/multi-edges.
            pass
        
        adjacency[u][v] = 1
        adjacency[v][u] = 1
    
    # Check for multi-edges (simple graph required)
    # If any entry > 1, it's a multi-edge.
    if np.any(adjacency > 1):
        # Try to fix with 2-switches?
        # Or just return None.
        # For now, return None.
        return None
    
    return adjacency

def _compute_girth(adjacency, max_girth):
    """
    Compute the girth of the graph.
    If girth >= max_girth, return max_girth (or larger).
    """
    n = adjacency.shape[0]
    # BFS from each vertex
    # Optimization: if we find a cycle of length L, girth <= L.
    # We want to know if girth >= max_girth.
    # So we stop if we find a cycle < max_girth.
    
    for start in range(n):
        # BFS from start
        # dist[v] = distance from start
        # parent[v] = parent in BFS tree
        dist = [-1] * n
        parent = [-1] * n
        q = deque()
        q.append(start)
        dist[start] = 0
        parent[start] = -2 # Mark root
        
        while q:
            u = q.popleft()
            d = dist[u]
            if d >= max_girth // 2:
                continue
            
            for v in range(n):
                if adjacency[u][v]:
                    if dist[v] == -1:
                        dist[v] = d + 1
                        parent[v] = u
                        q.append(v)
                    else:
                        # Found a cycle
                        # Cycle length = dist[u] + dist[v] + 1
                        # But we need to ensure it's a simple cycle.
                        # In BFS, if u and v are neighbors and dist[u] != dist[v] + 1 and dist[v] != dist[u] + 1?
                        # Actually, if u is at d and v is at d' and they are connected.
                        # If d' == d + 1, it's a tree edge (forward).
                        # If d' == d - 1, it's a back edge to parent.
                        # If d' == d, it's a cross edge. Cycle length 2d + 1? No.
                        # If d' < d, cycle length d - d' + 1?
                        
                        # Let's just compute the cycle length properly.
                        # The cycle is path start->...->u->v->...->start?
                        # No, path start->...->u and start->...->v.
                        # If u-v is an edge, cycle is path(start, u) + edge(u, v) + path(v, start).
                        # Length = dist[u] + 1 + dist[v].
                        # This is valid if u != parent[v] and v != parent[u].
                        
                        if v != parent[u] and u != parent[v]:
                            cycle_len = dist[u] + 1 + dist[v]
                            if cycle_len < max_girth:
                                return cycle_len
                            
    return max_girth

def _graph_to_edges(adjacency):
    n = adjacency.shape[0]
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if adjacency[u][v]:
                edges.append((u, v))
    return n, edges

def _luw(k, g):
    """Lazebnik-Ustimenko-Woldar construction CD(d, q)."""
    q = k
    if (g - 5) % 2 == 0:
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
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    from collections import deque as dq
    start = (0, tuple([None] + [0] * d))

    def neighbours(vertex):
        side, x = vertex
        x = list(x)
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    index = {start: 0}
    order = [start]
    edges = []

    for vi, vertex in enumerate(order):
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))

    return len(order), edges
