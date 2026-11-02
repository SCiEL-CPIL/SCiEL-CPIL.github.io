"""Probabilistic voltage-lift construction for k-regular graphs of large girth."""
import time
import random
import itertools
import sys

# EVOLVE-BLOCK-START

def _build_lift(base_adj, M):
    """Build the M-fold voltage lift.
    base_adj: list of lists of (neighbor_index, voltage)
    M: group size
    Returns: n, edges (list of (u, v))
    """
    n = M * len(base_adj)
    edges = []
    for i in range(len(base_adj)):
        for (j, v) in base_adj[i]:
            if i < j:
                for x in range(M):
                    u = i * M + x
                    w = j * M + (x + v) % M
                    edges.append((u, w))
            elif i == j:
                # Self-loop in base: for each x, edge to (x+v) mod M
                # Need to add each undirected edge once
                for x in range(M):
                    w = (x + v) % M
                    if x < w:
                        edges.append((i * M + x, i * M + w))
                    elif w == 0 and x == 0:
                        pass  # loop, skip
                    # Actually for self-loops in base, we need to be careful
                    # Since base is simple, no self-loops, so this won't happen
    return n, edges


def _girth_fast(n, edges, g_target):
    """Check if girth is at least g_target. Returns True if no cycle shorter than g_target."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    for s in range(n):
        # BFS with non-backtracking, stop at depth g_target-1
        # If we find a cycle of length < g_target, return False
        visited = {}
        queue = []
        for nb in adj[s]:
            state = (nb, s)
            visited[state] = 1
            queue.append((nb, s, 1))
        
        head = 0
        while head < len(queue):
            v, p, depth = queue[head]
            head += 1
            
            if depth >= g_target - 1:
                continue
            
            for nb in adj[v]:
                if nb == p:
                    continue
                if nb == s:
                    # Cycle of length depth+1 < g_target
                    return False
                state = (nb, v)
                if state not in visited:
                    visited[state] = depth + 1
                    queue.append((nb, v, depth + 1))
    
    return True


def _girth_non_backtracking(n, edges, max_girth):
    """Compute girth using non-backtracking walks from each vertex.
    Returns the girth (length of shortest cycle) or max_girth+1 if none found.
    """
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    best = max_girth + 1
    
    for s in range(n):
        visited = {}
        queue = []
        for nb in adj[s]:
            state = (nb, s)
            visited[state] = 1
            queue.append((nb, s, 1))
        
        head = 0
        while head < len(queue):
            v, p, depth = queue[head]
            head += 1
            
            if depth >= best:
                continue
            
            for nb in adj[v]:
                if nb == p:
                    continue
                if nb == s:
                    if depth + 1 < best:
                        best = depth + 1
                    continue
                state = (nb, v)
                if state not in visited:
                    visited[state] = depth + 1
                    queue.append((nb, v, depth + 1))
    
    return best


def _try_k5_lift(M, g, deadline):
    """Try to find a valid voltage assignment on K5 base with M-fold lift."""
    base_edges = [(i, j) for i in range(5) for j in range(i+1, 5)]
    
    # Number of random attempts based on time budget
    max_attempts = 100
    for attempt in range(max_attempts):
        if time.time() > deadline:
            return None
        
        # Random voltages in Z_M
        voltages = [random.randint(0, M-1) for _ in range(len(base_edges))]
        
        # Build base_adj
        base_adj = [[] for _ in range(5)]
        for idx, (i, j) in enumerate(base_edges):
            v = voltages[idx]
            base_adj[i].append((j, v))
            base_adj[j].append((i, M - v if v != 0 else 0))
        
        n, edges = _build_lift(base_adj, M)
        
        # Check girth
        if _girth_fast(n, edges, g):
            return (n, edges)
    
    return None


def _try_cage_knowledge(k, g):
    """
    For known cages, return the known optimal construction.
    For k=4, g=10, the cage has 320 vertices.
    We use a specific known construction if possible.
    """
    # The (4,10)-cage is known to have 320 vertices.
    # One known construction is via the generalized Petersen graph or
    # specific voltage assignments on K5.
    # Let's try a systematic approach for small M values.
    # Actually, let's just return the LUW construction which gives a valid graph.
    # The LUW graph for k=4, g=10 has 2*q^(d+1) vertices where d = g-5 = 5, q=4
    # That's 2*4^6 = 8192 vertices, which is much larger than 320.
    # So we need the voltage lift approach.
    
    # For now, return None to let the main function handle it.
    return None


def _luz(k, g):
    """Lazebnik-Ustimenko-Woldar graph as fallback."""
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


def construct(k, g):
    """Construct a k-regular graph with girth at least g on as few vertices as possible."""
    deadline = time.time() + 130  # safety margin before 150s kill
    
    # Moore bound for k=4, g=10 is 243
    # Known cage: 320 vertices
    
    # Strategy: Try voltage lifts on small base graphs.
    # For k=4, K5 is a natural 4-regular base graph with 5 vertices.
    # M-fold lift has 5*M vertices.
    # Try M from 49 (n=245) up to 64 (n=320)
    
    best_result = None
    best_n = float('inf')
    
    if k == 4:
        # Try K5 voltage lifts
        # Start with larger M (more likely to have high girth) and work down
        # Actually, start from the known optimum and try to find smaller
        # But we want the smallest n, so try smaller M first
        
        # Try M values: 49, 50, 51, ..., 64
        for M in range(49, 65):
            if time.time() > deadline - 5:
                break
            
            n = 5 * M
            if n >= best_n:
                continue
            
            # Try random voltage assignments
            result = _try_k5_lift(M, g, deadline - 5)
            if result is not None:
                n_found, edges_found = result
                if n_found < best_n:
                    best_n = n_found
                    best_result = result
                # If we found a valid graph, try smaller M
                # But we're already going from small to large M
                # So we can continue to try smaller M... wait, we're going up.
                # Let's reverse: try from large M down to small M
                # Actually, the loop goes 49, 50, ..., 64
                # If we find a valid graph at M=64, n=320, we should try M=49-63
                # But the loop already does that order.
                # The issue is: if M=49 doesn't work, we try M=50, etc.
                # This is correct: we want the smallest n, so smallest M first.
        
        # If no K5 lift worked, try other base graphs or fallback
        if best_result is None:
            # Try a different approach: systematic voltage assignment
            # For K5, there are 10 edges, each with voltage in Z_M
            # Total possibilities: M^10 which is huge
            # Let's try a few more random attempts for each M
            
            for M in range(49, 65):
                if time.time() > deadline - 5:
                    break
                n = 5 * M
                if n >= best_n:
                    continue
                
                # More attempts for smaller M
                max_attempts = 200 if M < 55 else 100
                for attempt in range(max_attempts):
                    if time.time() > deadline - 5:
                        break
                    
                    voltages = [random.randint(0, M-1) for _ in range(10)]
                    base_adj = [[] for _ in range(5)]
                    base_edges = [(i, j) for i in range(5) for j in range(i+1, 5)]
                    for idx, (i, j) in enumerate(base_edges):
                        v = voltages[idx]
                        base_adj[i].append((j, v))
                        base_adj[j].append((i, M - v if v != 0 else 0))
                    
                    n, edges = _build_lift(base_adj, M)
                    if _girth_fast(n, edges, g):
                        if n < best_n:
                            best_n = n
                            best_result = (n, edges)
                        break
    
    # Fallback: LUW construction
    if best_result is None:
        best_result = _luz(k, g)
    
    return best_result

# EVOLVE-BLOCK-END
