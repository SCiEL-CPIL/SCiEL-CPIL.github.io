"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # For k=5, g=9: use a bipartite construction over GF(5)
    # Points: (a, b) in GF(5)^2, Lines: (c, d) in GF(5)^2
    # Incidence: point (a,b) ~ line (c,d) iff b = c*a + d (a line in the affine plane)
    # This gives a 5-regular bipartite graph (each point on 5 lines, each line through 5 points)
    # Total vertices: 2 * 25 = 50
    # But girth may not be 9. Let's use a larger field or different construction.
    
    # Better: use a construction over GF(q) with q chosen to get good girth
    # For 5-regular, we need degree 5 on each side.
    
    # Use a bipartite graph where:
    # Left side: elements of GF(5^2) = 25 elements
    # Right side: elements of GF(5^2) = 25 elements  
    # Edge between x (left) and y (right) iff y = x^5 * a + b for specific a,b
    # This is getting complex. Let me use a simpler explicit construction.
    
    # Use the incidence graph of the affine plane AG(2,5):
    # Points: (i,j) for i,j in 0..4 -> 25 points
    # Lines: for each slope m in 0..4 and intercept b in 0..4, line {(i, m*i+b) : i in 0..4}
    #   plus vertical lines {(i, b) : i in 0..4} for b in 0..4 -> 25 + 5 = 30 lines? No.
    # Actually in AG(2,5): 25 points, 30 lines (5 slopes * 5 intercepts + 5 vertical)
    # Each point is on 6 lines (5 non-vertical + 1 vertical), so degree 6, not 5.
    
    # Let me use a different approach: a 5-regular bipartite graph with 2*n vertices
    # constructed from a 5-regular graph on n vertices via a covering or similar.
    
    # Simplest reliable approach: use a random 5-regular graph with many vertices
    # and verify girth. For n large enough, random 5-regular graphs have girth ~ 2*log_4(n)
    # For girth >= 9, we need 2*log_4(n) >= 9, so n >= 4^4.5 ~ 1024
    
    # Let me try n = 2000 vertices for a 5-regular graph
    # Expected girth ~ 2*log_4(2000) ~ 2*5.3 ~ 10.6, so girth >= 9 is likely
    
    def is_5_regular(n, edges):
        degree = [0] * n
        edge_set = set()
        for u, v in edges:
            if u == v:
                return False
            if (u, v) in edge_set or (v, u) in edge_set:
                return False
            edge_set.add((min(u,v), max(u,v)))
            degree[u] += 1
            degree[v] += 1
        return all(d == 5 for d in degree)
    
    def check_girth_nb(n, edges, min_girth):
        """Check girth using non-backtracking walks from each vertex.
        A cycle of length L exists iff there's a non-backtracking walk of length L
        from a vertex back to itself. We check all non-backtracking walks up to length
        min_girth-1 from each vertex."""
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # For each start vertex, do BFS on the directed edge graph (non-backtracking)
        # State: (current_vertex, previous_vertex) where prev is the vertex we came from
        # We want to find if we can return to start without backtracking
        for start in range(n):
            # visited: set of (node, parent) states reached
            # dist: dictionary mapping (node, parent) -> distance
            from collections import deque
            visited = set()
            queue = deque()
            # Initialize: walk from start to each neighbor
            for nb in adj[start]:
                state = (nb, start)
                visited.add(state)
                queue.append((nb, start, 1))
            
            while queue:
                node, parent, depth = queue.popleft()
                if depth >= min_girth:
                    continue
                for nb in adj[node]:
                    if nb == parent:
                        continue  # No backtracking
                    if nb == start:
                        # Found a cycle of length depth+1
                        if depth + 1 < min_girth:
                            return False
                        else:
                            continue
                    state = (nb, node)
                    if state not in visited:
                        visited.add(state)
                        queue.append((nb, node, depth + 1))
        return True
    
    def generate_random_5_regular(n, seed):
        """Generate a random 5-regular graph on n vertices."""
        if n < 5:
            return None
        if (n * 5) % 2 != 0:
            return None
        
        rng = random.Random(seed)
        
        # Use configuration model
        half_edges = [i for i in range(n) for _ in range(5)]
        rng.shuffle(half_edges)
        
        # Pair up half-edges
        edges = set()
        i = 0
        while i < len(half_edges):
            j = i + 1
            if half_edges[i] == half_edges[j]:
                # Self-loop, try to swap
                found = False
                for k in range(i + 2, len(half_edges)):
                    if half_edges[k] != half_edges[i] and half_edges[k] != half_edges[j]:
                        half_edges[j], half_edges[k] = half_edges[k], half_edges[j]
                        found = True
                        break
                if not found:
                    return None
            u, v = half_edges[i], half_edges[j]
            if u != v:
                edges.add((min(u,v), max(u,v)))
            i += 2
        
        if len(edges) != n * 5 // 2:
            return None
        
        return list(edges)
    
    # Try to find a 5-regular graph with girth >= 9
    # Start with reasonable sizes and try multiple seeds
    best_result = None
    best_n = float('inf')
    
    # Try sizes from smaller to larger, but we want minimum n with girth >= 9
    # Moore bound for k=5, g=9 (odd): n >= 1 + k * sum_{i=0}^{3} (k-1)^i = 1 + 5*(1+4+16+64) = 1 + 5*85 = 426
    # So n must be at least 426. Let's try from 426 upward.
    
    # For efficiency, try a few sizes with multiple random seeds
    sizes_to_try = [426, 430, 434, 438, 442, 446, 450, 454, 458, 462, 466, 470, 474, 478, 482, 486, 490, 494, 498, 502, 506, 510, 514, 518, 522, 526, 530, 534, 538, 542]
    
    for n in sizes_to_try:
        if (n * 5) % 2 != 0:
            continue
        if time.time() > deadline - 30:
            break
        
        # Try multiple random seeds for this n
        max_seeds = 30
        for seed in range(max_seeds):
            if time.time() > deadline - 15:
                break
            
            edges = generate_random_5_regular(n, seed)
            if edges is None:
                continue
            
            # Verify 5-regular
            degree = [0] * n
            for u, v in edges:
                degree[u] += 1
                degree[v] += 1
            if not all(d == 5 for d in degree):
                continue
            
            # Check girth
            if check_girth_nb(n, edges, g):
                # Valid graph found
                if n < best_n:
                    best_n = n
                    best_result = (n, edges)
                break  # Found one for this n, move to next n
    
    if best_result is not None:
        return best_result
    
    # Fallback: use the original LUW construction
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
# EVOLVE-BLOCK-END
