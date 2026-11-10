import time
import random
import math
import sys

def _luw_construct(k, g):
    # Seed logic
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

def _check_girth_fast(adj, n, min_girth):
    """Check if graph has girth >= min_girth using BFS from each vertex. Early exit."""
    max_depth = min_girth // 2 - 1
    for start in range(n):
        dist = [-1] * n
        dist[start] = 0
        queue = [start]
        qi = 0
        while qi < len(queue):
            node = queue[qi]
            qi += 1
            d = dist[node]
            if d >= max_depth:
                continue
            for nb in adj[node]:
                if dist[nb] == -1:
                    dist[nb] = d + 1
                    queue.append(nb)
                elif dist[nb] <= d:
                    cycle_len = d + dist[nb] + 1
                    if cycle_len < min_girth:
                        return False
    return True

def _count_4cycles(adj, n):
    """Count 4-cycles in a bipartite graph (A: 0..half-1, B: half..n-1)."""
    half = n // 2
    count = 0
    # For each pair of A vertices, count common B neighbors
    for u in range(half):
        au = adj[u]
        for v in range(u + 1, half):
            common = len(au & adj[v])
            if common >= 2:
                count += common * (common - 1) // 2
    return count

def construct(k, g):
    if k != 4 or g != 10:
        return _luw_construct(k, g)
    
    deadline = time.time() + 115
    
    # First, get the LUW construction as fallback
    luw_n, luw_edges = _luw_construct(k, g)
    best_n = luw_n
    best_edges = luw_edges
    
    # Target the known cage size: 320
    # We will try to find a valid graph of size 320.
    # If we fail, we try 322, 324, etc., up to luw_n.
    
    def generate_random_bipartite_4reg(n):
        """Generate a random 4-regular bipartite graph on n vertices (n even)."""
        half = n // 2
        adj = [set() for _ in range(n)]
        
        a_halfends = []
        for i in range(half):
            a_halfends.extend([i] * 4)
        b_halfends = []
        for i in range(half):
            b_halfends.extend([half + i] * 4)
        
        random.shuffle(b_halfends)
        for a, b in zip(a_halfends, b_halfends):
            adj[a].add(b)
            adj[b].add(a)
        
        # Check for multi-edges (should not happen with sets, but degree check is good)
        for i in range(n):
            if len(adj[i]) != k:
                return None
        
        return adj

    def local_search(adj, n, deadline):
        """Local search to maximize girth via edge swaps."""
        best_adj = [set(s) for s in adj]
        half = n // 2
        adj_lists = [list(s) for s in best_adj]
        
        def recompute_lists():
            nonlocal adj_lists
            adj_lists = [list(s) for s in best_adj]
        
        def swap(u1, v1, u2, v2):
            """Perform swap: (u1,v1)+(u2,v2) -> (u1,v2)+(u2,v1)"""
            best_adj[u1].remove(v1)
            best_adj[v1].remove(u1)
            best_adj[u2].remove(v2)
            best_adj[v2].remove(u2)
            best_adj[u1].add(v2)
            best_adj[v2].add(u1)
            best_adj[u2].add(v1)
            best_adj[v1].add(u2)
            
        def unswap(u1, v1, u2, v2):
            """Revert swap: (u1,v2)+(u2,v1) -> (u1,v1)+(u2,v2)"""
            best_adj[u1].remove(v2)
            best_adj[v2].remove(u1)
            best_adj[u2].remove(v1)
            best_adj[v1].remove(u2)
            best_adj[u1].add(v1)
            best_adj[v1].add(u1)
            best_adj[u2].add(v2)
            best_adj[v2].add(u2)
            
        # Initial girth check
        if _check_girth_fast(best_adj, n, g):
            edges = []
            for i in range(half):
                for j in best_adj[i]:
                    edges.append((i, j))
            return n, edges
        
        # Score: minimize 4-cycles
        score_4 = _count_4cycles(best_adj, n)
        best_score_4 = score_4
        best_adj_copy = [set(s) for s in best_adj]
        
        # Simulated Annealing style
        temp = 100.0
        cooling = 0.99
        
        for iteration in range(50000):
            if time.time() > deadline:
                break
            
            # Try multiple swaps per iteration? No, one at a time is fine.
            u1 = random.randint(0, half - 1)
            u2 = random.randint(0, half - 1)
            if u1 == u2:
                continue
            v1 = random.choice(adj_lists[u1])
            v2 = random.choice(adj_lists[u2])
            if v1 == v2:
                continue
            if v2 in best_adj[u1] or v1 in best_adj[u2]:
                continue
            
            # Perform swap
            swap(u1, v1, u2, v2)
            recompute_lists()
            
            # Quick check: count 4-cycles
            new_score_4 = _count_4cycles(best_adj, n)
            
            delta = new_score_4 - best_score_4
            
            if delta <= 0:
                # Improvement or equal
                if delta < 0:
                    best_score_4 = new_score_4
                    best_adj_copy = [set(s) for s in best_adj]
                    # Check girth if we improved significantly
                    if _check_girth_fast(best_adj, n, g):
                        edges = []
                        for i in range(half):
                            for j in best_adj[i]:
                                edges.append((i, j))
                        return n, edges
                else:
                    # Equal score: accept with probability 1 (or high)
                    # Check girth occasionally
                    if iteration % 100 == 0 and _check_girth_fast(best_adj, n, g):
                        edges = []
                        for i in range(half):
                            for j in best_adj[i]:
                                edges.append((i, j))
                        return n, edges
            else:
                # Worse: accept with probability exp(-delta/temp)
                if random.random() < math.exp(-delta / temp):
                    pass # Accept
                else:
                    # Revert
                    unswap(u1, v1, u2, v2)
                    recompute_lists()
            
            temp *= cooling
        
        # Final check on the best found so far
        if _check_girth_fast(best_adj_copy, n, g):
            edges = []
            half = n // 2
            for i in range(half):
                for j in best_adj_copy[i]:
                    edges.append((i, j))
            return n, edges
            
        return None

    # Try n=320 first (known cage size)
    # Then 322, 324, ...
    # Time budget: 115s total.
    # Allocate time based on n.
    
    # We know 320 is the target.
    # Let's try 320 with most of the time.
    
    t_start = time.time()
    # Give 320 a generous time budget
    time_for_320 = 60.0
    deadline_320 = t_start + time_for_320
    
    adj = generate_random_bipartite_4reg(320)
    if adj:
        result = local_search(adj, 320, deadline_320)
        if result:
            return result
    
    # If 320 failed, try 322, 324...
    # Remaining time
    time_left = deadline - time.time()
    if time_left < 5:
        return best_n, best_edges
        
    for n in range(322, min(luw_n + 1, 400), 2):
        if time.time() > deadline - 3:
            break
        
        # Allocate remaining time / number of steps
        # Just try a few attempts
        for attempt in range(5):
            if time.time() > deadline - 3:
                break
            adj = generate_random_bipartite_4reg(n)
            if adj is None:
                continue
            result = local_search(adj, n, deadline - 3)
            if result is not None:
                return result
    
    return best_n, best_edges
