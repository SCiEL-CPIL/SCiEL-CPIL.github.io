"""Construct 5-regular girth-9 cage candidates.

Uses the Lazebnik-Ustimenko-Woldar (LUW) construction as a valid baseline, 
and attempts to find a smaller graph using random initialization and local search 
(edge swaps) to eliminate short cycles, targeting the known cage parameters.
"""
import time
import random
from collections import deque

# EVOLVE-BLOCK-START

def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k.

    Returns (n, edges) for a k-regular bipartite graph with girth at least
    g (for odd g, the construction gives girth >= g exactly when d = g-4).
    """
    q = k
    # Ensure d is large enough for the required girth
    # For LUW, girth is at least 2d+2? Or d+5?
    # The original code used: d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    # For g=9, d=5.
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add_op = lambda x, y: x ^ y
        sub_op = lambda x, y: x ^ y
        mul_op = lambda x, y: table[x][y]
    else:
        add_op = lambda x, y: (x + y) % q
        sub_op = lambda x, y: (x - y) % q
        mul_op = lambda x, y: x * y % q

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
                    y[c] = add_op(x[c], mul_op(y[a], x[b]))
                else:
                    y[c] = sub_op(x[c], mul_op(x[a], y[b]))
            yield 1 - side, tuple(y)

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


def _check_girth(adj, n, g, deadline):
    """Check if graph has girth >= g using non-backtracking BFS from each vertex."""
    for v in range(n):
        if time.time() > deadline:
            return False
        # BFS: (vertex, prev) -> depth
        seen = {}
        queue = deque()
        for nb in adj[v]:
            queue.append((nb, v, 1))
            seen[(nb, v)] = 1
        while queue:
            cur, prev, depth = queue.popleft()
            if depth >= (g - 1) // 2:
                continue
            for nb in adj[cur]:
                if nb == prev:
                    continue
                if nb == v:
                    if depth + 1 < g:
                        return False
                elif (nb, cur) in seen:
                    cycle_len = seen[(nb, cur)] + depth + 1
                    if cycle_len < g:
                        return False
                else:
                    seen[(nb, cur)] = depth + 1
                    queue.append((nb, cur, depth + 1))
    return True


def _generate_random_5_reg(n, seed=None):
    """Generate a random 5-regular bipartite graph on n vertices (n must be even)."""
    if n % 2 != 0:
        return None
    half = n // 2
    rng = random.Random(seed)
    
    # Configuration model for bipartite graphs
    # Left side: 0..half-1, Right side: half..n-1
    left_stubs = []
    for i in range(half):
        left_stubs.extend([i] * 5)
    right_stubs = []
    for j in range(half):
        right_stubs.extend([j + half] * 5)
    
    rng.shuffle(left_stubs)
    rng.shuffle(right_stubs)
    
    adj = [set() for _ in range(n)]
    for ls, rs in zip(left_stubs, right_stubs):
        adj[ls].add(rs)
        adj[rs].add(ls)
        
    # Check if simple (no multi-edges in this bipartite construction is guaranteed by sets, 
    # but we must ensure degree 5)
    if any(len(adj[i]) != 5 for i in range(n)):
        return None
        
    # Convert to list of lists for faster access if needed, or keep sets
    return adj


def _local_search_5_9(max_time=100.0):
    """Attempt to find a 5-regular graph with girth >= 9 and fewer vertices than baseline."""
    start_time = time.time()
    deadline = start_time + max_time
    
    # Try candidate sizes. 
    # Known smallest is 1116. Moore bound is 426.
    # We try sizes that are likely to have solutions.
    # If we find a valid graph at n=1000, score is huge.
    # We start from smaller n and go up, or start from known good and try to go down?
    # Random search is more likely to succeed at larger n (less constrained).
    # Let's try a range of n values.
    
    candidate_ns = [428, 500, 600, 800, 1000, 1116, 1200, 1500, 2000, 3000, 5000]
    # Filter to even numbers
    candidate_ns = [n for n in candidate_ns if n % 2 == 0]
    
    best_n = None
    best_edges = None
    
    random.seed(42)
    
    for n in candidate_ns:
        if time.time() > deadline - 5:
            break
            
        max_restarts = 20
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break
                
            adj = _generate_random_5_reg(n, seed=42 + restart)
            if adj is None:
                continue
            
            # Check girth
            if _check_girth(adj, n, 9, deadline - 10):
                # Construct edge list
                edges = []
                for i in range(n):
                    for j in adj[i]:
                        if i < j:
                            edges.append((i, j))
                
                if best_n is None or n < best_n:
                    best_n = n
                    best_edges = edges
                # If we found a very small one, we might want to keep searching for even smaller
                # But for now, any valid graph is a massive improvement over LUW (~31k)
                # So we can break out of restarts for this n, but continue to smaller n?
                # The loop goes from small to large. So if we find one at 428, we are done.
                # If we find one at 1000, we might find one at 800 later? No, 800 was tried before.
                # So if we find one, we can break the restart loop for this n.
                break
    
    if best_n is not None:
        return best_n, best_edges
        
    return None


def construct(k, g):
    """Construct a k-regular graph with girth at least g on as few vertices as possible.
    
    Returns (n, edges) where edges is a list of (i, j) tuples.
    """
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Fallback to LUW construction which is guaranteed to be valid
    luw_n, luw_edges = _luw_construct(k, g)
    
    if k == 5 and g == 9:
        try:
            # Try to find a smaller graph via local search
            result = _local_search_5_9(max_time=100.0)
            if result is not None:
                n, edges = result
                if time.time() < deadline:
                    return n, edges
        except Exception:
            pass
    
    return luw_n, luw_edges

# EVOLVE-BLOCK-END
