"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import deque

def construct(k, g):
    """
    Construct a k-regular graph with girth at least g on as few vertices as possible.
    Uses the Lazebnik-Ustimenko-Woldar (LUW) construction with optimized parameters.
    """
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    # The LUW construction CD(d, q) is q-regular and has girth at least 2d+2.
    # We need 2d+2 >= g  =>  d >= (g-2)/2.
    # Since d must be an integer, d_min = ceil((g-2)/2).
    # However, the standard LUW rule definition in the seed starts at c=2 and goes to d.
    # The seed's logic: d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    # Let's check the seed's d for g=10:
    # g-5 = 5 (odd) -> d = 10-5 = 5.
    # Theoretical girth for d=5 is 2*5+2 = 12. This is safe but uses more vertices.
    # For g=10, we need girth >= 10.
    # If we use d=4, theoretical girth is 2*4+2 = 10. This is exactly what we need.
    # Does the LUW construction with d=4 work for q=4?
    # The rule indices c go from 2 to d.
    # For d=4, c in {2,3,4}.
    # Rule[2]=(1,1), Rule[3]=(2,1), Rule[4]=(1,2).
    # This is well-defined.
    
    # Calculate optimal d
    # We want the smallest d such that the girth bound is satisfied.
    # The bound is girth >= 2d+2.
    # So we need 2d+2 >= g => d >= (g-2)/2.
    d_min = (g - 2 + 1) // 2  # ceil((g-2)/2)
    if d_min < 3:
        d_min = 3
        
    # For k=4, q=4 is a prime power, so LUW is applicable.
    # For other k, we might need q=k to be a prime power.
    # The task specifies k=4, g=10 specifically in the prompt context, 
    # but the function takes k, g. 
    # We will try to use q=k if k is a prime power (4 is).
    # If k is not a prime power, LUW might not apply directly or we fall back.
    
    if k == 4:
        q = 4
        # Use the minimal d that satisfies the girth requirement
        d = d_min
        
        # Verify that d is at least 3 for the rule definitions to make sense in this context
        if d < 3:
            d = 3
            
        result = _luw_construct(q, d)
        if result is not None:
            n, edges = result
            # Verify the girth to be absolutely sure
            if _has_girth_at_least(edges, n, g):
                return n, edges
            else:
                # If verification fails (shouldn't happen for correct LUW), try larger d
                for d_try in range(d+1, 10):
                    result_try = _luw_construct(q, d_try)
                    if result_try is not None:
                        n_try, edges_try = result_try
                        if _has_girth_at_least(edges_try, n_try, g):
                            return n_try, edges_try
                # Fallback to original seed logic if all else fails
                return _luw_construct_original(k, g)
        
    # Fallback for other k values or if q=k is not a prime power
    # Try original seed logic which handles q=k generically (assuming prime or 4)
    return _luw_construct_original(k, g)


def _luw_construct(q, d):
    """
    Lazebnik-Ustimenko-Woldar graph construction with specific q and d.
    Returns (n, edges) or None if invalid.
    """
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        # Assume q is prime for simplicity in fallback, though LUW works for prime powers.
        # For this task k=4, so q=4 is the main path.
        if q < 2:
            return None
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
    
    # Define the rule for LUW construction
    # The rule maps c -> (a, b) such that l[c] - p[c] = l[a] * p[b]
    # Standard LUW rules:
    # c=2: (1, 1)
    # c=3: (2, 1)
    # c=4: (1, 2)
    # For c >= 5, the pattern repeats with a shift.
    rule = {}
    if d >= 2:
        rule[2] = (1, 1)
    if d >= 3:
        rule[3] = (2, 1)
    if d >= 4:
        rule[4] = (1, 2)
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
        
    # Ensure all required c in 2..d are in rule
    for c in range(2, d+1):
        if c not in rule:
            # Fallback rule if not defined (should not happen for d>=4)
            rule[c] = (1, c-1)

    def neighbours(vertex):
        side, x = vertex
        # x is a tuple of length d+1: (None, x_1, x_2, ..., x_d)
        for first in range(q):
            # y is a tuple of length d+1: (None, y_1, y_2, ..., y_d)
            # y_1 = first
            y_list = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    # Point p = x, Line l = y
                    # y[c] = x[c] + y[a] * x[b]
                    # Note: y[a] is known if a < c.
                    # In LUW, the rule ensures a < c.
                    y_list[c] = add(x[c], mul(y_list[a], x[b]))
                else:
                    # Line l = x, Point p = y
                    # x[c] - y[c] = x[a] * y[b]
                    # y[c] = x[c] - x[a] * y[b]
                    # Note: y[b] is known if b < c.
                    # In LUW, the rule ensures b < c.
                    y_list[c] = sub(x[c], mul(x[a], y_list[b]))
            yield 1 - side, tuple(y_list)
    
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


def _luw_construct_original(k, g):
    """Original seed logic for fallback."""
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


def _has_girth_at_least(edges, n, g):
    """
    Check if graph has girth at least g using non-backtracking walks.
    Returns True if no cycle of length < g is found.
    """
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Non-backtracking walk check:
    # A cycle of length L < g exists iff there is a non-backtracking walk
    # of length L that returns to its starting vertex.
    # We check all non-backtracking walks up to length g-1.
    
    max_depth = g - 1  # We need to check walks of length up to g-1
    
    # For each starting vertex and first step, do DFS of non-backtracking walks
    # up to depth max_depth. If we return to start at depth L < g, we have a cycle.
    
    for start in range(n):
        for first in adj[start]:
            # State: (current_vertex, previous_vertex, depth)
            # Start: (first, start, 1)
            # If we reach (start, _, L) with L < g, we found a cycle.
            stack = [(first, start, 1)]
            while stack:
                u, prev, depth = stack.pop()
                if u == start and depth >= 3:
                    # Found a cycle of length depth
                    if depth < g:
                        return False
                    continue
                if depth >= max_depth:
                    continue
                for v in adj[u]:
                    if v != prev:
                        stack.append((v, u, depth + 1))
    return True

# EVOLVE-BLOCK-END
