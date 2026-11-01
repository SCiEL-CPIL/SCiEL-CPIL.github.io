import time
import random
from collections import deque

def construct(k, g):
    """
    Construct a k-regular graph with girth >= g.
    For k=3, g=16, we use a voltage lift of the Petersen graph.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit

    if k != 3:
        # Fallback for other k (should not be hit for this task)
        return _luw_construct(k, g)

    # Petersen graph edges (0-9)
    petersen_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    petersen_adj = [[] for _ in range(10)]
    for u, v in petersen_edges:
        petersen_adj[u].append(v)
        petersen_adj[v].append(u)

    # Base edges (u < v)
    base_edges = []
    for u in range(10):
        for v in petersen_adj[u]:
            if u < v:
                base_edges.append((u, v))
    
    # Precompute edge indices for adjacency
    edge_idx_map = {}
    for i, (u, v) in enumerate(petersen_edges):
        edge_idx_map[(u, v)] = i
        edge_idx_map[(v, u)] = i
        
    adj_edges = [[] for _ in range(10)]
    for i, (u, v) in enumerate(petersen_edges):
        adj_edges[u].append((v, i))
        adj_edges[v].append((u, i))
    
    # Pre-compute forbidden closed walks (as sorted edge multisets) up to length g-1
    # We only care about walks that could potentially have voltage sum 0.
    # Since voltages are arbitrary, we must check ALL closed walks.
    # However, we can prune: if a walk contains a backtracking edge pair (e, e) with net 0 voltage?
    # No, voltage of (u,v) is s, (v,u) is -s. Sum is 0.
    # So any walk with immediate backtracking has a sub-walk of length 2 with sum 0.
    # But a cycle in the lift corresponds to a closed walk in the base.
    # If the walk backtracks, say u->v->u, the voltage is s - s = 0.
    # This means (u, i) -> (v, i+s) -> (u, i). This is a cycle of length 2 in the lift?
    # No, the lift has no multiple edges, so (u,i)-(v,i+s) is an edge.
    # The walk u->v->u in base lifts to (u,i) -> (v, i+s) -> (u, i).
    # This is a cycle of length 2? No, it's a path of length 2 returning to start.
    # In a simple graph, a cycle must have length >= 3.
    # So backtracking walks in the base do NOT create cycles in the lift.
    # They create "bubbles" that don't count as cycles.
    # Therefore, we only need to consider NON-BACKTRACKING closed walks.
    # This drastically reduces the number of walks.
    
    # Let's generate all non-backtracking closed walks of length 5 to g-1.
    # We represent them as sorted tuples of edge indices.
    # Note: A non-backtracking walk is determined by its sequence of edges.
    # Different sequences might have the same multiset.
    # But for voltage sum, only the multiset matters.
    # So we just need the set of multisets of non-backtracking closed walks.
    
    def gen_non_backtracking_closed_walks(max_len):
        walks = set()
        for start in range(10):
            # DFS: (current_vertex, previous_vertex, length, edge_path)
            # Start from start, go to a neighbor, then continue non-backtracking.
            for next_v, e_idx in adj_edges[start]:
                stack = [(next_v, start, 1, [e_idx])]
                while stack:
                    curr, prev, length, edge_path = stack.pop()
                    if curr == start and length >= 5:
                        if length <= max_len:
                            key = tuple(sorted(edge_path))
                            walks.add(key)
                        # Don't extend if length >= max_len
                        if length < max_len:
                            for next_v2, e_idx2 in adj_edges[curr]:
                                if next_v2 != prev:
                                    stack.append((next_v2, curr, length + 1, edge_path + [e_idx2]))
                    else:
                        if length < max_len:
                            for next_v2, e_idx2 in adj_edges[curr]:
                                if next_v2 != prev:
                                    stack.append((next_v2, curr, length + 1, edge_path + [e_idx2]))
        return walks

    # For g=16, max_len=15.
    # Number of non-backtracking walks grows as 2^L.
    # 2^15 = 32768. Times 10 starts * 3 first steps = ~1M. Manageable.
    forbidden_walks = gen_non_backtracking_closed_walks(g - 1)
    
    # Convert to list of edge count vectors for faster summing?
    # Or just keep as sorted tuples and sum shifts[edge_idx].
    
    def _verify_girth_fast_lift(m, shifts, g):
        """
        Verify girth of the lift using the pre-computed forbidden walks.
        Returns True if no forbidden walk has voltage sum 0 mod m.
        """
        for walk in forbidden_walks:
            # Sum shifts for edges in walk
            s = 0
            for e in walk:
                s += shifts[e]
            if s % m == 0:
                return False
        return True

    def _build_lift(base_edges, m, shifts):
        n = 10 * m
        edges = []
        for (u, v), shift in zip(base_edges, shifts):
            for i in range(m):
                u_id = u * m + i
                v_id = v * m + (i + shift) % m
                edges.append((u_id, v_id))
        return n, edges

    # Search for valid (m, shifts)
    random.seed(42)
    
    # We want to minimize n = 10m.
    # Start from m=52 (Moore bound 512 -> m=52).
    
    # Strategy:
    # For each m, try many random shifts.
    # The probability that a random shift assignment avoids all forbidden walks is:
    # For a single walk of length L, P(sum=0) = 1/m.
    # If there are N walks, P(any sum=0) ~ N/m (if independent).
    # We need P(no sum=0) > 0.
    # So we need m >> N.
    # N for g=16 is roughly the number of non-backtracking closed walks of length 5..15.
    # This is not huge. Let's estimate.
    # Length 5: 12 pentagons.
    # Length 6: ...
    # Total N is likely in the thousands.
    # So m needs to be maybe 10x N?
    # If N ~ 1000, m ~ 10000? That's n=100000. Too big.
    # Wait, the walks are not independent.
    # Also, we can use structured shifts to avoid sums.
    
    # Let's try increasing m.
    # For small m, random shifts are unlikely to work.
    # We can try to solve the CSP: sum(shifts[e] for e in walk) != 0 mod m for all walks.
    # This is a system of linear inequalities mod m.
    # We can try to find a solution by backtracking or local search.
    
    # Simpler: Just try random shifts for increasing m.
    # If m is large enough, random shifts will work.
    # The threshold m* is roughly N (number of constraints).
    # Let's find N.
    
    N_forbidden = len(forbidden_walks)
    # print(f"Number of forbidden walks: {N_forbidden}")
    
    # Let's try m from 52 upwards.
    # For each m, try 1000 random shifts.
    # If m is small, it will fail fast.
    # If m is large, it will succeed fast.
    
    # Optimization: Pre-compute the list of edge indices for each forbidden walk?
    # They are already sorted tuples.
    
    # Let's also try some heuristic shifts:
    # 1. Random
    # 2. Linear: shifts[i] = (i * m) // 15
    # 3. Quadratic: shifts[i] = (i*i * m) // 15
    
    for m in range(52, 5000):
        if time.time() > deadline:
            break
        
        # Determine attempts
        if m < 100:
            attempts = 100
        elif m < 500:
            attempts = 500
        else:
            attempts = 1000
            
        # Try deterministic first
        shifts_lin = [int((i + 1) * m / 15.0) for i in range(15)]
        if _verify_girth_fast_lift(m, shifts_lin, g):
            n, edges = _build_lift(base_edges, m, shifts_lin)
            return n, edges
            
        shifts_quad = [int(((i + 1) * (i + 1)) * m / 15.0) for i in range(15)]
        if _verify_girth_fast_lift(m, shifts_quad, g):
            n, edges = _build_lift(base_edges, m, shifts_quad)
            return n, edges

        # Random shifts
        for _ in range(attempts):
            if time.time() > deadline:
                break
            shifts_rand = [random.randint(0, m-1) for _ in range(15)]
            if _verify_girth_fast_lift(m, shifts_rand, g):
                n, edges = _build_lift(base_edges, m, shifts_rand)
                return n, edges

    # If not found, try larger m with more attempts
    if time.time() < deadline:
        for m in range(5000, 20000):
            if time.time() > deadline:
                break
            for _ in range(2000):
                if time.time() > deadline:
                    break
                shifts_rand = [random.randint(0, m-1) for _ in range(15)]
                if _verify_girth_fast_lift(m, shifts_rand, g):
                    n, edges = _build_lift(base_edges, m, shifts_rand)
                    return n, edges

    # Fallback: LUW construction
    return _luw_construct(k, g)


def _luw_construct(k, g):
    """
    Lazebnik-Ustimenko-Woldar construction for k-regular graphs.
    """
    if k != 3:
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
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

    # For k=3, we use q=3.
    q = 3

    def build_luw(d):
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

    # Try d=8 first. If girth < 16, try d=9, etc.
    for d in [8, 9, 10, 11]:
        n, edges = build_luw(d)
        if _check_girth_fast(n, edges, g):
            return n, edges
    # If all fail (unlikely for d=11), return the largest one (which should be valid by theory)
    return build_luw(11)


def _build_adj(n, edges):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj


def _check_girth_fast(n, edges, min_girth):
    """Fast girth check using BFS from each vertex with parent tracking."""
    adj = _build_adj(n, edges)
    max_depth = (min_girth - 1) // 2

    for start in range(n):
        visited = [-1] * n
        visited[start] = 0
        queue = deque([(start, 0, -1)])

        while queue:
            node, depth, parent = queue.popleft()
            if depth > max_depth:
                continue
            for neighbor in adj[node]:
                if neighbor == parent:
                    continue
                if visited[neighbor] == -1:
                    visited[neighbor] = depth + 1
                    queue.append((neighbor, depth + 1, node))
                else:
                    cycle_len = depth + visited[neighbor] + 1
                    if cycle_len < min_girth:
                        return False
    return True
