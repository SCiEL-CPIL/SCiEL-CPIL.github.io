"""Structural redesign: voltage assignment / lifting search for (3,16)-cage.

Instead of the explicit Lazebnik-Ustimenko-Woldar construction (which yields a
large, far-from-optimal graph), we search for a small 3-regular graph of girth
>= g by:
  1. Starting from a small base 3-regular graph B with girth >= g/2 (so that the
     base has no short cycles to lift over).
  2. Trying to assign voltages in a finite abelian group Z_m (the "voltage") to
     each directed edge so that the resulting voltage lift L(B, phi) is 3-regular,
     simple, and has girth >= g.
  3. The lift has n = m * |V(B)| vertices. We minimize n.

The key structural insight: a voltage lift of a graph B with voltage assignment
phi: E(B) -> Z_m produces a graph on m*|V(B)| vertices. A cycle in the lift
projects to a closed walk in B; the lift has a cycle of length L iff there is a
closed walk of length L in B whose total voltage is 0 mod m. By choosing m and
phi carefully we can eliminate all short closed walks, achieving girth >= g with
far fewer vertices than explicit constructions.

We use a randomized / greedy approach to find voltage assignments that avoid
short zero-voltage closed walks, validated by exact non-backtracking BFS.
"""
import time
import random
import math
from collections import deque

def _moore_bound(k, g):
    """Lower bound on n for k-regular girth-g graph."""
    if g % 2 == 0:
        r = g // 2
        # n >= k * (k-1)^(r-1) / (k-1) * ... for even girth
        # Moore bound: n >= k * ((k-1)^(r-1) - 1) / (k-2) for k>2
        if k <= 2:
            return g
        return k * ((k - 1) ** (r - 1) - 1) // (k - 2)
    else:
        r = (g - 1) // 2
        if k <= 2:
            return g
        return k * ((k - 1) ** (r - 1) - 1) // (k - 2)


def _girth_nonbacktracking(n, edges, max_girth):
    """Return the girth (shortest cycle length) up to max_girth via non-backtracking BFS.
    If no cycle shorter than max_girth is found, return max_girth (meaning girth >= max_girth)."""
    # Build adjacency with edge indices to prevent backtracking
    adj = [[] for _ in range(n)]
    for idx, (u, v) in enumerate(edges):
        adj[u].append((v, idx))
        adj[v].append((u, idx))

    # Non-backtracking BFS: state = (vertex, prev_edge_idx)
    # dist[state] = length of walk
    best = max_girth
    for start in range(n):
        visited = set()
        queue = deque()
        for (nb, eidx) in adj[start]:
            queue.append((nb, eidx, 1))
            visited.add((nb, eidx))
        while queue:
            v, prev_e, dist = queue.popleft()
            if dist >= best:
                continue
            for (nb, eidx) in adj[v]:
                if eidx == prev_e:
                    continue
                if nb == start:
                    best = min(best, dist + 1)
                    break
                if (nb, eidx) not in visited:
                    visited.add((nb, eidx))
                    if dist + 1 < best:
                        queue.append((nb, eidx, dist + 1))
    return best


def _is_valid(n, edges, k, g):
    """Check k-regular, simple, and girth >= g."""
    if len(edges) != n * k // 2:
        return False
    # Check simplicity
    edge_set = set()
    for u, v in edges:
        if u == v:
            return False
        key = (min(u, v), max(u, v))
        if key in edge_set:
            return False
        edge_set.add(key)
    # Check degree
    deg = [0] * n
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
    for d in deg:
        if d != k:
            return False
    # Check girth
    girth = _girth_nonbacktracking(n, edges, g)
    return girth >= g


def _make_cycle_base(k, g):
    """Create a base k-regular graph (cycle of triangles or similar) with good girth.
    For k=3, use a cycle of length L where each vertex has 3 edges.
    A simple approach: take a 2L-cycle and add chords to make it 3-regular.
    Actually, for voltage lifting we need a base that is k-regular.
    Let's use a known small 3-regular graph: the prism graph (2 x cycle) is 3-regular.
    Prism P_L: 2L vertices, girth = min(6, L) for L>=3... not great for large girth.
    
    Better: use a cycle C_L as a 2-regular base and add a perfect matching to make 3-regular.
    But voltage lifting requires the base to be k-regular for the lift to be k-regular.
    
    Alternative: just use a cycle of length L (2-regular) and lift it, then the lift is 2-regular.
    That doesn't help for k=3.
    
    Let me use a different approach: generate random 3-regular graphs of small size and check girth.
    """
    pass


def construct(k, g):
    deadline = time.time() + 145  # 145s out of 150s budget
    best_result = None
    best_n = float('inf')

    # Moore bound
    mb = _moore_bound(k, g)

    # Strategy: For k=3, g=16, try voltage lifts of small 3-regular base graphs.
    # Base graph B with |V|=b, m = n/b. We need b*m >= mb.
    # Try small b (like 4, 6, 8, 10, ...) and find m and voltage assignment.

    # First, generate a pool of small 3-regular base graphs with girth >= ceil(g/2)
    # For g=16, need base girth >= 8.
    # Small 3-regular graphs with girth >= 8:
    #   - The (3,8)-cage has 30 vertices (the 30-vertex graph).
    #   - But we can also use non-cage graphs.
    
    # Actually, a cleaner structural approach: 
    # Use the fact that a voltage lift of a cycle C_L by Z_m gives a 2-regular graph (a cycle or two cycles).
    # For 3-regular, we need a 3-regular base.
    
    # Let me try a different structural family: 
    # Random 3-regular graphs + local search to increase girth.
    # Start with random 3-regular graph of size n, check girth, if < g do edge swaps to increase girth.
    
    # But that's hard for n~500+.
    
    # Let me try the voltage lift approach more carefully.
    # Base: a 3-regular graph B. Lift by Z_m.
    # The lift L(B, phi) has vertices (v, t) for v in V(B), t in Z_m.
    # Edge ((u, t), (v, t + phi(u,v))) for each edge (u,v) in B.
    # This is 3-regular. Simple if phi is chosen so no 2-cycles or loops.
    # Girth of lift: shortest closed walk in B with voltage 0 mod m.
    
    # For this to work well, B should have girth >= g/2 (so no short closed walks at all),
    # OR we need the voltage to be non-zero for all short closed walks.
    
    # Let's generate random 3-regular graphs of size b (small, like 6, 8, 10, 12, ...) 
    # and for each, try random voltage assignments in Z_m for m = ceil(mb/b) or nearby.
    
    # Actually, for the lift to be simple, we need:
    # 1. No loops: phi(u,u) != 0 (no self-loops in B, so fine)
    # 2. No multiple edges: for edge (u,v) in B, the lift edges are ((u,t),(v,t+phi(u,v))) and ((v,t),(u,t+phi(v,u))).
    #    For simplicity, we need phi(u,v) != -phi(v,u) mod m (otherwise we'd get a 2-cycle in the lift for the same pair).
    #    Actually in a simple graph B, (u,v) and (v,u) are the same undirected edge. 
    #    In the voltage lift, we assign a voltage to each directed edge. For an undirected edge {u,v},
    #    we assign phi(u,v) and phi(v,u) = -phi(u,v) mod m (to make it a proper voltage assignment).
    #    Then the lift has edges ((u,t),(v,t+phi(u,v))) for all t. This is a set of m edges forming a matching-like structure.
    #    The lift is simple as long as B is simple and we use the antisymmetric voltage convention.
    #    Wait, actually the lift of a simple graph by Z_m with antisymmetric voltages is always simple
    #    (no loops, no multi-edges) because each undirected edge in B gives m edges in the lift,
    #    and they connect different "layers".
    
    # So the main constraint is girth. We need no closed walk of length < g in B with total voltage 0 mod m.
    
    # Let's try:
    # - Generate random 3-regular graphs B of size b for b in [6, 8, 10, 12, 14, 16, 18, 20, ...]
    # - For each B, compute all closed walks of length up to g-1 and their "voltage expressions"
    #   (each closed walk corresponds to a linear equation in the voltages).
    # - Try to assign voltages in Z_m such that none of these equations = 0 mod m.
    # - This is a constraint satisfaction problem.
    
    # Simpler: for small m, just try random assignments and check girth directly.
    
    # Let me implement a direct approach:
    # 1. For n from max(mb, 2*3*something) upward in steps, try to construct a 3-regular graph of girth >= g.
    # 2. Use random construction + local improvement.
    
    # Actually, let me try the most promising structural approach:
    # Voltage lift of a small base.
    
    def _random_3reg(b):
        """Generate a random 3-regular graph on b vertices (b must be even)."""
        if b < 4 or b % 2 != 0:
            return None
        # Configuration model
        half = b // 2
        # Each vertex gets 3 stubs
        stubs = list(range(b)) * 3
        random.shuffle(stubs)
        edges = []
        adj = [[] for _ in range(b)]
        i = 0
        while i < len(stubs):
            u = stubs[i]
            v = stubs[i + 1]
            if u == v:
                i += 1
                continue
            if v in adj[u]:
                i += 1
                continue
            edges.append((u, v))
            adj[u].append(v)
            adj[v].append(u)
            i += 2
        if len(edges) != b * 3 // 2:
            return None
        # Check 3-regular
        for v in range(b):
            if len(adj[v]) != 3:
                return None
        return edges
    
    def _build_lift(base_edges, b, m, voltages):
        """Build the voltage lift. base_edges: list of (u,v). voltages: dict {(u,v): val} for u<v.
        Returns (n, edges) of the lift."""
        n = b * m
        lift_edges = []
        for (u, v) in base_edges:
            if u > v:
                u, v = v, u
            vol = voltages[(u, v)]
            for t in range(m):
                # Edge from (u, t) to (v, t + vol mod m)
                a = u * m + t
                b2 = v * m + ((t + vol) % m)
                lift_edges.append((a, b2))
        return n, lift_edges
    
    # Try voltage lifts with small bases
    # For k=3, g=16: Moore bound is around 512.
    # If base has b vertices, m = ceil(512/b). For b=8, m=64. For b=16, m=32. Etc.
    
    base_sizes = [8, 10, 12, 14, 16, 20, 24, 30]
    best_found = None
    
    for b in base_sizes:
        if time.time() > deadline:
            break
        # Try multiple random base graphs
        for trial in range(20):
            if time.time() > deadline:
                break
            base_edges = _random_3reg(b)
            if base_edges is None:
                continue
            
            # Check base girth
            base_girth = _girth_nonbacktracking(b, base_edges, g)
            if base_girth < g:
                # Base has a short cycle; the lift will too (with voltage 0 for that cycle)
                # Unless we assign non-zero voltage. But a cycle of length L in base with voltage 0
                # gives a cycle of length L in the lift. So we need all cycles of length < g to have non-zero voltage.
                # This is hard if base girth < g. Let's skip bases with girth < g//2.
                continue
            
            # m such that n = b*m >= mb
            m_min = max(2, math.ceil(mb / b))
            
            # Try a few values of m
            for m in [m_min, m_min + 1, m_min + 2]:
                if time.time() > deadline:
                    break
                n = b * m
                
                # Try random voltage assignments
                num_edges = len(base_edges)
                for attempt in range(50):
                    if time.time() > deadline:
                        break
                    # Assign random voltages in Z_m (non-zero to avoid trivial issues)
                    voltages = {}
                    for (u, v) in base_edges:
                        if u > v:
                            u, v = v, u
                        voltages[(u, v)] = random.randint(1, m - 1)
                    
                    _, lift_edges = _build_lift(base_edges, b, m, voltages)
                    
                    # Quick check: is it valid?
                    if _is_valid(n, lift_edges, k, g):
                        if n < best_n:
                            best_n = n
                            best_result = (n, lift_edges)
                            best_found = (b, m, trial, attempt)
                        break  # Found valid for this (b, m), try smaller n next
    
    # If voltage lift didn't find anything good, fall back to the original construction
    if best_result is not None:
        return best_result
    
    # Fallback: original Lazebnik-Ustimenko-Woldar construction
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
