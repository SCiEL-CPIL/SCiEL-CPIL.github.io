"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import deque

def _deadline_guard(start_time, limit=140.0):
    """Return True if we should stop searching."""
    return (time.time() - start_time) > limit

def _girth_nonbacktracking(n, adj, g_required):
    """
    Check that the graph has no cycle of length < g_required.
    Uses non-backtracking BFS from every vertex.
    Returns True if girth >= g_required, False otherwise.
    """
    for u in range(n):
        # Multi-source BFS from (u, v) for all v in adj[u], depth 1
        dist = {}
        queue = deque()
        for v in adj[u]:
            key = (v, u)
            dist[key] = 1
            queue.append(key)
        
        while queue:
            cur, prev = queue.popleft()
            d = dist[(cur, prev)]
            if d >= g_required - 1:
                continue
            for nb in adj[cur]:
                if nb == prev:
                    continue
                nkey = (nb, cur)
                if nkey in dist:
                    continue
                dist[nkey] = d + 1
                # Check if we've returned to u
                if nb == u and d + 1 < g_required:
                    return False
                queue.append(nkey)
    
    return True

def _is_simple_regular(n, edges, k):
    """Check graph is simple and k-regular."""
    adj = [set() for _ in range(n)]
    for i, j in edges:
        if i == j:
            return False
        if j in adj[i]:
            return False
        adj[i].add(j)
        adj[j].add(i)
    for i in range(n):
        if len(adj[i]) != k:
            return False
    return True

def _build_luw(k, g):
    """Fallback: Lazebnik-Ustimenko-Woldar construction."""
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

def _voltage_lift(base_edges, m, labels):
    """
    Build the lifted graph from a base graph with voltage labels.
    base_edges: list of (u, v) for base graph
    m: cover size
    labels: dict from (u,v) to voltage in Z_m (symmetric: label(u,v) + label(v,u) = 0 mod m)
    Returns (n, edges) for the lifted graph.
    """
    # Get base vertices
    base_verts = set()
    for u, v in base_edges:
        base_verts.add(u)
        base_verts.add(v)
    base_verts = sorted(base_verts)
    base_n = len(base_verts)
    # Map base vertex to index
    bmap = {v: i for i, v in enumerate(base_verts)}
    
    n = base_n * m
    edges = []
    for u, v in base_edges:
        bu, bv = bmap[u], bmap[v]
        lab = labels.get((u, v), 0)
        for i in range(m):
            i1 = bu * m + i
            i2 = bv * m + ((i + lab) % m)
            edges.append((i1, i2))
    
    return n, edges

def _check_lifted(base_edges, m, labels, k, g):
    """Check if the lifted graph is valid: simple, k-regular, girth >= g."""
    n, edges = _voltage_lift(base_edges, m, labels)
    # Check if edge count matches k-regular graph requirement: 2*E = n*k
    if 2 * len(edges) != n * k:
        return None
    # Check simple and regular
    if not _is_simple_regular(n, edges, k):
        return None
    # Build adjacency
    adj = [[] for _ in range(n)]
    for i, j in edges:
        adj[i].append(j)
        adj[j].append(i)
    # Check girth
    if not _girth_nonbacktracking(n, adj, g):
        return None
    return n, edges

def construct(k, g):
    start_time = time.time()
    limit = 140.0
    
    # For k=5, g=9: try voltage lifts of small 5-regular base graphs
    # The base graph must be 5-regular. Smallest 5-regular graphs:
    # - 6 vertices: K_6 (but girth 3, too small)
    # - 8 vertices: no 5-regular simple graph on 8 vertices (sum degrees = 40, need 40/2=20 edges, max edges on 8 vertices is 28, but 5-regular on 8: each vertex degree 5, possible? 8*5=40, 40/2=20 edges. Yes, complement of 2-regular on 8 vertices. 2-regular on 8 is a union of cycles. So 5-regular on 8 exists.)
    # But base graph girth doesn't matter directly; the lift's girth depends on both base and voltages.
    
    # Strategy: try various base graphs and voltage assignments.
    # For simplicity, start with known small 5-regular graphs and search voltage labels.
    
    # Moore bound for (5,9): 1 + 5 + 5*4 + 5*4^2 + 5*4^3 + 5*4^4 = 1 + 5 + 20 + 80 + 320 + 1280 = 1706? 
    # Wait, Moore bound for odd girth g=2r+1: 1 + k * (1 + (k-1) + ... + (k-1)^r)
    # For k=5, g=9=2*4+1, r=4: 1 + 5*(1 + 4 + 16 + 64 + 256) = 1 + 5*341 = 1706.
    # Hmm, that's the lower bound. The current LUW gives more.
    
    # Actually for the cage problem n(5,9), the Moore bound is 1 + 5*(1+4+4^2+4^3+4^4) = 1+5*(1+4+16+64+256) = 1+5*341 = 1706.
    # But the best known is 1116, which is less than 1706? That can't be right.
    # Let me recalculate: Moore bound for girth g=2r+1 is 1 + k * sum_{i=0}^{r} (k-1)^i.
    # For k=5, r=4: 1 + 5*(1 + 4 + 16 + 64 + 256) = 1 + 5*341 = 1706.
    # So 1116 < 1706 means my formula is wrong.
    # Actually the Moore bound gives a LOWER bound on n. So n >= 1706? But 1116 is claimed as smallest known.
    # I must have the formula wrong. Let me look it up mentally: for odd girth g=2r+1, Moore bound M(k,g) = 1 + k * sum_{i=0}^{r} (k-1)^i.
    # For k=3, g=5 (r=2): 1 + 3*(1+2+4) = 1+21 = 22. Petersen graph has 10 vertices, girth 5. 10 < 22. So the Moore bound is not a lower bound on n? 
    # No wait, the Moore bound IS a lower bound. Petersen has 10 vertices, so M(3,5) should be <= 10.
    # M(3,5) = 1 + 3*(1 + 2 + 4) = 22. That's > 10, contradiction.
    # I think the formula is: M(k,g) for odd g=2r+1 is 1 + k * sum_{i=0}^{r-1} (k-1)^i.
    # For k=3, r=2: 1 + 3*(1 + 2) = 1 + 9 = 10. Yes! Petersen has 10 vertices.
    # For k=5, g=9=2*4+1, r=4: 1 + 5*(1 + 4 + 16 + 64) = 1 + 5*85 = 426.
    # So Moore bound is 426. Best known is 1116. Lower bound mentioned is 428.
    
    # So we want n as small as possible, >= 426.
    
    # Try voltage lifts. Base graph with b vertices, lift of size m gives n = b*m vertices.
    # We need n >= 426. So b*m >= 426.
    # Try small base graphs:
    # - b=6 (K6), m >= 71
    # - b=8, m >= 54
    # - b=10, m >= 43
    # - b=12, m >= 36
    # - b=14, m >= 31
    # - b=16, m >= 27
    # - b=18, m >= 24
    # - b=20, m >= 22
    # - b=22, m >= 20
    # - b=24, m >= 18
    # - b=26, m >= 17
    # - b=28, m >= 16
    # - b=30, m >= 15
    # - b=32, m >= 14
    # - b=34, m >= 13
    # - b=36, m >= 12
    # - b=38, m >= 12
    # - b=40, m >= 11
    # - b=42, m >= 11
    # - b=44, m >= 10
    # - b=46, m >= 10
    # - b=48, m >= 9
    # - b=50, m >= 9
    # - b=52, m >= 9
    # - b=54, m >= 8
    # - b=56, m >= 8
    # - b=58, m >= 8
    # - b=60, m >= 8
    
    # This is a huge search space. Let's be more targeted.
    # Known constructions: the (5,9)-cage is known to have 1116 vertices.
    # 1116 = 2 * 3 * 11 * 17 = 2*3*187 = 6*187 = 12*93.5 no. 1116/2=558, /3=372, /4=279, /6=186, /9=124, /12=93, /18=62, /23=48.5 no.
    # 1116 = 2^2 * 3 * 11 * 17? 4*3=12, 12*11=132, 132*17=2244. No.
    # 1116 / 2 = 558, / 2 = 279, / 3 = 93, / 3 = 31. So 1116 = 4 * 3 * 3 * 31 = 36 * 31.
    # So base 36, lift 31? Or base 31, lift 36? Or base 12, lift 93?
    
    # Let's try a different approach: use known good constructions or random search.
    # Given time constraints, let's try a few specific cases and fall back to LUW.
    
    # For now, let's just return the LUW construction as it's known to work,
    # but try to find something better with voltage lifts if time permits.
    
    # Try a specific known structure: the (5,9) cage might be related to certain voltage assignments.
    # Given the complexity, let's implement a quick random search over voltage assignments
    # for a few base graphs.
    
    if k != 5 or g != 9:
        # For other parameters, use LUW
        n, edges = _build_luw(k, g)
        return n, edges
    
    # For (5,9), try voltage lifts
    # Base graph: 5-regular on b vertices
    # Let's try b=6 (K6) with various m and random voltage assignments
    # K6 has 15 edges. Each edge gets a voltage in Z_m.
    # The lifted graph has 6*m vertices and 15*m edges. For it to be 5-regular, 
    # each vertex in the lift has degree 5 (since base is 5-regular).
    # We need girth >= 9.
    
    # This is still a large search. Let's try a few m values and random labels.
    # m must be such that 6*m >= 426, so m >= 71.
    
    # Actually, let's try a smarter approach: use the fact that the (5,9)-cage is known
    # and try to construct it via known methods. But since we don't have the explicit construction,
    # let's do a limited random search.
    
    best_n = None
    best_edges = None
    
    # Try base K6 with m in range 71 to 100
    # For each m, try a few random voltage assignments
    k6_edges = [(i, j) for i in range(6) for j in range(i+1, 6)]
    
    # Limit total search time to leave room for LUW construction if needed
    search_deadline = start_time + (limit - 30)
    
    for m in range(71, 110):
        if time.time() > search_deadline:
            break
        # Reduce trials per m to fit in time budget
        # 50 trials * 40 m-values was too many. Let's do 10 trials.
        for trial in range(10):
            if time.time() > search_deadline:
                break
            # Random voltage assignment
            labels = {}
            for u, v in k6_edges:
                lab = random.randint(0, m-1)
                labels[(u, v)] = lab
                labels[(v, u)] = (-lab) % m
            
            result = _check_lifted(k6_edges, m, labels, 5, 9)
            if result is not None:
                n, edges = result
                if best_n is None or n < best_n:
                    best_n = n
                    best_edges = edges
                # Do not break immediately, try to find smaller n with other m?
                # But m is increasing. Smaller m gives smaller n.
                # So if we found one at m=71, we should check if we can find smaller?
                # No, n = 6*m. Smaller m means smaller n.
                # So we should iterate m from small to large.
                # If we find a valid one at m=71, n=426.
                # If we find one at m=72, n=432.
                # We want the smallest n.
                # So if we find one at m, we can break the inner loop?
                # No, maybe another random assignment at same m is invalid?
                # We found a valid one. n is fixed for this m.
                # We can break the inner loop to save time.
                break
    
    if best_n is not None:
        return best_n, best_edges
    
    # Fallback to LUW
    n, edges = _build_luw(k, g)
    return n, edges
# EVOLVE-BLOCK-END
