import time
import random
import math

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
    """Check if graph has girth >= min_girth using non-backtracking BFS from each vertex."""
    max_depth = (min_girth + 1) // 2 - 1  # depth to check: for girth 10, depth 4
    for start in range(n):
        # state: (vertex, prev_vertex)
        dist = {}
        for nb in adj[start]:
            dist[(start, nb)] = 1
        queue = [(start, nb) for nb in adj[start]]
        for state in queue:
            v, prev = state
            d = dist[state]
            if d >= max_depth:
                continue
            for nb in adj[v]:
                if nb == prev:
                    continue
                new_state = (v, nb)
                if new_state not in dist:
                    dist[new_state] = d + 1
                    queue.append(new_state)
                elif dist[new_state] == d:
                    # cycle of length 2*d+1 (odd cycle)
                    if 2 * d + 1 < min_girth:
                        return False
                else:
                    # even cycle
                    cycle_len = d + dist[new_state]
                    if cycle_len < min_girth:
                        return False
    return True

def _count_short_cycles(adj, n, max_len=8):
    """Count 4-cycles and estimate 6-cycles for scoring."""
    count = 0
    half = n // 2
    # Count 4-cycles: for each pair of A vertices, count common neighbors in B
    for u in range(half):
        for v in range(u + 1, half):
            common = len(adj[u] & adj[v])
            if common >= 2:
                count += common * (common - 1) // 2
    # Count 6-cycles: A-u - B-v - A-w - B-x - A-y - B-z - A-u
    # For each pair (u,w) in A, count paths u->B->w of length 2
    # then for each such pair, count B-neighbors of u and w that form a 6-cycle
    # Simplified: count via common B-neighbors of A-pairs
    # A 6-cycle: u1-b1-u2-b2-u3-b3-u1 where u1,u2,u3 in A, b1,b2,b3 in B
    # This is complex; use a faster proxy
    for u in range(half):
        au = adj[u]
        for w in range(u + 1, half):
            common_b = au & adj[w]
            if len(common_b) >= 2:
                count += len(common_b) * (len(common_b) - 1) // 2
    return count

def _voltage_lift_construct(k, g):
    """
    Voltage graph (lift) construction for 4-regular graph of girth 10.
    
    Base graph: a small 4-regular graph (e.g., a tree-like structure or small cage).
    Lift by Z_m where m = n / |base|.
    
    For k=4, g=10: Moore bound gives n >= 242.
    We try base graphs of size 8, 10, 12, ... with lifts of size m = n/base_size.
    
    The key insight: if the base graph has girth >= g and we assign voltages
    such that no closed walk of length < g has voltage sum 0, then the lift
    has girth >= g.
    """
    deadline = time.time() + 110
    best_n = None
    best_edges = None
    
    # Try different base graph sizes and lift sizes
    # n = base_size * lift_size
    # For n=242: factors are 242 = 2 * 121 = 2 * 11 * 11
    # For n=244: 244 = 4 * 61
    # For n=246: 246 = 2 * 3 * 41 = 6 * 41
    # For n=248: 248 = 8 * 31
    # For n=250: 250 = 2 * 5^3 = 25 * 10
    # For n=252: 252 = 4 * 63 = 6 * 42 = 7 * 36 = 9 * 28 = 12 * 21 = 14 * 18
    # For n=254: 254 = 2 * 127
    # For n=256: 256 = 2^8 = 16 * 16 = 8 * 32 = 4 * 64
    # For n=258: 258 = 2 * 3 * 43 = 6 * 43
    # For n=260: 260 = 4 * 65 = 5 * 52 = 10 * 26 = 13 * 20
    # For n=262: 262 = 2 * 131
    # For n=264: 264 = 8 * 33 = 11 * 24 = 12 * 22
    # For n=266: 266 = 2 * 7 * 19
    # For n=268: 268 = 4 * 67
    # For n=270: 270 = 2 * 3^3 * 5 = 18 * 15 = 27 * 10
    # For n=272: 272 = 16 * 17
    # For n=274: 274 = 2 * 137
    # For n=276: 276 = 4 * 69 = 6 * 46 = 12 * 23
    # For n=278: 278 = 2 * 139
    # For n=280: 280 = 8 * 35 = 10 * 28 = 14 * 20
    # For n=282: 282 = 2 * 3 * 47
    # For n=284: 284 = 4 * 71
    # For n=286: 286 = 2 * 11 * 13
    # For n=288: 288 = 16 * 18 = 12 * 24 = 9 * 32
    # For n=290: 290 = 2 * 5 * 29
    # For n=292: 292 = 4 * 73
    # For n=294: 294 = 2 * 3 * 7^2 = 14 * 21
    # For n=296: 296 = 8 * 37
    # For n=298: 298 = 2 * 149
    # For n=300: 300 = 12 * 25 = 15 * 20 = 20 * 15
    
    # We need a 4-regular base graph. Small 4-regular graphs:
    # - Complete graph K5 (5 vertices, 4-regular)
    # - Complete bipartite K_{4,4} minus perfect matching (8 vertices, 4-regular)
    # - Various 4-regular graphs on 8, 10, 12, ... vertices
    
    # For the lift to work, we need the base graph to be 4-regular.
    # Let's try base graphs of size 8 (K_{4,4} minus matching), 10, 12, etc.
    
    def make_base_4reg_8():
        """K_{4,4} minus a perfect matching: 8 vertices, 4-regular."""
        # Vertices 0-3 in A, 4-7 in B
        # A_i connects to B_j for j != i
        edges = []
        for i in range(4):
            for j in range(4):
                if i != j:
                    edges.append((i, 4 + j))
        return edges
    
    def make_base_4reg_10():
        """4-regular graph on 10 vertices (Petersen-like or other)."""
        # Use the complement of a 5-cycle on 10 vertices? 
        # Actually, let's use a 4-regular graph: two K5's with some cross edges
        # Simpler: circulant graph on 10 vertices with jumps {1, 3, 7, 9}
        # That gives degree 4
        edges = []
        for i in range(10):
            for j in [1, 3]:
                v = (i + j) % 10
                if i < v:
                    edges.append((i, v))
        return edges
    
    def make_base_4reg_12():
        """4-regular graph on 12 vertices: circulant with jumps {1, 5, 7, 11}"""
        edges = []
        for i in range(12):
            for j in [1, 5]:
                v = (i + j) % 12
                if i < v:
                    edges.append((i, v))
        return edges
    
    base_graphs = {
        8: make_base_4reg_8,
        10: make_base_4reg_10,
        12: make_base_4reg_12,
    }
    
    def get_factorizations(n):
        """Return list of (base_size, lift_size) pairs where base_size * lift_size = n."""
        result = []
        for bs in sorted(base_graphs.keys()):
            if n % bs == 0:
                ls = n // bs
                if ls >= 2:
                    result.append((bs, ls))
        return result
    
    def build_lift(base_edges, m, voltages):
        """
        Build the voltage lift.
        base_edges: list of (u, v) edges in base graph
        m: lift size (Z_m)
        voltages: dict mapping (u, v) -> voltage value in Z_m
        Returns: list of edges in the lifted graph
        """
        base_n = max(max(u, v) for u, v in base_edges) + 1
        lifted_edges = []
        for s in range(m):
            for (u, v) in base_edges:
                w = voltages[(u, v)]
                # Edge from (u, s) to (v, (s + w) % m)
                lifted_edges.append((u * m + s, v * m + ((s + w) % m)))
                # Also add reverse edge with negative voltage
                # Actually, in undirected voltage graphs, we need to be careful
                # Let's just add each edge once and ensure the graph is undirected
        # Remove duplicates and sort
        lifted_edges = list(set(lifted_edges))
        # Ensure undirected: for each (a,b), also add (b,a) if not present
        final_edges = []
        edge_set = set()
        for a, b in lifted_edges:
            if (a, b) not in edge_set and (b, a) not in edge_set:
                final_edges.append((a, b))
                edge_set.add((a, b))
                edge_set.add((b, a))
        return final_edges
    
    def check_girth_edges(edges, n, min_girth):
        """Check girth using edge list."""
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        # Verify 4-regular
        for i in range(n):
            if len(adj[i]) != 4:
                return False
        return _check_girth_fast(adj, n, min_girth)
    
    # Try candidate n values from Moore bound upward
    moore_bound = 2 * (1 + 3 + 9 + 27 + 81)  # = 242
    
    for n in range(moore_bound, min(moore_bound + 60, 310), 2):
        if time.time() > deadline - 5:
            break
        
        factorizations = get_factorizations(n)
        if not factorizations:
            continue
        
        for base_size, lift_size in factorizations:
            if time.time() > deadline - 5:
                break
            
            base_edges = base_graphs[base_size]()
            
            # Try random voltage assignments
            for attempt in range(30):
                if time.time() > deadline - 5:
                    break
                
                # Assign random voltages
                voltages = {}
                for (u, v) in base_edges:
                    voltages[(u, v)] = random.randint(0, lift_size - 1)
                
                lifted_edges = build_lift(base_edges, lift_size, voltages)
                
                if check_girth_edges(lifted_edges, n, g):
                    return n, lifted_edges
    
    return None

def construct(k, g):
    if k != 4 or g != 10:
        return _luw_construct(k, g)
    
    deadline = time.time() + 115
    
    # First, get the LUW construction as fallback
    luw_n, luw_edges = _luw_construct(k, g)
    best_n = luw_n
    best_edges = luw_edges
    
    # Try voltage lift construction
    vl_result = _voltage_lift_construct(k, g)
    if vl_result is not None:
        vl_n, vl_edges = vl_result
        if vl_n < best_n:
            best_n = vl_n
            best_edges = vl_edges
    
    # Try to find a smaller graph using local search
    # Start with random 4-regular bipartite graphs and try to maximize girth
    
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
        
        for i in range(n):
            if len(adj[i]) != k:
                return None
        
        return adj
    
    def local_search(adj, n, deadline):
        """Local search to maximize girth via edge swaps."""
        best_adj = [set(s) for s in adj]
        best_score = -1
        half = n // 2
        
        for iteration in range(2000):
            if time.time() > deadline:
                break
            
            score = -_count_short_cycles(best_adj, n)
            if score > best_score:
                best_score = score
            
            for attempt in range(30):
                if time.time() > deadline:
                    break
                u1 = random.randint(0, half - 1)
                u2 = random.randint(0, half - 1)
                if u1 == u2:
                    continue
                v1 = random.choice(list(best_adj[u1]))
                v2 = random.choice(list(best_adj[u2]))
                if v1 == v2:
                    continue
                if v2 in best_adj[u1] or v1 in best_adj[u2]:
                    continue
                
                best_adj[u1].remove(v1)
                best_adj[v1].remove(u1)
                best_adj[u2].remove(v2)
                best_adj[v2].remove(u2)
                best_adj[u1].add(v2)
                best_adj[v2].add(u1)
                best_adj[u2].add(v1)
                best_adj[v1].add(u2)
                
                if _check_girth_fast(best_adj, n, g):
                    edges = []
                    for i in range(half):
                        for j in best_adj[i]:
                            edges.append((i, j))
                    return n, edges
                
                new_score = -_count_short_cycles(best_adj, n)
                if new_score <= score:
                    best_adj[u1].remove(v2)
                    best_adj[v2].remove(u1)
                    best_adj[u2].remove(v1)
                    best_adj[v1].remove(u2)
                    best_adj[u1].add(v1)
                    best_adj[v1].add(u1)
                    best_adj[u2].add(v2)
                    best_adj[v2].add(u2)
        
        return None
    
    # Try sizes from Moore bound up to best_n
    for n in range(244, min(best_n + 1, 320), 2):
        if time.time() > deadline - 5:
            break
        
        for attempt in range(5):
            if time.time() > deadline - 5:
                break
            
            adj = generate_random_bipartite_4reg(n)
            if adj is None:
                continue
            
            result = local_search(adj, n, deadline - 5)
            if result is not None:
                best_n, best_edges = result
                return best_n, best_edges
    
    return best_n, best_edges
