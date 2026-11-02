import time
import math
import numpy as np

def construct(k, g):
    _t0 = time.time()
    _DEADLINE = 110.0

    # ---- Fallback: Lazebnik-Ustimenko-Woldar graph CD(d, q) ----
    def luw_construct(k, g):
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

    def luw_ok(k, g):
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        return d + 5 >= g

    # ---- Exact girth check via non-backtracking walk counts ----
    def girth_ok(n, edges, g):
        A = np.zeros((n, n), dtype=np.int64)
        for u, v in edges:
            if u == v:
                return False
            if A[u, v]:
                return False
            A[u, v] = 1
            A[v, u] = 1
        deg = A.sum(axis=1)
        if not np.all(deg == k):
            return False
        B = A.astype(np.int64)
        for L in range(3, g):
            newB = B @ A - B @ np.diag(deg)
            if int(np.trace(newB)) > 0:
                return False
            B = newB
        return True

    # ---- Voltage graph lift of Petersen over Z_32 ----
    def petersen_voltage_lift():
        # Petersen graph: 10 vertices, 15 edges, 3-regular
        # Outer pentagon: 0-1-2-3-4-0
        # Inner star: 5-7-9-6-8-5
        # Spokes: 0-5, 1-6, 2-7, 3-8, 4-9
        pet_edges = [
            (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
            (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
            (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
        ]
        # Adjacency list for Petersen
        pet_adj = [[] for _ in range(10)]
        for u, v in pet_edges:
            pet_adj[u].append(v)
            pet_adj[v].append(u)

        # We assign voltages in Z_32 to each directed edge.
        # Use a deterministic assignment: voltage depends on edge index.
        # Key constraint: for every non-backtracking closed walk of length L < 10
        # in the Petersen graph, the sum of voltages (with sign depending on
        # direction) must be nonzero mod 32.
        
        # Strategy: assign voltage w(e) for edge e = (u,v) as follows.
        # We'll use a systematic search over a small set of voltage patterns.
        # Since the Petersen graph has girth 5, non-backtracking closed walks
        # of length 5-9 are the critical ones.
        
        # Let's try a specific known-good voltage assignment.
        # Voltage for edge (u,v): we'll use a pattern based on vertex labels.
        # For a 3-regular graph lifted over Z_32, we need 3 voltages per vertex.
        
        # Let's define voltages for each of the 15 undirected edges.
        # For the lift: vertex (v, h) connects to (u, h + w(v,u)) for each neighbor u.
        # Non-backtracking means we can't immediately go back.
        
        # Try voltage assignment: use distinct values from {1, 2, 4, 8, 16}
        # assigned to edges in a pattern that breaks all short cycles.
        
        # Edge voltages (for directed edge u->v, voltage is w[u][v])
        # We'll use a symmetric-ish assignment with some asymmetry.
        # Let's try: for edge (u,v) with u < v, assign voltage based on position.
        
        # A known approach: use voltages that are all odd or all even to
        # control parity, combined with distinct magnitudes.
        
        # Let me try a specific assignment and verify:
        # Edge list with assigned voltages (mod 32):
        # (0,1):1, (1,2):2, (2,3):4, (3,4):8, (4,0):16,
        # (5,7):1, (7,9):2, (9,6):4, (6,8):8, (8,5):16,
        # (0,5):1, (1,6):2, (2,7):4, (3,8):8, (4,9):16
        
        # For directed edges: w(u,v) for u->v, and w(v,u) = -w(u,v) mod 32
        # Actually for a proper voltage graph, we need w(u,v) + w(v,u) = 0 mod 32
        # for the lift to be consistent. Let's use w(v,u) = 32 - w(u,v) mod 32.
        
        edge_voltage = {}
        voltages = [1, 2, 4, 8, 16, 1, 2, 4, 8, 16, 1, 2, 4, 8, 16]
        for i, (u, v) in enumerate(pet_edges):
            edge_voltage[(u, v)] = voltages[i]
            edge_voltage[(v, u)] = (32 - voltages[i]) % 32

        # Now verify: enumerate all non-backtracking closed walks of length 5 to 9
        # and check that the voltage sum is nonzero mod 32.
        
        # A non-backtracking walk is a sequence v0, v1, ..., vL=v0 where
        # v_{i-1} != v_{i+1} for all i (no immediate backtracking).
        
        def voltage_sum(walk):
            """walk is a list of vertices [v0, v1, ..., vL] with v0=vL"""
            total = 0
            for i in range(len(walk) - 1):
                u, v = walk[i], walk[i + 1]
                total += edge_voltage[(u, v)]
            return total % 32

        def gen_nonbacktracking_closed_walks(start, length):
            """Generate all non-backtracking closed walks of given length from start."""
            if length == 0:
                if start == start:
                    yield [start]
                return
            # Build walks of given length starting from start, non-backtracking
            # State: (current, prev, path)
            stack = [(start, -1, [start])]
            while stack:
                curr, prev, path = stack.pop()
                if len(path) - 1 == length:
                    if curr == start:
                        yield path
                    continue
                for nxt in pet_adj[curr]:
                    if nxt == prev:
                        continue  # non-backtracking
                    stack.append((nxt, curr, path + [nxt]))

        # Check all non-backtracking closed walks of length 5 to 9
        valid = True
        for L in range(5, 10):
            for s in range(10):
                for walk in gen_nonbacktracking_closed_walks(s, L):
                    if voltage_sum(walk) == 0:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break

        if not valid:
            return None

        # Build the lifted graph: 10 * 32 = 320 vertices
        n_lift = 320
        edges = []
        for h in range(32):
            for u in range(10):
                for v in pet_adj[u]:
                    w = edge_voltage[(u, v)]
                    h2 = (h + w) % 32
                    vid1 = u * 32 + h
                    vid2 = v * 32 + h2
                    if vid1 < vid2:
                        edges.append((vid1, vid2))
                    # else: already added from the other side? No, we need to add all.
                    # Actually each directed edge gives one undirected edge in the lift.
                    # We should add each undirected edge once.
                    # Let me reconsider: for each vertex (u,h) and each neighbor v,
                    # we add edge ((u,h), (v, h+w(u,v))).
                    # This is directed; the undirected edge is the same regardless of direction.
                    # To avoid duplicates, only add if vid1 < vid2.
        # But wait, this might miss some edges or add duplicates. Let me redo.
        # For each undirected edge in the lift, it comes from a directed edge in the base.
        # Each base edge (u,v) with u<v gives, for each h, an edge in the lift:
        # ((u,h), (v, h+w(u,v))) and also ((v,h), (u, h+w(v,u))) = ((v,h), (u, h-w(u,v)))
        # These are different edges in general. So we have 2 * 15 * 32 = 960 directed edges,
        # which is 480 undirected edges... but we need 320*4/2 = 640 undirected edges.
        # Hmm, 10 vertices * 4 neighbors / 2 = 20 undirected edges per layer? No.
        # In the lift, each vertex (u,h) has degree 3 (same as base). So total edges = 320*3/2 = 480.
        # But we need 4-regular! The Petersen is 3-regular, so the lift is 3-regular.
        # We need 4-regular. This doesn't work directly.
        
        return None  # Petersen lift gives 3-regular, not 4-regular

    # ---- For k=4, g=10: use a 4-regular base graph ----
    # The complete bipartite K_{4,4} is 4-regular with 8 vertices.
    # Or use the 4-regular circulant C_8(1,3).
    # Actually, let's use a different approach: search for a 4-regular graph
    # on 8 vertices (the Moore bound for k=4, g=10 is 1+4+4*3+4*3*3+4*3*3*3 = 1+4+12+36+108=161... wait)
    # Moore bound: 1 + k*sum_{i=0}^{D-1}(k-1)^i where D=(g-2)/2=4
    # = 1 + 4*(1 + 3 + 9 + 27) = 1 + 4*40 = 161
    # So n must be >= 161. The known cage is 320.
    
    # Let's try the voltage lift approach with a 4-regular base graph.
    # The 4-regular graph on 8 vertices: C_8(1,3) (circulant with jumps 1 and 3).
    # Lift over Z_40 to get 320 vertices? No, 8*40=320. But we need to verify girth.
    
    # Actually, let me just try the circulant C_320(a,b) search more efficiently.
    # Use the fact that for a circulant, the girth can be checked by
    # finding the shortest solution to a*x + b*y = 0 mod n with |x|+|y| < g/2.
    
    def circulant_girth_check(n, a, b, g):
        """Check if circulant C_n(a,b) has girth >= g."""
        # A cycle of length L in C_n(a,b) corresponds to a sequence of steps
        # ±a, ±b that sums to 0 mod n, with no two consecutive steps being
        # inverses (non-backtracking).
        # For small L, we can check by BFS on the "step space".
        # State: (current_position, last_step) where last_step is in {a, -a, b, -b, None}
        # We want to find if there's a non-backtracking closed walk of length < g.
        
        # Simpler: for L from 4 to g-1, check if there exist integers
        # x1, x2, ..., xL each in {±a, ±b} such that:
        # 1. sum(xi) ≡ 0 mod n
        # 2. xi != -x(i+1) for all i (cyclic, so xL != -x1)
        # 3. The walk is non-backtracking (condition 2 ensures this)
        
        # For efficiency, enumerate all sequences of steps of length L.
        # There are 4^L possibilities, which is manageable for L <= 9.
        
        steps = [a, -a, b, -b]
        for L in range(4, g):
            # Use itertools.product for step sequences
            for seq in _product(steps, repeat=L):
                # Check non-backtracking: no two consecutive are inverses
                valid = True
                for i in range(L):
                    if seq[i] == -seq[(i + 1) % L]:
                        valid = False
                        break
                if not valid:
                    continue
                s = sum(seq) % n
                if s == 0:
                    return False
        return True

    def _product(iterable, repeat):
        """Simple product generator to avoid importing itertools in hot path."""
        import itertools
        return itertools.product(iterable, repeat=repeat)

    # ---- Main logic ----
    if k == 4 and g == 10:
        # Try circulant C_320(a,b) with efficient girth check
        n_c = 320
        # a and b must be coprime to n for the graph to be connected
        # and a, b must be distinct mod n/2 for 4-regular
        for a in range(1, n_c // 2 + 1):
            if time.time() - _t0 > _DEADLINE:
                break
            if math.gcd(a, n_c) != 1:
                continue
            for b in range(a + 1, n_c // 2 + 1):
                if math.gcd(b, n_c) != 1:
                    continue
                # Also need a != b mod n and a != -b mod n (already ensured by a<b<n/2)
                if circulant_girth_check(n_c, a, b, g):
                    edges = []
                    for i in range(n_c):
                        edges.append((i, (i + a) % n_c))
                        edges.append((i, (i + b) % n_c))
                    return n_c, edges
            if time.time() - _t0 > _DEADLINE:
                break

        # Fallback to LUW
        if luw_ok(k, g):
            return luw_construct(k, g)
        return luw_construct(k, g)

    # General case: use LUW
    if luw_ok(k, g):
        return luw_construct(k, g)
    return luw_construct(k, g)
