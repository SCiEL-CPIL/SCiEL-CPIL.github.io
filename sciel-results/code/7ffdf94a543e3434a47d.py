import time
import random
from collections import deque

def construct(k, g):
    deadline = time.time() + 115

    # Moore bound for (3,16): 2*(k-1)*(k-2)^(g/2 - 1) = 2*2*2^7 = 512
    # Smallest known: 936 (Exoo et al. 2026)

    def luw_construct(k, g, d):
        """LUW-type construction for k-regular graphs."""
        q = k
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add = lambda x, y: x ^ y
            sub = lambda x, y: x ^ y
            mul = lambda x, y: table[x][y]
        else:
            add = lambda x, y: (x + y) % q
            sub = lambda x, y: (x - y) % q
            mul = lambda x, y: x * y % q

        start = (0, (None,) + (0,) * d)
        index = {start: 0}
        order = [start]
        edges = []

        for vertex in order:
            side, x = vertex
            v_idx = index[vertex]
            for first in range(q):
                y = [None, first] + [0] * (d - 1)
                for c in range(2, d + 1):
                    a, b = rule[c]
                    if side == 0:
                        y[c] = add(x[c], mul(y[a], x[b]))
                    else:
                        y[c] = sub(x[c], mul(x[a], y[b]))
                other = (1 - side, tuple(y))
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if side == 0:
                    edges.append((v_idx, index[other]))

        return len(order), edges

    def verify_girth(n, edges, g):
        """Verify girth >= g using non-backtracking BFS from each vertex."""
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            q = deque([start])
            while q:
                u = q.popleft()
                if dist[u] >= g // 2:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        q.append(v)
                    elif v != parent[u]:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < g:
                            return False
        return True

    def generate_k33_lift(m, seed):
        """
        Generate a 3-regular graph as a voltage lift of K_{3,3} over Z_m.
        
        K_{3,3} has vertices {A0, A1, A2, B0, B1, B2} with edges Ai-Bj for all i,j.
        A voltage lift assigns each edge a voltage in Z_m. The lift has 6m vertices:
        (u, t) where u is a vertex of K_{3,3} and t in Z_m.
        Edge (u, t) -- (v, t + voltage(u,v)) for each edge uv in K_{3,3}.
        
        To avoid loops and multiedges, we need the voltage assignment to be a
        proper "girth-16" assignment. We use a structured approach:
        - Assign random voltages to the 9 edges of K_{3,3}
        - Check if the resulting lift is simple and has girth >= 16
        """
        rng = random.Random(seed)
        
        # K_{3,3} edges: A0-B0, A0-B1, A0-B2, A1-B0, A1-B1, A1-B2, A2-B0, A2-B1, A2-B2
        # We assign voltages to each edge.
        # To ensure the lift is simple (no loops, no multiedges), we need:
        # For each edge (u,v) in K_{3,3}, voltage(u,v) != 0 would cause a loop
        # Actually, a loop occurs if voltage = 0. A multiedge occurs if two different
        # edges in the base create the same pair in the lift.
        
        # Let's use a specific structured voltage assignment:
        # v(Ai, Bj) = a_i + b_j * c + d_ij (mod m) for some parameters
        # Actually, let's just try random voltages and check.
        
        # For simplicity and speed, use a structured approach:
        # v(Ai, Bj) = i * x + j * y + z_ij where x, y are chosen to avoid short cycles
        # Let's use: v(Ai, Bj) = i * p + j * q (mod m) with p, q random
        # This gives a circulant-like structure.
        
        # Actually, for K_{3,3}, the base has 9 edges. Let's assign:
        # v(Ai, Bj) = a[i] + b[j] (mod m)
        # where a[0]=0, a[1]=x, a[2]=y and b[0]=0, b[1]=x', b[2]=y'
        
        x = rng.randint(1, m - 1)
        y = rng.randint(1, m - 1)
        xp = rng.randint(1, m - 1)
        yp = rng.randint(1, m - 1)
        
        # Ensure we don't get trivial cases
        if x == y or xp == yp or x == xp or y == yp:
            return None
        
        n = 6 * m
        # Vertex (u, t) where u in 0..5 (0=A0, 1=A1, 2=A2, 3=B0, 4=B1, 5=B2), t in 0..m-1
        # Vertex id = u * m + t
        
        edges = []
        seen = set()
        
        # A vertices: 0, 1, 2
        # B vertices: 3, 4, 5
        # Edge Ai-Bj: voltage = a[i] + b[j]
        a_vals = [0, x, y]
        b_vals = [0, xp, yp]
        
        for i in range(3):
            for j in range(3):
                volt = (a_vals[i] + b_vals[j]) % m
                if volt == 0:
                    # This would create edges that might cause issues, but let's keep it
                    # Actually volt=0 means (Ai, t) connects to (Bj, t)
                    # This is fine, no loop since Ai != Bj
                    pass
                for t in range(m):
                    u = i  # Ai
                    v = 3 + j  # Bj
                    tu = t
                    tv = (t + volt) % m
                    uid = u * m + tu
                    vid = v * m + tv
                    if uid == vid:
                        continue
                    edge_key = (min(uid, vid), max(uid, vid))
                    if edge_key not in seen:
                        seen.add(edge_key)
                        edges.append(edge_key)
        
        # Check for duplicate edges (shouldn't happen with this construction but be safe)
        if len(edges) != len(seen):
            return None
        
        # Check simplicity: no self-loops, no multiedges
        # Already checked above
        
        # Now check girth >= 16
        if not verify_girth(n, edges, g):
            return None
        
        return n, edges

    # For k=3, g=16, try the K_{3,3} voltage lift approach
    if k == 3 and g == 16:
        best_n = float('inf')
        best_edges = None
        
        # Try various m values. n = 6m, so m=86 gives n=516 (just above Moore bound 512)
        # m=87 gives n=522, etc.
        # The LUW construction with d=11 gives n=2*3^11 = 354294, which is way too big.
        # Wait, let me recalculate: LUW with q=3, d=11 gives 2*3^11 = 354294 vertices.
        # That's not 936. The seed program must be using a different parameter.
        
        # Actually, looking at the seed program more carefully:
        # luw_construct(k=3, g=16, d) with d=11
        # The number of vertices is 2 * q^d = 2 * 3^11 = 354294
        # But the task says the seed scores 3.0 which means n=936.
        # So the LUW construction in the seed must be different or I'm misunderstanding.
        
        # Let me just focus on the K_{3,3} lift which gives n=6m vertices.
        # For n=936, m=156. Let's try m values around there and below.
        
        # Try m from 90 to 200, with multiple seeds each
        for m in range(90, 201):
            if time.time() > deadline - 5:
                break
            n_candidate = 6 * m
            if n_candidate >= best_n:
                continue
            
            for seed in range(5):
                if time.time() > deadline - 5:
                    break
                try:
                    result = generate_k33_lift(m, seed)
                    if result is not None:
                        n_found, edges_found = result
                        if n_found < best_n:
                            best_n = n_found
                            best_edges = edges_found
                            break  # Found a valid graph for this m
                except Exception:
                    continue
        
        if best_edges is not None:
            return best_n, best_edges
        
        # If K_{3,3} lift didn't work, try LUW as fallback
        for d in range(8, 13):
            if time.time() > deadline - 2:
                break
            try:
                n, edges = luw_construct(k, g, d)
                if n < 512 or n >= best_n:
                    continue
                if verify_girth(n, edges, g):
                    return n, edges
            except Exception:
                continue

    # General fallback
    d_safe = max(3, g - 5)
    try:
        n, edges = luw_construct(k, g, d_safe)
        return n, edges
    except Exception:
        pass

    # Last resort
    n = max(g, 4)
    edges = [(i, (i + 1) % n) for i in range(n)]
    return n, edges
