import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115

    def has_independent_set_k(n, adj, k):
        """Check if graph has independent set of size k using bitset DFS."""
        found = [False]
        
        def dfs(candidates, count):
            if found[0]:
                return
            if count == k:
                found[0] = True
                return
            if candidates.bit_count() < k - count:
                return
            lsb = candidates & -candidates
            v = lsb.bit_length() - 1
            new_cand = candidates & ~adj[v] & ~lsb
            dfs(new_cand, count + 1)
            if not found[0]:
                dfs(candidates & ~lsb, count)
        
        dfs((1 << n) - 1, 0)
        return found[0]

    def build_adj_from_edges(n, edges):
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj

    def check_no_triangle(n, adj):
        """Check triangle-free using adjacency bitmasks."""
        for u in range(n):
            neighbors_u = adj[u]
            for v in range(u + 1, n):
                if adj[u] & (1 << v):
                    common = adj[u] & adj[v]
                    if common & (1 << u) == 0 and common & (1 << v) == 0:
                        # Check if any common neighbor w > v to avoid double counting
                        # Actually just check if common has any bit set
                        if common:
                            return False
        return True

    def polar_graph_pq(q):
        """
        Construct polar graph from PG(2,q) with orthogonal relation.
        Vertices: points of PG(2,q) = (q^2 + q + 1) points.
        Two points are adjacent if they are orthogonal w.r.t. a non-degenerate symmetric form.
        For q=3: PG(2,3) has 13 points.
        For q=5: PG(2,5) has 31 points.
        For q=7: PG(2,7) has 57 points (too large).
        
        Actually, let me use a different approach. The polar graph on PG(2,q) with
        the standard bilinear form gives a strongly regular graph.
        
        Let me use the known construction: for prime power q, the graph where
        vertices are 1-dimensional subspaces of F_q^3, and two vertices are adjacent
        if the corresponding vectors are orthogonal under the standard dot product.
        """
        # Generate all 1-dim subspaces of F_q^3
        def gen_points(q):
            points = []
            seen = set()
            for a in range(q):
                for b in range(q):
                    for c in range(q):
                        if a == 0 and b == 0 and c == 0:
                            continue
                        # Canonical representative: first nonzero coordinate is 1
                        rep = None
                        if a != 0:
                            inv_a = pow(a, q - 2, q) if q > 2 else 1
                            rep = (1, (b * inv_a) % q, (c * inv_a) % q)
                        elif b != 0:
                            inv_b = pow(b, q - 2, q) if q > 2 else 1
                            rep = (0, 1, (c * inv_b) % q)
                        else:
                            rep = (0, 0, 1)
                        if rep not in seen:
                            seen.add(rep)
                            points.append(rep)
            return points

        def orthogonal(p1, p2, q):
            return (p1[0]*p2[0] + p1[1]*p2[1] + p1[2]*p2[2]) % q == 0

        points = gen_points(q)
        n = len(points)
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if orthogonal(points[i], points[j], q):
                    edges.append([i, j])
        return n, edges

    def circulant_graph(n, S):
        """Build circulant graph with shift set S."""
        S_set = set(S)
        S_set.add(-s % n for s in S)
        S_set.discard(0)
        edges = []
        for u in range(n):
            for s in S_set:
                v = (u + s) % n
                if u < v:
                    edges.append([u, v])
        return edges

    def is_triangle_free_circulant(n, S):
        """Check if circulant graph with step set S is triangle-free."""
        D = set()
        for s in S:
            D.add(s % n)
            D.add((-s) % n)
        D.discard(0)
        
        for a in D:
            for b in D:
                if a == b:
                    continue
                c1 = (a + b) % n
                if c1 != 0 and c1 in D:
                    return False
                c2 = (a - b) % n
                if c2 != 0 and c2 in D:
                    return False
        return True

    def greedy_circulant(n):
        """
        Greedy construction of triangle-free circulant graph with no independent 10-set.
        Try to build a shift set S such that the resulting graph is triangle-free
        and has independence number < 10.
        """
        max_step = n // 2
        steps = list(range(1, max_step + 1))
        
        # Precompute conflict graph: step i conflicts with step j if adding both
        # creates a triangle
        conflict = [set() for _ in range(len(steps))]
        for i, si in enumerate(steps):
            for j, sj in enumerate(steps):
                if i == j:
                    continue
                # Check if {si, sj} creates a triangle
                # Triangle: 0, si, si+sj or 0, si, si-sj
                if (si + sj) % n != 0 and ((si + sj) % n) in steps:
                    conflict[i].add(j)
                    conflict[j].add(i)
                if (si - sj) % n != 0 and ((si - sj) % n) in steps:
                    conflict[i].add(j)
                    conflict[j].add(i)
        
        # Greedy: add steps in order of decreasing "usefulness"
        # Usefulness: how many vertices does it cover? Each step covers n edges.
        # But we want high degree to reduce independence number.
        # Let's try different orderings.
        
        best_S = None
        best_alpha = 999
        
        # Try multiple random orderings
        for trial in range(50):
            if time.time() > deadline:
                break
            order = list(range(len(steps)))
            random.shuffle(order)
            
            S = []
            used = set()
            for idx in order:
                if time.time() > deadline:
                    break
                # Check if adding step idx creates a triangle with existing S
                ok = True
                for existing in S:
                    if idx in conflict[existing]:
                        ok = False
                        break
                if ok:
                    S.append(idx)
                    used.add(idx)
            
            if not S:
                continue
            
            S_steps = [steps[i] for i in S]
            edges = circulant_graph(n, S_steps)
            
            # Check triangle-free (should be by construction, but verify)
            adj = build_adj_from_edges(n, edges)
            if not check_no_triangle(n, adj):
                continue
            
            # Check independence number
            if has_independent_set_k(n, adj, 10):
                continue
            
            # Valid!
            return n, edges
        
        return None

    def try_polar():
        """Try polar graph constructions."""
        for q in [3, 5]:
            if time.time() > deadline:
                break
            n, edges = polar_graph_pq(q)
            if n < 20:
                continue
            adj = build_adj_from_edges(n, edges)
            if not check_no_triangle(n, adj):
                continue
            if not has_independent_set_k(n, adj, 10):
                return n, edges
        return None

    # Strategy 1: Try polar graph constructions (deterministic, algebraic)
    result = try_polar()
    if result:
        return {'n': result[0], 'edges': result[1]}

    # Strategy 2: Greedy circulant construction
    for n in [40, 39]:
        if time.time() > deadline:
            break
        result = greedy_circulant(n)
        if result:
            return {'n': result[0], 'edges': result[1]}

    # Strategy 3: Random circulant search
    for n in [40, 39]:
        if time.time() > deadline:
            break
        max_step = n // 2
        steps = list(range(1, max_step + 1))
        
        for attempt in range(3000):
            if time.time() > deadline:
                break
            size = random.randint(n // 5, n // 3)
            S = set(random.sample(steps, size))
            if not is_triangle_free_circulant(n, S):
                continue
            edges = circulant_graph(n, S)
            adj = build_adj_from_edges(n, edges)
            if not check_no_triangle(n, adj):
                continue
            if not has_independent_set_k(n, adj, 10):
                return {'n': n, 'edges': edges}

    # Fallback: known valid graph on 26 vertices
    n = 26
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if (j - i) % n % 3 == 1:
                edges.append([i, j])
    return {'n': n, 'edges': edges}
