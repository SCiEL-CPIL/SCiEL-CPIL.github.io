import random
import time

def popcount(x):
    return x.bit_count()

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    # Precompute all pairs
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    all_pairs_set = set(all_pairs)

    def is_valid_addition(adj, u, v):
        # Check if adding edge (u, v) keeps the graph K2,3-free
        # 1. Common neighbors of u and v must be < 3
        if (adj[u] & adj[v]).bit_count() >= 3:
            return False

        # 2. For any w in N(v)\{u}, v becomes a common neighbor of u and w.
        #    So common neighbors of (u, w) must currently be < 2.
        neighbors_v = adj[v] & ~(1 << u)
        while neighbors_v:
            lsb = neighbors_v & -neighbors_v
            w = (lsb.bit_length() - 1)
            if (adj[u] & adj[w]).bit_count() >= 2:
                return False
            neighbors_v ^= lsb

        # 3. Similarly for w in N(u)\{v}
        neighbors_u = adj[u] & ~(1 << v)
        while neighbors_u:
            lsb = neighbors_u & -neighbors_u
            w = (lsb.bit_length() - 1)
            if (adj[v] & adj[w]).bit_count() >= 2:
                return False
            neighbors_u ^= lsb

        return True

    def try_add_edge(adj, u, v):
        if is_valid_addition(adj, u, v):
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            return True
        return False

    def greedy_build(adj, order):
        for u, v in order:
            try_add_edge(adj, u, v)

    def local_search_add(adj, max_rounds=5):
        """Iteratively add any missing edges until no more can be added."""
        for _ in range(max_rounds):
            improved = False
            for u, v in all_pairs:
                if not ((adj[u] >> v) & 1):
                    if try_add_edge(adj, u, v):
                        improved = True
            if not improved:
                break

    def two_swap_local_search(adj, max_iters=50):
        """Try removing 2 edges and adding up to 3 to gain net +1 edge."""
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            # Pick 2 random existing edges to remove
            existing = [(i, j) for i in range(n) for j in range(i+1, n)
                        if (adj[i] >> j) & 1]
            if len(existing) < 2:
                break
            e1, e2 = random.sample(existing, 2)
            u1, v1 = e1
            u2, v2 = e2

            # Save state
            adj_save = list(adj)

            # Remove both edges
            adj[u1] &= ~(1 << v1)
            adj[v1] &= ~(1 << u1)
            adj[u2] &= ~(1 << v2)
            adj[v2] &= ~(1 << u2)

            # Try to add missing edges
            missing = [(i, j) for i in range(n) for j in range(i+1, n)
                       if not ((adj[i] >> j) & 1)]
            random.shuffle(missing)
            for a, b in missing:
                try_add_edge(adj, a, b)

            # Check if we gained net edges
            old_count = sum(adj_save[i].bit_count() for i in range(n)) // 2
            new_count = sum(adj[i].bit_count() for i in range(n)) // 2
            if new_count <= old_count:
                # Revert
                adj[:] = adj_save

    def one_swap_local_search(adj, max_iters=50):
        """Try removing 1 edge and adding up to 2 to gain net +1 edge."""
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            # Pick a random existing edge to remove
            existing = [(i, j) for i in range(n) for j in range(i+1, n)
                        if (adj[i] >> j) & 1]
            if not existing:
                break
            u, v = random.choice(existing)

            # Save state
            adj_save = list(adj)

            # Remove edge
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)

            # Try to add missing edges
            missing = [(i, j) for i in range(n) for j in range(i+1, n)
                       if not ((adj[i] >> j) & 1)]
            random.shuffle(missing)
            for a, b in missing:
                try_add_edge(adj, a, b)

            # Check if we gained net edges
            old_count = sum(adj_save[i].bit_count() for i in range(n)) // 2
            new_count = sum(adj[i].bit_count() for i in range(n)) // 2
            if new_count <= old_count:
                # Revert
                adj[:] = adj_save

    def count_edges(adj):
        return sum(adj[i].bit_count() for i in range(n)) // 2

    def extract_edges(adj):
        edges = []
        for i in range(n):
            mask = adj[i]
            while mask:
                lsb = mask & -mask
                j = (lsb.bit_length() - 1)
                if i < j:
                    edges.append([i, j])
                mask ^= lsb
        return edges

    def build_polarity_graph_pgf4():
        """
        Construct the polarity graph of PG(2,4) restricted to 20 vertices.
        
        PG(2,4) has 21 points. We use a specific polarity (symplectic form)
        to define adjacency, then delete one vertex to get 20 vertices.
        
        The collinearity graph of PG(2,4) has 21 vertices, each with degree 5,
        giving 52.5 edges... wait, that's not right.
        
        Actually, let me use a different approach. The polarity graph of a
        symplectic polarity on PG(2,4) gives a graph where two points are
        adjacent if they are orthogonal under the polarity.
        
        For q=4, PG(2,4) has (4^3-1)/(4-1) = 21 points.
        A symplectic polarity makes each point orthogonal to q+1=5 other points.
        So the graph has 21*5/2 = 52.5... still not integer.
        
        Let me reconsider. Actually for a polarity graph, the number of
        neighbors of a point depends on whether it's self-orthogonal.
        
        Let me use a concrete construction:
        Points of PG(2,4) can be represented as 1D subspaces of GF(4)^3.
        GF(4) = {0, 1, alpha, alpha^2} where alpha^2 = alpha + 1.
        
        I'll use the standard polarity given by the bilinear form:
        B(x,y) = x0*y2 + x1*y0 + x2*y1
        
        Two points [x] and [y] are adjacent if B(x,y) = 0.
        """
        # GF(4) arithmetic
        # Elements: 0, 1, a, a2 where a^2 = a + 1
        # Multiplication table for non-zero elements:
        # 1*1=1, 1*a=a, 1*a2=a2
        # a*a=a2, a*a2=1
        # a2*a2=a
        
        gf4_add = [
            [0, 1, 2, 3],
            [1, 0, 3, 2],
            [2, 3, 0, 1],
            [3, 2, 1, 0]
        ]
        
        gf4_mul = [
            [0, 0, 0, 0],
            [0, 1, 2, 3],
            [0, 2, 3, 1],
            [0, 3, 1, 2]
        ]
        
        def gf4_add_vec(v1, v2):
            return [gf4_add[v1[i]][v2[i]] for i in range(3)]
        
        def gf4_mul_scalar(s, v):
            return [gf4_mul[s][v[i]] for i in range(3)]
        
        def gf4_dot(v1, v2):
            result = 0
            for i in range(3):
                result = gf4_add[result][gf4_mul[v1[i]][v2[i]]]
            return result
        
        # Generate all 21 points of PG(2,4)
        # Each point is a 1D subspace, represented by a canonical representative
        points = []
        seen = set()
        
        for x0 in range(4):
            for x1 in range(4):
                for x2 in range(4):
                    if x0 == 0 and x1 == 0 and x2 == 0:
                        continue
                    # Canonical form: first non-zero coordinate is 1
                    # But in GF(4), we need to normalize
                    # Find first non-zero
                    first_nonzero = -1
                    for i in range(3):
                        if v[i] != 0 if False else (x0, x1, x2)[i] != 0:
                            first_nonzero = i
                            break
                    
                    if first_nonzero == -1:
                        continue
                    
                    # Scale so first non-zero is 1
                    scale_inv = [0, 1, 3, 2]  # inverse of 1,2,3 in GF(4)
                    # 1^-1=1, 2^-1=3, 3^-1=2
                    first_val = (x0, x1, x2)[first_nonzero]
                    inv_first = [0, 1, 3, 2][first_val]
                    
                    scaled = gf4_mul_scalar(inv_first, [x0, x1, x2])
                    # Zero out the first_nonzero position (it's now 1, but we want canonical)
                    # Actually for canonical form, we can just use the scaled vector
                    # and zero out coordinates before first_nonzero
                    canonical = [0, 0, 0]
                    for i in range(3):
                        if i >= first_nonzero:
                            canonical[i] = scaled[i]
                        else:
                            canonical[i] = 0
                    
                    # Make a hashable key
                    key = tuple(canonical)
                    if key not in seen:
                        seen.add(key)
                        points.append(canonical)
        
        # We should have 21 points
        # Now build the polarity graph
        # Adjacency: B(x,y) = x0*y2 + x1*y0 + x2*y1 = 0
        m = len(points)
        adj_full = [0] * m
        
        for i in range(m):
            for j in range(m):
                if i == j:
                    continue
                # Compute B(points[i], points[j])
                xi = points[i]
                xj = points[j]
                b_val = gf4_dot([xi[0], xi[1], xi[2]], [xj[2], xj[0], xj[1]])
                if b_val == 0:
                    adj_full[i] |= (1 << j)
        
        # Delete vertex 0 (or any vertex) to get 20 vertices
        # Remap vertices 1..20 to 0..19
        adj_20 = [0] * 20
        for i in range(1, m):
            for j in range(1, m):
                if adj_full[i] & (1 << j):
                    adj_20[i-1] |= (1 << (j-1))
        
        return adj_20

    best_edges = []
    best_count = 0

    # First, try the polarity graph construction
    adj_polarity = build_polarity_graph_pgf4()
    local_search_add(adj_polarity)
    edge_count = count_edges(adj_polarity)
    if edge_count > best_count:
        best_count = edge_count
        best_edges = extract_edges(adj_polarity)

    # Then do local search to improve
    one_swap_local_search(adj_polarity, max_iters=50)
    two_swap_local_search(adj_polarity, max_iters=30)
    local_search_add(adj_polarity)
    edge_count = count_edges(adj_polarity)
    if edge_count > best_count:
        best_count = edge_count
        best_edges = extract_edges(adj_polarity)

    # Also try random restarts with the polarity graph as a starting point
    for restart in range(500):
        if time.time() > deadline:
            break

        adj = list(adj_polarity)

        # Add some randomness by removing a few edges and re-adding
        existing = [(i, j) for i in range(n) for j in range(i+1, n)
                    if (adj[i] >> j) & 1]
        if len(existing) >= 3:
            to_remove = random.sample(existing, 3)
            for u, v in to_remove:
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)

        local_search_add(adj)
        one_swap_local_search(adj, max_iters=20)
        two_swap_local_search(adj, max_iters=15)
        local_search_add(adj)

        edge_count = count_edges(adj)
        if edge_count > best_count:
            best_count = edge_count
            best_edges = extract_edges(adj)

    return best_edges
