import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    # 1. Generate Clebsch Graph as initial solution
    vectors = []
    for i in range(32):
        if bin(i).count('1') % 2 == 0:
            vectors.append(i)

    adj = [0] * 16
    clebsch_edges = []
    for u_idx in range(16):
        for v_idx in range(u_idx + 1, 16):
            xor = vectors[u_idx] ^ vectors[v_idx]
            if bin(xor).count('1') == 4:
                clebsch_edges.append([u_idx, v_idx])
                adj[u_idx] |= (1 << v_idx)
                adj[v_idx] |= (1 << u_idx)

    best_edges = clebsch_edges
    best_count = len(clebsch_edges)  # 40

    # 2. Fast 3-colorability check using bitmasks
    def is_3_colorable(adj_mask):
        n = 16
        neighbors = [0] * n
        for i in range(n):
            neighbors[i] = adj_mask[i]

        order = sorted(range(n), key=lambda x: -bin(adj_mask[x]).count('1'))

        colors = [-1] * n
        color_classes = [0, 0, 0]

        def backtrack(idx):
            if idx == n:
                return True
            v = order[idx]
            for c in range(3):
                if neighbors[v] & color_classes[c]:
                    continue
                colors[v] = c
                color_classes[c] |= (1 << v)
                if backtrack(idx + 1):
                    return True
                color_classes[c] &= ~(1 << v)
                colors[v] = -1
            return False

        return backtrack(0)

    # 3. WalkSAT-style local search for Max-SAT formulation
    # Variables: x_{u,v} for each potential edge (u,v)
    # Hard constraints: triangle-freeness (no three mutually connected vertices)
    # Soft objective: maximize number of edges
    # Additional hard constraint: graph must not be 3-colorable

    all_pairs = [(u, v) for u in range(16) for v in range(u+1, 16)]
    edge_index = {}
    for i, (u, v) in enumerate(all_pairs):
        edge_index[(u, v)] = i

    n_edges_total = len(all_pairs)

    # Precompute triangles: for each pair of edges that share a vertex, 
    # find the third edge that would complete a triangle
    # Triangle constraint: for any triangle (a,b,c), not all three edges can be present
    # We encode this as: for each triangle, at least one edge must be absent

    # Build adjacency structure for quick triangle detection
    def get_triangles_from_adj(adj_mask):
        """Return list of triangles as tuples of edge indices"""
        triangles = []
        for u in range(16):
            for v in range(u+1, 16):
                if adj_mask[u] & (1 << v):
                    common = adj_mask[u] & adj_mask[v]
                    for w in range(16):
                        if common & (1 << w):
                            if w > v:
                                triangles.append((edge_index[(u,v)], edge_index[(v,w)], edge_index[(u,w)]))
                            elif w < u:
                                triangles.append((edge_index[(u,v)], edge_index[(u,w)], edge_index[(v,w)]))
        return triangles

    # WalkSAT solver
    def walksat_solve(adj_mask, max_steps=50000):
        """
        WalkSAT-style local search.
        State: adj_mask represents current graph
        Moves: add/remove an edge
        Objective: maximize edge count while maintaining triangle-freeness and non-3-colorability
        """
        nonlocal best_edges, best_count

        n = 16
        current_adj = adj_mask[:]
        current_edges = []
        for u in range(n):
            for v in range(u+1, n):
                if current_adj[u] & (1 << v):
                    current_edges.append([u, v])

        current_count = len(current_edges)

        # Precompute for each edge, which other edges form triangles with it
        # For edge (u,v), a triangle is formed if there exists w such that (u,w) and (v,w) are both present
        # So adding edge (u,v) is only valid if no w has both (u,w) and (v,w) present

        def can_add_edge(adj_m, u, v):
            common = adj_m[u] & adj_m[v]
            return common == 0

        def edge_count(adj_m):
            count = 0
            for u in range(n):
                count += bin(adj_m[u]).count('1')
            return count // 2

        # WalkSAT parameters
        walk_prob = 0.3
        best_local = current_count

        for step in range(max_steps):
            if time.time() > deadline:
                break

            # Check if current solution is valid (triangle-free and not 3-colorable)
            if not is_3_colorable(current_adj):
                if current_count > best_count:
                    best_edges = current_edges[:]
                    best_count = current_count

            # Try all possible moves: add or remove an edge
            best_move = None
            best_score = -1

            for u in range(n):
                for v in range(u+1, n):
                    is_present = bool(current_adj[u] & (1 << v))

                    if is_present:
                        # Try removing edge (u,v)
                        new_adj = current_adj[:]
                        new_adj[u] &= ~(1 << v)
                        new_adj[v] &= ~(1 << u)
                        new_count = current_count - 1

                        # Check triangle-freeness (removing edges never creates triangles)
                        # Check 3-colorability
                        if not is_3_colorable(new_adj):
                            # Score: we prefer to keep edges, so removing is bad
                            # But we might need to remove to allow more additions later
                            score = new_count  # lower is worse for removal
                            if score > best_score:
                                best_score = score
                                best_move = ('remove', u, v, new_adj, new_count)
                    else:
                        # Try adding edge (u,v)
                        if can_add_edge(current_adj, u, v):
                            new_adj = current_adj[:]
                            new_adj[u] |= (1 << v)
                            new_adj[v] |= (1 << u)
                            new_count = current_count + 1

                            # Check 3-colorability
                            if not is_3_colorable(new_adj):
                                score = new_count  # higher is better
                                if score > best_score:
                                    best_score = score
                                    best_move = ('add', u, v, new_adj, new_count)

            if best_move is None:
                # No improving move, do a random walk
                if random.random() < walk_prob:
                    # Pick a random edge to flip
                    u = random.randint(0, n-1)
                    v = random.randint(u+1, n-1)
                    is_present = bool(current_adj[u] & (1 << v))

                    if is_present:
                        new_adj = current_adj[:]
                        new_adj[u] &= ~(1 << v)
                        new_adj[v] &= ~(1 << u)
                        new_count = current_count - 1
                        if not is_3_colorable(new_adj):
                            current_adj = new_adj
                            current_count = new_count
                    else:
                        if can_add_edge(current_adj, u, v):
                            new_adj = current_adj[:]
                            new_adj[u] |= (1 << v)
                            new_adj[v] |= (1 << u)
                            new_count = current_count + 1
                            if not is_3_colorable(new_adj):
                                current_adj = new_adj
                                current_count = new_count
                else:
                    break  # Stuck, break out

            # Apply best move
            move_type, u, v, new_adj, new_count = best_move
            current_adj = new_adj
            current_count = new_count

            # Update edge list
            current_edges = []
            for uu in range(n):
                for vv in range(uu+1, n):
                    if current_adj[uu] & (1 << vv):
                        current_edges.append([uu, vv])

        return current_edges, current_count

    # 4. Greedy edge addition (fast, no 3-color check)
    def try_add_edges_greedy(adj_mask, current_edges, current_count):
        nonlocal best_edges, best_count
        n = 16
        changed = True
        while changed and time.time() < deadline:
            changed = False
            available = []
            for u in range(n):
                for v in range(u+1, n):
                    if not (adj_mask[u] & (1 << v)):
                        if can_add_edge(adj_mask, u, v):
                            available.append((u, v))

            if not available:
                break

            for u, v in available:
                if time.time() > deadline:
                    break
                adj_mask[u] |= (1 << v)
                adj_mask[v] |= (1 << u)
                new_edges = current_edges + [[u, v]]
                new_count = current_count + 1

                if not is_3_colorable(adj_mask):
                    if new_count > best_count:
                        best_edges = new_edges
                        best_count = new_count
                    new_edges, new_count = try_add_edges_greedy(adj_mask, new_edges, new_count)
                    current_edges = new_edges
                    current_count = new_count
                    changed = True
                else:
                    adj_mask[u] &= ~(1 << v)
                    adj_mask[v] &= ~(1 << u)

        return current_edges, current_count

    # 5. Run WalkSAT from Clebsch graph
    clebsch_adj = adj[:]
    walksat_solve(clebsch_adj[:], max_steps=30000)

    # 6. Random restarts with WalkSAT
    for iteration in range(200):
        if time.time() > deadline:
            break

        # Create a random starting graph
        adj_mask = [0] * 16
        edges = []

        # Start with a random subset of Clebsch edges
        for u, v in clebsch_edges:
            if random.random() < 0.7:
                adj_mask[u] |= (1 << v)
                adj_mask[v] |= (1 << u)
                edges.append([u, v])

        # Add some random edges that don't create triangles
        all_pairs = [(u, v) for u in range(16) for v in range(u+1, 16)]
        random.shuffle(all_pairs)
        for u, v in all_pairs:
            if time.time() > deadline:
                break
            if not (adj_mask[u] & (1 << v)):
                if can_add_edge(adj_mask, u, v):
                    adj_mask[u] |= (1 << v)
                    adj_mask[v] |= (1 << u)
                    edges.append([u, v])

        # Run WalkSAT from this random starting point
        walksat_solve(adj_mask[:], max_steps=20000)

    return best_edges
