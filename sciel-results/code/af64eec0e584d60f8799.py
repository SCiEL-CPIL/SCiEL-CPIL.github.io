"""Local search for graphs on 16 vertices with 65 edges minimizing max triangle degree-sum."""
import time

def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    N = 16
    TARGET = 65

    # Bitmask representation: adj[i] is a bitmask of neighbors of vertex i
    # Initial construction: K_{8,8} plus one edge inside part A
    # Vertices 0-7 in part A, 8-15 in part B
    adj = [0] * N
    # All cross edges (A-B): 8*8 = 64 edges
    for i in range(8):
        for j in range(8):
            adj[i] |= (1 << (8 + j))
            adj[8 + j] |= (1 << i)
    # One extra edge inside A: (0,1)
    adj[0] |= (1 << 1)
    adj[1] |= (1 << 0)
    # Total: 65 edges

    def get_score(adj):
        """Compute the maximum degree sum over all triangles."""
        deg = [bin(adj[i]).count('1') for i in range(N)]
        best = 0
        for i in range(N):
            for j in range(i+1, N):
                if adj[i] & (1 << j):
                    common = adj[i] & adj[j]
                    # Find common neighbors k > j to avoid duplicates
                    common &= ~((1 << (j+1)) - 1)
                    while common:
                        lsb = common & (-common)
                        k = lsb.bit_length() - 1
                        s = deg[i] + deg[j] + deg[k]
                        if s > best:
                            best = s
                        common ^= lsb
        return best

    def get_worst_triangles(adj, threshold=None):
        """Get triangles with degree sum >= threshold (or all if None)."""
        deg = [bin(adj[i]).count('1') for i in range(N)]
        tris = []
        for i in range(N):
            for j in range(i+1, N):
                if adj[i] & (1 << j):
                    common = adj[i] & adj[j]
                    common &= ~((1 << (j+1)) - 1)
                    while common:
                        lsb = common & (-common)
                        k = lsb.bit_length() - 1
                        s = deg[i] + deg[j] + deg[k]
                        if threshold is None or s >= threshold:
                            tris.append((s, i, j, k))
                        common ^= lsb
        return tris

    def try_swap(adj, u, v, w, x):
        """Try swapping edge (u,v) with non-edge (w,x). Return new adj if valid and improves."""
        # Check (u,v) is an edge
        if not (adj[u] & (1 << v)):
            return None
        # Check (w,x) is a non-edge
        if adj[w] & (1 << x):
            return None
        # Perform swap
        adj[u] ^= (1 << v)
        adj[v] ^= (1 << u)
        adj[w] ^= (1 << x)
        adj[x] ^= (1 << w)
        s = get_score(adj)
        # Undo swap
        adj[u] ^= (1 << v)
        adj[v] ^= (1 << u)
        adj[w] ^= (1 << x)
        adj[x] ^= (1 << w)
        return s

    def try_add_remove(adj, u, v, w, x):
        """Try removing edge (u,v) and adding edge (w,x)."""
        if not (adj[u] & (1 << v)):
            return None
        if adj[w] & (1 << x):
            return None
        adj[u] ^= (1 << v)
        adj[v] ^= (1 << u)
        adj[w] ^= (1 << x)
        adj[x] ^= (1 << w)
        s = get_score(adj)
        adj[u] ^= (1 << v)
        adj[v] ^= (1 << u)
        adj[w] ^= (1 << x)
        adj[x] ^= (1 << w)
        return s

    current_score = get_score(adj)
    best_score = current_score
    best_adj = list(adj)

    # Local search: try to reduce the maximum triangle degree-sum
    # Strategy: find the worst triangle, try to break it by swapping an edge
    import random
    rng = random.Random(42)

    iteration = 0
    no_improve_count = 0

    while time.time() < deadline:
        iteration += 1
        if time.time() > deadline:
            break

        # Get the worst triangles
        tris = get_worst_triangles(adj, threshold=current_score)
        if not tris:
            break

        # Sort by degree sum descending
        tris.sort(reverse=True)

        improved = False
        for (s, i, j, k) in tris[:5]:  # Try top 5 worst triangles
            if time.time() > deadline:
                break

            # For each edge in this triangle, try to remove it and add a different edge
            edges_in_tri = [(i, j), (i, k), (j, k)]
            rng.shuffle(edges_in_tri)

            for (u, v) in edges_in_tri:
                if time.time() > deadline:
                    break

                # Try removing (u,v) and adding a random non-edge
                # Prefer non-edges that don't create new high-degree-sum triangles
                candidates = []
                for a in range(N):
                    for b in range(a+1, N):
                        if not (adj[a] & (1 << b)):
                            # Check this doesn't create a triangle with high degree sum
                            # Quick check: just try it
                            candidates.append((a, b))

                # Limit candidates for speed
                if len(candidates) > 50:
                    candidates = rng.sample(candidates, 50)

                rng.shuffle(candidates)
                for (w, x) in candidates[:10]:
                    if time.time() > deadline:
                        break
                    new_score = try_add_remove(adj, u, v, w, x)
                    if new_score is not None and new_score < current_score:
                        # Accept the improvement
                        adj[u] ^= (1 << v)
                        adj[v] ^= (1 << u)
                        adj[w] ^= (1 << x)
                        adj[x] ^= (1 << w)
                        current_score = new_score
                        if current_score < best_score:
                            best_score = current_score
                            best_adj = list(adj)
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break

        if not improved:
            no_improve_count += 1
            # If stuck, try a random perturbation (kick)
            if no_improve_count % 3 == 0:
                # Random kick: do a random swap
                all_edges = []
                all_non_edges = []
                for i in range(N):
                    for j in range(i+1, N):
                        if adj[i] & (1 << j):
                            all_edges.append((i, j))
                        else:
                            all_non_edges.append((i, j))
                if all_edges and all_non_edges:
                    e1 = rng.choice(all_edges)
                    e2 = rng.choice(all_non_edges)
                    adj[e1[0]] ^= (1 << e1[1])
                    adj[e1[1]] ^= (1 << e1[0])
                    adj[e2[0]] ^= (1 << e2[1])
                    adj[e2[1]] ^= (1 << e2[0])
                    current_score = get_score(adj)
                    # Don't update best here; let the search find improvements

    # Return the best found
    adj = best_adj
    edges = []
    for i in range(N):
        for j in range(i+1, N):
            if adj[i] & (1 << j):
                edges.append([i, j])

    # Verify we have exactly 65 edges
    assert len(edges) == 65, f"Expected 65 edges, got {len(edges)}"

    return {'n': 16, 'edges': edges}
