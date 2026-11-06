"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools

# EVOLVE-BLOCK-START
def construct():
    n = 18
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin before 120s limit

    # --- Bitmask helpers ---
    def set_edge(adj, u, v, val):
        if val:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        else:
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)

    def can_add_edge(adj, u, v):
        """Check if adding edge (u,v) keeps graph K4-free."""
        c_mask = adj[u] & adj[v]
        c = c_mask
        while c:
            lsb = c & -c
            x = lsb.bit_length() - 1
            if adj[x] & c_mask:
                return False
            c ^= lsb
        return True

    def get_triangles(adj):
        """Return list of triangle masks (bitmask of 3 vertices)."""
        tris = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u] & (1 << v):
                    w_mask = adj[u] & adj[v]
                    # Only consider w > v to avoid duplicates
                    w_mask &= ~((1 << (v + 1)) - 1)
                    while w_mask:
                        lsb = w_mask & -w_mask
                        w = lsb.bit_length() - 1
                        tris.append((1 << u) | (1 << v) | (1 << w))
                        w_mask ^= lsb
        return tris

    def max_triangle_free_size(adj):
        """
        Find the size of the largest triangle-free induced subgraph.
        Uses branch and bound with triangle masks.
        """
        tri_masks = get_triangles(adj)
        if not tri_masks:
            return n

        # Precompute for each vertex, which triangles it belongs to
        # For efficiency, we'll just check subsets
        
        best = 1
        # Try sizes from n down to best+1
        for k in range(n, best, -1):
            found = False
            # Use combinations - for n=18, C(18,12) is manageable
            for subset in itertools.combinations(range(n), k):
                mask = 0
                for v in subset:
                    mask |= (1 << v)
                # Check if this subset contains any triangle
                has_tri = False
                for tm in tri_masks:
                    if (mask & tm) == tm:
                        has_tri = True
                        break
                if not has_tri:
                    best = k
                    found = True
                    break  # Found one of size k, no need to check others of same size
            if found:
                break
        return best

    # --- Initial construction: K6,6,6 ---
    def initial_graph():
        adj = [0] * n
        parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
        for i in range(3):
            for j in range(i + 1, 3):
                for a in parts[i]:
                    for b in parts[j]:
                        set_edge(adj, a, b, True)
        return adj

    best_adj = initial_graph()
    best_score = max_triangle_free_size(best_adj)

    # --- Local search ---
    current_adj = [x for x in best_adj]
    current_score = best_score

    # Precompute all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]

    while time.time() < deadline:
        # Try adding a random non-edge
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        if u == v:
            continue
        if u > v:
            u, v = v, u
        if not (current_adj[u] & (1 << v)):
            if can_add_edge(current_adj, u, v):
                # Temporarily add
                old_u = current_adj[u]
                old_v = current_adj[v]
                current_adj[u] = old_u | (1 << v)
                current_adj[v] = old_v | (1 << u)
                new_score = max_triangle_free_size(current_adj)
                if new_score < best_score:
                    best_score = new_score
                    best_adj = [x for x in current_adj]
                else:
                    # Revert
                    current_adj[u] = old_u
                    current_adj[v] = old_v

        # Occasionally try removing an edge
        if random.random() < 0.15:
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            if u > v:
                u, v = v, u
            if current_adj[u] & (1 << v):
                old_u = current_adj[u]
                old_v = current_adj[v]
                current_adj[u] = old_u & ~(1 << v)
                current_adj[v] = old_v & ~(1 << u)
                new_score = max_triangle_free_size(current_adj)
                if new_score < best_score:
                    best_score = new_score
                    best_adj = [x for x in current_adj]
                else:
                    # Revert
                    current_adj[u] = old_u
                    current_adj[v] = old_v

    # Reconstruct edges from best_adj
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_adj[u] & (1 << v):
                edges.append([u, v])
    return edges

# EVOLVE-BLOCK-END
