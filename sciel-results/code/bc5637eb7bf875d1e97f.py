import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    def check_c4_safe(adj, u, v):
        """Check if adding edge (u,v) creates a C4 using bitmasks."""
        mask_u = adj[u]
        mask_v = adj[v]
        # For each neighbor a of u, check if N(a) intersects N(v)
        m = mask_u
        while m:
            lsb = m & (-m)
            a = lsb.bit_length() - 1
            m ^= lsb
            if adj[a] & mask_v:
                return False
        return True

    def verify_graph(adj, n, min_deg_req):
        """Verify C4-free and min degree."""
        if min(d for d in [bin(adj[i]).count('1') for i in range(n)]) < min_deg_req:
            return False
        for i in range(n):
            for j in range(i + 1, n):
                if (adj[i] >> j) & 1:
                    common = adj[i] & adj[j]
                    if common & (common - 1):  # more than 1 common neighbor
                        return False
        return True

    def extract_edges(adj, n):
        edges = []
        for i in range(n):
            m = adj[i]
            while m:
                lsb = m & (-m)
                j = lsb.bit_length() - 1
                if i < j:
                    edges.append([i, j])
                m ^= lsb
        return edges

    # Try from high n down
    for n in [47, 46, 45, 44, 43, 42, 41, 40]:
        if time.time() > deadline:
            break

        min_deg_req = n - 39
        if min_deg_req <= 0:
            return {'n': n, 'edges': []}

        found = False
        best_edges = []

        # Multiple strategies with different random seeds
        for strategy in range(50):
            if time.time() > deadline:
                break

            random.seed(strategy * 7919 + n)  # Deterministic per strategy

            adj = [0] * n
            deg = [0] * n

            # Strategy: greedy with random tie-breaking and occasional repair
            stuck = False
            max_steps = 200  # Max edges to add
            steps = 0

            while steps < max_steps:
                # Find vertices below min degree
                low_verts = [i for i in range(n) if deg[i] < min_deg_req]
                if not low_verts:
                    break

                # Pick a low-degree vertex
                u = random.choice(low_verts)

                # Find safe neighbors
                safe_vs = []
                for v in range(n):
                    if v == u or (adj[u] >> v) & 1:
                        continue
                    if check_c4_safe(adj, u, v):
                        safe_vs.append(v)

                if not safe_vs:
                    # Try repair: remove a random edge incident to u to free up options
                    if deg[u] > 0:
                        # Find a neighbor to remove
                        m = adj[u]
                        neighbors = []
                        while m:
                            lsb = m & (-m)
                            v = lsb.bit_length() - 1
                            m ^= lsb
                            neighbors.append(v)
                        if neighbors:
                            v_remove = random.choice(neighbors)
                            # Remove edge (u, v_remove)
                            adj[u] &= ~(1 << v_remove)
                            adj[v_remove] &= ~(1 << u)
                            deg[u] -= 1
                            deg[v_remove] -= 1
                            steps += 1
                            continue
                    stuck = True
                    break

                # Pick v: prefer low degree to balance
                if random.random() < 0.7:
                    v = min(safe_vs, key=lambda x: deg[x])
                else:
                    v = random.choice(safe_vs)

                # Add edge
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                deg[u] += 1
                deg[v] += 1
                steps += 1

            if not stuck and verify_graph(adj, n, min_deg_req):
                found = True
                best_edges = extract_edges(adj, n)
                break

        if found:
            return {'n': n, 'edges': best_edges}

    # Fallback: return a valid small graph
    # n=39, min_deg_req=0, empty graph
    return {'n': 39, 'edges': []}
