import time
import random

def construct():
    start = time.time()
    DEADLINE = start + 145.0
    N = 16
    FULL = (1 << N) - 1
    adj = [0] * N
    edge_count = 0

    def time_ok():
        return time.time() < DEADLINE

    def is_3_colorable(adj, edge_count):
        # Fast exact 3-colorability test using bitmask backtracking
        if edge_count <= 2 * N - 3:
            return True
        
        # Order vertices by degree descending for better pruning
        deg = [bin(adj[i]).count("1") for i in range(N)]
        order = list(range(N))
        order.sort(key=lambda i: -deg[i])
        
        # Build adjacency masks in the new ordering
        pos = [0] * N
        for i, v in enumerate(order):
            pos[v] = i
        
        adj_mask = [0] * N
        for i in range(N):
            v = order[i]
            m = 0
            for j in range(i + 1, N):
                u = order[j]
                if (adj[v] >> u) & 1:
                    m |= (1 << j)
            adj_mask[i] = m
            # Also set the reverse direction for completeness
            for j in range(i):
                u = order[j]
                if (adj[v] >> u) & 1:
                    adj_mask[j] |= (1 << i)
        
        # Backtracking with bitmasks
        # colors[c] is a bitmask of vertices assigned color c
        def bt(i, c0, c1, c2):
            if i == N:
                return True
            
            mask = adj_mask[i]
            # Try each color
            if (c0 & mask) == 0:
                if bt(i + 1, c0 | (1 << i), c1, c2):
                    return True
            if (c1 & mask) == 0:
                if bt(i + 1, c0, c1 | (1 << i), c2):
                    return True
            if (c2 & mask) == 0:
                if bt(i + 1, c0, c1, c2 | (1 << i)):
                    return True
            return False
        
        return bt(0, 0, 0, 0)

    def add_edge(u, v):
        nonlocal edge_count
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        edge_count += 1

    def remove_edge(u, v):
        nonlocal edge_count
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        edge_count -= 1

    best_edges = None
    best_count = -1

    rng = random.Random(12345)
    attempts = 0
    while time_ok() and attempts < 5000:
        attempts += 1
        # Randomize edge order
        pairs = [(u, v) for u in range(N) for v in range(u + 1, N)]
        rng.shuffle(pairs)
        adj = [0] * N
        edge_count = 0
        for u, v in pairs:
            if not ((adj[u] >> v) & 1):
                if (adj[u] & adj[v]) == 0:
                    add_edge(u, v)
        
        if is_3_colorable(adj, edge_count):
            continue
        
        if edge_count > best_count:
            best_count = edge_count
            best_edges = [[u, v] for u in range(N) for v in range(u + 1, N) if (adj[u] >> v) & 1]
        
        # Local search: try adding edges one by one
        changed = True
        while changed and time_ok():
            changed = False
            for u in range(N):
                if not time_ok():
                    break
                for v in range(u + 1, N):
                    if not ((adj[u] >> v) & 1):
                        if (adj[u] & adj[v]) == 0:
                            add_edge(u, v)
                            if not is_3_colorable(adj, edge_count):
                                changed = True
                            else:
                                remove_edge(u, v)
        
        if best_edges and edge_count > best_count:
            best_count = edge_count
            best_edges = [[u, v] for u in range(N) for v in range(u + 1, N) if (adj[u] >> v) & 1]

    if best_edges is not None:
        return best_edges

    # Fallback: return Clebsch graph (40 edges)
    vertices = []
    for i in range(32):
        if bin(i).count('1') % 2 == 0:
            vertices.append(i)
    edges = []
    n = len(vertices)
    for u_idx in range(n):
        for v_idx in range(u_idx + 1, n):
            xor_val = vertices[u_idx] ^ vertices[v_idx]
            if bin(xor_val).count('1') == 4:
                edges.append([u_idx, v_idx])
    return edges
