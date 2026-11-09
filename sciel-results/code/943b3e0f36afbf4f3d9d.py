"""Construct a dense C4-rainbow edge-coloured graph on 20 vertices."""
import time
import random

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    deadline = start + 135.0
    n = 20
    num_colours = 20

    def elapsed():
        return time.time() - start

    def time_ok():
        return elapsed() < deadline - 3.0

    def build_greedy(seed):
        rng = random.Random(seed)
        edge_color = {}
        adj = {i: {} for i in range(n)}
        neighbors = {i: [] for i in range(n)}

        def add_edge(u, v, c):
            if u > v:
                u, v = v, u
            edge_color[(u, v)] = c
            adj[u][v] = c
            adj[v][u] = c
            if v not in neighbors[u]:
                neighbors[u].append(v)
            if u not in neighbors[v]:
                neighbors[v].append(u)

        def remove_edge(u, v):
            if u > v:
                u, v = v, u
            del edge_color[(u, v)]
            del adj[u][v]
            del adj[v][u]
            if v in neighbors[u]:
                neighbors[u].remove(v)
            if u in neighbors[v]:
                neighbors[v].remove(u)

        def is_valid_add(u, v, c):
            # Check all 4-cycles u-v-w-x-u
            # w in neighbors[v], x in neighbors[u], edge (w,x) exists
            for w in neighbors[v]:
                if w == u:
                    continue
                cw = adj[v][w]
                for x in neighbors[u]:
                    if x == v or x == w:
                        continue
                    cxu = adj[u][x]
                    if x in adj[w]:
                        cwx = adj[w][x]
                        if c == cw or c == cwx or c == cxu or cw == cwx or cw == cxu or cwx == cxu:
                            return False
            return True

        # Generate edge order with some structure
        all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        
        # Mix different ordering strategies based on seed
        if seed % 3 == 0:
            rng.shuffle(all_pairs)
        elif seed % 3 == 1:
            # Sort by sum of indices (prefer central edges first)
            all_pairs.sort(key=lambda p: (abs(p[0]-p[1]), p[0], p[1]))
            rng.shuffle(all_pairs)
        else:
            # Random but with slight preference for low-degree vertices
            all_pairs.sort(key=lambda p: rng.random())
            rng.shuffle(all_pairs)

        for (u, v) in all_pairs:
            if not time_ok():
                break
            cols = list(range(num_colours))
            rng.shuffle(cols)
            for c in cols:
                if is_valid_add(u, v, c):
                    add_edge(u, v, c)
                    break

        return edge_color, adj, neighbors

    def local_search(edge_color, adj, neighbors, rng):
        """Try to add back edges that were skipped."""
        all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        skipped = [(u, v) for (u, v) in all_pairs if (u, v) not in edge_color]
        rng.shuffle(skipped)
        
        improved = True
        while improved and time_ok():
            improved = False
            for (u, v) in skipped:
                if not time_ok():
                    break
                cols = list(range(num_colours))
                rng.shuffle(cols)
                for c in cols:
                    if is_valid_add_local(u, v, c, adj, neighbors):
                        edge_color[(u, v)] = c
                        adj[u][v] = c
                        adj[v][u] = c
                        if v not in neighbors[u]:
                            neighbors[u].append(v)
                        if u not in neighbors[v]:
                            neighbors[v].append(u)
                        skipped.remove((u, v))
                        improved = True
                        break
        return edge_color, adj, neighbors

    def is_valid_add_local(u, v, c, adj, neighbors):
        for w in neighbors[v]:
            if w == u:
                continue
            cw = adj[v][w]
            for x in neighbors[u]:
                if x == v or x == w:
                    continue
                cxu = adj[u][x]
                if x in adj[w]:
                    cwx = adj[w][x]
                    if c == cw or c == cwx or c == cxu or cw == cwx or cw == cxu or cwx == cxu:
                        return False
        return True

    def verify(edge_color):
        """Verify all 4-cycles are rainbow."""
        adj = {i: {} for i in range(n)}
        for (u, v), c in edge_color.items():
            adj[u][v] = c
            adj[v][u] = c
        for u in range(n):
            for v in adj[u]:
                if v <= u:
                    continue
                for w in adj[v]:
                    if w == u:
                        continue
                    for x in adj[u]:
                        if x == v or x == w:
                            continue
                        if x in adj[w]:
                            c1 = adj[u][v]
                            c2 = adj[v][w]
                            c3 = adj[w][x]
                            c4 = adj[x][u]
                            if len({c1, c2, c3, c4}) != 4:
                                return False
        return True

    best = None
    best_count = -1
    
    for seed in range(5000):
        if not time_ok():
            break
        ec, adj, neighbors = build_greedy(seed)
        cnt = len(ec)
        
        # Local search to add back skipped edges
        rng_ls = random.Random(seed + 10000)
        ec, adj, neighbors = local_search(ec, adj, neighbors, rng_ls)
        cnt = len(ec)
        
        if cnt > best_count:
            # Verify before accepting
            if verify(ec):
                best_count = cnt
                best = ec
            else:
                # Fall back to pre-local-search version
                ec2, _, _ = build_greedy(seed)
                if len(ec2) > best_count and verify(ec2):
                    best_count = len(ec2)
                    best = ec2
        if best_count >= 150:
            break

    if best is None:
        return [[0, v, 0] for v in range(1, n)]

    return [[u, v, c] for (u, v), c in best.items()]
# EVOLVE-BLOCK-END
