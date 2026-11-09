import time
import random

def construct():
    n = 15
    start = time.time()
    deadline = start + 85.0  # Safety margin for 90s limit

    # Start with K_{7,8}: vertices 0-6 in part A, 7-14 in part B
    # Part A: {0,1,2,3,4,5,6}, Part B: {7,8,9,10,11,12,13,14}
    part_a = list(range(7))
    part_b = list(range(7, 15))

    # Build initial edge set: all cross edges
    edges = set()
    for u in part_a:
        for v in part_b:
            edges.add((u, v) if u < v else (v, u))

    # Adjacency as bitmasks
    adj = [0] * n
    deg = [0] * n
    for (u, v) in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        deg[u] += 1
        deg[v] += 1

    def has_violation_fast(adj, deg):
        """Check if any two same-degree vertices have a 3-path between them.
        s-u-v-t: s~u, u~v, v~t, all distinct.
        Equivalent: exists u,v such that s~u, u~v, v~t, u!=v, u!=s,t, v!=s,t.
        Faster: for each pair (s,t) with same degree, check if there's a length-3 path.
        A length-3 path from s to t exists iff:
          there exists u in N(s) and v in N(t) such that u~v, u!=v, u!=t, v!=s
        Using bitmasks: for each u in N(s), check if (adj[u] & N(t) & ~(1<<s) & ~(1<<t) & ~(1<<u)) != 0
        But we also need v != u and v != s and v != t.
        """
        # Group vertices by degree
        deg_map = {}
        for i in range(n):
            deg_map.setdefault(deg[i], []).append(i)

        for d, verts in deg_map.items():
            if len(verts) < 2:
                continue
            for i in range(len(verts)):
                for j in range(i + 1, len(verts)):
                    s, t = verts[i], verts[j]
                    # Check if 3-path s-u-v-t exists
                    # For each u in N(s):
                    ns = adj[s]
                    nt = adj[t]
                    # u must be in N(s), u != t
                    # v must be in N(u) & N(t), v != s, v != t, v != u
                    # So: for u in N(s) \ {t}, check if adj[u] & N(t) & ~(1<<s) & ~(1<<t) & ~(1<<u) != 0
                    candidates_u = ns & ~(1 << t)
                    while candidates_u:
                        u_bit = candidates_u & (-candidates_u)
                        u = u_bit.bit_length() - 1
                        candidates_u ^= u_bit
                        # v in N(u) & N(t), v not in {s, t, u}
                        mask = adj[u] & nt & ~(1 << s) & ~(1 << t) & ~(1 << u)
                        if mask:
                            return True
        return False

    def try_add_edge(u, v, adj, deg, edges):
        """Try adding edge (u,v). Returns True if valid."""
        if u == v or (u, v) in edges or (v, u) in edges:
            return False
        # Temporarily add
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        deg[u] += 1
        deg[v] += 1
        valid = not has_violation_fast(adj, deg)
        # Revert
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        deg[u] -= 1
        deg[v] -= 1
        if valid:
            edges.add((u, v) if u < v else (v, u))
        return valid

    # Intra-part edges to try
    intra_edges_a = [(i, j) for i in range(len(part_a)) for j in range(i+1, len(part_a))]
    intra_edges_b = [(i, j) for i in range(len(part_b)) for j in range(i+1, len(part_b))]
    intra_edges = [(part_a[i], part_a[j]) for (i, j) in intra_edges_a] + \
                  [(part_b[i], part_b[j]) for (i, j) in intra_edges_b]

    # Greedy: try adding intra-part edges in random order, multiple restarts
    best_edges = set(edges)
    best_count = len(edges)

    for attempt in range(200):
        if time.time() > deadline - 15:
            break
        current_edges = set(best_edges)
        # Reset adj and deg from current_edges
        cur_adj = [0] * n
        cur_deg = [0] * n
        for (u, v) in current_edges:
            cur_adj[u] |= (1 << v)
            cur_adj[v] |= (1 << u)
            cur_deg[u] += 1
            cur_deg[v] += 1

        shuffled = intra_edges[:]
        random.shuffle(shuffled)
        for (u, v) in shuffled:
            if time.time() > deadline - 10:
                break
            key = (u, v) if u < v else (v, u)
            if key in current_edges:
                continue
            # Try adding
            cur_adj[u] |= (1 << v)
            cur_adj[v] |= (1 << u)
            cur_deg[u] += 1
            cur_deg[v] += 1
            if not has_violation_fast(cur_adj, cur_deg):
                current_edges.add(key)
            else:
                cur_adj[u] &= ~(1 << v)
                cur_adj[v] &= ~(1 << u)
                cur_deg[u] -= 1
                cur_deg[v] -= 1

        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_edges = current_edges

    # Local search: try removing one intra-part edge and adding others
    for ls_attempt in range(100):
        if time.time() > deadline - 5:
            break
        current_edges = set(best_edges)
        cur_adj = [0] * n
        cur_deg = [0] * n
        for (u, v) in current_edges:
            cur_adj[u] |= (1 << v)
            cur_adj[v] |= (1 << u)
            cur_deg[u] += 1
            cur_deg[v] += 1

        improved = False
        # Try removing each intra-part edge that's currently in the graph
        intra_in_graph = [(u, v) for (u, v) in current_edges if (u < 7 and v < 7) or (u >= 7 and v >= 7)]
        random.shuffle(intra_in_graph)
        for (ru, rv) in intra_in_graph:
            if time.time() > deadline - 3:
                break
            key = (ru, rv) if ru < rv else (rv, ru)
            if key not in current_edges:
                continue
            # Remove
            current_edges.discard(key)
            cur_adj[ru] &= ~(1 << rv)
            cur_adj[rv] &= ~(1 << ru)
            cur_deg[ru] -= 1
            cur_deg[rv] -= 1

            # Try to add other intra-part edges
            shuffled = intra_edges[:]
            random.shuffle(shuffled)
            for (u, v) in shuffled:
                k2 = (u, v) if u < v else (v, u)
                if k2 in current_edges:
                    continue
                cur_adj[u] |= (1 << v)
                cur_adj[v] |= (1 << u)
                cur_deg[u] += 1
                cur_deg[v] += 1
                if not has_violation_fast(cur_adj, cur_deg):
                    current_edges.add(k2)
                else:
                    cur_adj[u] &= ~(1 << v)
                    cur_adj[v] &= ~(1 << u)
                    cur_deg[u] -= 1
                    cur_deg[v] -= 1

            if len(current_edges) > len(best_edges):
                best_count = len(current_edges)
                best_edges = current_edges
                improved = True
                break

        if not improved and time.time() > deadline - 10:
            break

    out = [[u, v] for (u, v) in best_edges]
    return out
