import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    n_target = 16

    # 1. Construct Mycielski of Grötzsch (21 vertices)
    # Grötzsch graph (11 vertices)
    groetzsch_edges = [
        (0,1), (1,2), (2,3), (3,4), (4,0),
        (5,1), (5,4),
        (6,2), (6,0),
        (7,3), (7,1),
        (8,4), (8,2),
        (9,0), (9,3),
        (10,5), (10,6), (10,7), (10,8), (10,9)
    ]
    n_g = 11

    # Mycielski construction:
    # Vertices: V, V', v*
    # Edges: E, plus for each (u,v) in E, add (u, v') and (v, u').
    # v* is connected to all v' in V'(G).
    myc_g_edges = list(groetzsch_edges)
    n_myc = 23 # 11 original + 11 copies + 1 root
    # Add edges for copies
    for u, v in groetzsch_edges:
        myc_g_edges.append((u, v + 11))
        myc_g_edges.append((v, u + 11))
    # Add root edges
    root = 22
    for i in range(11):
        myc_g_edges.append((root, 11 + i))

    # Build adjacency for Myc graph (as sets for easy induced subgraph extraction)
    adj_myc = [set() for _ in range(n_myc)]
    for u, v in myc_g_edges:
        adj_myc[u].add(v)
        adj_myc[v].add(u)

    def is_3_colorable(adj, n):
        """Check if graph is 3-colorable using backtracking with bitmask color tracking."""
        # Precompute neighbor bitmasks
        neighbor_mask = [0] * n
        for u in range(n):
            for v in adj[u]:
                neighbor_mask[u] |= (1 << v)

        color = [-1] * n
        # Order vertices by degree (descending)
        order = sorted(range(n), key=lambda x: -len(adj[x]))

        def backtrack(idx):
            if idx == n:
                return True
            v = order[idx]
            forbidden = 0
            for u in adj[v]:
                cu = color[u]
                if cu != -1:
                    forbidden |= (1 << cu)

            for c in range(3):
                if not (forbidden & (1 << c)):
                    color[v] = c
                    if backtrack(idx + 1):
                        return True
                    color[v] = -1
            return False

        return backtrack(0)

    def build_adj_from_edges(edges, n):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def edge_count(adj):
        return sum(len(adj[i]) for i in range(len(adj))) // 2

    # Step 1: Find best induced subgraph of size 16 from Myc graph (23 vertices)
    # We need to remove 7 vertices.
    # C(23, 7) = 245157. This is feasible.
    best_edges_count = -1
    best_adj = None
    best_edges_list = []

    remove_count = n_myc - n_target
    vertices_to_remove = combinations(range(n_myc), remove_count)

    count = 0
    for rem in vertices_to_remove:
        if time.time() > deadline - 30:
            break
        rem_set = set(rem)
        # Build induced subgraph on remaining vertices
        remaining = [v for v in range(n_myc) if v not in rem_set]

        # Quick check: count edges
        edges_sub = []
        for u, v in myc_g_edges:
            if u not in rem_set and v not in rem_set:
                edges_sub.append((u, v))

        # If edge count is less than best, skip 3-color check
        if len(edges_sub) <= best_edges_count:
            continue

        # Check 3-colorability
        index_map = {v: i for i, v in enumerate(remaining)}
        adj_sub = [set() for _ in range(n_target)]
        for u, v in edges_sub:
            iu = index_map[u]
            iv = index_map[v]
            adj_sub[iu].add(iv)
            adj_sub[iv].add(iu)

        if not is_3_colorable(adj_sub, n_target):
            if len(edges_sub) > best_edges_count:
                best_edges_count = len(edges_sub)
                best_adj = adj_sub
                best_edges_list = [[index_map[u], index_map[v]] for u, v in edges_sub]

    # If we didn't find any, use a fallback (e.g., Grötzsch + isolated vertices)
    if best_adj is None:
        adj_fallback = [set() for _ in range(n_target)]
        for u, v in groetzsch_edges:
            adj_fallback[u].add(v)
            adj_fallback[v].add(u)
        best_adj = adj_fallback
        best_edges_count = edge_count(adj_fallback)
        best_edges_list = [[u, v] for u, v in groetzsch_edges]
    else:
        # Re-extract edges to be safe
        best_edges_list = []
        for u in range(n_target):
            for v in best_adj[u]:
                if u < v:
                    best_edges_list.append([u, v])

    # Step 2: Local Search to improve
    current_adj = [set(s) for s in best_adj]
    current_edges = edge_count(current_adj)

    all_pairs = [(i, j) for i in range(n_target) for j in range(i+1, n_target)]

    def try_add_edge(adj, u, v):
        # Check triangle-free
        if adj[u] & adj[v]:
            return False
        # Add
        adj[u].add(v)
        adj[v].add(u)
        # Check 3-colorable
        if is_3_colorable(adj, n_target):
            adj[u].remove(v)
            adj[v].remove(u)
            return False
        return True

    def local_search(adj, deadline):
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try adding edges
            non_edges = [(u, v) for u, v in all_pairs if v not in adj[u]]
            random.shuffle(non_edges)
            for u, v in non_edges:
                if time.time() >= deadline:
                    break
                if try_add_edge(adj, u, v):
                    improved = True
                    # Restart loop to find more additions
                    break

            if not improved:
                # Try swaps
                edges_list = [(u, v) for u in range(n_target) for v in adj[u] if u < v]
                random.shuffle(edges_list)
                for u, v in edges_list:
                    if time.time() >= deadline:
                        break
                    # Remove (u, v)
                    adj[u].remove(v)
                    adj[v].remove(u)

                    # Try adding a new edge
                    found = False
                    for a, b in all_pairs:
                        if (a, b) == (u, v) or (a, b) == (v, u):
                            continue
                        if b in adj[a]:
                            continue
                        if try_add_edge(adj, a, b):
                            improved = True
                            found = True
                            break
                    if not found:
                        # Restore
                        adj[u].add(v)
                        adj[v].add(u)
                    if found:
                        break

        return adj

    # Run local search with multiple restarts
    for _ in range(5):
        if time.time() > deadline - 10:
            break
        # Perturb current best
        adj_perturb = [set(s) for s in best_adj]
        # Remove 2 random edges
        edges_list = [(u, v) for u in range(n_target) for v in adj_perturb[u] if u < v]
        if len(edges_list) > 2:
            for _ in range(2):
                u, v = random.choice(edges_list)
                adj_perturb[u].remove(v)
                adj_perturb[v].remove(u)
                edges_list.remove((u, v))

        adj_searched = local_search(adj_perturb, deadline - 10)
        if edge_count(adj_searched) > current_edges:
            current_edges = edge_count(adj_searched)
            best_adj = adj_searched

    # Final result
    result_edges = []
    for u in range(n_target):
        for v in best_adj[u]:
            if u < v:
                result_edges.append([u, v])

    return result_edges
