import time
import itertools
import random
import numpy as np

def construct():
    start_time = time.time()
    n = 16
    DEADLINE = 115

    # Q3: vertices 0..7, edge if Hamming distance 1
    q3_adj = [
        {1, 2, 4},
        {0, 3, 5},
        {0, 3, 6},
        {1, 2, 7},
        {0, 5, 7},
        {1, 4, 6},
        {2, 5, 7},
        {3, 4, 6}
    ]

    adj = np.zeros((n, n), dtype=bool)

    def add_edge(u, v):
        adj[u, v] = True
        adj[v, u] = True

    def remove_edge(u, v):
        adj[u, v] = False
        adj[v, u] = False

    def contains_q3(verts):
        """Check if induced subgraph on 'verts' (8 vertices) contains Q3."""
        # Build local adjacency
        la = np.zeros((8, 8), dtype=bool)
        for i in range(8):
            for j in range(i + 1, 8):
                if adj[verts[i], verts[j]]:
                    la[i, j] = True
                    la[j, i] = True
        if la.sum() < 24:
            return False

        # Backtracking with symmetry breaking (fix Q3 vertex 0 to local vertex 0)
        used = [False] * 8
        mapping = [-1] * 8

        def bt(q3_v):
            if q3_v == 8:
                return True
            for ov in range(8):
                if used[ov]:
                    continue
                ok = True
                for qu in range(q3_v):
                    if qu in q3_adj[q3_v] and not la[mapping[q3_v]][mapping[qu]]:
                        ok = False
                        break
                if not ok:
                    continue
                mapping[q3_v] = ov
                used[ov] = True
                if bt(q3_v + 1):
                    return True
                used[ov] = False
                mapping[q3_v] = -1
            return False

        mapping[0] = 0
        used[0] = True
        res = bt(1)
        used[0] = False
        mapping[0] = -1
        return res

    def find_q3():
        for combo in itertools.combinations(range(n), 8):
            if contains_q3(combo):
                return combo
        return None

    def is_q3_free():
        return find_q3() is None

    # Start with a dense random graph
    rng = random.Random(777)
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    rng.shuffle(all_pairs)

    # Add 55 edges
    for u, v in all_pairs[:55]:
        add_edge(u, v)

    # Remove edges to destroy all Q3s
    for _ in range(300):
        if time.time() - start_time > DEADLINE - 15:
            break
        q3 = find_q3()
        if q3 is None:
            break
        edges_in_q3 = [(q3[i], q3[j]) for i in range(8) for j in range(i + 1, 8) if adj[q3[i], q3[j]]]
        if edges_in_q3:
            u, v = rng.choice(edges_in_q3)
            remove_edge(u, v)

    # Greedily add edges that don't create Q3
    non_edges = [(i, j) for i in range(n) for j in range(i + 1, n) if not adj[i, j]]
    rng.shuffle(non_edges)

    for u, v in non_edges:
        if time.time() - start_time > DEADLINE:
            break
        if adj[u, v]:
            continue
        add_edge(u, v)
        other = [i for i in range(n) if i != u and i != v]
        bad = False
        for combo in itertools.combinations(other, 6):
            verts = tuple(sorted([u, v] + list(combo)))
            if contains_q3(verts):
                bad = True
                break
        if bad:
            remove_edge(u, v)

    # Final cleanup: remove any remaining Q3s
    for _ in range(100):
        if time.time() - start_time > DEADLINE:
            break
        q3 = find_q3()
        if q3 is None:
            break
        edges_in_q3 = [(q3[i], q3[j]) for i in range(8) for j in range(i + 1, 8) if adj[q3[i], q3[j]]]
        if edges_in_q3:
            u, v = edges_in_q3[0]
            remove_edge(u, v)

    result = [[i, j] for i in range(n) for j in range(i + 1, n) if adj[i, j]]
    return result
