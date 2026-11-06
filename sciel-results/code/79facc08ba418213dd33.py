"""Q3-free graph on 16 vertices: maximize edge count via fast subgraph check + greedy + local search."""
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    n = 16
    # Q3 edges (12 edges on 8 vertices)
    q3_edge_pairs = [(0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7)]
    # Q3 adjacency
    q3_adj = {
        0: [1, 2, 4],
        1: [0, 3, 5],
        2: [0, 3, 6],
        3: [1, 2, 7],
        4: [0, 5, 6],
        5: [1, 4, 7],
        6: [2, 4, 7],
        7: [3, 5, 6]
    }

    # Precompute all 8-subsets of {0..15}
    all_8sets = list(combinations(range(16), 8))

    # Precompute edge index for fast lookup: use a set of frozensets or tuple
    # We'll maintain edge_set as a set of (min, max) tuples

    def check_q3_in_subset(vs, edge_set):
        """Check if the induced subgraph on 'vs' contains Q3.
        vs is a tuple of 8 vertices, edge_set is a set of (min,max) tuples."""
        # Build local adjacency: map vertex to its neighbors within vs
        # First, quick degree check
        for i in range(8):
            u = vs[i]
            deg = 0
            for j in range(8):
                if i == j:
                    continue
                v = vs[j]
                if (min(u,v), max(u,v)) in edge_set:
                    deg += 1
            if deg < 3:
                return False

        # Now do backtracking to find Q3 isomorphism
        # Try all 8! / 2 = 20160 assignments? No, use smarter approach.
        # Assign Q3 vertices 0..7 to vs[0..7] via backtracking.
        assignment = [-1] * 8  # assignment[q3_v] = index in vs
        vs_used = [False] * 8

        def backtrack(q3_v):
            if q3_v == 8:
                return True
            for vs_i in range(8):
                if vs_used[vs_i]:
                    continue
                # Check all neighbors of q3_v that are already assigned
                ok = True
                for nb in q3_adj[q3_v]:
                    if nb < q3_v:
                        # nb must be assigned
                        nb_vs_i = assignment[nb]
                        u = vs[vs_i]
                        w = vs[nb_vs_i]
                        if (min(u,w), max(u,w)) not in edge_set:
                            ok = False
                            break
                if not ok:
                    continue
                # Also check non-neighbor constraints? For Q3, we need non-adjacency too.
                # Actually for subgraph isomorphism (not induced), we only need edges to exist.
                # Wait, Q3 has 12 edges. If we map Q3 to vs, all 12 edges of Q3 must be present.
                # Non-edges in Q3 don't need to be non-edges in the graph.
                # So we only check that edges of Q3 map to edges in the graph.
                assignment[q3_v] = vs_i
                vs_used[vs_i] = True
                if backtrack(q3_v + 1):
                    return True
                vs_used[vs_i] = False
                assignment[q3_v] = -1
            return False

        return backtrack(0)

    def is_q3_free(edge_set):
        """Check if graph with edges in edge_set (set of (min,max)) on 16 vertices is Q3-free."""
        for vs in all_8sets:
            if check_q3_in_subset(vs, edge_set):
                return False
        return True

    # Baseline: K_{3,13} - partition into {0,1,2} and {3..15}, all cross edges
    k313 = set()
    for a in range(3):
        for b in range(3, 16):
            k313.add((min(a,b), max(a,b)))
    best_edges = k313
    best_count = len(k313)

    # Greedy edge addition
    current_edges = set(k313)
    # All possible edges not in current
    all_edges = [(i,j) for i in range(16) for j in range(i+1, 16)]
    remaining_edges = [e for e in all_edges if e not in current_edges]

    # Prioritize edges that are "safe" - try edges within the large part first?
    # Actually, let's try to add edges in order of how likely they are to not create Q3.
    # Edges within {0,1,2} (the small part) are safer because Q3 needs 8 vertices.
    # Let's try all remaining edges in a smart order.

    def edge_safety(e):
        u, v = e
        # Count how many edges u and v already have
        u_deg = sum(1 for (a,b) in current_edges if a == u or b == u)
        v_deg = sum(1 for (a,b) in current_edges if a == v or b == v)
        # Edges between high-degree vertices are more dangerous
        return -(u_deg + v_deg)

    remaining_edges.sort(key=edge_safety)

    for edge in remaining_edges:
        if time.time() > deadline - 30:
            break
        u, v = edge
        # Check if adding this edge creates a Q3
        # Only need to check 8-sets containing both u and v
        creates_q3 = False
        # Pick 6 other vertices from the remaining 14
        other_vertices = [x for x in range(16) if x != u and x != v]
        for other6 in combinations(other_vertices, 6):
            if time.time() > deadline - 28:
                break
            vs = (u, v) + tuple(other6)
            # Quick check: in this 8-set, u and v must each have degree >= 3 within the set
            # after adding the new edge
            u_deg_in_set = 1  # for the new edge to v
            v_deg_in_set = 1
            for x in other6:
                if (min(u,x), max(u,x)) in current_edges:
                    u_deg_in_set += 1
                if (min(v,x), max(v,x)) in current_edges:
                    v_deg_in_set += 1
            if u_deg_in_set < 3 or v_deg_in_set < 3:
                continue
            # Full check
            test_edges = current_edges | {edge}
            if check_q3_in_subset(vs, test_edges):
                creates_q3 = True
                break

        if not creates_q3:
            current_edges.add(edge)

    if len(current_edges) > best_count:
        best_count = len(current_edges)
        best_edges = set(current_edges)

    # Local search: try to swap edges to improve
    # Try random restarts with different initial graphs
    random.seed(42)

    for trial in range(200):
        if time.time() > deadline - 5:
            break

        # Start from current best or random dense graph
        if trial % 3 == 0:
            # Start from best
            cur = set(best_edges)
        else:
            # Random dense graph
            p = random.uniform(0.6, 0.8)
            cur = set()
            for i in range(16):
                for j in range(i+1, 16):
                    if random.random() < p:
                        cur.add((i,j))

        # If starting from best, do local search by trying to add edges
        improved = True
        while improved:
            if time.time() > deadline - 5:
                break
            improved = False
            # Try adding each missing edge
            for i in range(16):
                if time.time() > deadline - 5:
                    break
                for j in range(i+1, 16):
                    e = (i,j)
                    if e in cur:
                        continue
                    # Check if adding e creates Q3
                    u, v = i, j
                    other_vertices = [x for x in range(16) if x != u and x != v]
                    creates_q3 = False
                    for other6 in combinations(other_vertices, 6):
                        if time.time() > deadline - 5:
                            break
                        vs = (u, v) + tuple(other6)
                        u_deg_in_set = 1
                        v_deg_in_set = 1
                        for x in other6:
                            if (min(u,x), max(u,x)) in cur:
                                u_deg_in_set += 1
                            if (min(v,x), max(v,x)) in cur:
                                v_deg_in_set += 1
                        if u_deg_in_set < 3 or v_deg_in_set < 3:
                            continue
                        test_edges = cur | {e}
                        if check_q3_in_subset(vs, test_edges):
                            creates_q3 = True
                            break
                    if not creates_q3:
                        cur.add(e)
                        improved = True

        if len(cur) > best_count:
            best_count = len(cur)
            best_edges = set(cur)

    # Final verification
    # result = [[u, v] for (u, v) in sorted(best_edges)]
    return [[u, v] for (u, v) in best_edges]
