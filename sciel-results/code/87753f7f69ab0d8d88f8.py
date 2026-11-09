import time
import random
from collections import defaultdict

def construct():
    n = 15
    start = time.time()
    deadline = start + 50.0  # Safety margin for 60s limit

    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    num_total = len(all_edges)

    def build_adj_deg(edge_set):
        adj = [0] * n
        deg = [0] * n
        for (u, v) in edge_set:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            deg[u] += 1
            deg[v] += 1
        return adj, deg

    def check_violation_for_add(adj, deg, u, v):
        """
        Check if adding edge (u, v) creates a violation.
        adj/deg represent the graph BEFORE adding (u, v).
        """
        neighbors_u = adj[u]
        neighbors_v = adj[v]

        # Case 1: (u, v) is the middle edge. Path s-u-v-t.
        # s in N(u), t in N(v), s != t, deg_new(s) == deg_new(t).
        for s in range(n):
            if not (neighbors_u & (1 << s)):
                continue
            deg_s_new = deg[s] + (1 if s == u or s == v else 0)
            for t in range(n):
                if t == s:
                    continue
                if not (neighbors_v & (1 << t)):
                    continue
                deg_t_new = deg[t] + (1 if t == u or t == v else 0)
                if deg_s_new == deg_t_new:
                    return True

        # Case 2: (u, v) is the first edge. Path u-v-b-t.
        # b in N(v), t in N(b), t not in {u,v}, deg_new(u) == deg_new(t).
        deg_u_new = deg[u] + 1
        for b in range(n):
            if not (neighbors_v & (1 << b)):
                continue
            neighbors_b = adj[b]
            for t in range(n):
                if not (neighbors_b & (1 << t)):
                    continue
                if t == u or t == v:
                    continue
                deg_t_new = deg[t] + (1 if t == u or t == v else 0)
                if deg_u_new == deg_t_new:
                    return True

        # Case 3: (u, v) is the last edge. Path s-a-u-v.
        # a in N(u), s in N(a), s not in {u,v}, deg_new(s) == deg_new(v).
        deg_v_new = deg[v] + 1
        for a in range(n):
            if not (neighbors_u & (1 << a)):
                continue
            neighbors_a = adj[a]
            for s in range(n):
                if not (neighbors_a & (1 << s)):
                    continue
                if s == u or s == v:
                    continue
                deg_s_new = deg[s] + (1 if s == u or s == v else 0)
                if deg_v_new == deg_s_new:
                    return True

        return False

    def is_valid_graph(edge_set):
        """Check if a complete edge set has no violations."""
        adj, deg = build_adj_deg(edge_set)
        for u, v in edge_set:
            degs_u = set()
            for s in range(n):
                if s != v and (adj[u] & (1 << s)):
                    degs_u.add(deg[s])
            degs_v = set()
            for t in range(n):
                if t != u and (adj[v] & (1 << t)):
                    degs_v.add(deg[t])
            if degs_u & degs_v:
                return False
        return True

    rng = random.Random(42)

    # Generate multiple structured starting points
    def make_k78():
        return [(i, j) for i in range(7) for j in range(7, 15)]

    def make_turan3():
        parts = [[0,3,6,9,12], [1,4,7,10,13], [2,5,8,11,14]]
        edges = []
        for a in range(3):
            for b in range(a+1, 3):
                for i in parts[a]:
                    for j in parts[b]:
                        edges.append((min(i,j), max(i,j)))
        return edges

    def make_k87_plus_internal():
        # K(8,7) plus all internal edges in the 8-part
        edges = [(i, j) for i in range(8) for j in range(8, 15)]
        edges += [(i, j) for i in range(8, 15) for j in range(i+1, 15)]
        return edges

    def make_complement_star():
        # Complete graph minus star from vertex 0
        return [(i, j) for i in range(1, n) for j in range(i+1, n)]

    def make_random_graph(p):
        edges = []
        for e in all_edges:
            if rng.random() < p:
                edges.append(e)
        return edges

    starting_candidates = [
        make_k78(),
        make_turan3(),
        make_k87_plus_internal(),
        make_complement_star(),
    ]
    for p in [0.3, 0.4, 0.5]:
        starting_candidates.append(make_random_graph(p))

    # Filter to valid starting points
    valid_starting = []
    for cand in starting_candidates:
        if is_valid_graph(cand):
            valid_starting.append(cand)

    if not valid_starting:
        valid_starting = [make_k78()]

    best_edges = valid_starting[0][:]
    best_count = len(best_edges)

    # Greedy add from each valid starting point with multiple orderings
    for start_idx, base in enumerate(valid_starting):
        if time.time() > deadline - 15:
            break

        for attempt in range(50):
            if time.time() > deadline - 10:
                break

            current_edges = set(base)
            current_adj, current_deg = build_adj_deg(current_edges)

            # Different ordering strategies
            if attempt % 5 == 0:
                shuffled = all_edges[:]
                rng.shuffle(shuffled)
            elif attempt % 5 == 1:
                # Prefer edges within parts
                shuffled = sorted(all_edges, key=lambda e: (0 if e[0] < 7 and e[1] < 7 else 1 if e[0] >= 7 and e[1] >= 7 else 2, rng.random()))
            elif attempt % 5 == 2:
                # Prefer edges connecting low-degree vertices
                shuffled = sorted(all_edges, key=lambda e: (current_deg[e[0]] + current_deg[e[1]], rng.random()))
            elif attempt % 5 == 3:
                # Prefer edges that don't create degree conflicts
                def conflict_score(e):
                    u, v = e
                    # Count how many neighbors would have same degree as other side
                    score = 0
                    for s in range(n):
                        if s != v and (current_adj[u] & (1 << s)):
                            for t in range(n):
                                if t != s and t != u and (current_adj[v] & (1 << t)):
                                    ds = current_deg[s] + (1 if s == u or s == v else 0)
                                    dt = current_deg[t] + (1 if t == u or t == v else 0)
                                    if ds == dt:
                                        score += 1
                    return score
                shuffled = sorted(all_edges, key=lambda e: (conflict_score(e), rng.random()))
            else:
                shuffled = all_edges[:]
                rng.shuffle(shuffled)

            for edge in shuffled:
                if edge in current_edges:
                    continue
                u, v = edge
                if check_violation_for_add(current_adj, current_deg, u, v):
                    continue
                current_edges.add(edge)
                current_adj[u] |= (1 << v)
                current_adj[v] |= (1 << u)
                current_deg[u] += 1
                current_deg[v] += 1

            if len(current_edges) > best_count:
                best_count = len(current_edges)
                best_edges = list(current_edges)

    # Local search: try removing one edge and re-greedy
    no_improve_count = 0
    max_no_improve = 20

    for ls_attempt in range(200):
        if time.time() > deadline - 5:
            break
        if no_improve_count >= max_no_improve:
            break

        improved = False

        # Try removing one edge and re-greedy
        edges_to_try = best_edges[:]
        rng.shuffle(edges_to_try)

        for edge_to_remove in edges_to_try[:40]:
            if time.time() > deadline - 3:
                break

            current_edges = set(best_edges)
            current_edges.remove(edge_to_remove)
            current_adj, current_deg = build_adj_deg(current_edges)

            shuffled = all_edges[:]
            rng.shuffle(shuffled)

            for edge in shuffled:
                if edge in current_edges:
                    continue
                u, v = edge
                if check_violation_for_add(current_adj, current_deg, u, v):
                    continue
                current_edges.add(edge)
                current_adj[u] |= (1 << v)
                current_adj[v] |= (1 << u)
                current_deg[u] += 1
                current_deg[v] += 1

            if len(current_edges) > best_count:
                best_count = len(current_edges)
                best_edges = list(current_edges)
                improved = True
                no_improve_count = 0
                break

        # Try 2-edge swaps if no 1-edge improvement
        if not improved:
            for _ in range(20):
                if time.time() > deadline - 3:
                    break

                if len(best_edges) < 2:
                    break
                e1 = rng.choice(best_edges)
                e2 = rng.choice(best_edges)
                if e1 == e2:
                    continue

                current_edges = set(best_edges)
                current_edges.remove(e1)
                current_edges.remove(e2)
                current_adj, current_deg = build_adj_deg(current_edges)

                shuffled = all_edges[:]
                rng.shuffle(shuffled)

                for edge in shuffled:
                    if edge in current_edges:
                        continue
                    u, v = edge
                    if check_violation_for_add(current_adj, current_deg, u, v):
                        continue
                    current_edges.add(edge)
                    current_adj[u] |= (1 << v)
                    current_adj[v] |= (1 << u)
                    current_deg[u] += 1
                    current_deg[v] += 1

                if len(current_edges) > best_count:
                    best_count = len(current_edges)
                    best_edges = list(current_edges)
                    improved = True
                    no_improve_count = 0
                    break

        if not improved:
            no_improve_count += 1

    # Final validation
    if not is_valid_graph(best_edges):
        # Fallback to first valid starting point
        best_edges = valid_starting[0][:]
        best_count = len(best_edges)

    out = [[int(u), int(v)] for u, v in best_edges]
    return out
