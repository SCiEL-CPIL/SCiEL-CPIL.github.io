"""Rewritten program: greedy construction with smart colouring + local search."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def construct():
    t0 = time.time()
    DEADLINE = 150.0  # wall-clock limit; keep safety margin
    N = 20
    C = 20  # number of colours

    def time_left():
        return DEADLINE - (time.time() - t0)

    # ---------- helpers ----------
    def build_adjacency(edges):
        """edges: list of [u,v,c]. Returns adj dict: adj[u][v] = c"""
        adj = defaultdict(dict)
        for u, v, c in edges:
            adj[u][v] = c
            adj[v][u] = c
        return adj

    def check_4cycle_rainbow(adj):
        """Check every 4-cycle has 4 distinct colours. Returns True if valid."""
        for i in range(N):
            for j in range(i+1, N):
                if j not in adj[i]:
                    continue
                # find common neighbours of i and j
                common = []
                for k in range(N):
                    if k != i and k != j and k in adj[i] and k in adj[j]:
                        common.append(k)
                for idx, k in enumerate(common):
                    for l in common[idx+1:]:
                        # 4-cycle: i-k-j-l-i
                        c1 = adj[i][k]
                        c2 = adj[k][j]
                        c3 = adj[j][l]
                        c4 = adj[l][i]
                        if len({c1, c2, c3, c4}) != 4:
                            return False
        return True

    def get_4cycle_violations(adj):
        """Return list of (i,k,j,l, [c1,c2,c3,c4]) for violating 4-cycles."""
        violations = []
        for i in range(N):
            for j in range(i+1, N):
                if j not in adj[i]:
                    continue
                common = []
                for k in range(N):
                    if k != i and k != j and k in adj[i] and k in adj[j]:
                        common.append(k)
                for idx, k in enumerate(common):
                    for l in common[idx+1:]:
                        c1 = adj[i][k]
                        c2 = adj[k][j]
                        c3 = adj[j][l]
                        c4 = adj[l][i]
                        if len({c1, c2, c3, c4}) != 4:
                            violations.append((i, k, j, l, [c1, c2, c3, c4]))
        return violations

    def can_add_edge(adj, u, v, c):
        """Check if adding edge (u,v,c) keeps all 4-cycles rainbow."""
        if v in adj[u]:
            return False
        # A 4-cycle containing edge (u,v) has form u-v-x-y-u or u-v-y-x-u
        # i.e., x adjacent to v, y adjacent to u, and x adjacent to y
        for x in list(adj[v].keys()):
            if x == u:
                continue
            for y in list(adj[u].keys()):
                if y == v or y == x:
                    continue
                if x in adj[y]:
                    # cycle: u-v-x-y-u
                    colours = [c, adj[v][x], adj[x][y], adj[y][u]]
                    if len(set(colours)) != 4:
                        return False
        return True

    def add_edge(adj, edges, u, v, c):
        adj[u][v] = c
        adj[v][u] = c
        edges.append([u, v, c])

    def remove_edge(adj, edges, u, v):
        del adj[u][v]
        del adj[v][u]
        edges[:] = [e for e in edges if not ((e[0]==u and e[1]==v) or (e[0]==v and e[1]==u))]

    def best_colour_for_edge(adj, u, v):
        """Pick colour that minimises risk: count how many 4-cycles would be violated per colour."""
        best_c = None
        best_score = float('inf')
        for c in range(C):
            if not can_add_edge(adj, u, v, c):
                continue
            # Score: for each 4-cycle this edge participates in, check if colour c
            # clashes with existing colours in that cycle
            score = 0
            for x in list(adj[v].keys()):
                if x == u:
                    continue
                for y in list(adj[u].keys()):
                    if y == v or y == x:
                        continue
                    if x in adj[y]:
                        existing = {adj[v][x], adj[x][y], adj[y][u]}
                        if c in existing:
                            score += 10  # would be a violation (but can_add already checked)
                        else:
                            score += 1  # minor: adds another colour constraint
            if score < best_score:
                best_score = score
                best_c = c
        return best_c

    def local_search(adj, edges, max_iter=20000):
        """Try to add edges, and if stuck, recolor or remove-rebuild."""
        improved = True
        while improved and time_left() > 5.0:
            improved = False
            # Phase 1: try to add any edge
            for u in range(N):
                for v in range(u+1, N):
                    if v not in adj[u]:
                        c = best_colour_for_edge(adj, u, v)
                        if c is not None:
                            add_edge(adj, edges, u, v, c)
                            improved = True
            # Phase 2: if no edge could be added, try recoloring
            if not improved:
                recolored = False
                for u in range(N):
                    for v in list(adj[u].keys()):
                        if u < v:
                            old_c = adj[u][v]
                            for new_c in range(C):
                                if new_c == old_c:
                                    continue
                                # temporarily remove, try new colour
                                remove_edge(adj, edges, u, v)
                                if can_add_edge(adj, u, v, new_c):
                                    add_edge(adj, edges, u, v, new_c)
                                    recolored = True
                                    improved = True
                                    break
                                else:
                                    # restore
                                    add_edge(adj, edges, u, v, old_c)
                                    break
                            if time_left() < 5.0:
                                return
                if recolored:
                    continue
                # Phase 3: remove a random edge and try to rebuild
                if edges:
                    idx = random.randint(0, len(edges)-1)
                    u, v, c = edges[idx]
                    remove_edge(adj, edges, u, v)
                    # Now try to re-add possibly with different colour, and other edges
                    for trial in range(50):
                        any_added = False
                        for uu in range(N):
                            for vv in range(uu+1, N):
                                if vv not in adj[uu]:
                                    cc = best_colour_for_edge(adj, uu, vv)
                                    if cc is not None:
                                        add_edge(adj, edges, uu, vv, cc)
                                        any_added = True
                        if not any_added:
                            break
                    improved = True
        return adj, edges

    # ---------- main ----------
    random.seed(42)

    best_edges = None
    best_count = 0

    # Multiple restarts with different random orderings
    for restart in range(50):
        if time_left() < 10.0:
            break

        adj = defaultdict(dict)
        edges = []

        # Greedy: try to add edges in a random order
        pairs = [(u, v) for u in range(N) for v in range(u+1, N)]
        random.shuffle(pairs)

        for u, v in pairs:
            c = best_colour_for_edge(adj, u, v)
            if c is not None:
                add_edge(adj, edges, u, v, c)

        # Local search
        adj, edges = local_search(adj, edges, max_iter=20000)

        # Validate
        if check_4cycle_rainbow(adj):
            if len(edges) > best_count:
                best_count = len(edges)
                best_edges = [list(e) for e in edges]

        if time_left() < 10.0:
            break

    # Final safety: if no valid graph found, return the star
    if best_edges is None or best_count < 19:
        return [[0, v, 0] for v in range(1, N)]

    return best_edges
# EVOLVE-BLOCK-END
