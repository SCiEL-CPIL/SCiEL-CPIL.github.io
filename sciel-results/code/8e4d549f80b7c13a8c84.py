import time
import random
import sys
from collections import deque

try:
    from scipy.optimize import linprog
    _HAS_LP = True
except Exception:
    _HAS_LP = False

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}


def construct():
    t_start = time.time()
    DEADLINE = 85.0
    n = P['n']
    edges = P['edges']
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    def time_up():
        return (time.time() - t_start) > DEADLINE

    # ---------- fast fallback heuristic (always valid) ----------
    def bfs_path(s, t, banned):
        if s == t:
            return None
        if t in adj[s]:
            return [s, t]
        parent = {s: None}
        q = deque([s])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if v == s or v in parent or v in banned:
                    continue
                if v == t:
                    path = [v]
                    cur = u
                    while cur is not None:
                        path.append(cur)
                        cur = parent[cur]
                    path.reverse()
                    return path
                parent[v] = u
                q.append(v)
        return None

    def find_paths(branches, order=None):
        m = len(branches)
        if m < 2:
            return None
        used = set()
        paths = {}
        if order is None:
            pairs = [(i, j) for i in range(m) for j in range(i + 1, m)]
            pairs.sort(key=lambda p: (0 if branches[p[1]] in adj[branches[p[0]]] else 1,
                                      -len(adj[branches[p[0]]]), -len(adj[branches[p[1]]])))
        else:
            pairs = order
        for (i, j) in pairs:
            a, b = branches[i], branches[j]
            banned = set(used)
            banned.update(branches)
            banned.discard(a)
            banned.discard(b)
            path = bfs_path(a, b, banned)
            if path is None:
                return None
            used.update(path[1:-1])
            paths[(i, j)] = path
        return [paths[p] for p in pairs]

    def find_paths_multi(branches, k=5):
        p = find_paths(branches)
        if p is not None:
            return p
        for _ in range(k):
            pairs = [(i, j) for i in range(len(branches)) for j in range(i + 1, len(branches))]
            random.shuffle(pairs)
            p = find_paths(branches, pairs)
            if p is not None:
                return p
        return None

    def find_max_clique():
        best = []
        cands = sorted(range(n), key=lambda x: -len(adj[x]))
        def expand(clique, cand):
            nonlocal best
            if len(clique) > len(best):
                best = list(clique)
            if not cand:
                return
            for i, v in enumerate(cand):
                expand(clique + [v], [u for u in cand[i + 1:] if u in adj[v]])
        expand([], cands)
        return best

    clique = find_max_clique()
    heur_paths = find_paths_multi(clique, 8)
    heur_branches = list(clique)
    if heur_paths is None and len(clique) >= 2:
        for sz in range(len(clique) - 1, 1, -1):
            sub = clique[:sz]
            heur_paths = find_paths_multi(sub, 5)
            if heur_paths is not None:
                heur_branches = sub
                break
    if heur_paths is None:
        u, v = edges[0]
        heur_branches = [u, v]
        heur_paths = [[u, v]]

    best_branches = heur_branches
    best_paths = heur_paths
    best_count = len(heur_branches)

    def update(b, p):
        nonlocal best_branches, best_paths, best_count
        if p is not None and len(b) > best_count:
            best_branches, best_paths, best_count = list(b), list(p), len(b)

    # ---------- Path extraction from a selected edge subgraph ----------
    def extract_paths(branches, edge_set):
        """Given branch set and selected edges (set of frozenset({u,v})), extract k(k-1)/2 internally-disjoint paths.
        Uses flow decomposition: build multigraph of selected edges, decompose into paths between branch pairs."""
        m = len(branches)
        bset = set(branches)
        # Build adjacency from edge_set
        sub_adj = {v: [] for v in bset}
        for e in edge_set:
            u, v = list(e)
            if u in bset:
                sub_adj.setdefault(u, []).append(v)
            if v in bset:
                sub_adj.setdefault(v, []).append(u)
        # For interior vertices, track degree
        # We need to decompose the edge set into paths. Each branch has degree m-1 in the subgraph.
        # Each interior vertex has degree <= 2.
        # Build a multigraph and do edge decomposition.
        # Represent edges as a multiset; for each pair of branches, find a path using only remaining edges.
        # Simpler: since each interior vertex has degree exactly 2 (in a valid subdivision) or less,
        # the subgraph is a union of paths and cycles. We need to extract the paths.
        # Use a stack-based approach: for each branch, follow edges, pairing them up.
        
        # Build edge list with multiplicity
        edge_list = []
        for e in edge_set:
            u, v = list(e)
            edge_list.append((u, v))
        
        # For each branch, it has m-1 incident edges. We need to pair them into m-1 paths to other branches.
        # Approach: for each branch b, look at its incident edges. Each edge leads to either another branch
        # (direct edge) or an interior vertex. Follow the path from b through interior vertices until reaching
        # another branch.
        
        # Build adjacency as list of neighbors for the subgraph
        sub_adj2 = {v: [] for v in range(n)}
        for (u, v) in edge_list:
            sub_adj2[u].append(v)
            sub_adj2[v].append(u)
        
        # For each branch, find all paths starting from it
        paths_found = {}
        used_edges = set()  # frozenset of (min,max) to track used edges
        
        # For each branch, iterate over its incident edges in the subgraph
        for b in branches:
            for nb in sub_adj2.get(b, []):
                # Check if edge (b, nb) is already used
                ekey = frozenset([b, nb])
                if ekey in used_edges:
                    continue
                # Follow path from b through nb
                path = [b, nb]
                prev = b
                cur = nb
                used_edges.add(ekey)
                while cur not in bset:
                    # cur is interior, degree 2 in subgraph
                    neighbors = [x for x in sub_adj2.get(cur, []) if x != prev]
                    if not neighbors:
                        return None  # broken
                    nxt = neighbors[0]
                    ekey2 = frozenset([cur, nxt])
                    if ekey2 in used_edges:
                        return None  # conflict
                    used_edges.add(ekey2)
                    path.append(nxt)
                    prev = cur
                    cur = nxt
                # path ends at cur (a branch)
                a, c = path[0], path[-1]
                key = frozenset([a, c])
                if key in paths_found:
                    return None  # duplicate path for same pair
                paths_found[key] = path
        
        # Verify all pairs covered
        if len(paths_found) != m * (m - 1) // 2:
            return None
        
        # Order paths consistently
        result = []
        for i in range(m):
            for j in range(i + 1, m):
                key = frozenset([branches[i], branches[j]])
                if key not in paths_found:
                    return None
                result.append(paths_found[key])
        return result

    # ---------- ILP via branch-and-bound with LP relaxation ----------
    if _HAS_LP and not time_up():
        try:
            # Variables: x_v (branch indicator), y_e (edge selected)
            # For a fixed k, constraints:
            #   sum x_v = k
            #   For each vertex v: sum_{e incident to v} y_e <= 1 + (k-1)*x_v
            #   For each edge e: y_e <= x_u + x_v for endpoints u,v (only select edges if both endpoints are branches OR the edge is used as interior)
            #   Actually: y_e can be 1 only if the edge is "available". An edge (u,v) can be selected if:
            #     - both u,v are branches (direct edge in subdivision), OR
            #     - at least one of u,v is not a branch (interior edge)
            #   But we also need: if v is not a branch and y_e=1 for some edge e incident to v, then v is used as interior.
            #   The capacity constraint sum_{e incident to v} y_e <= 1 handles interior vertices.
            #   For branch vertices: sum_{e incident to v} y_e = (k-1) * x_v
            #   Wait, that's not quite right. A branch vertex can have edges going to other branches (direct) and edges going to interior vertices.
            #   In a clique subdivision, each branch has exactly k-1 incident subdivision edges.
            #   So: for branch v, sum_{e incident to v} y_e = k-1.
            #   For non-branch v, sum_{e incident to v} y_e <= 1 (used by at most one path).
            #   Combined: sum_{e incident to v} y_e <= 1 + (k-2)*x_v  ... no.
            #   If x_v = 1: sum = k-1
            #   If x_v = 0: sum <= 1
            #   So: sum_{e incident to v} y_e <= 1 + (k-2)*x_v  ... when x_v=1: 1+k-2 = k-1. When x_v=0: 1. 
            #   But we also need lower bound for branches: sum >= (k-1)*x_v.
            #   And y_e <= x_u + x_v is not needed; instead y_e <= 1 always, and the vertex constraints handle it.
            #   Actually we need: y_e = 0 if neither endpoint is a branch AND the edge is not used. 
            #   The vertex capacity for non-branch vertices (<=1) already limits interior usage.
            #   But an edge between two non-branch vertices could be selected with y_e=1, using up capacity of both.
            #   That's fine as long as the global structure works.
            #   However, we need the selected edges to form exactly k-1 paths from each branch.
            #   The degree constraints + connectivity is the hard part. The LP relaxation won't enforce connectivity.
            #   So this is a relaxation: we solve it, get a candidate edge set, then try to extract paths.
            #   If extraction fails, we add a cutting plane or just use the heuristic.
            
            # Let's try k from best_count+1 up to some max.
            # For each k, solve the LP relaxation of the ILP.
            # Variables: n (x) + m (y) = 24 + 89 = 113 vars.
            # This is small enough for linprog.
            
            edge_idx = {}
            for i, (u, v) in enumerate(edges):
                edge_idx[(u, v)] = i
                edge_idx[(v, u)] = i
            m_edges = len(edges)
            
            def solve_for_k(k):
                if k < 2:
                    return None
                if k > n:
                    return None
                # LP relaxation: 0 <= x_v <= 1, 0 <= y_e <= 1
                # Maximize sum x_v (but we fix sum x_v = k for feasibility check)
                # Actually, let's maximize sum x_v with the constraints, and see if we can get k.
                # Or: check feasibility of sum x_v >= k.
                # Let's just maximize sum x_v subject to the degree constraints with a given k.
                # Hmm, the constraints depend on k. Let's parameterize.
                
                # Variables: x_0..x_{n-1}, y_0..y_{m_edges-1}
                # Objective: maximize sum x_v  (c = -1 for x, 0 for y)
                c = [-1.0] * n + [0.0] * m_edges
                
                # Constraints (A_ub @ vars <= b_ub):
                A_ub = []
                b_ub = []
                
                # For each vertex v: sum_{e incident to v} y_e - (k-1)*x_v <= 1  (when x_v=0: sum y <= 1; when x_v=1: sum y <= k)
                # Wait, I want: if x_v=1, sum y_e = k-1; if x_v=0, sum y_e <= 1.
                # Upper: sum y_e <= 1 + (k-2)*x_v  ->  sum y_e - (k-2)*x_v <= 1
                # Lower: sum y_e >= (k-1)*x_v  ->  -sum y_e + (k-1)*x_v <= 0
                # Also sum x_v = k:  sum x_v <= k and -sum x_v <= -k
                # And 0 <= x_v <= 1, 0 <= y_e <= 1.
                
                for v in range(n):
                    # Upper: sum_{e incident to v} y_e - (k-2)*x_v <= 1
                    row = [0.0] * (n + m_edges)
                    row[v] = -(k - 2)  # coefficient of x_v
                    for (u, w) in edges:
                        if u == v or w == v:
                            row[n + edge_idx[(u, w)]] = 1.0
                    A_ub.append(row)
                    b_ub.append(1.0)
                    
                    # Lower: -sum_{e incident to v} y_e + (k-1)*x_v <= 0
                    row2 = [0.0] * (n + m_edges)
                    row2[v] = (k - 1)
                    for (u, w) in edges:
                        if u == v or w == v:
                            row2[n + edge_idx[(u, w)]] = -1.0
                    A_ub.append(row2)
                    b_ub.append(0.0)
                
                # sum x_v <= k
                row_sum = [0.0] * (n + m_edges)
                for v in range(n):
                    row_sum[v] = 1.0
                A_ub.append(row_sum)
                b_ub.append(float(k))
                
                # -sum x_v <= -k
                row_neg = [-row_sum[i] for i in range(len(row_sum))]
                A_ub.append(row_neg)
                b_ub.append(-float(k))
                
                # Bounds: 0 <= x_v <= 1, 0 <= y_e <= 1
                bounds = [(0, 1)] * (n + m_edges)
                
                res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
                if res.status != 0:
                    return None
                # Check if LP solution is close to integer
                x_vals = res.x[:n]
                y_vals = res.x[n:]
                # Round: if x_v > 0.5, treat as branch
                branches = [v for v in range(n) if x_vals[v] > 0.5]
                if len(branches) != k:
                    # Try different threshold
                    branches = sorted(range(n), key=lambda v: -x_vals[v])[:k]
                edge_set = set()
                for i, (u, v) in enumerate(edges):
                    if y_vals[i] > 0.5:
                        edge_set.add(frozenset([u, v]))
                return (branches, edge_set)
            
            # Try k values from best_count+1 upward
            max_k_to_try = min(n, best_count + 6)
            for k in range(max_k_to_try, best_count, -1):  # try largest first
                if time_up():
                    break
                sol = solve_for_k(k)
                if sol is None:
                    continue
                branches, edge_set = sol
                paths = extract_paths(branches, edge_set)
                if paths is not None:
                    update(branches, paths)
                    break
                # If extraction failed, try with the LP's fractional solution rounded differently
                # or just continue to next k
            
            # Also try the heuristic's best and see if ILP confirms
            if not time_up():
                k = best_count
                sol = solve_for_k(k)
                if sol is not None:
                    branches, edge_set = sol
                    paths = extract_paths(branches, edge_set)
                    if paths is not None:
                        update(branches, paths)

        except Exception:
            pass

    # ---------- Improved heuristic (same as before, as backup) ----------
    if not time_up():
        def can_pack(branches, tries=12):
            p = find_paths(branches)
            if p is not None:
                return p
            for _ in range(tries):
                pairs = [(i, j) for i in range(len(branches)) for j in range(i + 1, len(branches))]
                random.shuffle(pairs)
                p = find_paths(branches, pairs)
                if p is not None:
                    return p
            return None

        cur = list(clique)
        cur_paths = can_pack(cur, 10)
        if cur_paths is not None:
            update(cur, cur_paths)
        changed = True
        while changed and not time_up():
            changed = False
            candidates = [v for v in range(n) if v not in cur]
            candidates.sort(key=lambda v: len(adj[v]))
            for v in candidates:
                if time_up():
                    break
                trial = cur + [v]
                p = can_pack(trial, 8)
                if p is not None:
                    cur = trial
                    changed = True
                    update(cur, p)
                    break

        if not time_up():
            improved = True
            while improved and not time_up():
                improved = False
                cur_b = list(best_branches)
                cur_set = set(cur_b)
                for idx in range(len(cur_b)):
                    if time_up():
                        break
                    for v in range(n):
                        if v in cur_set:
                            continue
                        new_b = cur_b[:idx] + cur_b[idx + 1:] + [v]
                        p = can_pack(new_b, 8)
                        if p is not None and len(new_b) > best_count:
                            update(new_b, p)
                            improved = True
                            break
                    if improved:
                        break

        if not time_up():
            for _ in range(300):
                if time_up():
                    break
                sz = min(n, max(2, best_count + random.randint(-1, 2)))
                seed = random.sample(range(n), sz)
                cur = list(seed)
                p = can_pack(cur, 6)
                if p is not None:
                    update(cur, p)
                changed = True
                while changed and not time_up():
                    changed = False
                    for v in range(n):
                        if v in cur:
                            continue
                        trial = cur + [v]
                        p = can_pack(trial, 5)
                        if p is not None:
                            cur = trial
                            changed = True
                            update(cur, p)
                            break

    # final safety
    if best_paths is None or len(best_branches) < 2:
        u, v = edges[0]
        best_branches = [u, v]
        best_paths = [[u, v]]

    branches_out = [int(b) for b in best_branches]
    paths_out = [[int(x) for x in p] for p in best_paths]
    return {'branches': branches_out, 'paths': paths_out}
