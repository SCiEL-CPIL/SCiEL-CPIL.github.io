import time
import random
import sys
from collections import deque

try:
    from ortools.sat.python import cp_model
    _HAS_CP = True
except Exception:
    _HAS_CP = False

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}


def construct():
    t_start = time.time()
    DEADLINE = 70.0
    n = P['n']
    edges = P['edges']
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    def time_up():
        return (time.time() - t_start) > DEADLINE

    # ---------- fallback heuristic (fast, always valid) ----------
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

    # quick heuristic to have a valid answer immediately
    clique = find_max_clique()
    heur_branches = list(clique)
    heur_paths = find_paths_multi(clique, 8)
    if heur_paths is None and len(clique) >= 2:
        # try shrinking clique to a feasible subset
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

    # ---------- CP-SAT global solver ----------
    if _HAS_CP and not time_up():
        try:
            m = n
            # node ids in split graph
            def IN(v):  return v
            def OUT(v): return n + v
            Nnodes = 2 * n
            S = 2 * n
            T = 2 * n + 1

            model = cp_model.CpModel()
            x = [model.NewBoolVar(f'x_{v}') for v in range(n)]
            # objective: maximize number of branches
            model.Maximize(sum(x))

            # edge arc variables for each directed edge of the split graph
            # arc id -> (u_node, v_node, from_branch_a, to_branch_b, is_interior_edge)
            arcs = []          # list of (u_node, v_node)
            arc_vars = []      # x_{u_node,v_node}
            arc_index = {}     # (u_node, v_node) -> index
            def add_arc(u_node, v_node):
                if (u_node, v_node) in arc_index:
                    return arc_index[(u_node, v_node)]
                idx = len(arcs)
                arcs.append((u_node, v_node))
                var = model.NewBoolVar(f'a_{u_node}_{v_node}')
                arc_vars.append(var)
                arc_index[(u_node, v_node)] = idx
                return idx

            # Build arcs:
            # 1) S -> IN(a) for each branch a
            for a in range(n):
                add_arc(S, IN(a))
            # 2) OUT(a) -> T for each branch a
            for a in range(n):
                add_arc(OUT(a), T)
            # 3) IN(v) -> OUT(v) for each vertex v (capacity 1 if not branch)
            for v in range(n):
                add_arc(IN(v), OUT(v))
            # 4) OUT(u) -> IN(v) for each directed edge (u,v) in original graph
            for u, v in edges:
                add_arc(OUT(u), IN(v))
                add_arc(OUT(v), IN(u))

            # Flow conservation per branch a:
            #   supply at IN(a): 1 if x_a
            #   demand at OUT(a): 1 if x_a
            # For each node w in split graph, sum_in - sum_out = b(w)
            # b(IN(a)) = x_a, b(OUT(a)) = -x_a, others 0
            # S: sum_out(S) = sum_a x_a ; T: sum_in(T) = sum_a x_a
            for a in range(n):
                # IN(a): in arcs = [S->IN(a)]; out arcs = [IN(a)->OUT(a)] + [OUT(u)->IN(a) for u in adj[a]]
                in_arcs = [add_arc(S, IN(a))]
                out_arcs = [add_arc(IN(a), OUT(a))]
                for u in adj[a]:
                    out_arcs.append(add_arc(OUT(u), IN(a)))
                model.Add(sum(arc_vars[i] for i in in_arcs) - sum(arc_vars[i] for i in out_arcs) == x[a])
                # OUT(a): in arcs = [IN(a)->OUT(a)] + [OUT(u)->IN(v) ... no, in to OUT(a) is only IN(a)->OUT(a)]
                # actually OUT(a) receives from IN(a) only; sends to T and to IN(v) for v in adj[a]
                in_arcs2 = [add_arc(IN(a), OUT(a))]
                out_arcs2 = [add_arc(OUT(a), T)]
                for v in adj[a]:
                    out_arcs2.append(add_arc(OUT(a), IN(v)))
                model.Add(sum(arc_vars[i] for i in in_arcs2) - sum(arc_vars[i] for i in out_arcs2) == -x[a])
            # S: sum_out(S) == sum_a x_a
            model.Add(sum(arc_vars[i] for i in range(len(arcs)) if arcs[i][0] == S) == sum(x))
            # T: sum_in(T) == sum_a x_a
            model.Add(sum(arc_vars[i] for i in range(len(arcs)) if arcs[i][1] == T) == sum(x))

            # Capacity: each non-branch interior vertex used at most once.
            # Vertex v (non-branch) is used if flow passes through IN(v)->OUT(v).
            # The only arc through v's interior is IN(v)->OUT(v).
            # Constraint: if x_v == 0, then arc IN(v)->OUT(v) <= 1 (always true for bool) -- but we need
            # "v is interior of at most one path". Since flow is unit per branch and the only way to use v
            # is via IN(v)->OUT(v), and that arc is binary, it's automatically <=1. However, multiple
            # branches could route through v only if v is non-branch, and the arc is shared. Binary arc
            # already caps total flow through v at 1. Good.
            # Also branch vertices: flow through IN(a)->OUT(a) is part of a's own unit; fine.

            # Disallow trivial self-loop paths: a branch a must not send flow S->IN(a)->OUT(a)->T
            # (that would be a length-0 "path" for pair (a,a) which doesn't exist). Actually flow for
            # branch a is a single unit that must go from IN(a) to OUT(a) via the graph; the trivial
            # IN(a)->OUT(a) arc would make it a self path. We must forbid using IN(a)->OUT(a) for a.
            for a in range(n):
                model.Add(arc_vars[add_arc(IN(a), OUT(a))] <= 1 - x[a])

            # Also ensure that a unit for branch a actually reaches OUT(a) via at least one edge
            # (not the forbidden trivial arc). Since we forbid the trivial arc when x_a=1, the unit
            # must leave IN(a) via some OUT(u)->IN(a) arc? No: IN(a) has out arcs IN(a)->OUT(a) (forbidden)
            # and ... wait, IN(a) out-arcs are only IN(a)->OUT(a). In-arcs to IN(a) are S->IN(a) and
            # OUT(u)->IN(a). The flow enters IN(a) from S (supply) and possibly from OUT(u). But the
            # net supply at IN(a) is +x_a. If we forbid IN(a)->OUT(a), then the unit cannot leave IN(a),
            # violating conservation unless it enters and leaves via other arcs -- but IN(a) has no other
            # out-arcs. So the model is wrong: branch a's unit must travel from a to b and back? 
            # 
            # Re-think: The standard reduction for "a set of pairwise internally-disjoint paths between
            # all pairs of terminals" is NOT a simple single-commodity flow. Each pair (a,b) needs its own
            # path. This is a multi-commodity flow (one commodity per unordered pair), which is hard.
            #
            # Simplification: Instead, model it as: choose branches B, and for each v not in B, v is used
            # by at most one path. Use a different encoding: for each unordered pair {a,b}, a binary var
            # y_{a,b} indicating we have a path; but ensuring existence of internally-disjoint paths is
            # the hard part.
            #
            # Given complexity, fall back to a simpler CP model: maximize |B| subject to a *relaxation*
            # that each non-branch vertex has degree <= 2 in the union of chosen path edges, and each
            # branch has degree = |B|-1, using edge-selection variables with flow per pair approximated.
            # This is getting too complex for a robust 70s model. 
            #
            # -> Use a pragmatic CP model: select branches B and edge set E' such that the subgraph (V, E')
            #  has each non-branch vertex degree <= 2, each branch degree = |B|-1, and the subgraph is
            #  connected enough. This is a relaxation; then extract paths heuristically and verify.
            pass  # skip complex model; use improved heuristic below

        except Exception:
            pass

    # ---------- improved heuristic search (structural: branch-and-bound on branch set with
    #            path-extraction feasibility, using a better ordering + repair) ----------
    if not time_up():
        # Start from clique, try to grow with a smarter "least-constraining" vertex order
        def can_pack(branches, tries=12):
            # try default + random orders; also try sorting by pair difficulty
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

        # grow from clique
        cur = list(clique)
        cur_paths = can_pack(cur, 10)
        if cur_paths is not None:
            update(cur, cur_paths)
        changed = True
        while changed and not time_up():
            changed = False
            # try adding each non-branch vertex, in order of low degree (less constraining)
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

        # local search: swap to increase
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

        # random restarts with repair
        if not time_up():
            for _ in range(400):
                if time_up():
                    break
                sz = min(n, max(2, best_count + random.randint(-1, 2)))
                seed = random.sample(range(n), sz)
                cur = list(seed)
                p = can_pack(cur, 6)
                if p is not None:
                    update(cur, p)
                # grow
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
