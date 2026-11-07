"""Rewritten program: greedy clique subdivision with iterative branch augmentation."""
# EVOLVE-BLOCK-START
import random
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def _build_adj():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _bfs_path(adj, src, dst, blocked, seed):
    """Return a simple path (list of vertices) from src to dst avoiding `blocked`
    (except endpoints). Uses randomized BFS neighbor order with given seed."""
    if src == dst:
        return [src]
    rng = random.Random(seed)
    parent = {src: None}
    q = deque([src])
    # order neighbors deterministically but shuffle per call
    while q:
        u = q.popleft()
        if u == dst:
            break
        nbrs = [w for w in adj[u] if w not in blocked]
        rng.shuffle(nbrs)
        for w in nbrs:
            if w not in parent:
                parent[w] = u
                if w == dst:
                    q = deque()
                    break
                q.append(w)
        if not q:
            break
    if dst not in parent:
        return None
    path = []
    cur = dst
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return path

def _greedy_max_clique(adj, n, seed=0):
    rng = random.Random(seed)
    # start from a random vertex, greedily add vertex maximizing common neighbors
    start = rng.randrange(n)
    clique = [start]
    cand = set(range(n)) - {start}
    while cand:
        best = None
        best_score = -1
        for v in cand:
            common = sum(1 for c in clique if c in adj[v])
            # tie-break by degree
            score = (common, len(adj[v]))
            if score > (best_score, 0) or best is None:
                if best is None or score > best[0]:
                    best = (score, v)
        if best is None:
            break
        (sc, v) = best
        if sc[0] == 0:
            break
        clique.append(v)
        cand.discard(v)
    return clique

def _try_add_branch(adj, branches, used_interiors, candidate, seed):
    """Try to connect `candidate` to every branch by internally-disjoint paths
    avoiding all branches and used_interiors. Returns list of paths (one per branch)
    or None."""
    blocked = set(branches) | set(used_interiors)
    new_paths = []
    new_interiors = set()
    for b in branches:
        if b == candidate:
            continue
        # find path candidate -> b
        path = None
        for attempt in range(6):
            p = _bfs_path(adj, candidate, b, blocked, seed + attempt * 1000 + b)
            if p is not None:
                path = p
                break
        if path is None:
            return None
        interiors = set(path[1:-1])
        if interiors & new_interiors:
            return None
        if interiors & blocked:
            return None
        new_paths.append(path)
        new_interiors |= interiors
    return new_paths

def _score_solution(branches, paths):
    return len(branches)

def construct():
    adj = _build_adj()
    n = P['n']
    best_branches = None
    best_paths = None
    best_score = 0

    rng = random.Random(12345)

    # Try multiple seeds for initial clique and augmentation order
    for trial in range(40):
        seed = trial * 7919 + 13
        # initial clique
        clique = _greedy_max_clique(adj, n, seed)
        if len(clique) < 2:
            # fallback to an edge
            u, v = P['edges'][0]
            clique = [u, v]
        branches = list(clique)
        used_interiors = set()
        paths = {}
        # paths between existing branches: use direct edge if exists else BFS
        for i in range(len(branches)):
            for j in range(i + 1, len(branches)):
                a, b = branches[i], branches[j]
                if b in adj[a]:
                    p = [a, b]
                else:
                    p = _bfs_path(adj, a, b, set(branches), seed)
                    if p is None:
                        p = [a, b]  # shouldn't happen for clique
                paths[(a, b)] = p
                paths[(b, a)] = p
                used_interiors |= set(p[1:-1])

        # iteratively add branches
        added = True
        while added:
            added = False
            candidates = [v for v in range(n) if v not in branches and v not in used_interiors]
            # score candidates by sum of distances to branches
            scored = []
            for v in candidates:
                total = 0
                ok = True
                for b in branches:
                    p = _bfs_path(adj, v, b, set(branches) | used_interiors, seed)
                    if p is None:
                        ok = False
                        break
                    total += len(p)
                if ok:
                    scored.append((total, v))
            scored.sort()
            for total, v in scored:
                new_paths = _try_add_branch(adj, branches, used_interiors, v, seed)
                if new_paths is not None:
                    # commit
                    branches.append(v)
                    for b, p in zip(branches[:-1], new_paths):
                        # ensure path endpoints are v and b
                        if p[0] != v or p[-1] != b:
                            p = list(p)
                            if p[0] != v:
                                p.reverse()
                        paths[(v, b)] = p
                        paths[(b, v)] = list(p)
                        used_interiors |= set(p[1:-1])
                    added = True
                    break

        sc = _score_solution(branches, paths)
        if sc > best_score:
            best_score = sc
            best_branches = list(branches)
            best_paths = dict(paths)

    # Build final paths dict as list for unordered pairs
    if best_branches is None:
        u, v = P['edges'][0]
        best_branches = [u, v]
        best_paths = {(u, v): [u, v], (v, u): [u, v]}

    # Construct 'paths' as one simple path for each unordered pair of branches
    final_paths = []
    for i in range(len(best_branches)):
        for j in range(i + 1, len(best_branches)):
            a, b = best_branches[i], best_branches[j]
            p = best_paths.get((a, b)) or best_paths.get((b, a))
            if p is None:
                p = [a, b]
            # ensure endpoints
            if p[0] != a or p[-1] != b:
                p = list(p)
                if p[0] != a:
                    p.reverse()
            final_paths.append(list(p))

    # Convert to plain Python types
    branches_out = [int(v) for v in best_branches]
    paths_out = [[int(x) for x in p] for p in final_paths]

    return {'branches': branches_out, 'paths': paths_out}
# EVOLVE-BLOCK-END
