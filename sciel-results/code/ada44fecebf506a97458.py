import numpy as np
import time
from itertools import combinations

def kissing_number11() -> np.ndarray:
    d = 11
    start_time = time.time()
    deadline = start_time + 620  # safety margin under 630s wall-clock

    def two_nonzero_vectors():
        """All vectors with exactly two non-zero entries, each ±1 (norm 2)."""
        vecs = []
        for i, j in combinations(range(d), 2):
            for s1 in (1, -1):
                for s2 in (1, -1):
                    v = [0] * d
                    v[i] = s1
                    v[j] = s2
                    vecs.append(tuple(v))
        return vecs

    def hopcroft_karp(adj, n_left, n_right):
        """Max cardinality matching. adj[u] = list of right-vertices adjacent to u."""
        from collections import deque
        pair_u = [-1] * n_left
        pair_v = [-1] * n_right
        dist = [0] * n_left

        def bfs():
            q = deque()
            for u in range(n_left):
                if pair_u[u] == -1:
                    dist[u] = 0
                    q.append(u)
                else:
                    dist[u] = -1
            found = False
            while q:
                u = q.popleft()
                for v in adj[u]:
                    pu = pair_v[v]
                    if pu == -1:
                        found = True
                    elif dist[pu] == -1:
                        dist[pu] = dist[u] + 1
                        q.append(pu)
            return found

        def dfs(u):
            for v in adj[u]:
                pu = pair_v[v]
                if pu == -1 or (dist[pu] == dist[u] + 1 and dfs(pu)):
                    pair_u[u] = v
                    pair_v[v] = u
                    return True
            dist[u] = -1
            return False

        matching = 0
        while bfs():
            for u in range(n_left):
                if pair_u[u] == -1 and dfs(u):
                    matching += 1
        return matching

    def independent_set_from_split(adj, n_left, n_right):
        """Largest independent set in a split graph = all right + (left - min vertex cover)."""
        from collections import deque
        # Hopcroft-Karp
        pair_u = [-1] * n_left
        pair_v = [-1] * n_right
        dist = [0] * n_left

        def bfs():
            q = deque()
            for u in range(n_left):
                if pair_u[u] == -1:
                    dist[u] = 0
                    q.append(u)
                else:
                    dist[u] = -1
            found = False
            while q:
                u = q.popleft()
                for v in adj[u]:
                    pu = pair_v[v]
                    if pu == -1:
                        found = True
                    elif dist[pu] == -1:
                        dist[pu] = dist[u] + 1
                        q.append(pu)
            return found

        def dfs(u):
            for v in adj[u]:
                pu = pair_v[v]
                if pu == -1 or (dist[pu] == dist[u] + 1 and dfs(pu)):
                    pair_u[u] = v
                    pair_v[v] = u
                    return True
            dist[u] = -1
            return False

        while bfs():
            for u in range(n_left):
                if pair_u[u] == -1 and dfs(u):
                    pass

        # Min vertex cover from max matching (Kőnig)
        matched_u = [False] * n_left
        for u in range(n_left):
            if pair_u[u] != -1:
                matched_u[u] = True
        Z = [False] * n_left
        q = deque()
        for u in range(n_left):
            if not matched_u[u]:
                Z[u] = True
                q.append(u)
        while q:
            u = q.popleft()
            for v in adj[u]:
                pu = pair_v[v]
                if pu != -1 and not Z[pu]:
                    Z[pu] = True
                    q.append(pu)
        # vertex cover = (left - Z) ∪ (right ∩ matched)
        cover_left = [False] * n_left
        for u in range(n_left):
            cover_left[u] = matched_u[u] and not Z[u]
        cover_right = [False] * n_right
        for v in range(n_right):
            cover_right[v] = pair_v[v] != -1
        # independent set = all right not in cover + left not in cover
        iset = []
        for u in range(n_left):
            if not cover_left[u]:
                iset.append(u)
        for v in range(n_right):
            if not cover_right[v]:
                iset.append(n_left + v)
        return iset

    def solve_independent_set(conflict_mask):
        """conflict_mask: n x n symmetric bool. Return list of indices of a large independent set."""
        n = conflict_mask.shape[0]
        if n == 0:
            return []
        if n == 1:
            return [0]

        # Spectral/degree-based ordering: order vertices by (degree, then eigenvalue proxy)
        deg = conflict_mask.sum(axis=1)
        # Approximate eigenvector via power iteration on adjacency for a better split
        A = conflict_mask.astype(np.float64)
        vec = np.ones(n, dtype=np.float64)
        vec /= (np.linalg.norm(vec) or 1.0)
        for _ in range(30):
            vec = A @ vec
            nn = np.linalg.norm(vec)
            if nn == 0:
                break
            vec /= nn
        # Combine: low degree AND low |spectral coord| preferred as "right" (independent-friendly)
        score = deg + 0.001 * np.abs(vec)
        order = np.argsort(score)  # ascending
        n_left = n // 2
        left_set = set(int(i) for i in order[:n_left])
        left_pos = {v: p for p, v in enumerate(order[:n_left])}
        right_pos = {v: p for p, v in enumerate(order[n_left:])}

        adj = [[] for _ in range(n_left)]
        for u in range(n_left):
            u_idx = int(order[u])
            row = conflict_mask[u_idx]
            for v_idx in np.where(row)[0]:
                v_idx = int(v_idx)
                if v_idx in left_set:
                    continue  # edges within left are irrelevant for the split graph
                adj[u].append(right_pos[v_idx])

        iset = independent_set_from_split(adj, n_left, n_right := (n - n_left))
        result = []
        for x in iset:
            if x < n_left:
                result.append(int(order[x]))
            else:
                result.append(int(order[n_left + (x - n_left)]))
        return result

    def local_search(improving, conflict_mask, max_iter=200):
        """Try single swaps to enlarge the independent set."""
        n = conflict_mask.shape[0]
        cur = list(improving)
        cur_set = set(cur)
        size = len(cur)
        for _ in range(max_iter):
            if time.time() > deadline - 10:
                break
            improved = False
            for a in list(cur_set):
                for b in range(n):
                    if b in cur_set:
                        continue
                    # remove a, add b
                    can = True
                    for c in cur_set:
                        if c == a:
                            continue
                        if conflict_mask[b, c]:
                            can = False
                            break
                    if can:
                        # swapping keeps size same; only helps if it enables more later.
                        # accept if it reduces future conflicts (heuristic)
                        newset = (cur_set - {a}) | {b}
                        # count conflicts b has vs newset already 0; count a's removed conflicts
                        if sum(conflict_mask[a].tolist()) > sum(conflict_mask[b].tolist()):
                            cur_set = newset
                            cur = list(cur_set)
                            improved = True
                            break
                if improved or time.time() > deadline - 10:
                    break
            if not improved:
                break
        return cur

    def validate(points):
        if len(points) < 2:
            return False
        if len(points[0]) != 11:
            return False
        for p in points:
            if all(x == 0 for x in p):
                return False
        max_norm = 0
        for p in points:
            nn = sum(x * x for x in p)
            if nn > max_norm:
                max_norm = nn
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                d2 = 0
                for a, b in zip(points[i], points[j]):
                    d2 += (a - b) * (a - b)
                if d2 < max_norm:
                    return False
        return True

    # ---- Build the norm-2 shell (220 vectors) ----
    vecs = two_nonzero_vectors()
    n = len(vecs)
    arr = np.array(vecs, dtype=np.int64)

    # Conflict if squared distance < 2  (i.e. == 0, distinct vectors never 0)
    # dist^2 = 4 - 2*dot ; conflict iff dot > 1  (never)  -> no conflicts, but compute generally
    dots = arr @ arr.T
    dist_sq = 4 - 2 * dots
    conflict_mask = dist_sq < 2  # R=2
    np.fill_diagonal(conflict_mask, False)

    # Solve for a large independent set (here all 220, but run the structural solver)
    idx = solve_independent_set(conflict_mask)
    if len(idx) < 2:
        idx = list(range(n))
    # Local polish
    idx = local_search(idx, conflict_mask)
    idx = list(set(idx))

    points = [tuple(int(x) for x in arr[i]) for i in idx]

    # Exact verification
    if validate(points):
        result = np.array(points, dtype=np.int64)
        result = np.around(result).astype(np.int64)
        return result

    # Fallback: guaranteed valid pair
    return np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int64)

if __name__ == "__main__":
    res = kissing_number11()
    print("Size:", len(res))
