import numpy as np
import time
import sys

def kissing_number11() -> np.ndarray:
    """
    Finds a large set of integer points in 11 dimensions such that
    min pairwise squared distance >= max squared norm.
    Uses a coloring-based branch-and-bound Maximum Independent Set solver
    on the conflict graph of candidate integer vectors.
    """
    start_time = time.time()
    DEADLINE = 570.0  # safety margin before 600s limit

    def elapsed():
        return time.time() - start_time

    # Fallback safe result: 2 points
    def fallback():
        return np.array([[1,0,0,0,0,0,0,0,0,0,0],
                         [-1,0,0,0,0,0,0,0,0,0,0]], dtype=np.int64)

    d = 11

    # ---- Generate candidate pool: integer vectors with small coordinates ----
    # We want many candidates. Use coordinates in {-2,-1,0,1,2} with squared
    # norm up to some bound. All chosen points will share the same norm ideally,
    # but we allow the solver to pick any independent set; we then verify.
    # To keep the pool manageable but large, generate vectors with norm^2 in a
    # small set of values.
    #
    # Strategy: pick a target norm^2 = N. Points all with norm^2 = N.
    # Conflict: dist^2 < N  <=>  2N - 2*dot < N  <=>  dot > N/2.
    #
    # We'll try several norms and take the best independent set found.

    best_points = None
    best_k = 0

    from itertools import product

    # Candidate coordinate values
    coord_vals = [-2, -1, 0, 1, 2]

    # Generate all vectors with squared norm exactly equal to a target,
    # for a few target norms, using bounded search.
    # We'll generate by enumerating support patterns.

    def gen_vectors_with_norm2(target_n2, max_coord=2):
        """Generate all integer vectors in Z^d with each coord in [-max_coord,max_coord]
        and squared norm exactly target_n2."""
        vecs = []
        # recursive generation over coordinates
        vals = list(range(-max_coord, max_coord + 1))
        # prune: remaining coordinates must be able to fill the rest
        def rec(idx, remaining, cur):
            if idx == d:
                if remaining == 0:
                    vecs.append(tuple(cur))
                return
            # remaining must be achievable by (d-idx) coords each contributing 0..max_coord^2
            if remaining < 0:
                return
            if remaining > (d - idx) * (max_coord * max_coord):
                return
            for v in vals:
                cur[idx] = v
                rec(idx + 1, remaining - v * v, cur)
        rec(0, target_n2, [0] * d)
        return vecs

    # Try target norms that yield large pools.
    # norm2 = 8: coords in {-2,-1,0,1,2} -> many combos
    # We'll attempt a few and keep the best independent set.
    candidate_norms = [8, 9, 10, 12, 16]

    for target_n2 in candidate_norms:
        if elapsed() > DEADLINE:
            break
        if best_k >= 400:  # already great
            break
        try:
            vecs = gen_vectors_with_norm2(target_n2, max_coord=2)
        except Exception:
            continue
        if len(vecs) < 2:
            continue
        # Deduplicate
        vecs = list(set(vecs))
        n = len(vecs)
        if n < 2:
            continue
        A = np.array(vecs, dtype=np.int64)  # (n, d)

        # Build conflict adjacency: edge if dist^2 < target_n2
        # dist^2 = 2*target_n2 - 2*dot.  < target_n2  <=>  dot > target_n2/2
        # dot is integer, so dot >= floor(target_n2/2)+1
        threshold = target_n2 // 2  # dot > target_n2/2
        # dot must be > target_n2/2, i.e. dot >= floor(target_n2/2)+1
        dot_min_conflict = threshold + 1 if (target_n2 % 2 == 0) else target_n2 // 2 + 1
        # More simply: conflict if dot * 2 > target_n2
        # Compute dot products
        # For memory, n could be large. Compute in blocks.
        adj = [None] * n
        for i in range(n):
            if elapsed() > DEADLINE:
                break
            # dot of A[i] with all
            dots = A @ A[i]  # (n,)
            conflict = np.where((dots * 2) > target_n2)[0]
            # exclude self
            conflict = conflict[conflict != i]
            adj[i] = conflict
        if elapsed() > DEADLINE:
            break

        # Now solve Maximum Independent Set via coloring branch-and-bound.
        # Independent set in conflict graph = set of mutually non-conflicting points.
        # We want a large independent set.
        # Use the classical branch-and-bound (Tomita) algorithm.

        # Build neighbor sets as Python sets for speed
        neigh = [set(adj[i].tolist()) for i in range(n)]

        # Tomita's MCS / BB-MCS with coloring.
        # We'll implement a standard branch-and-bound.

        best_iset = []
        best_size = 0

        # color the graph to get an upper bound
        # We'll do greedy coloring of the candidate set R at each node.

        # Implement recursive search with bitsets for speed.
        # Use Python ints as bitsets for the candidate set.
        # Precompute neighbor bitsets.
        neigh_bits = [0] * n
        for i in range(n):
            b = 0
            for j in adj[i].tolist():
                b |= (1 << j)
            neigh_bits[i] = b

        # Greedy coloring on a set of vertices (as list) returns colors and max color.
        # We color in the order of a greedy coloring (Largest First not necessary).
        # Standard: color sequentially, assign smallest color not used by colored neighbors.
        # We'll color the candidate set R (list of vertices) and produce a color
        # ordering: vertices grouped by color, processed in reverse color order.

        sys.setrecursionlimit(1000000)

        # We'll implement BB with a stack to avoid deep recursion issues,
        # but recursion is fine with high limit.

        # To keep it simple and fast, use the well-known BB-MCS:
        #   color(R) -> list of color classes; then branch on vertices in
        #   reverse color-class order.

        def color_set(vertices):
            """vertices: list of ints. Returns list of color classes (lists),
            colored greedily."""
            color_class = []
            colored = {}
            # order vertices by descending degree within R for better bounds
            # (Largest First heuristic)
            Rset = set(vertices)
            deg = {}
            for v in vertices:
                deg[v] = len(neigh[v] & Rset)
            # sort by degree descending
            ordered = sorted(vertices, key=lambda v: -deg[v])
            for v in ordered:
                # find smallest color not used by colored neighbors
                used = set()
                for nb in neigh[v]:
                    if nb in colored:
                        used.add(colored[nb])
                c = 0
                while c in used:
                    c += 1
                colored[v] = c
                if c == len(color_class):
                    color_class.append([])
                color_class[c].append(v)
            return color_class

        # Global best
        best_iset = []
        best_size = 0

        def branch(R_list, P_list):
            """R_list: list of vertices in candidate set (to be colored/branched).
               P_list: list of vertices already in the independent set (partial)."""
            nonlocal best_iset, best_size
            if elapsed() > DEADLINE:
                return
            if len(R_list) == 0:
                if len(P_list) > best_size:
                    best_size = len(P_list)
                    best_iset = list(P_list)
                return
            if len(P_list) + len(R_list) <= best_size:
                return
            # Color R
            color_class = color_set(R_list)
            # Process color classes from last to first
            for ci in range(len(color_class) - 1, -1, -1):
                if elapsed() > DEADLINE:
                    return
                if len(P_list) + ci + 1 <= best_size:
                    break  # no point continuing
                for v in color_class[ci]:
                    if v not in R_list:
                        continue
                    # branch: include v
                    P_list.append(v)
                    # new R = R_list ∩ N(v)  (vertices compatible with v)
                    new_R = [u for u in R_list if u != v and u not in neigh[v]]
                    branch(new_R, P_list)
                    P_list.pop()
                    # remove v from R_list for subsequent branches
                    R_list.remove(v)
                    if len(P_list) + len(R_list) <= best_size:
                        break

        # Initial call: R = all vertices, P = []
        # This can be huge; use the coloring bound to prune.
        # For very large n, this may be slow. We rely on deadline guard.
        all_vertices = list(range(n))
        branch(all_vertices, [])

        if best_size > best_k:
            best_k = best_size
            best_points = A[best_iset]

    if best_points is None or best_k < 2:
        return fallback()

    pts = np.around(best_points).astype(np.int64)

    # Final verification
    k = len(pts)
    if k < 2:
        return fallback()
    if pts.shape[1] != 11:
        return fallback()
    if np.any(np.all(pts == 0, axis=1)):
        return fallback()
    norms = np.sum(pts ** 2, axis=1)
    max_norm = int(np.max(norms))
    min_dist = float('inf')
    for i in range(k):
        for j in range(i + 1, k):
            diff = pts[i] - pts[j]
            ds = int(np.dot(diff, diff))
            if ds < min_dist:
                min_dist = ds
    if min_dist < max_norm:
        return fallback()

    return pts
