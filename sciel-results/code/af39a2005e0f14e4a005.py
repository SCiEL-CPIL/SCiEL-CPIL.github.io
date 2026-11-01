"""Program for kissing number in dimension 11."""
import time
import numpy as np
import itertools
from collections import defaultdict

D = 11
DEADLINE_FRACTION = 0.85  # safety margin for 120s evaluator (kills at 150s)
TIME_LIMIT = 120.0 * DEADLINE_FRACTION


def _generate_candidates():
    """Generate candidate nonzero integer vectors in R^11."""
    candidates = []

    # Weight-1 vectors: ±e_i  (22 vectors)
    for i in range(D):
        v = [0] * D
        v[i] = 1
        candidates.append(v)
        v[i] = -1
        candidates.append(v)

    # Weight-2 vectors: ±e_i ± e_j  (132 vectors)
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i], v[j] = si, sj
                candidates.append(v)

    # Weight-3 vectors: ±e_i ± e_j ± e_k  (440 vectors)
    for i, j, k in itertools.combinations(range(D), 3):
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    v = [0] * D
                    v[i], v[j], v[k] = si, sj, sk
                    candidates.append(v)

    # Weight-4 vectors: ±e_i ± e_j ± e_k ± e_l  (990 vectors)
    for i, j, k, l in itertools.combinations(range(D), 4):
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    for sl in (1, -1):
                        v = [0] * D
                        v[i], v[j], v[k], v[l] = si, sj, sk, sl
                        candidates.append(v)

    return np.array(candidates, dtype=np.int64)


def _greedy_independent_set(adj, n, start_order=None):
    """Fast greedy independent set using degree ordering."""
    if start_order is None:
        deg = np.array([len(neigh) for neigh in adj])
        order = np.argsort(deg)  # low degree first
    else:
        order = start_order

    selected = set()
    forbidden = set()
    for v in order:
        if v in forbidden:
            continue
        if any(u in selected for u in adj[v]):
            continue
        selected.add(v)
        forbidden.update(adj[v])
    return selected


def _branch_and_bound_mis(adj, n, time_limit, deadline):
    """Branch and bound for maximum independent set with coloring bound."""
    best = [set()]

    def color_bound(vertex_list, adj):
        """Compute greedy coloring upper bound on independent set size."""
        if not vertex_list:
            return 0
        colors = {}
        color_count = 0
        for v in vertex_list:
            used = set()
            for u in adj[v]:
                if u in colors:
                    used.add(colors[u])
            c = 0
            while c in used:
                c += 1
            colors[v] = c
            if c + 1 > color_count:
                color_count = c + 1
        return color_count

    def bbs(candidates, current):
        nonlocal best
        if time.time() > deadline:
            return
        if not candidates:
            if len(current) > len(best[0]):
                best[0] = set(current)
            return

        # Upper bound: |current| + color_bound(candidates)
        ub = len(current) + color_bound(candidates, adj)
        if ub <= len(best[0]):
            return

        # Pick vertex with max degree in candidate subgraph
        cand_list = list(candidates)
        max_deg = -1
        pivot = cand_list[0]
        for v in cand_list:
            d = sum(1 for u in cand_list if u in adj[v])
            if d > max_deg:
                max_deg = d
                pivot = v

        # Branch 1: include pivot
        new_current = current | {pivot}
        new_candidates = candidates - {pivot} - set(adj[pivot])
        bbs(new_candidates, new_current)

        if time.time() > deadline:
            return

        # Branch 2: exclude pivot
        bbs(candidates - {pivot}, current)

    bbs(set(range(n)), set())
    return best[0]


def _construct_leech_based():
    """Construct kissing configuration using Leech lattice-inspired approach.
    
    In dimension 11, the kissing number is 560. We use the known construction
    from the Leech lattice: take all vectors of the D11 lattice with norm 4
    that satisfy certain congruence conditions, plus additional vectors.
    
    The D11 lattice consists of integer vectors with even sum.
    Norm-4 vectors in D11: weight-2 vectors (sum of 2 entries is even if both same sign or opposite)
    and weight-4 vectors.
    
    Actually, let's use a simpler known construction:
    The kissing number of 560 in d=11 comes from:
    - 22 vectors of form ±e_i (weight 1) - but these have norm 1, not 4
    - We need all vectors to have the same norm for equal-radius spheres
    
    Let me reconsider. The standard construction:
    In the D_n lattice (integer vectors with even coordinate sum), the minimal vectors
    have norm 2 (weight-2 vectors ±e_i ± e_j). The kissing number of D_n is 2n(n-1).
    For n=11: 2*11*10 = 220.
    
    But the kissing number in d=11 is 560, which exceeds 220.
    
    The 560 construction uses the Leech lattice projection. Let me use a different approach:
    Take all ±e_i ± e_j ± e_k ± e_l type vectors with appropriate constraints.
    
    Actually, the known optimal construction for k(11)=560 uses:
    560 vectors derived from the Leech lattice. One explicit construction:
    All vectors in Z^11 of form (±1, ±1, ±1, ±1, ±1, ±1, ±1, ±1, ±1, ±1, 0) 
    with even number of -1's, plus their permutations... that's too many.
    
    Let me just use the greedy/optimization approach on a good candidate set.
    """
    pass


def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT

    # Generate candidates: weight 1-4 vectors
    candidates = _generate_candidates()
    n = len(candidates)
    
    # Compute norms
    norms = np.sqrt(np.sum(candidates.astype(np.float64) ** 2, axis=1))
    
    # Max norm is for weight-4 vectors: sqrt(4) = 2
    max_norm = norms.max()
    
    # Two vectors conflict if their distance < max_norm
    # distance^2 = ||v||^2 + ||w||^2 - 2*v·w
    # We need distance >= max_norm, i.e., distance^2 >= max_norm^2
    # So: ||v||^2 + ||w||^2 - 2*v·w >= max_norm^2
    # Conflicts: ||v||^2 + ||w||^2 - 2*v·w < max_norm^2
    
    # Compute dot products
    dots = candidates.astype(np.float64) @ candidates.astype(np.float64).T
    norm_sq = norms ** 2
    
    # Conflict matrix: conflict[i,j] = True if distance(i,j) < max_norm
    # dist_sq[i,j] = norm_sq[i] + norm_sq[j] - 2*dots[i,j]
    dist_sq = norm_sq[:, None] + norm_sq[None, :] - 2.0 * dots
    conflict = dist_sq < (max_norm ** 2 - 1e-9)
    np.fill_diagonal(conflict, False)
    
    # Build adjacency list
    adj = [set() for _ in range(n)]
    for i in range(n):
        neighbors = np.where(conflict[i])[0]
        adj[i] = set(neighbors.tolist())
    
    # First, get a good initial solution via greedy
    deg = np.array([len(adj[i]) for i in range(n)])
    order = np.argsort(deg)
    initial_set = _greedy_independent_set(adj, n, order)
    
    best_set = set(initial_set)
    best_size = len(best_set)
    
    # Try to improve with local search (add/remove swaps)
    if time.time() < deadline:
        # Try adding vertices not in best_set
        for v in range(n):
            if v in best_set:
                continue
            if time.time() > deadline:
                break
            # Check if v can be added
            if not any(u in best_set for u in adj[v]):
                best_set.add(v)
                best_size += 1
    
    # Local search: try swapping one vertex out for multiple in
    if time.time() < deadline:
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for v in list(best_set):
                if time.time() > deadline:
                    break
                # Try removing v and adding all possible neighbors
                new_set = best_set - {v}
                added = 0
                for u in range(n):
                    if u == v or u in new_set:
                        continue
                    if not any(x in new_set for x in adj[u]):
                        new_set.add(u)
                        added += 1
                if len(new_set) > len(best_set):
                    best_set = new_set
                    best_size = len(best_set)
                    improved = True
                    break
    
    # Convert to array
    result_indices = sorted(best_set)
    result = candidates[result_indices]
    
    return result


# Keep the function name as required
def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT

    # Generate candidates
    candidates = _generate_candidates()
    n = len(candidates)

    # Compute norms
    norms = np.sqrt(np.sum(candidates.astype(np.float64) ** 2, axis=1))
    max_norm = norms.max()

    # Conflict: dist(i,j) < max_norm
    dots = candidates.astype(np.float64) @ candidates.astype(np.float64).T
    norm_sq = norms ** 2
    dist_sq = norm_sq[:, None] + norm_sq[None, :] - 2.0 * dots
    conflict = dist_sq < (max_norm ** 2 - 1e-9)
    np.fill_diagonal(conflict, False)

    # Build adjacency
    adj = [set(np.where(conflict[i])[0].tolist()) for i in range(n)]

    # Greedy initial solution (low-degree first)
    deg = np.array([len(adj[i]) for i in range(n)])
    order = np.argsort(deg)
    best_set = _greedy_independent_set(adj, n, order)

    # Greedy add
    for v in range(n):
        if v in best_set:
            continue
        if not any(u in best_set for u in adj[v]):
            best_set.add(v)

    # Local search: single-swap improvement
    if time.time() < deadline:
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for v in list(best_set):
                if time.time() > deadline:
                    break
                new_set = best_set - {v}
                for u in range(n):
                    if u == v or u in new_set:
                        continue
                    if not any(x in new_set for x in adj[u]):
                        new_set.add(u)
                if len(new_set) > len(best_set):
                    best_set = new_set
                    improved = True
                    break

    # Try branch-and-bound on a subset if time permits
    if time.time() < deadline - 5.0 and n <= 500:
        bb_set = _branch_and_bound_mis(adj, n, time.time(), deadline)
        if len(bb_set) > len(best_set):
            best_set = bb_set

    result_indices = sorted(best_set)
    return candidates[result_indices]
