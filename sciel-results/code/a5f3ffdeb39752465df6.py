"""Improved search for minimum maximal no-isosceles set on N x N grid."""
N = 16
import time
import random
import itertools
import numpy as np


def _d2(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def _is_iso(a, b, c):
    d1 = _d2(a, b)
    d2_ = _d2(a, c)
    d3 = _d2(b, c)
    return d1 == d2_ or d1 == d3 or d2_ == d3


def _can_add_np(p, chosen_np):
    """Vectorized check if point p can be added to chosen (numpy array) without creating isosceles triangle."""
    if len(chosen_np) == 0:
        return True
    # Compute distances from p to all chosen points
    dx = p[0] - chosen_np[:, 0]
    dy = p[1] - chosen_np[:, 1]
    d_to_p = dx * dx + dy * dy  # shape (n,)
    
    # Compute all pairwise distances among chosen points
    # d2[i,j] = distance^2 between chosen[i] and chosen[j]
    dx_chosen = chosen_np[:, 0][:, None] - chosen_np[:, 0][None, :]
    dy_chosen = chosen_np[:, 1][:, None] - chosen_np[:, 1][None, :]
    d_chosen = dx_chosen * dx_chosen + dy_chosen * dy_chosen  # shape (n, n)
    
    # For each pair (i, j) with i < j, check if triangle (p, chosen[i], chosen[j]) is isosceles
    # d1 = d_to_p[i], d2 = d_to_p[j], d3 = d_chosen[i,j]
    # Isosceles if d1 == d2 or d1 == d3 or d2 == d3
    
    n = len(chosen_np)
    # Create index arrays for upper triangle
    i_idx, j_idx = np.triu_indices(n, k=1)
    if len(i_idx) == 0:
        return True
    
    d1 = d_to_p[i_idx]  # distance from p to chosen[i]
    d2 = d_to_p[j_idx]  # distance from p to chosen[j]
    d3 = d_chosen[i_idx, j_idx]  # distance between chosen[i] and chosen[j]
    
    # Check for any isosceles condition
    if np.any(d1 == d2) or np.any(d1 == d3) or np.any(d2 == d3):
        return False
    return True


def _can_add(p, chosen):
    """Check if point p can be added to chosen without creating isosceles triangle."""
    if not chosen:
        return True
    chosen_np = np.array(chosen, dtype=np.int32)
    return _can_add_np(p, chosen_np)


def _is_maximal(chosen, all_points):
    """Check if no point can be added."""
    chosen_set = set(chosen)
    for p in all_points:
        if p in chosen_set:
            continue
        if _can_add(p, chosen):
            return False
    return True


def _is_valid_no_iso(chosen):
    """Verify no three points form isosceles triangle."""
    if len(chosen) < 3:
        return True
    n = len(chosen)
    # Precompute squared distances
    d2 = [[0]*n for _ in range(n)]
    for i in range(n):
        xi, yi = chosen[i]
        for j in range(i+1, n):
            xj, yj = chosen[j]
            dist = (xi - xj)**2 + (yi - yj)**2
            d2[i][j] = dist
            d2[j][i] = dist
            
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                d_ij = d2[i][j]
                d_ik = d2[i][k]
                d_jk = d2[j][k]
                if d_ij == d_ik or d_ij == d_jk or d_ik == d_jk:
                    return False
    return True


def _greedy(order, chosen=None):
    """Greedy construction following given point order."""
    if chosen is None:
        chosen = []
    else:
        chosen = list(chosen)
    chosen_set = set(chosen)
    chosen_np = np.array(chosen, dtype=np.int32) if chosen else np.empty((0, 2), dtype=np.int32)
    
    for p in order:
        if p not in chosen_set:
            if _can_add_np(p, chosen_np):
                chosen.append(p)
                chosen_set.add(p)
                chosen_np = np.vstack([chosen_np, np.array(p, dtype=np.int32)])
    return chosen


def _construct_maximal(chosen, all_points):
    """Add points greedily until maximal."""
    chosen = list(chosen)
    chosen_set = set(chosen)
    chosen_np = np.array(chosen, dtype=np.int32) if chosen else np.empty((0, 2), dtype=np.int32)
    changed = True
    while changed:
        changed = False
        for p in all_points:
            if p not in chosen_set:
                if _can_add_np(p, chosen_np):
                    chosen.append(p)
                    chosen_set.add(p)
                    chosen_np = np.vstack([chosen_np, np.array(p, dtype=np.int32)])
                    changed = True
    return chosen


def _remove_and_remaximize(current, all_points, idx):
    """Remove point at idx, then make maximal again. Return new set if valid."""
    new_set = list(current)
    new_set.pop(idx)
    
    # Check validity (no isosceles) - removing a point can't create new isosceles triangles
    # so this is always valid if current was valid
    
    # Make maximal
    extended = _construct_maximal(new_set, all_points)
    return extended


def _local_search(all_points, start_set, max_iters=5000, deadline=None):
    """Local search: try removing each point and re-maximizing, keep if better."""
    current = list(start_set)
    best = list(current)
    best_len = len(current)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        for idx in range(len(current)):
            if deadline is not None and time.time() > deadline:
                break
            extended = _remove_and_remaximize(current, all_points, idx)
            if extended is not None and len(extended) < len(current):
                current = extended
                improved = True
                break
        
        if not improved:
            break
        
        if len(current) < best_len:
            best = list(current)
            best_len = len(current)
    
    return best


def _simulated_annealing(all_points, start_set, max_iters=20000, deadline=None):
    """
    Simulated annealing to minimize the size of a maximal no-isosceles set.
    Moves: remove a point, then try to re-maximize.
    """
    current = list(start_set)
    best = list(start_set)
    best_len = len(current)
    
    T = 2.0
    T_min = 0.01
    cooling = 0.999
    
    rng = random.Random(12345)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        if len(current) < best_len:
            best = list(current)
            best_len = len(current)
        
        if T < T_min:
            break
        
        # Random move: try to remove a random point and re-maximize
        if current:
            idx = rng.randint(0, len(current) - 1)
            extended = _remove_and_remaximize(current, all_points, idx)
            
            if extended is not None:
                delta = len(extended) - len(current)
                if delta <= 0 or rng.random() < (1.0 - delta) / T:
                    current = extended
        
        T *= cooling
    
    return best


def _backtracking_search(all_points, best_len, max_nodes=200000, deadline=None):
    """
    Backtracking search with strong pruning.
    Tries to find a solution smaller than best_len.
    """
    best = [None]
    best_len_ref = [best_len]
    nodes = [0]
    
    def _can_add_fast(p, chosen_np):
        return _can_add_np(p, chosen_np)
    
    def _is_maximal_fast(chosen, chosen_np):
        chosen_set = set(chosen)
        for p in all_points:
            if p not in chosen_set:
                if _can_add_np(p, chosen_np):
                    return False
        return True
    
    def _search(chosen, chosen_np, remaining):
        if deadline is not None and time.time() > deadline:
            return
        if nodes[0] >= max_nodes:
            return
        nodes[0] += 1
        
        if len(chosen) >= best_len_ref[0]:
            return
        
        if not remaining:
            if _is_maximal_fast(chosen, chosen_np):
                if len(chosen) < best_len_ref[0]:
                    best_len_ref[0] = len(chosen)
                    best[0] = list(chosen)
            return
        
        p = remaining[0]
        rest = remaining[1:]
        
        # Option 1: include p if valid
        if _can_add_np(p, chosen_np):
            new_chosen_np = np.vstack([chosen_np, np.array(p, dtype=np.int32)]) if len(chosen_np) > 0 else np.array([p], dtype=np.int32)
            _search(chosen + [p], new_chosen_np, rest)
        
        # Option 2: exclude p
        _search(chosen, chosen_np, rest)
    
    # Use a heuristic ordering
    pts = list(all_points)
    pts.sort(key=lambda p: (p[0] + p[1], p[0]))
    
    empty_np = np.empty((0, 2), dtype=np.int32)
    _search([], empty_np, pts)
    
    return best[0] if best[0] is not None else None


def _diversified_greedy(all_points, rng, num_orderings=5):
    """Generate multiple diverse greedy orderings and return the best maximal set."""
    best = None
    best_len = float('inf')
    
    for _ in range(num_orderings):
        pts = all_points[:]
        rng.shuffle(pts)
        chosen = _greedy(pts)
        extended = _construct_maximal(chosen, all_points)
        if len(extended) < best_len:
            best = extended
            best_len = len(extended)
    
    return best


def construct():
    start_time = time.time()
    deadline = start_time + 280  # safety margin before 300s limit
    
    all_points = list(itertools.product(range(N), repeat=2))
    
    best = None
    best_len = float('inf')
    
    # Phase 1: Quick greedy constructions with various deterministic orderings
    orderings = [
        ("row_major", list(itertools.product(range(N), repeat=2))),
        ("col_major", sorted(all_points, key=lambda p: (p[1], p[0]))),
        ("diag", sorted(all_points, key=lambda p: (p[0] + p[1], p[0]))),
        ("checker", [p for p in all_points if (p[0] + p[1]) % 2 == 0] +
                   [p for p in all_points if (p[0] + p[1]) % 2 == 1]),
        ("anti_diag", sorted(all_points, key=lambda p: (abs(p[0] - p[1]), p[0]))),
    ]
    
    for name, order in orderings:
        if time.time() > deadline:
            break
        chosen = _greedy(order)
        extended = _construct_maximal(chosen, all_points)
        if len(extended) < best_len:
            best = extended
            best_len = len(extended)
    
    # Phase 2: Backtracking search to find smaller solution
    if best is not None and time.time() < deadline - 30:
        bt_result = _backtracking_search(all_points, best_len, max_nodes=100000,
                                         deadline=deadline - 30)
        if bt_result is not None and len(bt_result) < best_len:
            best = bt_result
            best_len = len(best)
    
    # Phase 3: Local search from best
    if best is not None and time.time() < deadline - 20:
        local_best = _local_search(all_points, best, max_iters=20000,
                                   deadline=deadline - 20)
        if len(local_best) < best_len:
            best = local_best
            best_len = len(best)
    
    # Phase 4: Simulated annealing
    if best is not None and time.time() < deadline - 10:
        sa_best = _simulated_annealing(all_points, best, max_iters=30000,
                                       deadline=deadline - 10)
        if len(sa_best) < best_len:
            best = sa_best
            best_len = len(best)
    
    # Phase 5: Randomized restarts with local search
    rng = random.Random(42)
    attempts = 0
    while time.time() < deadline - 5:
        attempts += 1
        
        # Diversified greedy
        extended = _diversified_greedy(all_points, rng, num_orderings=3)
        if extended is not None and len(extended) < best_len:
            best = extended
            best_len = len(best)
        
        # Occasionally do local search
        if attempts % 5 == 0 and time.time() < deadline - 10:
            local = _local_search(all_points, extended, max_iters=3000,
                                  deadline=deadline - 10)
            if len(local) < best_len:
                best = local
                best_len = len(best)
    
    # Phase 6: Final polish - try more local search iterations
    if best is not None and time.time() < deadline - 2:
        final = _local_search(all_points, best, max_iters=10000,
                              deadline=deadline - 2)
        if len(final) < best_len:
            best = final
            best_len = len(final)
    
    # Final verification and fallback
    if best is None:
        # Fallback to a simple greedy construction if no result was found
        best = _construct_maximal(_greedy(all_points), all_points)
        
    if not _is_valid_no_iso(best):
        best = _construct_maximal(_greedy(all_points), all_points)
        
    if not _is_maximal(best, all_points):
        best = _construct_maximal(best, all_points)
    
    return best
