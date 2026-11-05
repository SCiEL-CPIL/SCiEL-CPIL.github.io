"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16
import random
import time
import itertools

# EVOLVE-BLOCK-START

def _precompute_iso():
    """
    Precompute:
    - points: list of (x, y) tuples
    - covers[p]: list of pair indices (a, b) such that (a, b, p) is isosceles
    - pair_to_idx: dict mapping (a, b) -> index
    - pair_covers[pair_idx]: list of points p covered by that pair
    """
    points = list(itertools.product(range(N), repeat=2))
    n_pts = len(points)
    
    # Build list of all unordered pairs
    pairs = []
    pair_to_idx = {}
    for i in range(n_pts):
        for j in range(i+1, n_pts):
            pair_to_idx[(i, j)] = len(pairs)
            pairs.append((i, j))
            
    n_pairs = len(pairs)
    
    # covers[p] = list of pair indices that form isosceles with p
    covers = [[] for _ in range(n_pts)]
    # pair_covers[pair_idx] = list of points p that form isosceles with this pair
    pair_covers = [[] for _ in range(n_pairs)]
    
    # Iterate over all pairs and find points that form isosceles triangles
    for idx, (a, b) in enumerate(pairs):
        ax, ay = points[a]
        bx, by = points[b]
        d_ab_sq = (ax - bx) ** 2 + (ay - by) ** 2
        
        for p_idx in range(n_pts):
            if p_idx == a or p_idx == b:
                continue
            px, py = points[p_idx]
            d_pa_sq = (px - ax) ** 2 + (py - ay) ** 2
            d_pb_sq = (px - bx) ** 2 + (py - by) ** 2
            
            if d_pa_sq == d_pb_sq or d_pa_sq == d_ab_sq or d_pb_sq == d_ab_sq:
                covers[p_idx].append(idx)
                pair_covers[idx].append(p_idx)
                
    # Inverse map: pair_idx -> (a, b)
    pair_to_idx_inv = [None] * n_pairs
    for (a, b), idx in pair_to_idx.items():
        pair_to_idx_inv[idx] = (a, b)
        
    return points, covers, pair_to_idx, pair_covers, pair_to_idx_inv

def _is_independent(S, points):
    """Check if S is an independent set (no isosceles triangles)."""
    S_list = list(S)
    n = len(S_list)
    if n < 3:
        return True
    for i in range(n):
        a = S_list[i]
        ax, ay = points[a]
        for j in range(i+1, n):
            b = S_list[j]
            bx, by = points[b]
            d_ab_sq = (ax - bx) ** 2 + (ay - by) ** 2
            for k in range(j+1, n):
                c = S_list[k]
                cx, cy = points[c]
                d_ac_sq = (cx - ax) ** 2 + (cy - ay) ** 2
                d_bc_sq = (cx - bx) ** 2 + (cy - by) ** 2
                if d_ac_sq == d_bc_sq or d_ac_sq == d_ab_sq or d_bc_sq == d_ab_sq:
                    return False
    return True

def _is_maximal(S, points, covers, pair_to_idx_inv):
    """Check if S is maximal: no point outside S can be added without forming an isosceles."""
    S_set = set(S)
    n_pts = len(points)
    for p in range(n_pts):
        if p in S_set:
            continue
        # Check if p is covered by any pair in S
        covered = False
        for pair_idx in covers[p]:
            a, b = pair_to_idx_inv[pair_idx]
            if a in S_set and b in S_set:
                covered = True
                break
        if not covered:
            return False
    return True

def _greedy_random_maximal(points, n_pts):
    """Greedy construction of a random maximal independent set."""
    S = set()
    remaining = list(range(n_pts))
    random.shuffle(remaining)
    for p in remaining:
        # Check if p can be added (independence)
        if not _is_independent(S | {p}, points):
            continue
        S.add(p)
    return S

def _minimize_by_deletion(S, points, covers, pair_to_idx_inv):
    """Repeatedly remove points if the set remains maximal after removal."""
    S_list = list(S)
    changed = True
    while changed:
        changed = False
        # Try removing each point in random order
        random.shuffle(S_list)
        for v in S_list:
            if v not in S:
                continue
            S_new = S - {v}
            if _is_maximal(S_new, points, covers, pair_to_idx_inv):
                S = S_new
                S_list.remove(v)
                changed = True
                break  # Restart loop since set changed
    return S

def _local_search_swap(S, points, covers, pair_to_idx_inv, n_pts, max_swaps=20):
    """Try to swap a point v in S with a point u not in S to escape local minimum."""
    S_set = set(S)
    candidates = [u for u in range(n_pts) if u not in S_set]
    random.shuffle(candidates)
    
    improved = False
    S_list = list(S)
    random.shuffle(S_list)
    
    for v in S_list:
        if len(candidates) == 0:
            break
        for u in candidates[:10]:  # Check only a few candidates
            S_trial = (S - {v}) | {u}
            if _is_independent(S_trial, points) and _is_maximal(S_trial, points, covers, pair_to_idx_inv):
                S = S_trial
                improved = True
                candidates.remove(u)
                break
        if improved:
            break
            
    return S, improved

def construct():
    start_time = time.time()
    deadline = start_time + 290  # Safety margin
    
    points, covers, pair_to_idx, pair_covers, pair_to_idx_inv = _precompute_iso()
    n_pts = len(points)
    
    best_set = None
    best_size = n_pts + 1
    
    random.seed(42)
    
    # Initial construction with multiple random orderings
    for attempt in range(500):
        if time.time() > deadline - 30:
            break
            
        S = _greedy_random_maximal(points, n_pts)
        
        # Verify maximality
        if not _is_maximal(S, points, covers, pair_to_idx_inv):
            continue
            
        if len(S) < best_size:
            best_size = len(S)
            best_set = set(S)
            
        # Local Search: Minimize by deletion
        S = _minimize_by_deletion(S, points, covers, pair_to_idx_inv)
        
        if len(S) < best_size:
            best_size = len(S)
            best_set = set(S)
            
        # Perturbation: Swap
        if time.time() > deadline - 10:
            break
            
        for _ in range(5):
            S, improved = _local_search_swap(S, points, covers, pair_to_idx_inv, n_pts)
            if improved:
                # Continue local search after swap
                S = _minimize_by_deletion(S, points, covers, pair_to_idx_inv)
                if len(S) < best_size:
                    best_size = len(S)
                    best_set = set(S)
            else:
                break
                
    # Final polish: try more deletion orders on the best set
    if best_set is not None and time.time() < deadline - 20:
        trial = set(best_set)
        for _ in range(20):
            if time.time() > deadline - 10:
                break
            trial_new = _minimize_by_deletion(trial, points, covers, pair_to_idx_inv)
            if len(trial_new) < best_size:
                best_size = len(trial_new)
                best_set = trial_new
                trial = best_set
            else:
                # Try a swap
                trial, improved = _local_search_swap(trial, points, covers, pair_to_idx_inv, n_pts)
                if improved:
                    trial = _minimize_by_deletion(trial, points, covers, pair_to_idx_inv)
                    if len(trial) < best_size:
                        best_size = len(trial)
                        best_set = trial
    
    if best_set is None:
        # Fallback
        best_set = _greedy_random_maximal(points, n_pts)
        
    return [points[i] for i in best_set]

# EVOLVE-BLOCK-END
