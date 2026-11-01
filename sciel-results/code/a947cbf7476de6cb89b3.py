import numpy as np
import time
from itertools import combinations

N = 32

def _build_candidates(resolution=40, radii_per_cell=5):
    """
    Generate candidate circles on a grid.
    For each grid position, try several radii.
    A candidate is (x, y, r) where the circle fits in [0,1]^2.
    """
    candidates = []
    # Grid positions
    xs = np.linspace(0.5 / resolution, 1.0 - 0.5 / resolution, resolution)
    ys = np.linspace(0.5 / resolution, 1.0 - 0.5 / resolution, resolution)
    
    # Radii to try: from small to max possible at that position
    # We'll use a set of discrete radii
    r_values = np.linspace(0.01, 0.25, radii_per_cell)
    
    for x in xs:
        for y in ys:
            max_r_here = min(x, 1.0 - x, y, 1.0 - y)
            for r in r_values:
                if r <= max_r_here + 1e-9:
                    candidates.append((x, y, r))
    
    return np.array(candidates)


def _build_conflict_matrix(candidates, tol=1e-9):
    """
    Build a boolean conflict matrix: conflict[i,j] = True if candidates i and j overlap.
    Two circles overlap if dist(center_i, center_j) < r_i + r_j - tol.
    """
    n = len(candidates)
    # Vectorized distance computation in chunks to save memory
    conflict = np.zeros((n, n), dtype=bool)
    
    # Process in blocks to limit memory
    block_size = 500
    for start in range(0, n, block_size):
        end = min(start + block_size, n)
        block = candidates[start:end]
        # Distance from each block point to all candidates
        dx = block[:, 0:1] - candidates[None, :, 0]  # (block, n)
        dy = block[:, 1:2] - candidates[None, :, 1]
        dist = np.sqrt(dx**2 + dy**2)
        # Sum of radii
        sum_r = block[:, 2:3] + candidates[None, :, 2]
        # Conflict where dist < sum_r
        conf = dist < (sum_r - tol)
        conflict[start:end, :] = conf
    
    return conflict


def _greedy_select(candidates, conflict, n_select=N, seed=0):
    """
    Greedy: repeatedly pick the highest-radius candidate that doesn't conflict with already-selected.
    """
    rng = np.random.RandomState(seed)
    n = len(candidates)
    radii = candidates[:, 2]
    
    # Shuffle indices by radius (descending) with slight randomness for diversity
    idx_order = np.argsort(-radii)
    # Add small random perturbation to break ties and explore
    perturbed = radii + rng.uniform(0, 0.001, size=n)
    idx_order = np.argsort(-perturbed)
    
    selected = []
    selected_set = set()
    
    for idx in idx_order:
        if len(selected) >= n_select:
            break
        idx = int(idx)
        if idx in selected_set:
            continue
        # Check conflict with all selected
        ok = True
        for s in selected:
            if conflict[idx, s]:
                ok = False
                break
        if ok:
            selected.append(idx)
            selected_set.add(idx)
    
    # If we couldn't select enough, try a different strategy:
    # Allow lower-radius candidates
    if len(selected) < n_select:
        # Try adding from remaining in order of radius
        remaining = [i for i in range(n) if i not in selected_set]
        remaining.sort(key=lambda i: -radii[i])
        for idx in remaining:
            if len(selected) >= n_select:
                break
            ok = True
            for s in selected:
                if conflict[idx, s]:
                    ok = False
                    break
            if ok:
                selected.append(idx)
                selected_set.add(idx)
    
    return selected


def _local_search(candidates, conflict, selected, n_select=N, max_swaps=2000, seed=0):
    """
    Local search: try swapping one selected candidate with one unselected to improve total radius.
    """
    rng = np.random.RandomState(seed)
    selected_set = set(selected)
    current_score = sum(candidates[i, 2] for i in selected)
    best_score = current_score
    best_selected = list(selected)
    
    unselected = [i for i in range(len(candidates)) if i not in selected_set]
    
    for it in range(max_swaps):
        # Randomly pick a selected and an unselected
        if not unselected or not selected:
            break
        i_sel = rng.choice(selected)
        i_unsel = rng.choice(unselected)
        
        # Try the swap: remove i_sel, add i_unsel
        new_set = selected_set - {i_sel} | {i_unsel}
        # Check that i_unsel doesn't conflict with any other selected (excluding i_sel)
        ok = True
        for s in new_set:
            if s != i_unsel and conflict[i_unsel, s]:
                ok = False
                break
        if not ok:
            continue
        
        new_score = current_score - candidates[i_sel, 2] + candidates[i_unsel, 2]
        if new_score > best_score + 1e-9:
            # Accept
            selected_set = new_set
            selected = list(selected_set)
            current_score = new_score
            best_score = new_score
            best_selected = list(selected)
            # Update unselected
            unselected = [i for i in range(len(candidates)) if i not in selected_set]
    
    return best_selected, best_score


def _local_search_multi(candidates, conflict, selected, n_select=N, max_swaps=3000, seed=0):
    """
    More aggressive local search: try all single swaps in each iteration.
    """
    rng = np.random.RandomState(seed)
    selected_set = set(selected)
    current_score = sum(candidates[i, 2] for i in selected)
    best_score = current_score
    best_selected = list(selected)
    
    unselected = [i for i in range(len(candidates)) if i not in selected_set]
    
    improved = True
    while improved and max_swaps > 0:
        improved = False
        # Try each unselected against each selected (but limit for speed)
        # Sample a subset for speed
        if len(unselected) > 200:
            sample_unsel = rng.choice(unselected, size=200, replace=False)
        else:
            sample_unsel = unselected
        
        if len(selected) > 100:
            sample_sel = rng.choice(selected, size=100, replace=False)
        else:
            sample_sel = selected
        
        for i_unsel in sample_unsel:
            for i_sel in sample_sel:
                if i_unsel == i_sel:
                    continue
                new_set = selected_set - {i_sel} | {i_unsel}
                # Check conflict
                ok = True
                for s in new_set:
                    if s != i_unsel and conflict[i_unsel, s]:
                        ok = False
                        break
                if not ok:
                    continue
                
                new_score = current_score - candidates[i_sel, 2] + candidates[i_unsel, 2]
                if new_score > best_score + 1e-9:
                    selected_set = new_set
                    selected = list(selected_set)
                    current_score = new_score
                    best_score = new_score
                    best_selected = list(selected)
                    improved = True
                    max_swaps -= 1
                    break
            if improved:
                unselected = [i for i in range(len(candidates)) if i not in selected_set]
                if max_swaps <= 0:
                    break
    
    return best_selected, best_score


def _is_valid(xy, r, tol=1e-9):
    n = len(r)
    if np.any(r < -tol):
        return False
    if np.any(xy[:, 0] - r < -tol) or np.any(1 - xy[:, 0] - r < -tol):
        return False
    if np.any(xy[:, 1] - r < -tol) or np.any(1 - xy[:, 1] - r < -tol):
        return False
    dx = xy[:, 0][:, None] - xy[:, 0][None, :]
    dy = xy[:, 1][:, None] - xy[:, 1][None, :]
    dist = np.sqrt(dx**2 + dy**2)
    min_dist = r[:, None] + r[None, :]
    iu = np.triu_indices(n, k=1)
    if np.any(dist[iu] < min_dist[iu] - tol):
        return False
    return True


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Build candidate set
    # Use a moderate resolution to keep candidate count manageable
    # 30x30 grid with 5 radii = 4500 candidates
    # Conflict matrix: 4500x4500 = 20M booleans = 20MB, OK
    resolution = 30
    radii_per_cell = 6
    candidates = _build_candidates(resolution, radii_per_cell)
    n_cand = len(candidates)
    
    # Build conflict matrix
    conflict = _build_conflict_matrix(candidates)
    
    best_result = None
    best_score = -np.inf
    
    # Multiple greedy + local search restarts
    n_restarts = 50
    for restart in range(n_restarts):
        if time.time() > deadline - 10:
            break
        
        seed = restart * 7 + 1
        selected = _greedy_select(candidates, conflict, N, seed=seed)
        
        if len(selected) < N:
            continue
        
        # Local search to improve
        selected, score = _local_search_multi(candidates, conflict, selected, N, 
                                               max_swaps=2000, seed=seed)
        
        if score > best_score:
            best_score = score
            best_result = selected[:]
        
        # Periodic check
        if restart % 10 == 9 and time.time() > deadline - 30:
            break
    
    # If we have a result, convert to output format
    if best_result is not None:
        result = candidates[best_result]
        xy = result[:, 0:2]
        r = result[:, 2]
        
        # Validate
        if not _is_valid(xy, r):
            # Try to fix by shrinking
            for scale in np.linspace(1.0, 0.5, 20):
                r_test = r * scale
                if _is_valid(xy, r_test):
                    r = r_test
                    break
        
        return np.column_stack([xy, r])
    
    # Fallback: simple grid
    side = 6
    xy = np.zeros((N, 2))
    r = np.zeros(N)
    for i in range(N):
        row = i // side
        col = i % side
        xy[i] = [(col + 0.5) / side, (row + 0.5) / side]
        r[i] = 0.5 / side - 0.005
    return np.column_stack([xy, r])
