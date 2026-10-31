import numpy as np
from scipy.optimize import minimize
import time

N = 26

def _check_validity(circles, tol=1e-9):
    """Check if circles are within [0,1]^2 and non-overlapping."""
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]
    
    if np.any(x < -tol) or np.any(x > 1 + tol) or np.any(y < -tol) or np.any(y > 1 + tol):
        return False
    if np.any(r < -tol):
        return False
        
    x_diff = x[:, None] - x[None, :]
    y_diff = y[:, None] - y[None, :]
    dist = np.sqrt(x_diff**2 + y_diff**2)
    r_sum = r[:, None] + r[None, :]
    mask = ~np.eye(N, dtype=bool)
    
    if np.any(dist[mask] < r_sum[mask] - tol):
        return False
    return True

def _max_radius_at(x, y, other_circles):
    """
    Given a center (x,y) and other circles, compute the maximum radius
    such that the new circle doesn't overlap any other and stays in bounds.
    """
    # Boundary constraint
    r_max = min(x, y, 1 - x, 1 - y)
    
    if len(other_circles) > 0:
        ox = other_circles[:, 0]
        oy = other_circles[:, 1]
        or_ = other_circles[:, 2]
        dists = np.sqrt((x - ox)**2 + (y - oy)**2)
        # r <= dist - other_r for each other circle
        r_max = min(r_max, np.min(dists - or_))
    
    return max(0.0, r_max)

def _greedy_place(candidates, existing_circles):
    """
    Greedily place circles from candidates into existing_circles.
    candidates: list of (x, y, preferred_r)
    existing_circles: list of (x, y, r)
    Returns updated existing_circles.
    """
    for cx, cy, pref_r in candidates:
        # Compute max radius at this position given existing circles
        if len(existing_circles) > 0:
            arr = np.array(existing_circles)
            r_max = _max_radius_at(cx, cy, arr)
        else:
            r_max = min(cx, cy, 1 - cx, 1 - cy)
        
        # Use min of preferred and max possible
        r = min(pref_r, r_max)
        
        if r > 1e-6:
            existing_circles.append((cx, cy, r))
    
    return existing_circles

def _local_search(circles, time_limit=5.0):
    """
    Local search: try to grow each circle by repositioning or expanding.
    Uses a simple hill-climbing approach.
    """
    circles = [list(c) for c in circles]
    best_score = sum(c[2] for c in circles)
    start_time = time.time()
    
    while time.time() - start_time < time_limit:
        improved = False
        
        # Try growing each circle
        for i in range(N):
            x, y, r = circles[i]
            others = np.array([c for j, c in enumerate(circles) if j != i])
            
            # Try to expand radius
            new_r = _max_radius_at(x, y, others)
            if new_r > r + 1e-9:
                circles[i] = [x, y, new_r]
                improved = True
        
        # Try moving each circle to a better position
        if not improved:
            for i in range(N):
                x, y, r = circles[i]
                others = np.array([c for j, c in enumerate(circles) if j != i])
                
                # Try small random perturbations
                for _ in range(20):
                    dx = np.random.uniform(-0.1, 0.1)
                    dy = np.random.uniform(-0.1, 0.1)
                    nx, ny = x + dx, y + dy
                    
                    if nx < 0 or nx > 1 or ny < 0 or ny > 1:
                        continue
                    
                    new_r = _max_radius_at(nx, ny, others)
                    if new_r > r + 1e-9:
                        circles[i] = [nx, ny, new_r]
                        improved = True
                        break
        
        if not improved:
            # Try swapping: remove one circle, re-optimize others, re-add
            for i in range(N):
                removed = circles[i]
                remaining = circles[:i] + circles[i+1:]
                
                # Re-optimize remaining circles
                for j in range(len(remaining)):
                    x, y, r = remaining[j]
                    others = np.array([c for k, c in enumerate(remaining) if k != j])
                    new_r = _max_radius_at(x, y, others)
                    if new_r > r + 1e-9:
                        remaining[j] = [x, y, new_r]
                
                # Try to re-add the removed circle with optimal radius
                if len(remaining) > 0:
                    arr = np.array(remaining)
                    new_r = _max_radius_at(removed[0], removed[1], arr)
                else:
                    new_r = min(removed[0], removed[1], 1-removed[0], 1-removed[1])
                
                new_circles = remaining + [[removed[0], removed[1], new_r]]
                new_score = sum(c[2] for c in new_circles)
                
                if new_score > best_score + 1e-9:
                    circles = new_circles
                    best_score = new_score
                    improved = True
        
        if not improved:
            # Try random restarts within local search
            np.random.seed(int(time.time() * 1000) % 10000)
            # Perturb all positions slightly
            for i in range(N):
                x, y, r = circles[i]
                nx = np.clip(x + np.random.normal(0, 0.05), 0.05, 0.95)
                ny = np.clip(y + np.random.normal(0, 0.05), 0.05, 0.95)
                circles[i] = [nx, ny, r]
            
            # Re-optimize
            for _ in range(5):
                for i in range(N):
                    x, y, r = circles[i]
                    others = np.array([c for j, c in enumerate(circles) if j != i])
                    new_r = _max_radius_at(x, y, others)
                    if new_r > r + 1e-9:
                        circles[i] = [x, y, new_r]
            
            new_score = sum(c[2] for c in circles)
            if new_score > best_score:
                best_score = new_score
    
    return np.array(circles)

def _generate_candidates():
    """
    Generate a diverse set of candidate circle positions and preferred radii.
    Uses multiple geometric patterns.
    """
    candidates = []
    
    # 1. Grid-based candidates
    for grid_size in [4, 5, 6]:
        side = grid_size
        for i in range(side):
            for j in range(side):
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r_pref = 0.5 / side * 0.95
                candidates.append((x, y, r_pref))
    
    # 2. Hexagonal packing candidates
    for rows in [4, 5]:
        for j in range(rows):
            for i in range(rows + 1):
                x = (i + 0.5) / rows + (0.5 / rows if j % 2 == 1 else 0)
                y = (j + 0.5) / rows
                if 0 <= x <= 1 and 0 <= y <= 1:
                    r_pref = 0.5 / rows * 0.9
                    candidates.append((x, y, r_pref))
    
    # 3. Corner candidates
    for cx in [0.15, 0.35, 0.65, 0.85]:
        for cy in [0.15, 0.35, 0.65, 0.85]:
            r_pref = min(cx, cy, 1-cx, 1-cy) * 0.95
            candidates.append((cx, cy, r_pref))
    
    # 4. Center candidates
    for r_pref in [0.2, 0.25, 0.3]:
        candidates.append((0.5, 0.5, r_pref))
    
    # 5. Ring candidates
    for radius_frac in [0.25, 0.35, 0.45]:
        n_ring = int(2 * np.pi * radius_frac)
        for k in range(n_ring):
            angle = 2 * np.pi * k / n_ring
            x = 0.5 + radius_frac * np.cos(angle)
            y = 0.5 + radius_frac * np.sin(angle)
            if 0.05 <= x <= 0.95 and 0.05 <= y <= 0.95:
                r_pref = 0.12
                candidates.append((x, y, r_pref))
    
    # 6. Random candidates
    np.random.seed(123)
    for _ in range(200):
        x = np.random.uniform(0.1, 0.9)
        y = np.random.uniform(0.1, 0.9)
        r_pref = np.random.uniform(0.08, 0.2)
        candidates.append((x, y, r_pref))
    
    # 7. Edge-midpoint candidates
    for t in np.linspace(0.15, 0.85, 7):
        candidates.append((t, 0.1, 0.08))
        candidates.append((t, 0.9, 0.08))
        candidates.append((0.1, t, 0.08))
        candidates.append((0.9, t, 0.08))
    
    # 8. Diagonal candidates
    for t in np.linspace(0.15, 0.85, 7):
        candidates.append((t, t, 0.1))
        candidates.append((t, 1-t, 0.1))
    
    return candidates

def _greedy_select(candidates, max_circles=N):
    """
    Greedy selection: pick candidates one by one, always choosing the one
    that gives the largest radius when placed.
    """
    selected = []
    remaining = list(candidates)
    
    while len(selected) < max_circles and len(remaining) > 0:
        best_idx = -1
        best_r = -1
        best_cand = None
        
        for idx, (cx, cy, pref_r) in enumerate(remaining):
            if len(selected) > 0:
                arr = np.array(selected)
                r_max = _max_radius_at(cx, cy, arr)
            else:
                r_max = min(cx, cy, 1 - cx, 1 - cy)
            
            r = min(pref_r, r_max)
            
            if r > best_r:
                best_r = r
                best_idx = idx
                best_cand = (cx, cy, r)
        
        if best_r < 1e-6:
            break
        
        selected.append(best_cand)
        remaining.pop(best_idx)
    
    return selected

def _greedy_select_diverse(candidates, max_circles=N):
    """
    Greedy selection with diversity: prefer candidates that are far from
    already selected ones, to avoid clustering.
    """
    selected = []
    remaining = list(candidates)
    
    # First, pick the single best candidate
    if not remaining:
        return selected
    
    best_r = -1
    best_idx = 0
    for idx, (cx, cy, pref_r) in enumerate(remaining):
        r_max = min(cx, cy, 1 - cx, 1 - cy)
        r = min(pref_r, r_max)
        if r > best_r:
            best_r = r
            best_idx = idx
    
    selected.append(remaining.pop(best_idx))
    
    while len(selected) < max_circles and len(remaining) > 0:
        best_idx = -1
        best_score = -1
        best_cand = None
        
        for idx, (cx, cy, pref_r) in enumerate(remaining):
            if len(selected) > 0:
                arr = np.array(selected)
                r_max = _max_radius_at(cx, cy, arr)
            else:
                r_max = min(cx, cy, 1 - cx, 1 - cy)
            
            r = min(pref_r, r_max)
            
            # Diversity bonus: distance to nearest selected
            if selected:
                sel_arr = np.array(selected)
                dists = np.sqrt((cx - sel_arr[:, 0])**2 + (cy - sel_arr[:, 1])**2)
                min_dist = np.min(dists)
            else:
                min_dist = 1.0
            
            # Combined score: radius + diversity bonus
            score = r + 0.3 * min_dist
            
            if score > best_score:
                best_score = score
                best_idx = idx
                best_cand = (cx, cy, r)
        
        if best_r < 1e-6 and best_cand is None:
            break
        
        if best_cand is not None:
            selected.append(best_cand)
            remaining.pop(best_idx)
    
    return selected

def _force_valid(circles, tol=1e-10):
    """
    Force validity by shrinking radii until no overlaps.
    """
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2].copy()
    
    # Boundary fix
    for i in range(N):
        min_r = min(x[i], y[i], 1 - x[i], 1 - y[i])
        if r[i] > min_r:
            r[i] = min_r * (1 - 1e-8)
    
    # Overlap fix: iterative shrink
    for _ in range(50):
        x_diff = x[:, None] - x[None, :]
        y_diff = y[:, None] - y[None, :]
        dist = np.sqrt(x_diff**2 + y_diff**2)
        r_sum = r[:, None] + r[None, :]
        mask = ~np.eye(N, dtype=bool)
        
        overlaps = r_sum - dist
        max_overlap = np.max(overlaps[mask]) if N > 1 else 0
        
        if max_overlap < tol:
            break
        
        # Shrink all radii proportionally
        r *= (1 - max_overlap * 0.5)
    
    return np.column_stack((x, y, r))

def _polish_continuous(circles, time_limit=10.0):
    """
    Use scipy minimize to polish the solution in continuous space.
    """
    def objective(params):
        x = params[:N]
        y = params[N:2*N]
        r = params[2*N:]
        
        obj = -np.sum(r)
        
        # Boundary penalties
        bound_pen = 0
        for i in range(N):
            if x[i] < r[i]:
                bound_pen += (r[i] - x[i])**2
            if x[i] > 1 - r[i]:
                bound_pen += (x[i] - (1 - r[i]))**2
            if y[i] < r[i]:
                bound_pen += (r[i] - y[i])**2
            if y[i] > 1 - r[i]:
                bound_pen += (y[i] - (1 - r[i]))**2
        
        # Overlap penalties
        x_diff = x[:, None] - x[None, :]
        y_diff = y[:, None] - y[None, :]
        dist = np.sqrt(x_diff**2 + y_diff**2)
        r_sum = r[:, None] + r[None, :]
        mask = ~np.eye(N, dtype=bool)
        
        overlaps = np.maximum(0, r_sum - dist)
        overlap_pen = np.sum(overlaps[mask]**2)
        
        return obj + 100.0 * (bound_pen + overlap_pen)
    
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]
    
    params = np.concatenate([x, y, r])
    
    try:
        res = minimize(objective, params, method='L-BFGS-B', 
                       options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-10})
        if res.fun < objective(params):
            return np.column_stack([res.x[:N], res.x[N:2*N], res.x[2*N:]])
    except Exception:
        pass
    
    return circles

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    best_solution = None
    best_score = -np.inf
    
    # Generate candidate set
    candidates = _generate_candidates()
    
    # Strategy 1: Pure greedy selection
    if time.time() < deadline - 20:
        greedy1 = _greedy_select(candidates, N)
        if len(greedy1) == N:
            circles1 = np.array(greedy1)
            circles1 = _force_valid(circles1)
            if _check_validity(circles1):
                score1 = np.sum(circles1[:, 2])
                if score1 > best_score:
                    best_score = score1
                    best_solution = circles1.copy()
    
    # Strategy 2: Greedy with diversity
    if time.time() < deadline - 20:
        greedy2 = _greedy_select_diverse(candidates, N)
        if len(greedy2) == N:
            circles2 = np.array(greedy2)
            circles2 = _force_valid(circles2)
            if _check_validity(circles2):
                score2 = np.sum(circles2[:, 2])
                if score2 > best_score:
                    best_score = score2
                    best_solution = circles2.copy()
    
    # Strategy 3: Local search from greedy results
    if time.time() < deadline - 10:
        for init_name, init_circles in [("greedy1", _greedy_select(candidates, N)),
                                        ("greedy2", _greedy_select_diverse(candidates, N))]:
            if len(init_circles) < N:
                continue
            
            circles = np.array(init_circles)
            remaining_time = deadline - 10 - time.time()
            if remaining_time < 5:
                break
            
            ls_circles = _local_search(circles, time_limit=min(remaining_time / 2, 15))
            ls_circles = _force_valid(ls_circles)
            
            if _check_validity(ls_circles):
                score = np.sum(ls_circles[:, 2])
                if score > best_score:
                    best_score = score
                    best_solution = ls_circles.copy()
    
    # Strategy 4: Continuous polish from best found
    if best_solution is not None and time.time() < deadline - 5:
        polished = _polish_continuous(best_solution, time_limit=min(deadline - 5 - time.time(), 15))
        polished = _force_valid(polished)
        
        if _check_validity(polished):
            score = np.sum(polished[:, 2])
            if score > best_score:
                best_score = score
                best_solution = polished.copy()
    
    # Strategy 5: Multiple random restarts with greedy + local search
    np.random.seed(42)
    for restart in range(5):
        if time.time() > deadline - 10:
            break
        
        # Random initial placement
        rand_circles = np.zeros((N, 3))
        rand_circles[:, 0] = np.random.uniform(0.1, 0.9, N)
        rand_circles[:, 1] = np.random.uniform(0.1, 0.9, N)
        rand_circles[:, 2] = np.random.uniform(0.05, 0.12, N)
        
        # Force valid
        rand_circles = _force_valid(rand_circles)
        
        if not _check_validity(rand_circles):
            continue
        
        # Local search
        remaining_time = deadline - 10 - time.time()
        if remaining_time < 5:
            break
        
        ls_circles = _local_search(rand_circles, time_limit=min(remaining_time / 3, 10))
        ls_circles = _force_valid(ls_circles)
        
        if _check_validity(ls_circles):
            score = np.sum(ls_circles[:, 2])
            if score > best_score:
                best_score = score
                best_solution = ls_circles.copy()
    
    # Fallback
    if best_solution is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        best_solution = np.array(cells[:N])
    
    return best_solution
