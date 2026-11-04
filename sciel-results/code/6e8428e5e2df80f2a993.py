"""Circle packing in unit square: pack 26 circles, maximize sum of radii."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import least_squares
from scipy.spatial import Delaunay
import random


def _eval_packing(xy, r):
    """
    Evaluate a packing: returns (total_radius, is_valid).
    Valid means all circles inside [0,1]^2 and disjoint interiors.
    """
    n = len(xy)
    for i in range(n):
        x, y, ri = xy[i, 0], xy[i, 1], r[i]
        if ri < -1e-9:
            return 0.0, False
        if x - ri < -1e-9 or 1.0 - x - ri < -1e-9 or y - ri < -1e-9 or 1.0 - y - ri < -1e-9:
            return 0.0, False
    for i in range(n):
        for j in range(i + 1, n):
            dx = xy[i, 0] - xy[j, 0]
            dy = xy[i, 1] - xy[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - 1e-9:
                return 0.0, False
    return float(np.sum(r)), True


def _generate_random_contact_graph(n, rng):
    """
    Generate a random contact graph (edges = potential contacts).
    Strategy: random geometric graph with some structure.
    """
    # Start with a random set of points
    pts = rng.uniform(0.1, 0.9, (n, 2))
    
    # Use Delaunay to get a planar contact graph
    try:
        tri = Delaunay(pts)
        # Extract edges from triangulation
        edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i + 1, 3):
                    a, b = simplex[i], simplex[j]
                    edges.add((min(a, b), max(a, b)))
        edge_list = list(edges)
    except Exception:
        # Fallback: random edges
        edge_list = []
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < 0.3:
                    edge_list.append((i, j))
    
    # Also add some random long-range contacts to make it more interesting
    for _ in range(n // 3):
        i, j = rng.choice(n, 2, replace=False)
        if (min(i, j), max(i, j)) not in edge_list:
            edge_list.append((min(i, j), max(i, j)))
    
    return edge_list


def _build_contact_system(xy, r, edge_list, wall_contacts):
    """
    Build the residual system for contact constraints.
    Variables: x_i, y_i, r_i for each circle.
    Residuals:
      - For each edge (i,j): |p_i - p_j| - (r_i + r_j) = 0  (contact)
      - For each wall contact: appropriate constraint = 0
    """
    n = len(xy)
    nv = 3 * n  # x1..xn, y1..yn, r1..rn
    
    def residuals(v):
        x = v[0:n]
        y = v[n:2*n]
        r = v[2*n:3*n]
        res = []
        
        # Pairwise contact residuals
        for (i, j) in edge_list:
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            res.append(dist - (r[i] + r[j]))
        
        # Wall contact residuals
        for (i, wall) in wall_contacts:
            if wall == 0:  # left: x_i - r_i = 0
                res.append(x[i] - r[i])
            elif wall == 1:  # right: 1 - x_i - r_i = 0
                res.append(1.0 - x[i] - r[i])
            elif wall == 2:  # bottom: y_i - r_i = 0
                res.append(y[i] - r[i])
            elif wall == 3:  # top: 1 - y_i - r_i = 0
                res.append(1.0 - y[i] - r[i])
        
        return np.array(res)
    
    return residuals


def _solve_contact_graph(xy_init, r_init, edge_list, wall_contacts, deadline):
    """
    Solve the contact system using least squares, then expand radii.
    """
    n = len(xy_init)
    v0 = np.concatenate([xy_init[:, 0], xy_init[:, 1], r_init])
    
    # Bounds: x,y in [0,1], r in [0.001, 0.5]
    lower = np.concatenate([np.zeros(n), np.zeros(n), np.full(n, 0.001)])
    upper = np.concatenate([np.ones(n), np.ones(n), np.full(n, 0.5)])
    
    residuals_fn = _build_contact_system(xy_init, r_init, edge_list, wall_contacts)
    
    try:
        if time.time() > deadline:
            return xy_init, r_init, 0.0, False
        
        result = least_squares(residuals_fn, v0, bounds=(lower, upper),
                               method='trf', max_nfev=5000, ftol=1e-12,
                               xtol=1e-12, gtol=1e-12)
        v_sol = result.x
        new_xy = np.column_stack([v_sol[0:n], v_sol[n:2*n]])
        new_r = v_sol[2*n:3*n]
    except Exception:
        return xy_init, r_init, 0.0, False
    
    # Clip to valid
    for i in range(n):
        new_r[i] = min(new_r[i], new_xy[i, 0], 1.0 - new_xy[i, 0], new_xy[i, 1], 1.0 - new_xy[i, 1])
        new_r[i] = max(new_r[i], 0.001)
        new_xy[i, 0] = np.clip(new_xy[i, 0], new_r[i], 1.0 - new_r[i])
        new_xy[i, 1] = np.clip(new_xy[i, 1], new_r[i], 1.0 - new_r[i])
    
    total, valid = _eval_packing(new_xy, new_r)
    return new_xy, new_r, total, valid


def _radial_expansion(xy, r, edge_list, wall_contacts, deadline, max_iters=200):
    """
    Iteratively expand radii while maintaining contact constraints.
    At each step, increase all radii by a small amount and re-solve.
    """
    n = len(xy)
    best_total, best_valid = _eval_packing(xy, r)
    best_xy = xy.copy()
    best_r = r.copy()
    
    # Determine which walls each circle is touching
    # We'll use a simpler approach: just try to grow and re-solve
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Try to increase all radii by a small factor
        growth = 0.002 + 0.003 * (it / max_iters)
        new_r = r * (1.0 + growth)
        new_r = np.clip(new_r, 0.001, 0.5)
        
        # Re-solve with slightly larger radii as target
        # Modify the system: add a penalty for not matching target radii
        v0 = np.concatenate([xy[:, 0], xy[:, 1], new_r])
        
        def residuals_with_target(v):
            x = v[0:n]
            y = v[n:2*n]
            r_v = v[2*n:3*n]
            res = []
            
            # Contact constraints (soft)
            for (i, j) in edge_list:
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx * dx + dy * dy)
                res.append(dist - (r_v[i] + r_v[j]))
            
            # Wall constraints (soft)
            for (i, wall) in wall_contacts:
                if wall == 0:
                    res.append(x[i] - r_v[i])
                elif wall == 1:
                    res.append(1.0 - x[i] - r_v[i])
                elif wall == 2:
                    res.append(y[i] - r_v[i])
                elif wall == 3:
                    res.append(1.0 - y[i] - r_v[i])
            
            # Radius target (strong)
            for i in range(n):
                res.append(10.0 * (r_v[i] - new_r[i]))
            
            return np.array(res)
        
        lower = np.concatenate([np.zeros(n), np.zeros(n), np.full(n, 0.001)])
        upper = np.concatenate([np.ones(n), np.ones(n), np.full(n, 0.5)])
        
        try:
            result = least_squares(residuals_with_target, v0, bounds=(lower, upper),
                                   method='trf', max_nfev=2000, ftol=1e-10,
                                   xtol=1e-10, gtol=1e-10)
            v_sol = result.x
            cur_xy = np.column_stack([v_sol[0:n], v_sol[n:2*n]])
            cur_r = v_sol[2*n:3*n]
        except Exception:
            break
        
        # Clip
        for i in range(n):
            cur_r[i] = min(cur_r[i], cur_xy[i, 0], 1.0 - cur_xy[i, 0], cur_xy[i, 1], 1.0 - cur_xy[i, 1])
            cur_r[i] = max(cur_r[i], 0.001)
            cur_xy[i, 0] = np.clip(cur_xy[i, 0], cur_r[i], 1.0 - cur_r[i])
            cur_xy[i, 1] = np.clip(cur_xy[i, 1], cur_r[i], 1.0 - cur_r[i])
        
        total, valid = _eval_packing(cur_xy, cur_r)
        if valid and total > best_total:
            best_total = total
            best_valid = True
            best_xy = cur_xy.copy()
            best_r = cur_r.copy()
            xy = cur_xy.copy()
            r = cur_r.copy()
        else:
            # Revert and try smaller growth
            r = best_r.copy()
            xy = best_xy.copy()
    
    return best_xy, best_r, best_total, best_valid


def _local_search_refine(xy, r, deadline, grid_size=60):
    """
    Local search: iteratively move each circle to best position/radius.
    """
    n = len(xy)
    best_total, best_valid = _eval_packing(xy, r)
    best_xy = xy.copy()
    best_r = r.copy()
    
    gx = np.linspace(0.0, 1.0, grid_size)
    gy = np.linspace(0.0, 1.0, grid_size)
    GX, GY = np.meshgrid(gx, gy, indexing='ij')
    candidates = np.column_stack([GX.ravel(), GY.ravel()])
    M = len(candidates)
    
    wall_clear = np.minimum(np.minimum(candidates[:, 0], 1.0 - candidates[:, 0]),
                            np.minimum(candidates[:, 1], 1.0 - candidates[:, 1]))
    
    max_sweeps = 8
    for sweep in range(max_sweeps):
        if time.time() > deadline:
            break
        improved = False
        for i in range(n):
            if time.time() > deadline:
                break
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            others_xy = xy[mask]
            others_r = r[mask]
            
            if len(others_xy) > 0:
                dx = candidates[:, 0:1] - others_xy[:, 0][np.newaxis, :]
                dy = candidates[:, 1:2] - others_xy[:, 1][np.newaxis, :]
                dists = np.sqrt(dx * dx + dy * dy)
                circle_clear = np.min(dists - others_r[np.newaxis, :], axis=1)
                r_candidates = np.minimum(wall_clear, circle_clear)
            else:
                r_candidates = wall_clear.copy()
            
            idx = int(np.argmax(r_candidates))
            new_x, new_y = candidates[idx, 0], candidates[idx, 1]
            new_r = r_candidates[idx]
            
            if new_r > r[i] + 1e-10:
                xy[i, 0] = new_x
                xy[i, 1] = new_y
                r[i] = new_r
                improved = True
        
        total, valid = _eval_packing(xy, r)
        if valid and total > best_total + 1e-10:
            best_total = total
            best_xy = xy.copy()
            best_r = r.copy()
        
        if not improved:
            break
    
    return best_xy, best_r, best_total, best_valid


def _high_precision_polish(xy, r, deadline):
    """
    Fine-grained local search with adaptive grid.
    """
    n = len(xy)
    best_total, best_valid = _eval_packing(xy, r)
    best_xy = xy.copy()
    best_r = r.copy()
    
    # Adaptive local grid around each circle
    for i in range(n):
        if time.time() > deadline:
            break
        
        cx, cy, cr = xy[i, 0], xy[i, 1], r[i]
        # Local search region
        margin = cr * 2 + 0.05
        x_min = max(0.0, cx - margin)
        x_max = min(1.0, cx + margin)
        y_min = max(0.0, cy - margin)
        y_max = min(1.0, cy + margin)
        
        if x_max - x_min < 0.01 or y_max - y_min < 0.01:
            continue
        
        local_gx = np.linspace(x_min, x_max, 20)
        local_gy = np.linspace(y_min, y_max, 20)
        LGX, LGY = np.meshgrid(local_gx, local_gy, indexing='ij')
        local_candidates = np.column_stack([LGX.ravel(), LGY.ravel()])
        
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        others_xy = xy[mask]
        others_r = r[mask]
        
        if len(others_xy) > 0:
            dx = local_candidates[:, 0:1] - others_xy[:, 0][np.newaxis, :]
            dy = local_candidates[:, 1:2] - others_xy[:, 1][np.newaxis, :]
            dists = np.sqrt(dx * dx + dy * dy)
            circle_clear = np.min(dists - others_r[np.newaxis, :], axis=1)
        else:
            circle_clear = np.ones(len(local_candidates))
        
        wall_clear = np.minimum(np.minimum(local_candidates[:, 0], 1.0 - local_candidates[:, 0]),
                                np.minimum(local_candidates[:, 1], 1.0 - local_candidates[:, 1]))
        r_candidates = np.minimum(wall_clear, circle_clear)
        
        idx = int(np.argmax(r_candidates))
        new_x, new_y = local_candidates[idx, 0], local_candidates[idx, 1]
        new_r = r_candidates[idx]
        
        if new_r > r[i] + 1e-10:
            xy[i, 0] = new_x
            xy[i, 1] = new_y
            r[i] = new_r
    
    total, valid = _eval_packing(xy, r)
    if valid and total > best_total:
        best_total = total
        best_valid = True
        best_xy = xy.copy()
        best_r = r.copy()
    
    return best_xy, best_r, best_total, best_valid


def _voronoi_greedy_init(n, rng):
    """
    Greedy initialization using largest empty circle concept.
    Place circles one at a time at the point maximizing minimum distance to existing circles and walls.
    """
    xy_list = []
    r_list = []
    
    # Start with a few random points
    for _ in range(min(5, n)):
        x = rng.uniform(0.1, 0.9)
        y = rng.uniform(0.1, 0.9)
        xy_list.append([x, y])
    
    for _ in range(n - len(xy_list)):
        if time.time() > 100:
            break
        # Find best position for next circle
        best_r = 0
        best_pos = [0.5, 0.5]
        
        # Sample candidate positions
        for _ in range(500):
            cx = rng.uniform(0.05, 0.95)
            cy = rng.uniform(0.05, 0.95)
            
            # Max radius at this position
            r_max = min(cx, 1.0 - cx, cy, 1.0 - cy)
            for (px, py), pr in zip(xy_list, r_list):
                d = np.sqrt((cx - px) ** 2 + (cy - py) ** 2)
                r_max = min(r_max, d - pr)
            
            if r_max > best_r:
                best_r = r_max
                best_pos = [cx, cy]
        
        if best_r > 0.005:
            xy_list.append(best_pos)
            r_list.append(best_r)
        else:
            # Place at random with small radius
            x = rng.uniform(0.1, 0.9)
            y = rng.uniform(0.1, 0.9)
            r_max = min(x, 1.0 - x, y, 1.0 - y)
            for (px, py), pr in zip(xy_list, r_list):
                d = np.sqrt((x - px) ** 2 + (y - py) ** 2)
                r_max = min(r_max, d - pr)
            r_max = max(r_max, 0.005)
            xy_list.append([x, y])
            r_list.append(r_max)
    
    xy = np.array(xy_list[:n])
    r = np.array(r_list[:n])
    return xy, r


def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s limit
    
    best_xy = None
    best_r = None
    best_total = 0.0
    best_valid = False
    
    rng = np.random.RandomState(42)
    
    # Phase 1: Voronoi greedy init + local search
    if time.time() < deadline - 60:
        xy, r = _voronoi_greedy_init(N, rng)
        out_xy, out_r, total, valid = _local_search_refine(xy, r, deadline - 30, grid_size=60)
        if valid and total > best_total:
            best_total = total
            best_valid = True
            best_xy = out_xy
            best_r = out_r
    
    # Phase 2: Contact graph solving with multiple random graphs
    n_graphs = 15
    for g in range(n_graphs):
        if time.time() > deadline - 20:
            break
        
        # Generate random contact graph
        edge_list = _generate_random_contact_graph(N, rng)
        
        # Determine wall contacts: assign each circle to a wall based on position
        # Use the greedy init positions as reference
        if best_xy is not None:
            ref_xy = best_xy
        else:
            ref_xy = rng.uniform(0.1, 0.9, (N, 2))
        
        wall_contacts = []
        for i in range(N):
            x, y = ref_xy[i, 0], ref_xy[i, 1]
            # Assign to nearest wall
            d_left = x
            d_right = 1.0 - x
            d_bottom = y
            d_top = 1.0 - y
            walls = [d_left, d_right, d_bottom, d_top]
            min_d = min(walls)
            # Assign to 1-2 nearest walls
            sorted_walls = sorted(enumerate(walls), key=lambda t: t[1])
            for w_idx, w_d in sorted_walls[:2]:
                if w_d < min_d + 0.15:
                    wall_contacts.append((i, w_idx))
        
        # Initialize with current best or random
        if best_xy is not None:
            xy_init = best_xy.copy()
            r_init = best_r.copy()
        else:
            xy_init = rng.uniform(0.1, 0.9, (N, 2))
            r_init = rng.uniform(0.02, 0.04, N)
        
        # Solve contact system
        sol_xy, sol_r, sol_total, sol_valid = _solve_contact_graph(
            xy_init, r_init, edge_list, wall_contacts, deadline - 10)
        
        if sol_valid and sol_total > best_total:
            best_total = sol_total
            best_valid = True
            best_xy = sol_xy
            best_r = sol_r
        
        # Radial expansion
        if sol_valid:
            exp_xy, exp_r, exp_total, exp_valid = _radial_expansion(
                sol_xy, sol_r, edge_list, wall_contacts, deadline - 10, max_iters=50)
            if exp_valid and exp_total > best_total:
                best_total = exp_total
                best_valid = True
                best_xy = exp_xy
                best_r = exp_r
    
    # Phase 3: Local search refinement on best
    if best_valid and best_xy is not None and time.time() < deadline - 15:
        ls_xy, ls_r, ls_total, ls_valid = _local_search_refine(
            best_xy.copy(), best_r.copy(), deadline - 5, grid_size=80)
        if ls_valid and ls_total > best_total + 1e-10:
            best_total = ls_total
            best_xy = ls_xy
            best_r = ls_r
    
    # Phase 4: High precision polish
    if best_valid and best_xy is not None and time.time() < deadline - 5:
        hp_xy, hp_r, hp_total, hp_valid = _high_precision_polish(
            best_xy.copy(), best_r.copy(), deadline)
        if hp_valid and hp_total > best_total + 1e-10:
            best_total = hp_total
            best_xy = hp_xy
            best_r = hp_r
    
    # Fallback
    if best_xy is None or not best_valid:
        side = int(np.ceil(np.sqrt(N)))
        result = np.zeros((N, 3))
        for i in range(N):
            gx = i % side
            gy = i // side
            result[i, 0] = (gx + 0.5) / side
            result[i, 1] = (gy + 0.5) / side
            result[i, 2] = 0.4 / side
        return result
    
    # Final validation and clipping
    n = len(best_xy)
    result = np.zeros((n, 3))
    for i in range(n):
        x, y, ri = best_xy[i, 0], best_xy[i, 1], best_r[i]
        ri = min(ri, x, 1.0 - x, y, 1.0 - y)
        ri = max(ri, 0.001)
        result[i, 2] = ri
        result[i, 0] = np.clip(x, ri, 1.0 - ri)
        result[i, 1] = np.clip(y, ri, 1.0 - ri)
    
    # Final check
    final_xy = result[:, :2]
    final_r = result[:, 2]
    total, valid = _eval_packing(final_xy, final_r)
    if not valid:
        side = int(np.ceil(np.sqrt(N)))
        result = np.zeros((N, 3))
        for i in range(N):
            gx = i % side
            gy = i // side
            result[i, 0] = (gx + 0.5) / side
            result[i, 1] = (gy + 0.5) / side
            result[i, 2] = 0.4 / side
    
    return result
# EVOLVE-BLOCK-END
