"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

DEADLINE = 140.0


def _check_validity(circles, W, H):
    """Check if circles are valid: inside rectangle, no overlaps."""
    if len(circles) != N:
        return False
    for i in range(N):
        x, y, r = circles[i]
        if r <= 0:
            return False
        if x - r < -1e-9 or y - r < -1e-9 or x + r > W + 1e-9 or y + r > H + 1e-9:
            return False
    for i in range(N):
        for j in range(i + 1, N):
            dx = circles[i][0] - circles[j][0]
            dy = circles[i][1] - circles[j][1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < circles[i][2] + circles[j][2] - 1e-9:
                return False
    return True


def _separate_circles(circles, W, H, max_iter=3000, tol=1e-12):
    """Directly resolve overlaps and boundary violations using pairwise repulsion."""
    n = len(circles)
    x = circles[:, 0].copy()
    y = circles[:, 1].copy()
    r = circles[:, 2].copy()

    for it in range(max_iter):
        moved = False
        for i in range(n):
            if r[i] <= 0:
                r[i] = 1e-4
            x[i] = np.clip(x[i], r[i], W - r[i])
            y[i] = np.clip(y[i], r[i], H - r[i])

        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist_sq = dx * dx + dy * dy
                min_dist = r[i] + r[j]
                if dist_sq < (min_dist - tol) ** 2:
                    dist = np.sqrt(dist_sq)
                    if dist < 1e-12:
                        angle = np.random.uniform(0, 2 * np.pi)
                        dx = np.cos(angle) * 1e-6
                        dy = np.sin(angle) * 1e-6
                        dist = 1e-6
                    overlap = min_dist - dist
                    total_r = r[i] + r[j]
                    wi = r[i] / total_r if total_r > 0 else 0.5
                    wj = r[j] / total_r if total_r > 0 else 0.5
                    push = overlap + 1e-8
                    x[i] += wi * push * dx / dist
                    y[i] += wi * push * dy / dist
                    x[j] -= wj * push * dx / dist
                    y[j] -= wj * push * dy / dist
                    moved = True

        for i in range(n):
            max_r_x = min(x[i], W - x[i])
            max_r_y = min(y[i], H - y[i])
            max_r = max(1e-4, min(max_r_x, max_r_y))
            if r[i] > max_r:
                r[i] = max_r
            x[i] = np.clip(x[i], r[i], W - r[i])
            y[i] = np.clip(y[i], r[i], H - r[i])

        if not moved:
            break

    result = np.column_stack([x, y, r])
    valid = _check_validity(result, W, H)
    return result, valid


def _grid_initialization(W, H):
    """Create a grid-based initial configuration."""
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) >= N:
                break
            cx = (i + 0.5) * W / side
            cy = (j + 0.5) * H / side
            r = min(W, H) / (2 * side) - 1e-4
            cells.append([cx, cy, r])
    return np.array(cells[:N])


def _random_initialization(W, H, rng):
    """Create a random initial configuration with moderate circles."""
    circles = []
    max_r = min(W, H) / 4
    for _ in range(N):
        r = rng.uniform(0.02, max_r)
        x = rng.uniform(r, W - r)
        y = rng.uniform(r, H - r)
        circles.append([x, y, r])
    return np.array(circles)


def _optimize_positions_fixed_r(circles, W, H, max_iter=2000):
    """
    Optimize only x,y positions with radii fixed.
    Uses penalty method with L-BFGS-B.
    """
    n = N
    r = circles[:, 2].copy()
    x0 = circles[:, 0].copy()
    y0 = circles[:, 1].copy()
    
    def pos_obj(params):
        x = params[:n]
        y = params[n:2*n]
        obj = 0.0
        # Boundary penalties
        for i in range(n):
            if x[i] < r[i]:
                obj += 100.0 * (r[i] - x[i])**2
            if x[i] > W - r[i]:
                obj += 100.0 * (x[i] - (W - r[i]))**2
            if y[i] < r[i]:
                obj += 100.0 * (r[i] - y[i])**2
            if y[i] > H - r[i]:
                obj += 100.0 * (y[i] - (H - r[i]))**2
        # Overlap penalties
        for i in range(n):
            for j in range(i+1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                min_d = r[i] + r[j]
                if dist < min_d:
                    obj += 100.0 * (min_d - dist)**2
        return obj
    
    def pos_grad(params):
        x = params[:n]
        y = params[n:2*n]
        grad = np.zeros(2*n)
        for i in range(n):
            if x[i] < r[i]:
                grad[i] += -200.0 * (r[i] - x[i])
            if x[i] > W - r[i]:
                grad[i] += 200.0 * (x[i] - (W - r[i]))
            if y[i] < r[i]:
                grad[n+i] += -200.0 * (r[i] - y[i])
            if y[i] > H - r[i]:
                grad[n+i] += 200.0 * (y[i] - (H - r[i]))
        for i in range(n):
            for j in range(i+1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist_sq = dx*dx + dy*dy
                min_d = r[i] + r[j]
                if dist_sq < min_d*min_d:
                    dist = np.sqrt(dist_sq)
                    if dist < 1e-12:
                        continue
                    overlap = min_d - dist
                    if overlap > 0:
                        fx = 200.0 * overlap * dx / dist
                        fy = 200.0 * overlap * dy / dist
                        grad[i] += fx
                        grad[n+i] += fy
                        grad[j] -= fx
                        grad[n+j] -= fy
        return grad
    
    params0 = np.concatenate([x0, y0])
    bounds = [(r[i]*0.5, W - r[i]*0.5) for i in range(n)] + [(r[i]*0.5, H - r[i]*0.5) for i in range(n)]
    
    result = minimize(pos_obj, params0, jac=pos_grad, method='L-BFGS-B', bounds=bounds,
                      options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-10})
    
    x_opt = result.x[:n]
    y_opt = result.x[n:2*n]
    x_opt = np.clip(x_opt, r, W - r)
    y_opt = np.clip(y_opt, r, H - r)
    return np.column_stack([x_opt, y_opt, r])


def _optimize_radii_fixed_pos(circles, W, H, max_iter=2000):
    """
    Optimize only radii with positions fixed.
    Each circle's max radius is constrained by neighbors and walls.
    Uses penalty method with L-BFGS-B.
    """
    n = N
    x = circles[:, 0].copy()
    y = circles[:, 1].copy()
    r0 = circles[:, 2].copy()
    
    def rad_obj(params):
        r = params
        obj = -np.sum(r)
        # Boundary constraints
        for i in range(n):
            max_r_wall = min(x[i], W - x[i], y[i], H - y[i])
            if r[i] > max_r_wall:
                obj += 100.0 * (r[i] - max_r_wall)**2
            if r[i] < 0:
                obj += 100.0 * r[i]**2
        # Overlap constraints
        for i in range(n):
            for j in range(i+1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                if r[i] + r[j] > dist:
                    obj += 100.0 * (r[i] + r[j] - dist)**2
        return obj
    
    def rad_grad(params):
        r = params
        grad = np.full(n, -1.0)  # d(-sum(r))/dr = -1
        for i in range(n):
            max_r_wall = min(x[i], W - x[i], y[i], H - y[i])
            if r[i] > max_r_wall:
                grad[i] += 200.0 * (r[i] - max_r_wall)
            if r[i] < 0:
                grad[i] += -200.0 * r[i]
        for i in range(n):
            for j in range(i+1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist_sq = dx*dx + dy*dy
                dist = np.sqrt(dist_sq) if dist_sq > 1e-12 else 1e-12
                overlap = r[i] + r[j] - dist
                if overlap > 0:
                    grad[i] += 200.0 * overlap
                    grad[j] += 200.0 * overlap
        return grad
    
    # Compute bounds for each radius
    bounds = []
    for i in range(n):
        max_r_wall = min(x[i], W - x[i], y[i], H - y[i])
        max_r_wall = max(1e-6, max_r_wall)
        bounds.append((1e-6, max_r_wall))
    
    result = minimize(rad_obj, r0, jac=rad_grad, method='L-BFGS-B', bounds=bounds,
                      options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-10})
    
    r_opt = result.x
    r_opt = np.clip(r_opt, 1e-6, None)
    return np.column_stack([x, y, r_opt])


def _hierarchical_optimize(circles, W, H, time_budget=30.0):
    """
    Hierarchical optimization: alternate between position optimization (fixed r)
    and radius optimization (fixed x,y). This decouples the 63-variable problem
    into two 42-variable subproblems that are easier to solve.
    """
    start = time.time()
    current = circles.copy()
    best_val = -np.sum(current[:, 2])
    best_circles = current.copy()
    
    n_iter = 0
    while time.time() - start < time_budget - 5:
        n_iter += 1
        # Step 1: Optimize positions with fixed radii
        pos_opt = _optimize_positions_fixed_r(current, W, H, max_iter=1500)
        pos_opt, valid_pos = _separate_circles(pos_opt, W, H, max_iter=1000)
        if valid_pos:
            current = pos_opt
        
        if time.time() - start > time_budget - 5:
            break
        
        # Step 2: Optimize radii with fixed positions
        rad_opt = _optimize_radii_fixed_pos(current, W, H, max_iter=1500)
        rad_opt, valid_rad = _separate_circles(rad_opt, W, H, max_iter=1000)
        if valid_rad:
            current = rad_opt
        
        # Track best
        s = np.sum(current[:, 2])
        if s > best_val:
            best_val = s
            best_circles = current.copy()
        
        # Check convergence
        if n_iter > 1:
            # Could add convergence check here
            pass
    
    # Final polish with joint optimization if time permits
    if time.time() - start < time_budget - 3:
        polished = _joint_polish(best_circles, W, H, max_iter=3000)
        polished, valid_polish = _separate_circles(polished, W, H, max_iter=1000)
        if valid_polish:
            s = np.sum(polished[:, 2])
            if s > best_val:
                best_val = s
                best_circles = polished.copy()
    
    return best_circles, best_val


def _joint_polish(circles, W, H, max_iter=3000):
    """
    Final joint optimization using penalty method with smooth constraints.
    """
    n = N
    params = circles.flatten()
    r_max = min(W, H) / 2
    bounds = []
    for i in range(N):
        bounds.append((1e-6, W - 1e-6))
        bounds.append((1e-6, H - 1e-6))
        bounds.append((1e-6, r_max))
    
    def obj_and_grad(params):
        p = params.reshape(n, 3)
        x = p[:, 0]
        y = p[:, 1]
        r = p[:, 2]
        
        obj = -np.sum(r)
        grad = np.zeros_like(params)
        grad[2::3] = -1.0
        
        eps = 1e-4
        bw = 200.0
        
        def smax_grad(v):
            sm = 0.5 * (v + np.sqrt(v**2 + eps**2))
            gm = 0.5 * (1.0 + v / np.sqrt(v**2 + eps**2))
            return sm, gm
        
        # Boundary penalties
        v = r - x
        sm, gm = smax_grad(v)
        obj += bw * np.sum(sm**2)
        grad[0::3] += -2.0 * bw * sm * gm
        grad[2::3] += 2.0 * bw * sm * gm
        
        v = r - y
        sm, gm = smax_grad(v)
        obj += bw * np.sum(sm**2)
        grad[1::3] += -2.0 * bw * sm * gm
        grad[2::3] += 2.0 * bw * sm * gm
        
        v = x + r - W
        sm, gm = smax_grad(v)
        obj += bw * np.sum(sm**2)
        grad[0::3] += 2.0 * bw * sm * gm
        grad[2::3] += 2.0 * bw * sm * gm
        
        v = y + r - H
        sm, gm = smax_grad(v)
        obj += bw * np.sum(sm**2)
        grad[1::3] += 2.0 * bw * sm * gm
        grad[2::3] += 2.0 * bw * sm * gm
        
        # Overlap penalties
        ow = 200.0
        iu, ju = np.triu_indices(n, k=1)
        dx = x[iu] - x[ju]
        dy = y[iu] - y[ju]
        dist = np.sqrt(dx**2 + dy**2)
        dist_safe = np.where(dist < 1e-12, 1e-12, dist)
        r_sum = r[iu] + r[ju]
        v = r_sum - dist_safe
        sm, gm = smax_grad(v)
        obj += ow * np.sum(sm**2)
        
        factor = 2.0 * ow * sm * gm
        nx = -dx / dist_safe
        ny = -dy / dist_safe
        
        np.add.at(grad, 3*iu, factor * nx)
        np.add.at(grad, 3*iu+1, factor * ny)
        np.add.at(grad, 3*iu+2, factor)
        np.add.at(grad, 3*ju, -factor * nx)
        np.add.at(grad, 3*ju+1, -factor * ny)
        np.add.at(grad, 3*ju+2, factor)
        
        return obj, grad
    
    result = minimize(obj_and_grad, params, method='L-BFGS-B', jac=True, bounds=bounds,
                      options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 100})
    return result.x.reshape(n, 3)


def _successive_convex_relaxation(circles, W, H, max_sweeps=15, time_budget=20.0):
    """
    Successive Convex Relaxation: linearize overlap constraints at current point,
    solve convex subproblem, iterate.
    """
    start = time.time()
    n = N
    x = circles[:, 0].copy()
    y = circles[:, 1].copy()
    r = circles[:, 2].copy()
    
    best_val = np.sum(r)
    best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
    
    r_max = min(W, H) / 2
    bounds = []
    for i in range(n):
        bounds.append((1e-6, W - 1e-6))
        bounds.append((1e-6, H - 1e-6))
        bounds.append((1e-6, r_max))
    
    for sweep in range(max_sweeps):
        if time.time() - start > time_budget - 3:
            break
        
        def sc_obj_and_grad(params):
            p = params.reshape(n, 3)
            px = p[:, 0]
            py = p[:, 1]
            pr = p[:, 2]
            
            obj = -np.sum(pr)
            grad = np.zeros_like(params)
            grad[2::3] = -1.0
            
            eps = 1e-4
            bw = 200.0
            
            def smax_grad(v):
                sm = 0.5 * (v + np.sqrt(v**2 + eps**2))
                gm = 0.5 * (1.0 + v / np.sqrt(v**2 + eps**2))
                return sm, gm
            
            # Boundary penalties
            v = pr - px
            sm, gm = smax_grad(v)
            obj += bw * np.sum(sm**2)
            grad[0::3] += -2.0 * bw * sm * gm
            grad[2::3] += 2.0 * bw * sm * gm
            
            v = pr - py
            sm, gm = smax_grad(v)
            obj += bw * np.sum(sm**2)
            grad[1::3] += -2.0 * bw * sm * gm
            grad[2::3] += 2.0 * bw * sm * gm
            
            v = px + pr - W
            sm, gm = smax_grad(v)
            obj += bw * np.sum(sm**2)
            grad[0::3] += 2.0 * bw * sm * gm
            grad[2::3] += 2.0 * bw * sm * gm
            
            v = py + pr - H
            sm, gm = smax_grad(v)
            obj += bw * np.sum(sm**2)
            grad[1::3] += 2.0 * bw * sm * gm
            grad[2::3] += 2.0 * bw * sm * gm
            
            # Linearized overlap penalties
            ow = 200.0
            for i in range(n):
                for j in range(i+1, n):
                    dx0 = x[i] - x[j]
                    dy0 = y[i] - y[j]
                    dist0 = np.sqrt(dx0*dx0 + dy0*dy0)
                    if dist0 < 1e-10:
                        dist0 = 1e-10
                        dx0 = 1e-10
                        dy0 = 0.0
                    gx_i = dx0 / dist0
                    gy_i = dy0 / dist0
                    
                    lin_dist = dist0 + gx_i*(px[i]-x[i]) + gy_i*(py[i]-y[i]) \
                                 - gx_i*(px[j]-x[j]) - gy_i*(py[j]-y[j])
                    v = (pr[i] + pr[j]) - lin_dist
                    sm, gm = smax_grad(v)
                    obj += ow * sm**2
                    
                    d_obj = 2.0 * ow * sm * gm
                    grad[3*i] += d_obj * (-gx_i)
                    grad[3*i+1] += d_obj * (-gy_i)
                    grad[3*i+2] += d_obj
                    grad[3*j] += d_obj * gx_i
                    grad[3*j+1] += d_obj * gy_i
                    grad[3*j+2] += d_obj
            
            return obj, grad
        
        params = np.column_stack([x, y, r]).flatten()
        result = minimize(sc_obj_and_grad, params, method='L-BFGS-B', jac=True, bounds=bounds,
                          options={'maxiter': 800, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50})
        
        new_params = result.x.reshape(n, 3)
        new_x = np.clip(new_params[:, 0], new_params[:, 2], W - new_params[:, 2])
        new_y = np.clip(new_params[:, 1], new_params[:, 2], H - new_params[:, 2])
        new_r = np.clip(new_params[:, 2], 1e-4, r_max)
        
        new_val = np.sum(new_r)
        if new_val > best_val:
            best_val = new_val
            best_x, best_y, best_r = new_x.copy(), new_y.copy(), new_r.copy()
        
        # Check convergence
        if np.max(np.abs(new_x - x)) < 1e-6 and np.max(np.abs(new_y - y)) < 1e-6 and np.max(np.abs(new_r - r)) < 1e-6:
            break
        
        x, y, r = new_x, new_y, new_r
    
    return np.column_stack([best_x, best_y, best_r]), best_val


def _best_for_aspect(W, H, time_budget=25.0):
    """Find best configuration for given aspect ratio."""
    start = time.time()
    rng = np.random.default_rng(42)
    
    best_score = -np.inf
    best_circles = None
    
    # Try multiple initializations
    initializations = [
        _grid_initialization(W, H),
    ]
    
    # Add a few random initializations
    n_rand = min(3, max(1, int((time_budget - 10) / 5)))
    for _ in range(n_rand):
        initializations.append(_random_initialization(W, H, rng))
    
    for init in initializations:
        if time.time() - start > time_budget - 8:
            break
        
        # Hierarchical optimization (main structural innovation)
        hier_circles, hier_score = _hierarchical_optimize(init, W, H, time_budget=time_budget - 10)
        if hier_score > best_score:
            best_score = hier_score
            best_circles = hier_circles.copy()
        
        # Successive convex relaxation from best so far
        if time.time() - start < time_budget - 5 and best_circles is not None:
            sc_circles, sc_score = _successive_convex_relaxation(best_circles, W, H, 
                                                                 max_sweeps=10, time_budget=10.0)
            sc_circles, valid_sc = _separate_circles(sc_circles, W, H, max_iter=1000)
            if valid_sc:
                s = np.sum(sc_circles[:, 2])
                if s > best_score:
                    best_score = s
                    best_circles = sc_circles.copy()
    
    # Final joint polish
    if best_circles is not None and time.time() - start < time_budget - 3:
        polished = _joint_polish(best_circles, W, H, max_iter=4000)
        polished, valid_polish = _separate_circles(polished, W, H, max_iter=1000)
        if valid_polish:
            s = np.sum(polished[:, 2])
            if s > best_score:
                best_score = s
                best_circles = polished.copy()
    
    if best_circles is None:
        best_circles = _grid_initialization(W, H)
        best_score = np.sum(best_circles[:, 2])
    
    return best_circles, best_score


def construct():
    start_time = time.time()
    best_score = -np.inf
    best_result = None
    best_W, best_H = 1.0, 1.0
    
    # Aspect ratios to try
    aspect_ratios = [0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 1.8]
    
    # Allocate time budget based on aspect ratio proximity to square
    for ar in aspect_ratios:
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 12:
            break
        
        if 0.9 <= ar <= 1.1:
            time_budget = 20.0
        elif 0.8 <= ar <= 1.3:
            time_budget = 15.0
        else:
            time_budget = 12.0
        
        W = 2.0 * ar / (ar + 1.0)
        H = 2.0 / (ar + 1.0)
        
        circles, score = _best_for_aspect(W, H, time_budget=time_budget)
        if score > best_score:
            best_score = score
            best_result = circles.copy()
            best_W, best_H = W, H
    
    # Final polish on the best result
    if best_result is not None and time.time() - start_time < DEADLINE - 5:
        polished = _joint_polish(best_result, best_W, best_H, max_iter=6000)
        polished, valid_final = _separate_circles(polished, best_W, best_H, max_iter=1500)
        if valid_final:
            s = np.sum(polished[:, 2])
            if s > best_score:
                best_score = s
                best_result = polished.copy()
    
    # Fallback
    if best_result is None:
        side = 5
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N:
                    break
                cx = (i + 0.5) / side
                cy = (j + 0.5) / side
                r = 0.5 / side - 1e-3
                cells.append([cx, cy, r])
        best_result = np.array(cells[:N])
        best_W, best_H = 1.0, 1.0
    
    # Ensure validity
    if not _check_validity(best_result, best_W, best_H):
        side = 5
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N:
                    break
                cx = (i + 0.5) / side
                cy = (j + 0.5) / side
                r = 0.5 / side - 1e-3
                cells.append([cx, cy, r])
        best_result = np.array(cells[:N])
        best_W, best_H = 1.0, 1.0
    
    return best_result
# EVOLVE-BLOCK-END
