"""Circle packing in perimeter-4 rectangle: pack 21 circles, maximize sum of radii."""
import numpy as np
import time
import random
from scipy.optimize import minimize
from scipy.spatial import Delaunay

N = 21

def _softplus(x):
    return np.where(x > 20, x, np.log1p(np.exp(x)))

def _sigmoid(x):
    return np.where(x > 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))

def _delaunay_edges(points):
    """Get unique edges from Delaunay triangulation."""
    try:
        tri = Delaunay(points)
        edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                a, b = simplex[i], simplex[(i+1) % 3]
                edges.add((min(a,b), max(a,b)))
        return list(edges)
    except Exception:
        # Fallback: all pairs
        n = len(points)
        return [(i, j) for i in range(n) for j in range(i+1, n)]

def _solve_lp_radii(xs, ys, w, h):
    """
    Solve for maximum sum of radii given fixed centers using LP.
    Constraints:
      r_i + r_j <= dist(i,j)  for all pairs
      r_i <= x_i, r_i <= w - x_i, r_i <= y_i, r_i <= h - y_i
      r_i >= 0
    Maximize sum(r_i)
    """
    from scipy.optimize import linprog
    n = len(xs)
    # Objective: maximize sum(r) => minimize -sum(r)
    c = -np.ones(n)
    
    A_ub = []
    b_ub = []
    
    # Pairwise distance constraints: r_i + r_j <= d_ij
    for i in range(n):
        for j in range(i+1, n):
            d = np.sqrt((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2)
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            A_ub.append(row)
            b_ub.append(d)
    
    # Boundary constraints: r_i <= min(x_i, w-x_i, y_i, h-y_i)
    for i in range(n):
        max_r = min(xs[i], w - xs[i], ys[i], h - ys[i])
        max_r = max(max_r, 0.0)
        row = np.zeros(n)
        row[i] = 1.0
        A_ub.append(row)
        b_ub.append(max_r)
    
    bounds = [(0, None)] * n
    
    res = linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub), bounds=bounds, method='highs')
    if res.success:
        return res.x
    return np.zeros(n)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill
    
    def softplus(x):
        return _softplus(x)

    def sigmoid(x):
        return _sigmoid(x)

    W_BOUNDARY = 100.0
    W_OVERLAP = 50.0
    
    def make_hex(w, h, n_circles):
        """Create hexagonal packing initialization."""
        best_score = -1
        best_arr = None
        for rows in range(1, n_circles + 1):
            cols = (n_circles + rows - 1) // rows
            xs = np.zeros(n_circles)
            ys = np.zeros(n_circles)
            rs = np.zeros(n_circles)
            for i in range(n_circles):
                row = i // cols
                col = i % cols
                x = (col + 0.5) * (w / cols)
                y = (row + 0.5) * (h / rows)
                if row % 2 == 1:
                    x += 0.5 * (w / cols)
                x = np.clip(x, 0.1 * w, 0.9 * w)
                y = np.clip(y, 0.1 * h, 0.9 * h)
                cell_w = w / cols
                cell_h = h / rows
                r = 0.45 * min(cell_w, cell_h)
                xs[i] = x
                ys[i] = y
                rs[i] = r
            valid = True
            for i in range(n_circles):
                if xs[i] < rs[i] or xs[i] > w - rs[i]:
                    valid = False
                    break
                if ys[i] < rs[i] or ys[i] > h - rs[i]:
                    valid = False
                    break
            if valid:
                for i in range(n_circles):
                    for j in range(i + 1, n_circles):
                        dist = np.sqrt((xs[i] - xs[j])**2 + (ys[i] - ys[j])**2)
                        if dist < rs[i] + rs[j]:
                            valid = False
                            break
                    if not valid:
                        break
            if valid:
                score = np.sum(rs)
                if score > best_score:
                    best_score = score
                    best_arr = np.concatenate([xs, ys, rs])
        if best_arr is None:
            side = int(np.ceil(np.sqrt(n_circles)))
            xs = np.zeros(n_circles)
            ys = np.zeros(n_circles)
            rs = np.zeros(n_circles)
            for i in range(n_circles):
                row = i // side
                col = i % side
                xs[i] = (col + 0.5) / side * w
                ys[i] = (row + 0.5) / side * h
                rs[i] = 0.45 / side * min(w, h)
            best_arr = np.concatenate([xs, ys, rs])
        return best_arr

    def make_grid(w, h, n_circles, side=5):
        """Create grid packing initialization."""
        xs = np.zeros(n_circles)
        ys = np.zeros(n_circles)
        rs = np.zeros(n_circles)
        for i in range(n_circles):
            row = i // side
            col = i % side
            xs[i] = (col + 0.5) / side * w
            ys[i] = (row + 0.5) / side * h
            rs[i] = 0.48 / side * min(w, h)
        return np.concatenate([xs, ys, rs])

    def make_random(w, h, n_circles):
        """Create random initialization."""
        xs = np.random.uniform(0.1*w, 0.9*w, n_circles)
        ys = np.random.uniform(0.1*h, 0.9*h, n_circles)
        rs = np.full(n_circles, 0.05 * min(w, h))
        return np.concatenate([xs, ys, rs])

    def make_initial(w, h, n_circles):
        """Create a hexagonal-like packing initialization."""
        if n_circles == 0:
            return np.zeros(0)
        
        best_score = -1
        best_arr = None
        
        for rows in range(1, n_circles + 1):
            cols = (n_circles + rows - 1) // rows
            xs = np.zeros(n_circles)
            ys = np.zeros(n_circles)
            rs = np.zeros(n_circles)
            
            for i in range(n_circles):
                row = i // cols
                col = i % cols
                x = (col + 0.5) * (w / cols)
                y = (row + 0.5) * (h / rows)
                if row % 2 == 1:
                    x += 0.5 * (w / cols)
                x = np.clip(x, 0.1 * w, 0.9 * w)
                y = np.clip(y, 0.1 * h, 0.9 * h)
                cell_w = w / cols
                cell_h = h / rows
                r = 0.45 * min(cell_w, cell_h)
                xs[i] = x
                ys[i] = y
                rs[i] = r
            
            valid = True
            for i in range(n_circles):
                if xs[i] < rs[i] or xs[i] > w - rs[i]:
                    valid = False
                    break
                if ys[i] < rs[i] or ys[i] > h - rs[i]:
                    valid = False
                    break
            if valid:
                for i in range(n_circles):
                    for j in range(i + 1, n_circles):
                        dist = np.sqrt((xs[i] - xs[j])**2 + (ys[i] - ys[j])**2)
                        if dist < rs[i] + rs[j]:
                            valid = False
                            break
                    if not valid:
                        break
            
            if valid:
                score = np.sum(rs)
                if score > best_score:
                    best_score = score
                    best_arr = np.concatenate([xs, ys, rs])
        
        if best_arr is None:
            side = int(np.ceil(np.sqrt(n_circles)))
            xs = np.zeros(n_circles)
            ys = np.zeros(n_circles)
            rs = np.zeros(n_circles)
            for i in range(n_circles):
                row = i // side
                col = i % side
                xs[i] = (col + 0.5) / side * w
                ys[i] = (row + 0.5) / side * h
                rs[i] = 0.45 / side * min(w, h)
            best_arr = np.concatenate([xs, ys, rs])
        
        return best_arr

    def objective_and_grad(params):
        """Penalty-based objective with analytic gradient. Uses softplus for smoothness."""
        w = params[0]
        w_clipped = np.clip(w, 0.3, 1.7)
        h = 2.0 - w_clipped
        
        circles = params[1:]
        obj_val = 0.0
        grad = np.zeros_like(params)
        
        for i in range(N):
            xi = circles[3*i]
            yi = circles[3*i+1]
            ri = circles[3*i+2]
            if ri < 1e-5:
                ri = 1e-5
            obj_val -= ri
            grad[3*i+2] -= 1.0

            v_left = ri - xi
            if v_left > -20:
                sp_grad = sigmoid(v_left)
                obj_val += W_BOUNDARY * softplus(v_left)
                grad[3*i] -= W_BOUNDARY * sp_grad
                grad[3*i+2] += W_BOUNDARY * sp_grad
            
            v_right = xi + ri - w_clipped
            if v_right > -20:
                sp_grad = sigmoid(v_right)
                obj_val += W_BOUNDARY * softplus(v_right)
                grad[3*i] += W_BOUNDARY * sp_grad
                grad[3*i+2] += W_BOUNDARY * sp_grad
                
            v_bot = ri - yi
            if v_bot > -20:
                sp_grad = sigmoid(v_bot)
                obj_val += W_BOUNDARY * softplus(v_bot)
                grad[3*i+1] -= W_BOUNDARY * sp_grad
                grad[3*i+2] += W_BOUNDARY * sp_grad

            v_top = yi + ri - h
            if v_top > -20:
                sp_grad = sigmoid(v_top)
                obj_val += W_BOUNDARY * softplus(v_top)
                grad[3*i+1] += W_BOUNDARY * sp_grad
                grad[3*i+2] += W_BOUNDARY * sp_grad

        for i in range(N):
            xi = circles[3*i]
            yi = circles[3*i+1]
            ri = circles[3*i+2]
            for j in range(i+1, N):
                xj = circles[3*j]
                yj = circles[3*j+1]
                rj = circles[3*j+2]
                
                dx = xi - xj
                dy = yi - yj
                dist_sq = dx*dx + dy*dy
                min_dist = ri + rj
                
                if dist_sq < min_dist * min_dist:
                    dist = np.sqrt(dist_sq) if dist_sq > 1e-12 else 1e-12
                    v = min_dist - dist
                    if v > -20:
                        sp_grad = sigmoid(v)
                        obj_val += W_OVERLAP * softplus(v)
                        
                        coeff = W_OVERLAP * sp_grad / dist
                        grad[3*i] -= coeff * dx
                        grad[3*i+1] -= coeff * dy
                        grad[3*i+2] += W_OVERLAP * sp_grad
                        
                        grad[3*j] += coeff * dx
                        grad[3*j+1] += coeff * dy
                        grad[3*j+2] += W_OVERLAP * sp_grad

        grad_w = 0.0
        for i in range(N):
            yi = circles[3*i+1]
            ri = circles[3*i+2]
            v_top = yi + ri - h
            if v_top > -20:
                grad_w += W_BOUNDARY * sigmoid(v_top)
        grad[0] = grad_w

        return obj_val, grad

    def is_valid(params):
        """Check if configuration is valid."""
        w = np.clip(params[0], 0.1, 1.9)
        h = 2.0 - w
        circles = params[1:]
        
        for i in range(N):
            xi = circles[3*i]
            yi = circles[3*i+1]
            ri = circles[3*i+2]
            if ri < 1e-6:
                return False
            if xi - ri < -1e-5 or xi + ri > w + 1e-5:
                return False
            if yi - ri < -1e-5 or yi + ri > h + 1e-5:
                return False
        
        for i in range(N):
            xi = circles[3*i]
            yi = circles[3*i+1]
            ri = circles[3*i+2]
            for j in range(i+1, N):
                xj = circles[3*j]
                yj = circles[3*j+1]
                rj = circles[3*j+2]
                dist_sq = (xi - xj)**2 + (yi - yj)**2
                if dist_sq < (ri + rj - 1e-5)**2:
                    return False
        return True

    def get_value(params):
        """Return sum of radii if valid, else -inf."""
        if not is_valid(params):
            return -np.inf
        w = np.clip(params[0], 0.1, 1.9)
        h = 2.0 - w
        circles = params[1:]
        xs = circles[0::3]
        ys = circles[1::3]
        rs = circles[2::3]
        min_x = np.min(xs - rs)
        max_x = np.max(xs + rs)
        min_y = np.min(ys - rs)
        max_y = np.max(ys + rs)
        bbox_w = max_x - min_x
        bbox_h = max_y - min_y
        bbox_perim = 2 * (bbox_w + bbox_h)
        if bbox_perim > 4.0 + 1e-5:
            return -np.inf
        return np.sum(rs)

    def _contact_graph_constraints(params, contact_pairs):
        """
        Build constraint functions for contact-graph-constrained optimization.
        Variables: [w, x_0, y_0, r_0, x_1, y_1, r_1, ..., x_N-1, y_N-1, r_N-1]
        Constraints:
          - Non-overlap: r_i + r_j <= dist(i,j) for ALL pairs (hard constraint)
          - Boundary: circles stay within [0,w] x [0,h]
        """
        w = params[0]
        w_clipped = np.clip(w, 0.3, 1.7)
        h = 2.0 - w_clipped
        circles = params[1:]
        
        n_constraints = 0
        # Count constraints
        for i in range(N):
            # 4 boundary constraints per circle
            n_constraints += 4
        for i in range(N):
            for j in range(i+1, N):
                n_constraints += 1  # non-overlap
        
        constraints = []
        
        # Boundary constraints: g_i >= 0 means xi - ri >= 0, etc.
        for i in range(N):
            xi = circles[3*i]
            yi = circles[3*i+1]
            ri = circles[3*i+2]
            
            # Left: xi - ri >= 0
            constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[3*i+1] - p[3*i+3]})
            # Right: w - xi - ri >= 0
            constraints.append({'type': 'ineq', 'fun': lambda p, i=i: np.clip(p[0], 0.3, 1.7) - p[3*i+1] - p[3*i+3]})
            # Bottom: yi - ri >= 0
            constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[3*i+2] - p[3*i+3]})
            # Top: h - yi - ri >= 0
            constraints.append({'type': 'ineq', 'fun': lambda p, i=i: (2.0 - np.clip(p[0], 0.3, 1.7)) - p[3*i+2] - p[3*i+3]})
        
        # Non-overlap: dist(i,j) - ri - rj >= 0
        for i in range(N):
            for j in range(i+1, N):
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda p, i=i, j=j: (
                        np.sqrt((p[3*i+1] - p[3*j+1])**2 + (p[3*i+2] - p[3*j+2])**2) 
                        - p[3*i+3] - p[3*j+3]
                    )
                })
        
        return constraints

    def _solve_with_contact_graph(init_params, contact_pairs, maxiter=500):
        """
        Solve the constrained optimization problem with a given contact graph.
        The contact graph informs which pairs are likely to be in contact,
        but we enforce non-overlap for ALL pairs.
        """
        # Objective: minimize -sum(r_i)
        def obj(p):
            w = np.clip(p[0], 0.3, 1.7)
            h = 2.0 - w
            circles = p[1:]
            r_sum = 0.0
            for i in range(N):
                ri = circles[3*i+2]
                r_sum += ri
            return -r_sum
        
        def obj_grad(p):
            grad = np.zeros_like(p)
            for i in range(N):
                grad[3*i+2] = -1.0
            return grad
        
        # Build constraints
        constraints = _contact_graph_constraints(init_params, contact_pairs)
        
        bounds = [(0.3, 1.7)] + [(-0.5, 2.5)] * (2*N) + [(0.01, 1.0)] * N
        
        try:
            result = minimize(
                obj,
                init_params,
                method='SLSQP',
                jac=obj_grad,
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': maxiter, 'ftol': 1e-10}
            )
            if result.success:
                return result.x
            else:
                # Try with fewer iterations or different initial point
                return init_params
        except Exception:
            return init_params

    def _get_contact_graph_from_packing(params):
        """Extract contact graph from a packing: pairs that are nearly touching."""
        w = np.clip(params[0], 0.1, 1.9)
        h = 2.0 - w
        circles = params[1:]
        xs = circles[0::3]
        ys = circles[1::3]
        rs = circles[2::3]
        
        contact_pairs = []
        for i in range(N):
            for j in range(i+1, N):
                dist = np.sqrt((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2)
                if dist <= rs[i] + rs[j] + 0.01:
                    contact_pairs.append((i, j))
        return contact_pairs

    best_value = -np.inf
    best_params = None

    random.seed(42)
    np.random.seed(42)
    
    # Phase 1: Generate diverse initial configurations and get contact graphs
    candidates = []
    
    # 1. Hexagonal patterns at various aspect ratios
    for w_try in [0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15]:
        h_try = 2.0 - w_try
        arr = make_hex(w_try, h_try, N)
        params = np.zeros(1 + 3*N)
        params[0] = w_try
        params[1:] = arr
        candidates.append(params.copy())
    
    # 2. Grid patterns
    for w_try in [0.8, 0.9, 1.0, 1.1, 1.2]:
        h_try = 2.0 - w_try
        arr = make_grid(w_try, h_try, N)
        params = np.zeros(1 + 3*N)
        params[0] = w_try
        params[1:] = arr
        candidates.append(params.copy())
    
    # 3. Random initializations
    for _ in range(20):
        w_try = random.uniform(0.8, 1.2)
        h_try = 2.0 - w_try
        arr = make_random(w_try, h_try, N)
        params = np.zeros(1 + 3*N)
        params[0] = w_try
        params[1:] = arr
        candidates.append(params.copy())

    # Phase 2: For each candidate, do a quick penalty-based optimization to get a decent packing,
    # then extract its contact graph and use that for constrained optimization
    bounds_penalty = [(0.3, 1.7)] + [(-1.0, 3.0)] * (2*N) + [(1e-5, 1.0)] * N
    
    for idx, init_params in enumerate(candidates):
        if time.time() > deadline - 5:
            break
        
        # Check if valid as-is
        val = get_value(init_params)
        if val > best_value:
            best_value = val
            best_params = init_params.copy()
        
        # Quick penalty-based optimization to improve the packing
        try:
            result = minimize(
                objective_and_grad,
                init_params,
                method='L-BFGS-B',
                jac=True,
                bounds=bounds_penalty,
                options={'maxiter': 300, 'ftol': 1e-8}
            )
            opt_params = result.x
            opt_val = get_value(opt_params)
            if opt_val > best_value:
                best_value = opt_val
                best_params = opt_params.copy()
        except Exception:
            pass
    
    # Phase 3: Use contact-graph-constrained optimization from best packings
    if best_params is not None and time.time() < deadline - 30:
        # Extract contact graph from best packing
        contact_pairs = _get_contact_graph_from_packing(best_params)
        
        # Try constrained optimization from multiple starting points
        start_points = [best_params.copy()]
        
        # Add some perturbed versions
        for _ in range(5):
            perturbed = best_params.copy()
            perturbed[1:] += np.random.randn(len(perturbed)-1) * 0.05
            start_points.append(perturbed)
        
        for sp in start_points:
            if time.time() > deadline - 10:
                break
            
            # Try constrained optimization with the contact graph
            try:
                result = _solve_with_contact_graph(sp, contact_pairs, maxiter=800)
                result_val = get_value(result)
                if result_val > best_value:
                    best_value = result_val
                    best_params = result.copy()
            except Exception:
                pass
            
            # Also try without contact graph (just all-pairs non-overlap)
            if time.time() > deadline - 10:
                break
            try:
                result2 = _solve_with_contact_graph(sp, [], maxiter=500)
                result2_val = get_value(result2)
                if result2_val > best_value:
                    best_value = result2_val
                    best_params = result2.copy()
            except Exception:
                pass
    
    # Phase 4: Final polish with penalty-based SLSQP
    if best_params is not None and time.time() < deadline - 5:
        try:
            result = minimize(
                objective_and_grad,
                best_params,
                method='SLSQP',
                jac=True,
                bounds=bounds_penalty,
                options={'maxiter': 2000, 'ftol': 1e-12}
            )
            opt_val = get_value(result.x)
            if opt_val > best_value:
                best_value = opt_val
                best_params = result.x.copy()
        except Exception:
            pass

    # Fallback if no valid solution found
    if best_params is None or best_value < 0:
        w = 1.0
        h = 1.0
        side = 5
        result = np.zeros((N, 3))
        for i in range(N):
            row = i // side
            col = i % side
            result[i, 0] = (col + 0.5) / side
            result[i, 1] = (row + 0.5) / side
            result[i, 2] = 0.48 / side
        return result

    # Extract final result
    w_final = np.clip(best_params[0], 0.1, 1.9)
    h_final = 2.0 - w_final
    circles = best_params[1:]
    
    result = np.zeros((N, 3))
    for i in range(N):
        result[i, 0] = circles[3*i]
        result[i, 1] = circles[3*i+1]
        result[i, 2] = circles[3*i+2]
    
    # Final verification: ensure bounding box perimeter <= 4
    xs = result[:, 0]
    ys = result[:, 1]
    rs = result[:, 2]
    min_x = np.min(xs - rs)
    max_x = np.max(xs + rs)
    min_y = np.min(ys - rs)
    max_y = np.max(ys + rs)
    bbox_w = max_x - min_x
    bbox_h = max_y - min_y
    bbox_perim = 2 * (bbox_w + bbox_h)
    
    if bbox_perim > 4.0 + 1e-5:
        scale = 4.0 / bbox_perim
        cx = (min_x + max_x) / 2
        cy = (min_y + max_y) / 2
        for i in range(N):
            result[i, 0] = cx + (result[i, 0] - cx) * scale
            result[i, 1] = cy + (result[i, 1] - cy) * scale
            result[i, 2] = result[i, 2] * scale
    
    return result
