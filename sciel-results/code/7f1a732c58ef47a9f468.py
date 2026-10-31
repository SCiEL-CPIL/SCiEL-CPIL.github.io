"""Circle packing in unit square - 26 circles maximizing sum of radii.
Strategy: Contact-graph-based convex QP approach.
1. Start with a reasonable initial packing.
2. Build a contact graph from current circle positions.
3. Solve a convex quadratic program to maximize sum of radii subject to
   linearized non-overlap constraints on the contact graph and boundary constraints.
4. Update positions, rebuild contact graph, iterate.
5. Use multiple restarts and a final SLSQP polish.
"""

import numpy as np
import time
import random
from scipy.optimize import minimize
from scipy.spatial import Delaunay

N = 26

# EVOLVE-BLOCK-START

def _is_valid(circles, tol=1e-8):
    """Check validity: inside unit square, non-overlapping, positive radii."""
    n = len(circles)
    for i in range(n):
        x, y, r = circles[i]
        if r < -tol:
            return False
        if x - r < -tol or x + r > 1.0 + tol:
            return False
        if y - r < -tol or y + r > 1.0 + tol:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            xi, yi, ri = circles[i]
            xj, yj, rj = circles[j]
            dist = np.sqrt((xi - xj)**2 + (yi - yj)**2)
            if dist < ri + rj - tol:
                return False
    return True


def _post_process(circles, tol=1e-8):
    """Shrink circles slightly to ensure strict validity, clip to bounds."""
    out = []
    for x, y, r in circles:
        r = max(0.0, r - tol)
        x = min(1.0 - r, max(r, x))
        y = min(1.0 - r, max(r, y))
        out.append((x, y, r))
    return out


def _sum_radii(circles):
    return sum(c[2] for c in circles)


def _init_hexagonal():
    """Hexagonal packing initial layout for 26 circles."""
    circles = []
    r = 0.075
    rows = [5, 5, 5, 5, 6]
    row_spacing = r * np.sqrt(3)
    col_spacing = 2 * r
    
    row_idx = 0
    for count in rows:
        offset = 0 if row_idx % 2 == 0 else col_spacing / 2
        total_width = col_spacing * (count - 1) + offset
        start_x = (1.0 - total_width) / 2.0
        y = r + row_spacing * row_idx + 0.03
        for col in range(count):
            x = start_x + offset + col * col_spacing
            if len(circles) < N:
                circles.append((x, y, r))
        row_idx += 1
    
    circles = circles[:N]
    while len(circles) < N:
        circles.append((0.5, 0.5, 0.01))
    return circles


def _init_grid():
    """Simple grid initialization."""
    side = int(np.ceil(np.sqrt(N)))
    circles = []
    for k in range(N):
        i = k % side
        j = k // side
        x = (i + 0.5) / side
        y = (j + 0.5) / side
        r = 0.4 / side
        circles.append((x, y, r))
    return circles


def _init_corners():
    """Initialization with circles at corners and edges."""
    circles = []
    circles.append((0.08, 0.08, 0.08))
    circles.append((0.92, 0.08, 0.08))
    circles.append((0.08, 0.92, 0.08))
    circles.append((0.92, 0.92, 0.08))
    
    circles.append((0.5, 0.05, 0.06))
    circles.append((0.5, 0.95, 0.06))
    circles.append((0.05, 0.5, 0.06))
    circles.append((0.95, 0.5, 0.06))
    
    random.seed(42)
    while len(circles) < N:
        x = random.uniform(0.15, 0.85)
        y = random.uniform(0.15, 0.85)
        r = random.uniform(0.04, 0.07)
        circles.append((x, y, r))
    return circles


def _init_random(seed=None):
    """Random initialization."""
    if seed is not None:
        random.seed(seed)
    circles = []
    for _ in range(N):
        x = random.uniform(0.1, 0.9)
        y = random.uniform(0.1, 0.9)
        r = random.uniform(0.03, 0.08)
        circles.append((x, y, r))
    return circles


def _build_contact_graph(circles, margin=1.05):
    """Build contact graph: edges between circles that are close to touching.
    Also include boundary contacts (circle touching a wall).
    Returns list of (i, j) pairs for circle-circle contacts,
    and list of (i, wall) for boundary contacts where wall in {0,1,2,3}
    (0=left, 1=right, 2=bottom, 3=top).
    """
    n = len(circles)
    cc_edges = []
    bc_edges = []  # (i, wall)
    
    for i in range(n):
        xi, yi, ri = circles[i]
        # Boundary contacts
        if xi - ri < 1e-6 * margin:
            bc_edges.append((i, 0))  # left
        if xi + ri > 1.0 - 1e-6 * margin:
            bc_edges.append((i, 1))  # right
        if yi - ri < 1e-6 * margin:
            bc_edges.append((i, 2))  # bottom
        if yi + ri > 1.0 - 1e-6 * margin:
            bc_edges.append((i, 3))  # top
        
        for j in range(i + 1, n):
            xj, yj, rj = circles[j]
            dist = np.sqrt((xi - xj)**2 + (yi - yj)**2)
            if dist < (ri + rj) * margin:
                cc_edges.append((i, j))
    
    return cc_edges, bc_edges


def _solve_qp_for_radii(circles, cc_edges, bc_edges, max_iter=50, tol=1e-8):
    """Solve a convex QP: maximize sum of radii subject to linearized constraints.
    
    Variables: r_0, r_1, ..., r_{n-1}
    Objective: max sum(r_i)
    Constraints:
      For each (i,j) in cc_edges:
        Linearized: (xi-xj)*dr_i + (yi-yj)*dr_j ... actually we fix centers
        and only optimize radii. The constraint is:
        dist(i,j) >= r_i + r_j
        So: r_i + r_j <= dist(i,j)
      For each (i, wall) in bc_edges:
        r_i <= dist_to_wall(i, wall)
      r_i >= 0
    """
    n = len(circles)
    
    # Build constraint matrix A and vector b for: A @ r <= b
    constraints = []
    rhs = []
    
    for i, j in cc_edges:
        xi, yi, _ = circles[i]
        xj, yj, _ = circles[j]
        dist = np.sqrt((xi - xj)**2 + (yi - yj)**2)
        a = np.zeros(n)
        a[i] = 1.0
        a[j] = 1.0
        constraints.append(a)
        rhs.append(dist)
    
    for i, wall in bc_edges:
        xi, yi, _ = circles[i]
        if wall == 0:  # left
            d = xi
        elif wall == 1:  # right
            d = 1.0 - xi
        elif wall == 2:  # bottom
            d = yi
        else:  # top
            d = 1.0 - yi
        a = np.zeros(n)
        a[i] = 1.0
        constraints.append(a)
        rhs.append(d)
    
    if not constraints:
        return [c[2] for c in circles]
    
    A = np.array(constraints)
    b = np.array(rhs)
    
    # Solve: max c^T r  s.t.  A r <= b, r >= 0
    # Use scipy.optimize.linprog (LP since objective and constraints are linear in r)
    from scipy.optimize import linprog
    
    c = -np.ones(n)  # minimize -sum(r)
    # Bounds: r_i >= 0
    bounds = [(0, None)] * n
    
    try:
        result = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')
        if result.success:
            return list(result.x)
    except Exception:
        pass
    
    # Fallback: simple iterative projection
    radii = [c[2] for c in circles]
    for _ in range(max_iter):
        changed = False
        for i, j in cc_edges:
            xi, yi, _ = circles[i]
            xj, yj, _ = circles[j]
            dist = np.sqrt((xi - xj)**2 + (yi - yj)**2)
            s = radii[i] + radii[j]
            if s > dist:
                scale = dist / s if s > 0 else 0.5
                radii[i] *= scale
                radii[j] *= scale
                changed = True
        for i, wall in bc_edges:
            xi, yi, _ = circles[i]
            if wall == 0:
                d = xi
            elif wall == 1:
                d = 1.0 - xi
            elif wall == 2:
                d = yi
            else:
                d = 1.0 - yi
            if radii[i] > d:
                radii[i] = d
                changed = True
        if not changed:
            break
    
    return radii


def _update_positions(circles, cc_edges, bc_edges, radii, step_size=0.1):
    """Update circle centers to better accommodate new radii.
    Use gradient descent on a penalty function for overlaps and boundary violations.
    """
    n = len(circles)
    centers = np.array([[c[0], c[1]] for c in circles])
    
    for _ in range(30):
        grad = np.zeros_like(centers)
        
        for i, j in cc_edges:
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            required = radii[i] + radii[j]
            if dist < 1e-10:
                continue
            if dist < required:
                # Push apart
                force = (required - dist) * 0.5
                fx = force * dx / dist
                fy = force * dy / dist
                grad[i, 0] += fx
                grad[i, 1] += fy
                grad[j, 0] -= fx
                grad[j, 1] -= fy
        
        for i, wall in bc_edges:
            xi, yi = centers[i]
            r = radii[i]
            if wall == 0 and xi < r:
                grad[i, 0] += (r - xi)
            elif wall == 1 and xi > 1.0 - r:
                grad[i, 0] -= (xi - (1.0 - r))
            elif wall == 2 and yi < r:
                grad[i, 1] += (r - yi)
            elif wall == 3 and yi > 1.0 - r:
                grad[i, 1] -= (yi - (1.0 - r))
        
        centers += step_size * grad
        # Clip to valid range
        for i in range(n):
            r = radii[i]
            centers[i, 0] = np.clip(centers[i, 0], r, 1.0 - r)
            centers[i, 1] = np.clip(centers[i, 1], r, 1.0 - r)
        
        step_size *= 0.9
    
    new_circles = [(centers[i, 0], centers[i, 1], radii[i]) for i in range(n)]
    return new_circles


def _contact_graph_optimize(circles, deadline, max_outer=20):
    """Iteratively solve contact-graph QP and update positions."""
    current = list(circles)
    best = list(circles)
    best_sum = _sum_radii(best)
    
    for outer in range(max_outer):
        if time.time() > deadline - 15.0:
            break
        
        cc_edges, bc_edges = _build_contact_graph(current, margin=1.15)
        
        # Solve for optimal radii given current centers and contact graph
        radii = _solve_qp_for_radii(current, cc_edges, bc_edges)
        
        # Update positions to accommodate new radii
        new_circles = _update_positions(current, cc_edges, bc_edges, radii, step_size=0.05)
        
        # Check validity
        if _is_valid(new_circles, tol=1e-6):
            new_sum = _sum_radii(new_circles)
            if new_sum > best_sum:
                best = new_circles
                best_sum = new_sum
            current = new_circles
        
        # If no improvement in a few iterations, break
        if outer > 0 and abs(_sum_radii(current) - best_sum) < 1e-10:
            break
    
    return best


def _slsqp_polish(circles, deadline):
    """Final SLSQP polish with hard constraints."""
    params = np.array([val for c in circles for val in c])
    
    bounds = [(0.0, 1.0), (0.0, 1.0), (0.001, 0.5)] * N
    
    constraints = []
    
    for i in range(N):
        constraints.append({
            'type': 'ineq',
            'fun': lambda p, i=i: p[3*i] - p[3*i+2],
        })
        constraints.append({
            'type': 'ineq',
            'fun': lambda p, i=i: 1.0 - p[3*i] - p[3*i+2],
        })
        constraints.append({
            'type': 'ineq',
            'fun': lambda p, i=i: p[3*i+1] - p[3*i+2],
        })
        constraints.append({
            'type': 'ineq',
            'fun': lambda p, i=i: 1.0 - p[3*i+1] - p[3*i+2],
        })
    
    for i in range(N):
        for j in range(i+1, N):
            constraints.append({
                'type': 'ineq',
                'fun': lambda p, i=i, j=j: (p[3*i]-p[3*j])**2 + (p[3*i+1]-p[3*j+1])**2 - (p[3*i+2]+p[3*j+2])**2
            })
    
    def objective(p):
        return -sum(p[3*i+2] for i in range(N))
    
    try:
        result = minimize(
            objective, params,
            method='SLSQP', bounds=bounds, constraints=constraints,
            options={'maxiter': 300, 'ftol': 1e-12}
        )
        if result.success:
            new_circles = [(result.x[3*i], result.x[3*i+1], result.x[3*i+2]) for i in range(N)]
            return new_circles
    except Exception:
        pass
    
    return circles


def _softplus(x, beta=50):
    """Smooth approximation of max(0, x)."""
    return np.log1p(np.exp(beta * x)) / beta


def _penalty_objective(params, penalty_weight=100.0):
    """Penalty-based objective for unconstrained optimization."""
    n = N
    circles = [(params[3*i], params[3*i+1], params[3*i+2]) for i in range(n)]
    
    penalty = 0.0
    
    for i in range(n):
        x, y, r = circles[i]
        # Boundary penalties
        penalty += _softplus(r - x)
        penalty += _softplus(r - y)
        penalty += _softplus(r - (1.0 - x))
        penalty += _softplus(r - (1.0 - y))
    
    for i in range(n):
        for j in range(i+1, n):
            xi, yi, ri = circles[i]
            xj, yj, rj = circles[j]
            dist = np.sqrt((xi-xj)**2 + (yi-yj)**2)
            penalty += _softplus(ri + rj - dist)
    
    return -sum(c[2] for c in circles) + penalty_weight * penalty


def _lbfgs_optimize(circles, deadline, max_iter=200):
    """L-BFGS-B optimization with penalty method."""
    params = np.array([val for c in circles for val in c])
    bounds = [(0.0, 1.0), (0.0, 1.0), (0.001, 0.5)] * N
    
    try:
        result = minimize(
            _penalty_objective, params,
            method='L-BFGS-B', bounds=bounds,
            options={'maxiter': max_iter, 'ftol': 1e-12}
        )
        if result.success:
            new_circles = [(result.x[3*i], result.x[3*i+1], result.x[3*i+2]) for i in range(N)]
            return new_circles
    except Exception:
        pass
    
    return circles


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Multiple diverse initializations
    inits = [
        _init_hexagonal(),
        _init_grid(),
        _init_corners(),
        _init_random(42),
        _init_random(123),
        _init_random(777),
        _init_random(999),
    ]
    
    best_circles = inits[0]
    best_sum = _sum_radii(best_circles)
    
    # Phase 1: Contact graph optimization from each init
    for init_circles in inits:
        if time.time() > deadline - 40.0:
            break
        
        # First do a quick L-BFGS to get a decent starting point
        if time.time() < deadline - 40.0:
            lbfgs_result = _lbfgs_optimize(init_circles, deadline, max_iter=50)
            if _is_valid(lbfgs_result, tol=1e-6) and _sum_radii(lbfgs_result) > _sum_radii(init_circles):
                init_circles = lbfgs_result
        
        # Then contact graph optimization
        cg_result = _contact_graph_optimize(init_circles, deadline, max_outer=15)
        cg_sum = _sum_radii(cg_result)
        
        if cg_sum > best_sum and _is_valid(cg_result):
            best_circles = cg_result
            best_sum = cg_sum
    
    # Phase 2: SLSQP polish on the best result
    if time.time() < deadline - 10.0:
        polished = _slsqp_polish(best_circles, deadline)
        pol_sum = _sum_radii(polished)
        
        if pol_sum > best_sum and _is_valid(polished):
            best_circles = polished
            best_sum = pol_sum
    
    # Phase 3: One more round of contact graph + SLSQP
    if time.time() < deadline - 5.0:
        cg2 = _contact_graph_optimize(best_circles, deadline, max_outer=10)
        if _is_valid(cg2) and _sum_radii(cg2) > best_sum:
            best_circles = cg2
            best_sum = _sum_radii(best_circles)
        
        if time.time() < deadline - 3.0:
            polished2 = _slsqp_polish(best_circles, deadline)
            pol2_sum = _sum_radii(polished2)
            if pol2_sum > best_sum and _is_valid(polished2):
                best_circles = polished2
                best_sum = pol2_sum
    
    # Post-process
    best_circles = _post_process(best_circles)
    
    # Verify and return
    if not _is_valid(best_circles):
        # Fallback
        best_circles = _post_process(_init_grid())
    
    return np.array(best_circles)

# EVOLVE-BLOCK-END
