"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize, LinearConstraint

N = 26

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def _initial_grid(n):
    """Create an initial grid placement for n circles."""
    side = int(np.ceil(np.sqrt(n)))
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) < n:
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side - 0.001
                cells.append((x, y, r))
    return np.array(cells)


def _check_validity(circles, tol=1e-7):
    """Check if all circles are valid (inside unit square, non-overlapping)."""
    n = len(circles)
    for i in range(n):
        x, y, r = circles[i]
        if x - r < -tol or x + r > 1 + tol or y - r < -tol or y + r > 1 + tol:
            return False
        if r <= 0:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            xi, yi, ri = circles[i]
            xj, yj, rj = circles[j]
            dist_sq = (xi - xj) ** 2 + (yi - yj) ** 2
            if dist_sq < (ri + rj - tol) ** 2:
                return False
    return True


def _objective_and_constraints(circles, n):
    """
    Set up the optimization problem.
    Variables: x1, y1, r1, x2, y2, r2, ...
    Objective: maximize sum(r_i) -> minimize -sum(r_i)
    Constraints:
      1. x_i - r_i >= 0
      2. x_i + r_i <= 1
      3. y_i - r_i >= 0
      4. y_i + r_i <= 1
      5. r_i > 0 (handled by bounds)
      6. (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2 for i < j
    """
    def objective(v):
        r = v[2::3]
        return -np.sum(r)

    def jac_objective(v):
        j = np.zeros_like(v)
        j[2::3] = -1.0
        return j

    # Build constraints
    # Wall constraints: for each circle i
    # x_i - r_i >= 0 -> -(x_i - r_i) <= 0
    # x_i + r_i <= 1 -> (x_i + r_i - 1) <= 0
    # y_i - r_i >= 0 -> -(y_i - r_i) <= 0
    # y_i + r_i <= 1 -> (y_i + r_i - 1) <= 0

    constraints = []

    # Wall constraints
    for i in range(n):
        xi_idx = 3 * i
        yi_idx = 3 * i + 1
        ri_idx = 3 * i + 2

        # x_i - r_i >= 0
        def wall_x_min(v, i=i):
            return -(v[3*i] - v[3*i+2])
        def wall_x_min_jac(v, i=i):
            j = np.zeros_like(v)
            j[3*i] = -1.0
            j[3*i+2] = 1.0
            return j
        constraints.append({'type': 'ineq', 'fun': wall_x_min, 'jac': wall_x_min_jac})

        # x_i + r_i <= 1
        def wall_x_max(v, i=i):
            return v[3*i] + v[3*i+2] - 1
        def wall_x_max_jac(v, i=i):
            j = np.zeros_like(v)
            j[3*i] = 1.0
            j[3*i+2] = 1.0
            return j
        constraints.append({'type': 'ineq', 'fun': wall_x_max, 'jac': wall_x_max_jac})

        # y_i - r_i >= 0
        def wall_y_min(v, i=i):
            return -(v[3*i+1] - v[3*i+2])
        def wall_y_min_jac(v, i=i):
            j = np.zeros_like(v)
            j[3*i+1] = -1.0
            j[3*i+2] = 1.0
            return j
        constraints.append({'type': 'ineq', 'fun': wall_y_min, 'jac': wall_y_min_jac})

        # y_i + r_i <= 1
        def wall_y_max(v, i=i):
            return v[3*i+1] + v[3*i+2] - 1
        def wall_y_max_jac(v, i=i):
            j = np.zeros_like(v)
            j[3*i+1] = 1.0
            j[3*i+2] = 1.0
            return j
        constraints.append({'type': 'ineq', 'fun': wall_y_max, 'jac': wall_y_max_jac})

    # Non-overlap constraints: (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2
    # Equivalent: (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 >= 0
    for i in range(n):
        for j in range(i + 1, n):
            xi_idx = 3 * i
            yi_idx = 3 * i + 1
            ri_idx = 3 * i + 2
            xj_idx = 3 * j
            yj_idx = 3 * j + 1
            rj_idx = 3 * j + 2

            def non_overlap(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                rsum = v[3*i+2] + v[3*j+2]
                return dx*dx + dy*dy - rsum*rsum

            def non_overlap_jac(v, i=i, j=j):
                jv = np.zeros_like(v)
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                rsum = v[3*i+2] + v[3*j+2]
                # d/dx_i: 2*dx
                jv[3*i] = 2*dx
                # d/dx_j: -2*dx
                jv[3*j] = -2*dx
                # d/dy_i: 2*dy
                jv[3*i+1] = 2*dy
                # d/dy_j: -2*dy
                jv[3*j+1] = -2*dy
                # d/dr_i: -2*rsum
                jv[3*i+2] = -2*rsum
                # d/dr_j: -2*rsum
                jv[3*j+2] = -2*rsum
                return jv

            constraints.append({'type': 'ineq', 'fun': non_overlap, 'jac': non_overlap_jac})

    # Bounds: r_i > 0, x_i in [0,1], y_i in [0,1]
    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.0))  # x_i
        bounds.append((0.0, 1.0))  # y_i
        bounds.append((1e-6, 0.5))  # r_i

    return objective, jac_objective, constraints, bounds


def _solve_once(x0, n, time_budget=30.0):
    """Solve the optimization problem from initial guess x0."""
    objective, jac_objective, constraints, bounds = _objective_and_constraints(x0, n)

    start_time = time.time()
    
    try:
        result = minimize(
            objective,
            x0,
            jac=jac_objective,
            bounds=bounds,
            constraints=constraints,
            method='SLSQP',
            options={'maxiter': 200, 'ftol': 1e-10, 'disp': False}
        )
        
        elapsed = time.time() - start_time
        if elapsed > time_budget:
            return None
        
        if result.success or result.fun < 0:
            return result.x
        else:
            # Still return the result if it's valid
            return result.x
    except Exception:
        return None


def construct():
    n = N
    start_time = time.time()
    deadline = 120.0  # seconds
    safety_margin = 10.0  # seconds
    
    # Initial grid placement
    circles0 = _initial_grid(n)
    x0 = circles0.flatten()
    
    best_circles = circles0.copy()
    best_value = np.sum(circles0[:, 2])
    
    # Try multiple random restarts
    np.random.seed(42)
    
    # First try the grid
    result = _solve_once(x0, n, time_budget=30.0)
    if result is not None:
        circles = result.reshape(n, 3)
        if _check_validity(circles):
            val = np.sum(circles[:, 2])
            if val > best_value:
                best_value = val
                best_circles = circles.copy()
    
    # Check time
    if time.time() - start_time > deadline - safety_margin:
        return best_circles
    
    # Try random restarts
    max_restarts = 5
    for restart in range(max_restarts):
        if time.time() - start_time > deadline - safety_margin:
            break
        
        # Random initial placement
        x_rand = np.random.uniform(0.1, 0.9, size=n)
        y_rand = np.random.uniform(0.1, 0.9, size=n)
        r_rand = np.random.uniform(0.02, 0.08, size=n)
        x0_rand = np.column_stack([x_rand, y_rand, r_rand]).flatten()
        
        result = _solve_once(x0_rand, n, time_budget=30.0)
        if result is not None:
            circles = result.reshape(n, 3)
            if _check_validity(circles):
                val = np.sum(circles[:, 2])
                if val > best_value:
                    best_value = val
                    best_circles = circles.copy()
    
    # Final polish: run SLSQP with tighter tolerance on the best solution
    if time.time() - start_time < deadline - safety_margin:
        x_best = best_circles.flatten()
        objective, jac_objective, constraints, bounds = _objective_and_constraints(x_best, n)
        try:
            result = minimize(
                objective,
                x_best,
                jac=jac_objective,
                bounds=bounds,
                constraints=constraints,
                method='SLSQP',
                options={'maxiter': 500, 'ftol': 1e-12, 'disp': False}
            )
            circles = result.x.reshape(n, 3)
            if _check_validity(circles):
                val = np.sum(circles[:, 2])
                if val > best_value:
                    best_value = val
                    best_circles = circles.copy()
        except Exception:
            pass
    
    return best_circles
# EVOLVE-BLOCK-END
