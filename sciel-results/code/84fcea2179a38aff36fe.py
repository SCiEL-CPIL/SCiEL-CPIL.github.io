"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import linprog, minimize
import time

N = 26

# EVOLVE-BLOCK-START
def construct():
    """
    Pack 26 circles in [0,1]^2 maximizing sum of radii.
    
    Strategy:
    1. Generate multiple random initial configurations.
    2. For each, use gradient-based optimization (L-BFGS-B) with a smooth
       penalty for overlaps and boundary violations to maximize sum of radii.
    3. Use a graph-based LP polish: build contact graph, then solve an LP
       to maximize sum of radii given fixed positions (or allow small position
       adjustments via a final local search).
    4. Return the best valid packing found within the time budget.
    """
    deadline = time.time() + 140.0  # 10s safety margin before 150s kill
    
    def validate(circles):
        """Check if circles are valid (inside [0,1]^2, disjoint interiors)."""
        if len(circles) != N:
            return False
        for x, y, r in circles:
            if r < 0:
                return False
            if x - r < -1e-9 or x + r > 1 + 1e-9:
                return False
            if y - r < -1e-9 or y + r > 1 + 1e-9:
                return False
        for i in range(N):
            for j in range(i+1, N):
                xi, yi, ri = circles[i]
                xj, yj, rj = circles[j]
                dist = np.sqrt((xi-xj)**2 + (yi-yj)**2)
                if dist < ri + rj - 1e-9:
                    return False
        return True
    
    def sum_radii(circles):
        return np.sum([c[2] for c in circles])
    
    def penalty_objective(params, N_circles):
        """
        Objective: -sum(r_i) + penalty for overlaps and boundary violations.
        params: array of shape (3*N_circles) = [x0,y0,r0, x1,y1,r1, ...]
        """
        x = params[0::3]
        y = params[1::3]
        r = params[2::3]
        
        # Sum of radii to maximize -> minimize negative
        obj = -np.sum(r)
        
        # Boundary penalty: circles must be inside [0,1]^2
        # x - r >= 0, x + r <= 1, y - r >= 0, y + r <= 1
        bound_viol = 0.0
        bound_viol += np.sum(np.maximum(0, r - x))**2
        bound_viol += np.sum(np.maximum(0, r - (1 - x)))**2
        bound_viol += np.sum(np.maximum(0, r - y))**2
        bound_viol += np.sum(np.maximum(0, r - (1 - y)))**2
        obj += 100.0 * bound_viol
        
        # Overlap penalty: for each pair, dist >= ri + rj
        overlap_viol = 0.0
        for i in range(N_circles):
            for j in range(i+1, N_circles):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy + 1e-12)
                viol = r[i] + r[j] - dist
                if viol > 0:
                    overlap_viol += viol**2
        obj += 100.0 * overlap_viol
        
        return obj
    
    def lp_polish(circles):
        """
        Given fixed positions, solve LP to maximize sum of radii
        subject to:
          x_i - r_i >= 0
          x_i + r_i <= 1
          y_i - r_i >= 0
          y_i + r_i <= 1
          (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2  [nonlinear, but we linearize]
        
        For the LP polish, we use the contact graph approach:
        Build a graph where edges connect circles that are "close" (within some threshold).
        For each edge (i,j), the constraint is:
          r_i + r_j <= dist_ij  (since positions are fixed, dist is known)
        
        This is a linear constraint! So we can solve:
          maximize sum(r_i)
          subject to:
            r_i <= x_i
            r_i <= 1 - x_i
            r_i <= y_i
            r_i <= 1 - y_i
            r_i + r_j <= dist_ij for all close pairs (i,j)
            r_i >= 0
        """
        x = np.array([c[0] for c in circles])
        y = np.array([c[1] for c in circles])
        
        # Compute all pairwise distances
        dists = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                if i != j:
                    dists[i,j] = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
        
        # LP: maximize sum(r)
        # c = -ones(N) (for minimization)
        c = -np.ones(N)
        
        # Inequality constraints: A_ub @ r <= b_ub
        A_rows = []
        b_rows = []
        
        # r_i <= x_i  ->  r_i <= x_i
        for i in range(N):
            row = np.zeros(N)
            row[i] = 1
            A_rows.append(row)
            b_rows.append(x[i])
        
        # r_i <= 1 - x_i
        for i in range(N):
            row = np.zeros(N)
            row[i] = 1
            A_rows.append(row)
            b_rows.append(1 - x[i])
        
        # r_i <= y_i
        for i in range(N):
            row = np.zeros(N)
            row[i] = 1
            A_rows.append(row)
            b_rows.append(y[i])
        
        # r_i <= 1 - y_i
        for i in range(N):
            row = np.zeros(N)
            row[i] = 1
            A_rows.append(row)
            b_rows.append(1 - y[i])
        
        # r_i + r_j <= dist_ij for all pairs
        for i in range(N):
            for j in range(i+1, N):
                row = np.zeros(N)
                row[i] = 1
                row[j] = 1
                A_rows.append(row)
                b_rows.append(dists[i,j])
        
        A_ub = np.array(A_rows)
        b_ub = np.array(b_rows)
        
        # Bounds: r_i >= 0
        bounds = [(0, None)] * N
        
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if res.success:
            new_radii = res.x
            # Verify validity
            new_circles = [(x[i], y[i], new_radii[i]) for i in range(N)]
            if validate(new_circles):
                return new_circles
        return circles
    
    def local_search(circles, max_iters=200):
        """
        Simple local search: randomly perturb positions and radii,
        accept if sum of radii increases and configuration is valid.
        """
        best = circles
        best_sum = sum_radii(best)
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            
            # Perturb a random circle
            idx = np.random.randint(N)
            dx = np.random.normal(0, 0.02)
            dy = np.random.normal(0, 0.02)
            dr = np.random.normal(0, 0.01)
            
            new_circles = list(best)
            x, y, r = new_circles[idx]
            new_circles[idx] = (x + dx, y + dy, max(0.001, r + dr))
            
            if validate(new_circles):
                s = sum_radii(new_circles)
                if s > best_sum:
                    best = new_circles
                    best_sum = s
        
        return best
    
    # Initial configurations
    best_circles = None
    best_sum = -1
    
    # Configuration 1: Grid
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
    grid_circles = cells[:N]
    if validate(grid_circles):
        best_circles = grid_circles
        best_sum = sum_radii(grid_circles)
    
    # Try multiple random starts with gradient optimization
    n_starts = 30
    for start in range(n_starts):
        if time.time() > deadline:
            break
        
        # Random initialization: place circles with small radii
        x = np.random.uniform(0.1, 0.9, N)
        y = np.random.uniform(0.1, 0.9, N)
        r = np.full(N, 0.02)
        
        params = np.zeros(3*N)
        params[0::3] = x
        params[1::3] = y
        params[2::3] = r
        
        # Use L-BFGS-B to maximize sum of radii with penalties
        try:
            res = minimize(
                penalty_objective,
                params,
                args=(N,),
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-8}
            )
            if time.time() > deadline:
                break
            
            opt_params = res.x
            new_circles = [(opt_params[3*i], opt_params[3*i+1], max(0, opt_params[3*i+2])) for i in range(N)]
            
            if validate(new_circles):
                s = sum_radii(new_circles)
                if s > best_sum:
                    best_circles = new_circles
                    best_sum = s
        except Exception:
            pass
    
    # LP polish on the best configuration
    if best_circles is not None and time.time() < deadline:
        polished = lp_polish(best_circles)
        s = sum_radii(polished)
        if s > best_sum:
            best_circles = polished
            best_sum = s
    
    # Local search on the best configuration
    if best_circles is not None and time.time() < deadline:
        searched = local_search(best_circles, max_iters=100)
        s = sum_radii(searched)
        if s > best_sum:
            best_circles = searched
            best_sum = s
        
        # Another LP polish after local search
        if time.time() < deadline:
            polished2 = lp_polish(best_circles)
            s = sum_radii(polished2)
            if s > best_sum:
                best_circles = polished2
                best_sum = s
    
    # Final fallback: return grid if nothing better
    if best_circles is None:
        best_circles = grid_circles
    
    # Final validation and return
    result = np.array(best_circles)
    if not validate(result.tolist()):
        # Fallback to grid
        result = np.array(grid_circles)
    
    return result
# EVOLVE-BLOCK-END
