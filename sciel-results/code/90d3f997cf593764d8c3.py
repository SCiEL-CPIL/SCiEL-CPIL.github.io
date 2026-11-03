"""Circle packing in unit square: 26 circles, maximize sum of radii."""
import numpy as np
from scipy.optimize import minimize
import time

N = 26

# EVOLVE-BLOCK-START

def _compute_radii(pos, n):
    """
    Compute radii for each circle given positions.
    r_i = min(
        min(x_i, y_i, 1-x_i, 1-y_i),  # boundary
        min over j != i of (dist(i,j) - 0)  # this is wrong, we need r_i + r_j <= dist
    )
    
    Actually, the constraint is r_i + r_j <= dist(i,j) and r_i <= boundary_dist(i).
    This is a system of linear inequalities. We want to maximize sum(r_i).
    This is a linear program!
    
    However, solving an LP for every objective evaluation is too slow.
    
    Alternative: Use a fixed point iteration or a simpler heuristic.
    For a given set of positions, the optimal radii can be found by solving:
    Maximize sum(r_i)
    Subject to:
      r_i <= b_i  (boundary distance)
      r_i + r_j <= d_ij  (distance between i and j)
      r_i >= 0
    
    This is a standard LP. We can use scipy.optimize.linprog.
    But linprog might be slow if called too frequently.
    
    Let's try a different approach: 
    1. Start with small radii.
    2. Iteratively increase radii as much as possible (greedy).
    3. Use this sum as the objective for position optimization.
    
    Or, even better:
    Use the fact that for a fixed topology (which contacts exist), the radii are determined.
    But topology changes.
    
    Let's stick to the LP approach but make it efficient.
    Variables: r_0, ..., r_{n-1}
    Maximize: sum(r_i)
    Constraints:
      r_i <= b_i
      r_i + r_j <= d_ij
      r_i >= 0
    
    This is a dense LP with n variables and ~n + n(n-1)/2 constraints.
    For n=26, that's 26 variables and ~351 constraints. This is very small.
    linprog should be fast enough.
    """
    # Boundary distances
    b = np.minimum(np.minimum(pos[:, 0], pos[:, 1]), 1.0 - np.minimum(pos[:, 0], pos[:, 1]))
    
    # Pairwise distances
    d = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            dist = np.linalg.norm(pos[i] - pos[j])
            d[i, j] = dist
            d[j, i] = dist
    
    # LP formulation
    # c = -1 for all (maximize sum => minimize -sum)
    c = -np.ones(n)
    
    # A_ub, b_ub
    # r_i + r_j <= d_ij
    A_ub = []
    b_ub = []
    
    for i in range(n):
        for j in range(i+1, n):
            row = np.zeros(n)
            row[i] = 1
            row[j] = 1
            A_ub.append(row)
            b_ub.append(d[i, j])
    
    # r_i <= b_i
    # This can be handled by bounds, but let's add to A_ub for clarity if needed.
    # Actually, bounds are more efficient.
    
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    # Bounds: 0 <= r_i <= b_i
    bounds = [(0, b[i]) for i in range(n)]
    
    try:
        res = minimize(lambda r: -np.sum(r), np.zeros(n), method='SLSQP',
                       bounds=bounds,
                       constraints=[{'type': 'ineq', 'fun': lambda r, i=i, j=j: d[i,j] - r[i] - r[j]} for i in range(n) for j in range(i+1, n)])
        # SLSQP might be slow. Let's try linprog.
        # scipy.optimize.linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        # linprog is better for LPs.
        
        # Let's use linprog properly.
        # The objective is max sum(r) => min -sum(r)
        # c = -1
        # A_ub * r <= b_ub
        # bounds
        
        # Wait, the previous code used minimize. Let's switch to linprog for speed.
        pass
    except:
        pass
    
    # Let's just use a simple greedy update which is faster and often good enough.
    # Or, solve the LP with linprog.
    
    # Let's implement the linprog version correctly.
    # c = -1
    # A_ub: r_i + r_j <= d_ij
    # bounds: (0, b_i)
    
    # For n=26, A_ub is 325 x 26. This is tiny.
    
    from scipy.optimize import linprog
    
    # c: minimize c.T @ x
    c = -np.ones(n)
    
    # A_ub: inequality constraints A_ub @ x <= b_ub
    # We already built A_ub and b_ub above.
    
    # bounds: (min, max) for each variable
    # b can be negative if positions are outside [0,1], but we assume valid positions.
    # Clip b to 0 just in case.
    b = np.maximum(b, 0)
    bounds = [(0, b[i]) for i in range(n)]
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return -res.fun, res.x
        else:
            # Fallback: greedy
            radii = np.zeros(n)
            for _ in range(50):
                changed = False
                for i in range(n):
                    max_r = b[i]
                    for j in range(n):
                        if i != j:
                            max_r = min(max_r, d[i, j] - radii[j])
                    if max_r > radii[i] + 1e-6:
                        radii[i] = max_r
                        changed = True
                if not changed:
                    break
            return np.sum(radii), radii
    except Exception:
        # Fallback: greedy
        radii = np.zeros(n)
        for _ in range(50):
            changed = False
            for i in range(n):
                max_r = b[i]
                for j in range(n):
                    if i != j:
                        max_r = min(max_r, d[i, j] - radii[j])
                if max_r > radii[i] + 1e-6:
                    radii[i] = max_r
                    changed = True
            if not changed:
                break
        return np.sum(radii), radii

def _objective(params, n):
    """
    Objective function for position optimization.
    params: (n, 2) array of positions.
    Returns negative sum of radii (to be minimized).
    """
    pos = params.reshape(n, 2)
    
    # Check boundaries (soft penalty)
    # We assume the optimizer stays within [0,1] via bounds.
    
    sum_r, radii = _compute_radii(pos, n)
    
    # Small penalty for positions too close to boundary? No, bounds handle it.
    
    return -sum_r

def _initial_grid(n):
    """Create a grid layout as initial guess."""
    # 5x5 grid = 25, plus 1?
    # Or 5x6 = 30, take 26.
    # Let's use a 5x5 grid and place the 26th in the center of one cell?
    # Or just a 5x6 grid cropped.
    
    side = int(np.ceil(np.sqrt(n)))
    # 5x5=25. 5x6=30.
    # Let's use 5 rows, 6 cols.
    rows = 5
    cols = 6
    positions = []
    for i in range(rows):
        for j in range(cols):
            if len(positions) >= n:
                break
            x = (j + 0.5) / cols
            y = (i + 0.5) / rows
            positions.append([x, y])
    
    if len(positions) < n:
        # Fill remaining with random
        rng = np.random.RandomState(42)
        while len(positions) < n:
            positions.append(rng.uniform(0.1, 0.9, 2))
            
    return np.array(positions[:n])

def _is_valid(pos, radii, n, tol=1e-7):
    """Check if packing is valid."""
    # Boundary
    for i in range(n):
        x, y = pos[i]
        r = radii[i]
        if x - r < -tol or y - r < -tol or x + r > 1 + tol or y + r > 1 + tol:
            return False
    # Overlap
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(pos[i] - pos[j])
            if d < radii[i] + radii[j] - tol:
                return False
    return True

def _fix_to_boundary(pos, radii, n):
    """
    Fix any small violations by reducing radii.
    """
    radii = radii.copy()
    
    # Boundary clamp
    for i in range(n):
        max_r = min(pos[i, 0], pos[i, 1], 1 - pos[i, 0], 1 - pos[i, 1])
        if radii[i] > max_r:
            radii[i] = max_r
    
    # Overlap fix
    for _ in range(20):
        changed = False
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(pos[i] - pos[j])
                if radii[i] + radii[j] > d:
                    # Reduce both
                    excess = radii[i] + radii[j] - d
                    # Distribute reduction
                    reduction = excess / 2.0 + 1e-6
                    radii[i] -= reduction
                    radii[j] -= reduction
                    if radii[i] < 0: radii[i] = 0
                    if radii[j] < 0: radii[j] = 0
                    changed = True
        if not changed:
            break
    
    return radii

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_score = -np.inf
    best_result = None
    
    # Initial positions
    pos0 = _initial_grid(N)
    
    # Bounds for positions: [0, 1]
    bounds = [(0, 1)] * (2 * N)
    
    # Optimization
    # Use L-BFGS-B or Nelder-Mead?
    # L-BFGS-B is good for bound-constrained problems.
    # But the objective is non-smooth (due to min in radius calculation).
    # Nelder-Mead might be more robust.
    # Let's try L-BFGS-B first, then Nelder-Mead if it fails.
    
    # We need to provide the objective function.
    # _objective returns -sum(r). We want to minimize it.
    
    # To improve robustness, we can run multiple restarts.
    # But time is limited.
    
    # Let's try a single run with Nelder-Mead for robustness against non-smoothness.
    # Or L-BFGS-B with finite differences?
    
    # Let's use L-BFGS-B.
    
    try:
        res = minimize(_objective, pos0.flatten(), method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12})
        
        if res.success:
            pos_opt = res.x.reshape(N, 2)
            sum_r, radii_opt = _compute_radii(pos_opt, N)
            
            # Fix any numerical issues
            radii_opt = _fix_to_boundary(pos_opt, radii_opt, N)
            
            if _is_valid(pos_opt, radii_opt, N):
                score = np.sum(radii_opt)
                if score > best_score:
                    best_score = score
                    result = np.zeros((N, 3))
                    result[:, :2] = pos_opt
                    result[:, 2] = radii_opt
                    best_result = result
    except Exception as e:
        pass
    
    # If L-BFGS-B failed or didn't find a good solution, try Nelder-Mead
    # Or try a different initial configuration
    
    if time.time() < deadline - 20:
        # Try a random restart
        rng = np.random.RandomState(123)
        pos_rand = rng.uniform(0.1, 0.9, (N, 2))
        
        try:
            res = minimize(_objective, pos_rand.flatten(), method='Nelder-Mead',
                           options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-12})
            
            if res.success:
                pos_opt = res.x.reshape(N, 2)
                # Clip to [0,1]
                pos_opt = np.clip(pos_opt, 0, 1)
                sum_r, radii_opt = _compute_radii(pos_opt, N)
                radii_opt = _fix_to_boundary(pos_opt, radii_opt, N)
                
                if _is_valid(pos_opt, radii_opt, N):
                    score = np.sum(radii_opt)
                    if score > best_score:
                        best_score = score
                        result = np.zeros((N, 3))
                        result[:, :2] = pos_opt
                        result[:, 2] = radii_opt
                        best_result = result
        except Exception:
            pass
    
    # Fallback
    if best_result is None:
        # Return a valid small packing
        pos = _initial_grid(N)
        radii = np.full(N, 0.05)
        radii = _fix_to_boundary(pos, radii, N)
        best_result = np.zeros((N, 3))
        best_result[:, :2] = pos
        best_result[:, 2] = radii
    
    return best_result

# EVOLVE-BLOCK-END
