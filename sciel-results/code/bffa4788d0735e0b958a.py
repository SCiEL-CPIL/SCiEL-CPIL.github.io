"""Circle packing in unit square: pack 26 circles, maximize sum of radii.
Strategy: Combinatorial contact graph search + Newton solver.
"""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import minimize

def _eval_packing(xy, r):
    """
    Evaluate a packing: returns (total_radius, is_valid).
    Valid means all circles inside [0,1]^2 and disjoint interiors.
    """
    n = len(xy)
    # Check walls
    for i in range(n):
        x, y, ri = xy[i, 0], xy[i, 1], r[i]
        if ri < -1e-9:
            return 0.0, False
        if x - ri < -1e-9 or 1.0 - x - ri < -1e-9 or y - ri < -1e-9 or 1.0 - y - ri < -1e-9:
            return 0.0, False
    # Check pairwise distances
    for i in range(n):
        for j in range(i + 1, n):
            dx = xy[i, 0] - xy[j, 0]
            dy = xy[i, 1] - xy[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - 1e-9:
                return 0.0, False
    return float(np.sum(r)), True

def _grid_init(side):
    """Create a grid of up to N circles."""
    positions = []
    for i in range(side):
        for j in range(side):
            if len(positions) < N:
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side - 0.002
                positions.append([x, y, r])
    while len(positions) < N:
        positions.append([0.5, 0.5, 0.01])
    return np.array(positions[:N])

def _extract_graph(xy, r, tol=1e-4):
    """
    Extract contact graph from a valid packing.
    Returns adjacency matrix (n x n boolean) and wall contact matrix (n x 4 boolean).
    Wall indices: 0=Left, 1=Right, 2=Bottom, 3=Top.
    """
    n = len(xy)
    adj = np.zeros((n, n), dtype=bool)
    walls = np.zeros((n, 4), dtype=bool)
    
    for i in range(n):
        # Wall contacts
        if xy[i, 0] - r[i] < tol: walls[i, 0] = True
        if 1.0 - xy[i, 0] - r[i] < tol: walls[i, 1] = True
        if xy[i, 1] - r[i] < tol: walls[i, 2] = True
        if 1.0 - xy[i, 1] - r[i] < tol: walls[i, 3] = True
        
        for j in range(i + 1, n):
            dx = xy[i, 0] - xy[j, 0]
            dy = xy[i, 1] - xy[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            if abs(dist - (r[i] + r[j])) < tol:
                adj[i, j] = True
                adj[j, i] = True
    return adj, walls

def _solve_contact_system(adj, walls, max_iter=100, tol=1e-10):
    """
    Solve for positions and radii given a contact graph and wall contacts.
    Uses Newton's method on the system of equations.
    Variables: x_i, y_i, r_i for i=0..n-1. Total 3n variables.
    Equations:
    1. For each contact (i,j): (x_i-x_j)^2 + (y_i-y_j)^2 - (r_i+r_j)^2 = 0
    2. For each wall contact (i, w):
       Left: x_i - r_i = 0
       Right: 1 - x_i - r_i = 0
       Bottom: y_i - r_i = 0
       Top: 1 - y_i - r_i = 0
       
    We need 3n equations for 3n unknowns.
    If we have fewer, the system is underdetermined. We add "slack" constraints
    by fixing some variables or using least squares. 
    However, a standard rigid body packing is usually determined by the graph.
    To make it solvable, we assume the graph is "stiff" enough.
    We will use scipy's fsolve or a custom Newton step.
    
    For robustness, we initialize with a random valid packing close to the graph structure.
    """
    n = len(adj)
    # Initialize variables randomly
    # We need a good initialization. Let's use the current best packing if available,
    # otherwise random. Since this function is called from construct, we can pass init.
    # But for now, let's just try to solve from a random valid state that matches the topology?
    # That's hard. Instead, we solve the system directly.
    
    # Number of equations:
    n_contacts = np.sum(adj) // 2
    n_walls = np.sum(walls)
    n_eqs = n_contacts + n_walls
    
    if n_eqs < 3 * n:
        # Underdetermined. We need to fix 3n - n_eqs variables.
        # Strategy: Fix the first few circles to specific positions to break symmetry?
        # Or add penalty for deviation from a "canonical" form?
        # For simplicity, we will add dummy constraints: r_i > 0 is handled by bounds.
        # We will use scipy.optimize.least_squares to minimize residuals.
        pass

    def equations(v):
        x = v[0:n]
        y = v[n:2*n]
        r = v[2*n:3*n]
        
        res = []
        # Contact constraints
        for i in range(n):
            for j in range(i+1, n):
                if adj[i, j]:
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    res.append(dx*dx + dy*dy - (r[i] + r[j])**2)
        
        # Wall constraints
        for i in range(n):
            if walls[i, 0]: res.append(x[i] - r[i])
            if walls[i, 1]: res.append(1.0 - x[i] - r[i])
            if walls[i, 2]: res.append(y[i] - r[i])
            if walls[i, 3]: res.append(1.0 - y[i] - r[i])
            
        return np.array(res)

    def jacobian(v):
        x = v[0:n]
        y = v[n:2*n]
        r = v[2*n:3*n]
        J = np.zeros((len(equations(v)), 3*n))
        
        row = 0
        for i in range(n):
            for j in range(i+1, n):
                if adj[i, j]:
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    # d/dx_i: 2*dx, d/dx_j: -2*dx
                    # d/dy_i: 2*dy, d/dy_j: -2*dy
                    # d/dr_i: -2*(r_i+r_j), d/dr_j: -2*(r_i+r_j)
                    J[row, i] = 2*dx
                    J[row, n+j] = -2*dx
                    J[row, n+i] = 2*dy
                    J[row, 2*n+j] = -2*dy
                    J[row, 2*n+i] = -2*(r[i] + r[j])
                    J[row, 2*n+j] = -2*(r[i] + r[j])
                    row += 1
        
        for i in range(n):
            if walls[i, 0]:
                J[row, i] = 1.0
                J[row, 2*n+i] = -1.0
                row += 1
            if walls[i, 1]:
                J[row, i] = -1.0
                J[row, 2*n+i] = -1.0
                row += 1
            if walls[i, 2]:
                J[row, n+i] = 1.0
                J[row, 2*n+i] = -1.0
                row += 1
            if walls[i, 3]:
                J[row, n+i] = -1.0
                J[row, 2*n+i] = -1.0
                row += 1
                
        return J

    # Initial guess: Use a simple grid or random valid packing
    # We'll use a random valid packing as init to increase chance of convergence
    rng = np.random.RandomState(42)
    x0 = rng.uniform(0.1, 0.9, n)
    y0 = rng.uniform(0.1, 0.9, n)
    r0 = rng.uniform(0.02, 0.05, n)
    v0 = np.concatenate([x0, y0, r0])
    
    try:
        from scipy.optimize import fsolve
        # fsolve is robust for square systems. If underdetermined, it might fail or find one solution.
        # We add bounds via least_squares if fsolve fails or for better control.
        result = fsolve(equations, v0, fprime=jacobian, maxfev=1000)
        # Check if it converged
        if not np.all(np.abs(equations(result)) < 1e-6):
             return None, None
             
        x_sol = result[0:n]
        y_sol = result[n:2*n]
        r_sol = result[2*n:3*n]
        
        # Check validity
        if np.any(r_sol <= 0) or np.any(x_sol < 0) or np.any(x_sol > 1) or np.any(y_sol < 0) or np.any(y_sol > 1):
             return None, None
             
        # Verify all constraints (not just those in graph)
        # Check walls
        for i in range(n):
            if x_sol[i] - r_sol[i] < -1e-6 or 1.0 - x_sol[i] - r_sol[i] < -1e-6:
                return None, None
            if y_sol[i] - r_sol[i] < -1e-6 or 1.0 - y_sol[i] - r_sol[i] < -1e-6:
                return None, None
        # Check overlaps
        for i in range(n):
            for j in range(i+1, n):
                dx = x_sol[i] - x_sol[j]
                dy = y_sol[i] - y_sol[j]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < r_sol[i] + r_sol[j] - 1e-6:
                    return None, None
        
        return np.column_stack([x_sol, y_sol]), r_sol

    except Exception:
        return None, None

def _local_search_graph(adj, walls, deadline):
    """
    Perform local search on the contact graph.
    Mutate graph (add/remove edges, add/remove wall contacts) and solve.
    """
    current_adj = adj.copy()
    current_walls = walls.copy()
    current_xy, current_r = _solve_contact_system(current_adj, current_walls)
    
    if current_xy is None:
        # Try to recover from a valid random packing
        return None, None, 0.0, False

    current_total, _ = _eval_packing(current_xy, current_r)
    best_xy = current_xy.copy()
    best_r = current_r.copy()
    best_total = current_total
    
    temp = 0.1
    cooling = 0.99
    
    for step in range(500):
        if time.time() > deadline:
            break
            
        # Mutate graph
        new_adj = current_adj.copy()
        new_walls = current_walls.copy()
        
        # Randomly flip an edge
        i = random.randint(0, N-1)
        j = random.randint(0, N-1)
        if i != j:
            new_adj[i, j] = not new_adj[i, j]
            new_adj[j, i] = new_adj[i, j]
            
        # Randomly flip a wall contact
        if random.random() < 0.5:
            k = random.randint(0, N-1)
            w = random.randint(0, 3)
            new_walls[k, w] = not new_walls[k, w]
            
        # Solve
        new_xy, new_r = _solve_contact_system(new_adj, new_walls)
        
        if new_xy is not None:
            new_total, valid = _eval_packing(new_xy, new_r)
            if valid:
                delta = new_total - current_total
                if delta > 0 or random.random() < np.exp(delta / temp):
                    current_adj = new_adj
                    current_walls = new_walls
                    current_xy = new_xy
                    current_r = new_r
                    current_total = new_total
                    
                    if new_total > best_total:
                        best_total = new_total
                        best_xy = new_xy.copy()
                        best_r = new_r.copy()
                else:
                    # Revert
                    pass
            else:
                # Invalid, revert
                pass
        else:
            # Unsolvable, revert
            pass
            
        temp *= cooling
        
    return best_xy, best_r, best_total, True

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_xy = None
    best_r = None
    best_total = 0.0
    best_valid = False
    
    # Phase 1: Get a valid initial packing using the old grid/local search method
    # We reuse the logic from the seed program to get a good start.
    
    def _eval_packing_fast(xy, r):
        n = len(xy)
        for i in range(n):
            x, y, ri = xy[i, 0], xy[i, 1], r[i]
            if ri < 0: return 0.0, False
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

    def _grid_init_fast(side):
        positions = []
        for i in range(side):
            for j in range(side):
                if len(positions) < N:
                    x = (i + 0.5) / side
                    y = (j + 0.5) / side
                    r = 0.5 / side - 0.002
                    positions.append([x, y, r])
        while len(positions) < N:
            positions.append([0.5, 0.5, 0.01])
        return np.array(positions[:N])

    def _best_fit_local_search_fast(xy, r, deadline, grid_size=80):
        n = len(xy)
        best_total, best_valid = _eval_packing_fast(xy, r)
        best_xy = xy.copy()
        best_r = r.copy()
        
        gx = np.linspace(0.0, 1.0, grid_size)
        gy = np.linspace(0.0, 1.0, grid_size)
        GX, GY = np.meshgrid(gx, gy, indexing='ij')
        candidates = np.column_stack([GX.ravel(), GY.ravel()])
        M = len(candidates)
        
        wall_clear = np.minimum(np.minimum(candidates[:, 0], 1.0 - candidates[:, 0]),
                                np.minimum(candidates[:, 1], 1.0 - candidates[:, 1]))
        
        max_sweeps = 10
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
                
                old_r = r[i]
                if new_r > old_r + 1e-10:
                    xy[i, 0] = new_x
                    xy[i, 1] = new_y
                    r[i] = new_r
                    improved = True
                    
            total, valid = _eval_packing_fast(xy, r)
            if valid and total > best_total + 1e-10:
                best_total = total
                best_valid = True
                best_xy = xy.copy()
                best_r = r.copy()
                
            if not improved:
                break
                
        return best_xy, best_r, best_total, best_valid

    # Run Phase 1 to get a seed
    for side in [5, 6]:
        if time.time() > deadline - 20:
            break
        xy = _grid_init_fast(side)
        r = xy[:, 2].copy()
        xy = xy[:, :2].copy()
        out_xy, out_r, total, valid = _best_fit_local_search_fast(xy, r, deadline - 20)
        if valid and total > best_total:
            best_total = total
            best_valid = True
            best_xy = out_xy
            best_r = out_r
            
    if not best_valid:
        # Fallback
        side = int(np.ceil(np.sqrt(N)))
        xy = _grid_init_fast(side)
        r = xy[:, 2].copy()
        xy = xy[:, :2].copy()
        best_xy = xy
        best_r = r
        best_total, best_valid = _eval_packing_fast(xy, r)
        if not best_valid:
             # Hard fallback
             result = np.zeros((N, 3))
             for i in range(N):
                 gx = i % side
                 gy = i // side
                 result[i, 0] = (gx + 0.5) / side
                 result[i, 1] = (gy + 0.5) / side
                 result[i, 2] = 0.4 / side
             return result

    # Phase 2: Extract graph and run graph-based local search
    adj, walls = _extract_graph(best_xy, best_r, tol=1e-4)
    
    # Run multiple rounds of graph search
    for round_idx in range(5):
        if time.time() > deadline - 10:
            break
        new_xy, new_r, new_total, success = _local_search_graph(adj, walls, deadline - 10)
        if success and new_total > best_total:
            best_total = new_total
            best_xy = new_xy
            best_r = new_r
            # Update graph
            adj, walls = _extract_graph(best_xy, best_r, tol=1e-4)
        else:
            # If it failed, try perturbing the initial graph slightly before retrying?
            # Or just break.
            pass

    # Phase 3: Final polish with gradient descent on the continuous variables
    # We use the best graph solution as a starting point for a standard gradient polish
    # to squeeze out any remaining value.
    
    def _smooth_objective(v, n, beta=50.0, penalty_weight=100.0):
        x = v[0:n]
        y = v[n:2*n]
        r = v[2*n:3*n]
        wall_min = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
        wall_viol = r - wall_min
        wall_pen = np.sum(np.logaddexp(0.0, beta * wall_viol) / beta)
        total_viol = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx * dx + dy * dy)
                viol = r[i] + r[j] - dist
                total_viol += np.logaddexp(0.0, beta * viol) / beta
        return -np.sum(r) + penalty_weight * (wall_pen + total_viol)

    def _smooth_grad(v, n, beta=50.0, penalty_weight=100.0):
        x = v[0:n].copy()
        y = v[n:2*n].copy()
        r = v[2*n:3*n].copy()
        grad = np.zeros(3 * n)
        wall_min = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
        wall_viol = r - wall_min
        sig_wall = 1.0 / (1.0 + np.exp(-beta * wall_viol))
        for i in range(n):
            s = sig_wall[i]
            if s < 1e-10: continue
            dists = np.array([x[i], 1.0 - x[i], y[i], 1.0 - y[i]])
            min_idx = int(np.argmin(dists))
            pw = penalty_weight
            if min_idx == 0:
                grad[2*n + i] += pw * s
                grad[i] -= pw * s
            elif min_idx == 1:
                grad[2*n + i] += pw * s
                grad[i] += pw * s
            elif min_idx == 2:
                grad[2*n + i] += pw * s
                grad[n + i] -= pw * s
            else:
                grad[2*n + i] += pw * s
                grad[n + i] += pw * s
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist_sq = dx * dx + dy * dy
                if dist_sq < 1e-20: continue
                dist = np.sqrt(dist_sq)
                viol = r[i] + r[j] - dist
                s = 1.0 / (1.0 + np.exp(-beta * viol))
                if s < 1e-10: continue
                factor = penalty_weight * s
                grad[i] += factor * dx / dist
                grad[n + i] += factor * dy / dist
                grad[j] -= factor * dx / dist
                grad[n + j] -= factor * dy / dist
                grad[2*n + i] += factor
                grad[2*n + j] += factor
        grad[2*n:3*n] -= 1.0
        return grad

    if time.time() < deadline - 5:
        v0 = np.concatenate([best_xy[:, 0], best_xy[:, 1], best_r])
        bounds = []
        for i in range(N):
            bounds.append((0.0001, 0.9999))
        for i in range(N):
            bounds.append((0.0001, 0.9999))
        for i in range(N):
            bounds.append((0.0001, 0.4999))
            
        try:
            result = minimize(_smooth_objective, v0, jac=_smooth_grad, method='L-BFGS-B',
                              bounds=bounds, options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-8})
            v_opt = result.x
            new_xy = np.column_stack([v_opt[0:N], v_opt[N:2*N]])
            new_r = v_opt[2*N:3*N]
            total, valid = _eval_packing(new_xy, new_r)
            if valid and total > best_total:
                best_total = total
                best_xy = new_xy
                best_r = new_r
        except Exception:
            pass

    # Final Validation and Clipping
    n = len(best_xy)
    result = np.zeros((n, 3))
    for i in range(n):
        x, y, ri = best_xy[i, 0], best_xy[i, 1], best_r[i]
        ri = min(ri, x, 1.0 - x, y, 1.0 - y)
        ri = max(ri, 0.001)
        result[i, 2] = ri
        result[i, 0] = np.clip(x, ri, 1.0 - ri)
        result[i, 1] = np.clip(y, ri, 1.0 - ri)
        
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
