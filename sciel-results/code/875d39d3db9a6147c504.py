"""Circle packing in unit square: pack 26 circles, maximize sum of radii.
Strategy: Successive Convex Programming (SCP) with Taylor-linearized non-overlap
constraints, combined with multi-restart and radius-growth annealing."""
N = 26
import numpy as np
from scipy.optimize import minimize
import time


def _is_valid(x, tol=1e-7):
    """Check if configuration is valid."""
    pos = x.reshape(N, 3)
    xs, ys, rs = pos[:, 0], pos[:, 1], pos[:, 2]
    if np.any(rs < -tol):
        return False
    if np.any(xs < -tol) or np.any(xs > 1 + tol) or np.any(ys < -tol) or np.any(ys > 1 + tol):
        return False
    if np.any(rs > xs + tol) or np.any(rs > 1 - xs + tol) or np.any(rs > ys + tol) or np.any(rs > 1 - ys + tol):
        return False
    for i in range(N):
        for j in range(i + 1, N):
            d2 = (xs[i] - xs[j])**2 + (ys[i] - ys[j])**2
            if d2 < (rs[i] + rs[j])**2 - tol:
                return False
    return True


def _sum_radii(x):
    return np.sum(x.reshape(N, 3)[:, 2])


def _project_feasible(pos):
    """Project a configuration to feasibility by shrinking radii and adjusting positions."""
    xs = pos[:, 0].copy()
    ys = pos[:, 1].copy()
    rs = pos[:, 2].copy()
    # Clip positions to [0,1]
    xs = np.clip(xs, 0, 1)
    ys = np.clip(ys, 0, 1)
    # Shrink radii to fit within boundaries
    for i in range(N):
        max_r = min(xs[i], 1.0 - xs[i], ys[i], 1.0 - ys[i])
        rs[i] = min(rs[i], max(0.0, max_r))
    # Iteratively resolve overlaps by shrinking the smaller circle
    for _ in range(200):
        changed = False
        for i in range(N):
            for j in range(i + 1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                d = np.sqrt(dx*dx + dy*dy)
                required = rs[i] + rs[j]
                if d < required - 1e-12:
                    # Shrink both radii proportionally
                    overlap = required - d
                    # Move apart if possible
                    if d > 1e-10:
                        ux, uy = dx/d, dy/d
                        shift = overlap / 2.0 + 1e-10
                        # Move i away from j and j away from i
                        nx_i = xs[i] + ux * shift
                        ny_i = ys[i] + uy * shift
                        nx_j = xs[j] - ux * shift
                        ny_j = ys[j] - uy * shift
                        # Check if moves keep within bounds
                        if (0 <= nx_i <= 1 and 0 <= ny_i <= 1 and
                            0 <= nx_j <= 1 and 0 <= ny_j <= 1):
                            # Check if radii still fit at new positions
                            max_ri = min(nx_i, 1-nx_i, ny_i, 1-ny_i)
                            max_rj = min(nx_j, 1-nx_j, ny_j, 1-ny_j)
                            if rs[i] <= max_ri + 1e-12 and rs[j] <= max_rj + 1e-12:
                                xs[i], ys[i] = nx_i, ny_i
                                xs[j], ys[j] = nx_j, ny_j
                                changed = True
                                continue
                    # If can't move, shrink radii
                    total_r = rs[i] + rs[j]
                    if total_r > 1e-12:
                        scale = d / total_r if d > 1e-12 else 0.0
                        rs[i] = max(0.0, rs[i] * scale * 0.999)
                        rs[j] = max(0.0, rs[j] * scale * 0.999)
                        changed = True
        if not changed:
            break
    pos[:, 0] = xs
    pos[:, 1] = ys
    pos[:, 2] = rs
    return pos


def _scp_subproblem(pos, deadline):
    """
    Solve one SCP iteration: maximize sum of radii subject to
    linearized non-overlap constraints and boundary constraints.
    
    At the current point, for each pair (i,j), the non-overlap constraint
    ||p_i - p_j|| >= r_i + r_j is linearized as:
    (p_i - p_j) · (p_i - p_j)_0 / ||p_i - p_j||_0 >= r_i + r_j
    which becomes a linear constraint in (p_i, p_j, r_i, r_j).
    """
    xs0 = pos[:, 0].copy()
    ys0 = pos[:, 1].copy()
    rs0 = pos[:, 2].copy()
    
    # Build constraints: list of (linear_func, lb, ub)
    # Variables: x1,y1,r1, x2,y2,r2, ...
    n_vars = 3 * N
    
    # Boundary constraints (linear):
    # r_i <= x_i, r_i <= 1 - x_i, r_i <= y_i, r_i <= 1 - y_i, r_i >= 0
    A_ub = []
    b_ub = []
    
    for i in range(N):
        # r_i - x_i <= 0  =>  -x_i + r_i <= 0
        row = np.zeros(n_vars)
        row[3*i] = -1.0
        row[3*i+2] = 1.0
        A_ub.append(row)
        b_ub.append(0.0)
        # r_i - (1-x_i) <= 0  =>  x_i + r_i <= 1
        row = np.zeros(n_vars)
        row[3*i] = 1.0
        row[3*i+2] = 1.0
        A_ub.append(row)
        b_ub.append(1.0)
        # r_i - y_i <= 0
        row = np.zeros(n_vars)
        row[3*i+1] = -1.0
        row[3*i+2] = 1.0
        A_ub.append(row)
        b_ub.append(0.0)
        # r_i - (1-y_i) <= 0  =>  y_i + r_i <= 1
        row = np.zeros(n_vars)
        row[3*i+1] = 1.0
        row[3*i+2] = 1.0
        A_ub.append(row)
        b_ub.append(1.0)
    
    # Non-overlap linearized constraints:
    # For pair (i,j), direction u = (p_i - p_j)/||p_i - p_j||
    # Constraint: u · (p_i - p_j) >= r_i + r_j
    # => u_x*(x_i - x_j) + u_y*(y_i - y_j) - r_i - r_j >= 0
    # => -u_x*(x_i - x_j) - u_y*(y_i - y_j) + r_i + r_j <= 0
    for i in range(N):
        for j in range(i + 1, N):
            dx = xs0[i] - xs0[j]
            dy = ys0[i] - ys0[j]
            d = np.sqrt(dx*dx + dy*dy)
            if d < 1e-10:
                # Degenerate: use arbitrary direction
                ux, uy = 1.0, 0.0
            else:
                ux, uy = dx/d, dy/d
            row = np.zeros(n_vars)
            # -ux*x_i + ux*x_j - uy*y_i + uy*y_j + r_i + r_j <= 0
            row[3*i] = -ux
            row[3*i+1] = -uy
            row[3*i+2] = 1.0
            row[3*j] = ux
            row[3*j+1] = uy
            row[3*j+2] = 1.0
            A_ub.append(row)
            b_ub.append(0.0)
    
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    # Objective: maximize sum(r_i) => minimize -sum(r_i)
    c = np.zeros(n_vars)
    for i in range(N):
        c[3*i+2] = -1.0
    
    # Bounds
    bounds = []
    for i in range(N):
        bounds.append((0, 1))  # x
        bounds.append((0, 1))  # y
        bounds.append((0, 0.5))  # r
    
    x0 = pos.flatten().copy()
    
    try:
        res = minimize(
            lambda x: c @ x,
            x0,
            jac=lambda x: c,
            method='SLSQP',
            bounds=bounds,
            constraints=[{'type': 'ineq', 'fun': lambda x: b_ub - A_ub @ x}],
            options={'maxiter': 100, 'ftol': 1e-10}
        )
        if res.success or _is_valid(res.x):
            return res.x.reshape(N, 3)
    except Exception:
        pass
    
    return pos


def _scp_solve(pos, rng, deadline, n_iters=30):
    """Run SCP iterations with radius growth annealing."""
    pos = pos.copy()
    best_pos = pos.copy()
    best_val = _sum_radii(pos.flatten())
    if not _is_valid(pos.flatten()):
        pos = _project_feasible(pos)
        best_pos = pos.copy()
        best_val = _sum_radii(pos.flatten())
    
    for it in range(n_iters):
        if time.time() > deadline:
            break
        
        # Grow radii slightly to encourage expansion
        growth = 0.02 * (1.0 - it / max(n_iters, 1))
        for i in range(N):
            max_r = min(pos[i,0], 1-pos[i,0], pos[i,1], 1-pos[i,1])
            pos[i,2] = min(pos[i,2] * (1 + growth), max_r)
        
        # Solve SCP subproblem
        new_pos = _scp_subproblem(pos, deadline)
        
        # Project to feasibility
        new_pos = _project_feasible(new_pos)
        
        val = _sum_radii(new_pos.flatten())
        if val > best_val:
            best_val = val
            best_pos = new_pos.copy()
        
        pos = new_pos
    
    return best_pos, best_val


def _local_best_fit(pos, rng, deadline, n_passes=20):
    """Best-fit local search: for each circle, try to expand its radius."""
    pos = pos.copy()
    best_val = _sum_radii(pos.flatten())
    if not _is_valid(pos.flatten()):
        pos = _project_feasible(pos)
        best_val = _sum_radii(pos.flatten())
    
    for p in range(n_passes):
        if time.time() > deadline:
            break
        improved = False
        order = rng.permutation(N)
        for i in order:
            if time.time() > deadline:
                break
            # Max radius at current position
            max_r = min(pos[i,0], 1-pos[i,0], pos[i,1], 1-pos[i,1])
            for j in range(N):
                if j == i:
                    continue
                d = np.sqrt((pos[i,0]-pos[j,0])**2 + (pos[i,1]-pos[j,1])**2)
                avail = d - pos[j,2]
                if avail < max_r:
                    max_r = avail
            max_r = max(0.0, max_r)
            if max_r > pos[i,2] + 1e-8:
                pos[i,2] = max_r
                improved = True
            
            # Try small random moves
            for _ in range(10):
                dx = rng.uniform(-0.05, 0.05)
                dy = rng.uniform(-0.05, 0.05)
                nx = np.clip(pos[i,0] + dx, 0.01, 0.99)
                ny = np.clip(pos[i,1] + dy, 0.01, 0.99)
                max_r = min(nx, 1-nx, ny, 1-ny)
                for j in range(N):
                    if j == i:
                        continue
                    d = np.sqrt((nx-pos[j,0])**2 + (ny-pos[j,1])**2)
                    avail = d - pos[j,2]
                    if avail < max_r:
                        max_r = avail
                max_r = max(0.0, max_r)
                if max_r > pos[i,2] + 1e-6:
                    pos[i,0], pos[i,1], pos[i,2] = nx, ny, max_r
                    improved = True
        
        val = _sum_radii(pos.flatten())
        if val > best_val:
            best_val = val
    
    return pos, best_val


def _initial_hex_config(rng):
    """Hexagonal packing initial configuration."""
    pos = np.zeros((N, 3))
    # Hexagonal lattice
    spacing = 0.2
    row_h = spacing * np.sqrt(3) / 2
    idx = 0
    for row in range(6):
        for col in range(6):
            if idx >= N:
                break
            x = 0.1 + col * spacing + (row % 2) * spacing / 2
            y = 0.1 + row * row_h
            if x < 0.95 and y < 0.95:
                pos[idx, 0] = x + rng.uniform(-0.01, 0.01)
                pos[idx, 1] = y + rng.uniform(-0.01, 0.01)
                pos[idx, 2] = spacing / 2 - 0.01
                idx += 1
    # Fill remaining randomly
    while idx < N:
        pos[idx, 0] = rng.uniform(0.1, 0.9)
        pos[idx, 1] = rng.uniform(0.1, 0.9)
        pos[idx, 2] = 0.05
        idx += 1
    pos = _project_feasible(pos)
    return pos


def _initial_random_config(rng):
    """Random initial configuration with small radii."""
    pos = np.zeros((N, 3))
    for i in range(N):
        pos[i, 0] = rng.uniform(0.05, 0.95)
        pos[i, 1] = rng.uniform(0.05, 0.95)
        pos[i, 2] = rng.uniform(0.03, 0.08)
    pos = _project_feasible(pos)
    return pos


def _initial_grid_config(rng):
    """Perturbed grid initial configuration."""
    side = int(np.ceil(np.sqrt(N)))
    pos = np.zeros((N, 3))
    for i in range(N):
        row = i // side
        col = i % side
        pos[i, 0] = (col + 0.5) / side + rng.uniform(-0.02, 0.02)
        pos[i, 1] = (row + 0.5) / side + rng.uniform(-0.02, 0.02)
        pos[i, 2] = 0.5 / side - 0.01
    pos = _project_feasible(pos)
    return pos


def _sa_refine(pos, rng, deadline):
    """Simulated annealing refinement."""
    pos = pos.copy()
    best_val = _sum_radii(pos.flatten())
    if not _is_valid(pos.flatten()):
        pos = _project_feasible(pos)
        best_val = _sum_radii(pos.flatten())
    
    T = 0.005
    step = 0.008
    for it in range(2000):
        if time.time() > deadline:
            break
        i = rng.integers(N)
        dim = rng.integers(3)
        old = pos[i, dim]
        pos[i, dim] += rng.normal(0, step)
        if dim == 0:
            pos[i, 0] = np.clip(pos[i, 0], 0, 1)
        elif dim == 1:
            pos[i, 1] = np.clip(pos[i, 1], 0, 1)
        else:
            pos[i, 2] = np.clip(pos[i, 2], 0, 0.5)
        
        if _is_valid(pos.flatten()):
            new_val = _sum_radii(pos.flatten())
            delta = new_val - best_val
            if delta > 0 or (T > 1e-8 and rng.random() < np.exp(delta / T)):
                best_val = new_val
            else:
                pos[i, dim] = old
        else:
            pos[i, dim] = old
        T *= 0.9995
        step *= 0.9998
    
    return pos, best_val


def construct():
    deadline = time.time() + 140  # Safety margin
    rng = np.random.default_rng(42)
    
    best_pos = None
    best_val = -np.inf
    
    # Phase 1: SCP from multiple diverse inits
    inits = [
        _initial_hex_config(rng),
        _initial_grid_config(rng),
        _initial_random_config(rng),
        _initial_hex_config(rng),
        _initial_random_config(rng),
    ]
    
    for init_i, init_pos in enumerate(inits):
        if time.time() > deadline - 30:
            break
        
        # SCP solve
        scp_pos, scp_val = _scp_solve(init_pos, rng, deadline - 25, n_iters=25)
        
        # Local best-fit polish
        lf_pos, lf_val = _local_best_fit(scp_pos, rng, deadline - 20, n_passes=15)
        
        # SA refinement
        sa_pos, sa_val = _sa_refine(lf_pos, rng, deadline - 15)
        
        val = max(scp_val, lf_val, sa_val)
        if val > best_val:
            best_val = val
            if sa_val == val:
                best_pos = sa_pos.copy()
            elif lf_val == val:
                best_pos = lf_pos.copy()
            else:
                best_pos = scp_pos.copy()
    
    # Phase 2: Final SCP + local polish on best
    if best_pos is not None and time.time() < deadline - 10:
        final_pos, final_val = _scp_solve(best_pos, rng, deadline - 8, n_iters=15)
        final_pos, final_val = _local_best_fit(final_pos, rng, deadline - 5, n_passes=10)
        if final_val > best_val:
            best_val = final_val
            best_pos = final_pos.copy()
    
    # Phase 3: Final SA
    if best_pos is not None and time.time() < deadline - 3:
        sa_pos, sa_val = _sa_refine(best_pos, rng, deadline - 2)
        if sa_val > best_val:
            best_val = sa_val
            best_pos = sa_pos.copy()
    
    # Fallback
    if best_pos is None or not _is_valid(best_pos.flatten()):
        side = int(np.ceil(np.sqrt(N)))
        pos = np.zeros((N, 3))
        for i in range(N):
            row = i // side
            col = i % side
            pos[i, 0] = (col + 0.5) / side
            pos[i, 1] = (row + 0.5) / side
            pos[i, 2] = 0.5 / side - 0.01
        best_pos = _project_feasible(pos)
    
    return best_pos.reshape(N, 3)
