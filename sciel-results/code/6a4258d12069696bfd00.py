import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 50

# EVOLVE-BLOCK-START

def _compute_r(cents):
    """Compute the maximum feasible radius for given centers."""
    n = len(cents)
    wall_min = np.min(np.minimum(0.5 - np.abs(cents[:, 0]), 0.5 - np.abs(cents[:, 1])))
    if n > 1:
        diff = cents[:, np.newaxis, :] - cents[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        pair_min = np.min(dists) / 2.0
    else:
        pair_min = 0.5
    return min(wall_min, pair_min)

def _hex_init():
    """Create a hexagonal initial packing."""
    best_cents = None
    best_r = 0
    
    for n_rows in range(5, 9):
        base = N // n_rows
        extra = N % n_rows
        row_counts = [base + (1 if i < extra else 0) for i in range(n_rows)]
        
        max_rc = max(row_counts)
        r_horiz = 1.0 / (2.0 * max_rc)
        r_vert = 1.0 / (2.0 + (n_rows - 1) * np.sqrt(3.0))
        r = min(r_horiz, r_vert)
        
        if r <= best_r:
            continue
            
        centers = []
        total_y = (n_rows - 1) * np.sqrt(3.0) * r
        y_start = -0.5 + r + total_y / 2.0
        
        for row_idx in range(n_rows):
            rc = row_counts[row_idx]
            y = y_start + row_idx * np.sqrt(3.0) * r
            x_span = (rc - 1) * 2 * r
            x_start = -0.5 + r + x_span / 2.0
            offset = np.sqrt(3.0) * r * 0.5 * (row_idx % 2)
            for col in range(rc):
                x = x_start + col * 2 * r + offset
                centers.append([x, y])
                
        centers = np.array(centers[:N])
        r_actual = _compute_r(centers)
        
        if r_actual > best_r:
            best_r = r_actual
            best_cents = centers.copy()
            
    if best_cents is None:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_cents = grid[:N]
        best_r = _compute_r(best_cents)
        
    return best_cents, best_r

def _grid_init():
    """Create a grid initial packing."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    cents = grid[:N]
    r = _compute_r(cents)
    return cents, r

def _force_relax(cents, r, time_limit=20.0, rng=None):
    """
    Force-directed relaxation: overlapping circles repel, 
    circles are pulled toward walls to maximize r.
    Uses velocity-Verlet integration with adaptive step size.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    
    start_time = time.time()
    n = len(cents)
    pos = cents.copy()
    vel = np.zeros_like(pos)
    best_pos = pos.copy()
    best_r = r
    dt = 0.005
    damping = 0.98
    k_repel = 50.0  # repulsion stiffness
    k_attract = 5.0  # attraction to walls
    
    while time.time() - start_time < time_limit:
        # Compute forces
        forces = np.zeros_like(pos)
        
        # Repulsion between overlapping/near circles
        for i in range(n):
            for j in range(i + 1, n):
                diff = pos[i] - pos[j]
                d = np.linalg.norm(diff)
                if d < 2.0 * r and d > 1e-10:
                    # Strong repulsion when overlapping
                    overlap = 2.0 * r - d
                    f = k_repel * overlap * diff / d
                    forces[i] += f
                    forces[j] -= f
                elif d < 2.5 * r:
                    # Gentle repulsion to maintain spacing
                    f = k_repel * 0.1 * (2.5 * r - d) * diff / d
                    forces[i] += f
                    forces[j] -= f
        
        # Attraction to walls (push circles toward boundary to increase r)
        for i in range(n):
            x, y = pos[i]
            # Push toward nearest wall
            dist_left = x + 0.5
            dist_right = 0.5 - x
            dist_bot = y + 0.5
            dist_top = 0.5 - y
            
            min_dist = min(dist_left, dist_right, dist_bot, dist_top)
            
            if min_dist == dist_left:
                forces[i, 0] += k_attract * (r - dist_left) if dist_left < r else k_attract * 0.1
            elif min_dist == dist_right:
                forces[i, 0] -= k_attract * (r - dist_right) if dist_right < r else k_attract * 0.1
            elif min_dist == dist_bot:
                forces[i, 1] += k_attract * (r - dist_bot) if dist_bot < r else k_attract * 0.1
            elif min_dist == dist_top:
                forces[i, 1] -= k_attract * (r - dist_top) if dist_top < r else k_attract * 0.1
        
        # Velocity Verlet
        vel = vel * damping + forces * dt
        pos = pos + vel * dt
        
        # Clip to square
        pos[:, 0] = np.clip(pos[:, 0], -0.49, 0.49)
        pos[:, 1] = np.clip(pos[:, 1], -0.49, 0.49)
        
        # Check improvement
        new_r = _compute_r(pos)
        if new_r > best_r:
            best_r = new_r
            best_pos = pos.copy()
        
        # Adaptive step size: reduce if things are moving too fast
        max_vel = np.max(np.linalg.norm(vel, axis=1))
        if max_vel > 0.05:
            dt *= 0.9
        elif max_vel < 0.001 and dt < 0.005:
            dt *= 1.05
        
        # Occasional random perturbation to escape local minima
        if rng.random() < 0.01:
            mask = rng.random(n) < 0.2
            if not np.any(mask):
                mask[rng.integers(n)] = True
            pos[mask] += rng.normal(0, 0.01, size=(np.sum(mask), 2))
    
    return best_pos, best_r

def _newton_polish(cents, r, time_limit=15.0, max_iter=200):
    """
    Newton's method on the contact graph.
    For each pair of touching circles and wall contacts,
    solve the linear system to move circles apart/increase r.
    """
    start_time = time.time()
    n = len(cents)
    pos = cents.copy()
    current_r = r
    tol = 1e-10
    
    for iteration in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Build contact graph: pairs that are at distance <= 2r + epsilon
        # and wall contacts
        epsilon = 1e-6
        A = np.zeros((2 * n + 1, 2 * n + 1))  # 2n position vars + 1 radius var
        b = np.zeros(2 * n + 1)
        
        n_constraints = 0
        contact_pairs = []
        
        # Pairwise contacts
        for i in range(n):
            for j in range(i + 1, n):
                diff = pos[i] - pos[j]
                d = np.linalg.norm(diff)
                if d < 2.0 * current_r + epsilon and d > 1e-10:
                    # Constraint: |c_i - c_j| = 2r
                    # Gradient w.r.t. c_i: (c_i - c_j) / d
                    # Gradient w.r.t. c_j: -(c_i - c_j) / d
                    # Gradient w.r.t. r: -2
                    u = diff / d
                    idx_i = 2 * i
                    idx_j = 2 * j
                    idx_r = 2 * n
                    
                    A[idx_i, idx_i] += u[0] ** 2
                    A[idx_i, idx_i + 1] += u[0] * u[1]
                    A[idx_i + 1, idx_i] += u[0] * u[1]
                    A[idx_i + 1, idx_i + 1] += u[1] ** 2
                    A[idx_i, idx_j] -= u[0] ** 2
                    A[idx_i, idx_j + 1] -= u[0] * u[1]
                    A[idx_i + 1, idx_j] -= u[0] * u[1]
                    A[idx_i + 1, idx_j + 1] -= u[1] ** 2
                    A[idx_j, idx_i] -= u[0] ** 2
                    A[idx_j, idx_i + 1] -= u[0] * u[1]
                    A[idx_j + 1, idx_i] -= u[0] * u[1]
                    A[idx_j + 1, idx_j + 1] -= u[1] ** 2
                    A[idx_i, idx_r] += 2 * u[0]
                    A[idx_i + 1, idx_r] += 2 * u[1]
                    A[idx_j, idx_r] -= 2 * u[0]
                    A[idx_j + 1, idx_r] -= 2 * u[1]
                    A[idx_r, idx_i] += 2 * u[0]
                    A[idx_r, idx_i + 1] += 2 * u[1]
                    A[idx_r, idx_j] -= 2 * u[0]
                    A[idx_r, idx_j + 1] -= 2 * u[1]
                    A[idx_r, idx_r] += 4
                    b[idx_i] += 0  # We want to satisfy equality, so b=0 for the constraint
                    n_constraints += 1
                    contact_pairs.append((i, j))
        
        # Wall contacts
        for i in range(n):
            x, y = pos[i]
            idx_i = 2 * i
            idx_r = 2 * n
            
            # Left wall: x + 0.5 = r
            if x + 0.5 < current_r + epsilon:
                A[idx_i, idx_i] += 1
                A[idx_i, idx_r] += -1
                A[idx_r, idx_i] += -1
                A[idx_r, idx_r] += 1
                b[idx_i] += 0
                n_constraints += 1
            
            # Right wall: 0.5 - x = r
            if 0.5 - x < current_r + epsilon:
                A[idx_i, idx_i] += 1
                A[idx_i, idx_r] += -1
                A[idx_r, idx_i] += -1
                A[idx_r, idx_r] += 1
                b[idx_i] += 0
                n_constraints += 1
            
            # Bottom wall
            if y + 0.5 < current_r + epsilon:
                A[idx_i + 1, idx_i + 1] += 1
                A[idx_i + 1, idx_r] += -1
                A[idx_r, idx_i + 1] += -1
                A[idx_r, idx_r] += 1
                b[idx_i + 1] += 0
                n_constraints += 1
            
            # Top wall
            if 0.5 - y < current_r + epsilon:
                A[idx_i + 1, idx_i + 1] += 1
                A[idx_i + 1, idx_r] += -1
                A[idx_r, idx_i + 1] += -1
                A[idx_r, idx_r] += 1
                b[idx_i + 1] += 0
                n_constraints += 1
        
        if n_constraints == 0:
            break
        
        # Solve the least squares system
        try:
            # Add regularization for stability
            lam = 1e-8
            A_reg = A + lam * np.eye(2 * n + 1)
            delta = np.linalg.lstsq(A_reg, b, rcond=None)[0]
        except np.linalg.LinAlgError:
            break
        
        # Limit step size
        step_norm = np.linalg.norm(delta[:2 * n])
        max_step = 0.01
        if step_norm > max_step:
            delta[:2 * n] *= max_step / step_norm
        
        # Update
        pos = pos + delta[:2 * n].reshape(n, 2)
        current_r = current_r + delta[2 * n]
        
        # Ensure feasibility
        pos[:, 0] = np.clip(pos[:, 0], -0.49, 0.49)
        pos[:, 1] = np.clip(pos[:, 1], -0.49, 0.49)
        current_r = max(current_r, 0.01)
        
        # Check convergence
        if np.linalg.norm(delta) < tol:
            break
    
    actual_r = _compute_r(pos)
    if actual_r >= current_r * 0.99:
        return pos, actual_r
    else:
        return pos, actual_r

def _deformation_field_init(n_base=7):
    """
    Initialize using a deformation field on a regular grid.
    The base grid has n_base x n_base points, and we select N of them.
    The deformation is a smooth (low-frequency) perturbation.
    """
    # Create base grid
    m = n_base
    xs = (np.arange(m) + 0.5) / m - 0.5
    ys = (np.arange(m) + 0.5) / m - 0.5
    X, Y = np.meshgrid(xs, ys)
    base_grid = np.column_stack([X.ravel(), Y.ravel()])
    
    # Select N points (first N or best subset)
    cents = base_grid[:N]
    r = _compute_r(cents)
    return cents, r

def _lbfgs_polish(cents, r, time_limit=10.0, beta=500):
    """Polish the packing using L-BFGS with soft constraints."""
    start_time = time.time()
    n = len(cents)
    
    x0 = np.zeros(2 * n + 1)
    x0[:2 * n] = cents.flatten()
    x0[2 * n] = r
    
    bounds = [(-0.5, 0.5)] * (2 * n) + [(0.01, 0.5)]
    
    def obj(p):
        c = p[:2 * n].reshape(n, 2)
        rr = p[2 * n]
        val = -rr
        for i in range(n):
            for j in range(i + 1, n):
                d2 = np.dot(c[i] - c[j], c[i] - c[j])
                g = d2 - 4 * rr * rr
                if g < 0:
                    val += (1.0 / beta) * np.log1p(np.exp(-beta * g))
        for i in range(n):
            x, y = c[i]
            for g in [x + 0.5 - rr, 0.5 - rr - x, y + 0.5 - rr, 0.5 - rr - y]:
                if g < 0:
                    val += (1.0 / beta) * np.log1p(np.exp(-beta * g))
        return val
    
    def grad(p):
        c = p[:2 * n].reshape(n, 2)
        rr = p[2 * n]
        g = np.zeros_like(p)
        g[2 * n] = -1.0
        for i in range(n):
            for j in range(i + 1, n):
                diff = c[i] - c[j]
                d2 = np.dot(diff, diff)
                gv = d2 - 4 * rr * rr
                if gv < 0:
                    sig = 1.0 / (1.0 + np.exp(beta * gv))
                    g[2 * i:2 * i + 2] += -sig * 2 * diff
                    g[2 * j:2 * j + 2] += -sig * (-2 * diff)
                    g[2 * n] += -sig * (-8 * rr)
        for i in range(n):
            x, y = c[i]
            for gv, ix, iy, ir in [(x + 0.5 - rr, 1, 0, -1), (0.5 - rr - x, -1, 0, -1),
                                    (y + 0.5 - rr, 0, 1, -1), (0.5 - rr - y, 0, -1, -1)]:
                if gv < 0:
                    sig = 1.0 / (1.0 + np.exp(beta * gv))
                    g[2 * i + (0 if ix != 0 else 1)] += -sig * (ix if ix != 0 else iy)
                    g[2 * n] += -sig * ir
        return g
    
    try:
        res = minimize(
            obj, x0, jac=grad,
            method='L-BFGS-B', bounds=bounds,
            options={'maxiter': 1500, 'ftol': 1e-15, 'gtol': 1e-12, 'maxfun': 8000, 'disp': False}
        )
        new_cents = res.x[:2 * n].reshape(n, 2)
        new_r = res.x[2 * n]
        actual_r = _compute_r(new_cents)
        if actual_r >= r * 0.99:
            return new_cents, actual_r
    except Exception:
        pass
    
    return cents, r

def _simulated_annealing(cents, r, time_limit=25.0, rng=None):
    """Simulated annealing to improve the packing."""
    if rng is None:
        rng = np.random.default_rng(42)
    
    start_time = time.time()
    n = len(cents)
    best_cents = cents.copy()
    best_r = r
    current_cents = cents.copy()
    current_r = r
    
    T = 0.01
    T_min = 1e-6
    cooling = 0.995
    
    while time.time() - start_time < time_limit:
        mask = rng.random(n) < 0.15
        if not np.any(mask):
            mask[rng.integers(n)] = True
        current_cents[mask] += rng.normal(0, T * 0.5, size=(np.sum(mask), 2))
        current_cents[:, 0] = np.clip(current_cents[:, 0], -0.45, 0.45)
        current_cents[:, 1] = np.clip(current_cents[:, 1], -0.45, 0.45)
        
        new_r = _compute_r(current_cents)
        
        if new_r > current_r:
            current_r = new_r
            if new_r > best_r:
                best_r = new_r
                best_cents = current_cents.copy()
        else:
            delta = new_r - current_r
            if rng.random() < np.exp(delta / T):
                current_r = new_r
                if new_r > best_r:
                    best_r = new_r
                    best_cents = current_cents.copy()
        
        T *= cooling
        if T < T_min:
            T = T_min
    
    return best_cents, best_r

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    # 1. Initial packings
    hex_cents, hex_r = _hex_init()
    grid_cents, grid_r = _grid_init()
    def_cents, def_r = _deformation_field_init()
    
    best_cents = hex_cents.copy()
    best_r = hex_r
    if grid_r > best_r:
        best_r = grid_r
        best_cents = grid_cents.copy()
    if def_r > best_r:
        best_r = def_r
        best_cents = def_cents.copy()
    
    rng = np.random.default_rng(42)
    
    # 2. Force relaxation from multiple starts
    initializations = [(hex_cents, hex_r), (grid_cents, grid_r), (def_cents, def_r)]
    
    # Add random initializations
    for _ in range(4):
        rc = rng.uniform(-0.4, 0.4, size=(N, 2))
        rr = _compute_r(rc)
        initializations.append((rc, rr))
    
    for cents, r in initializations:
        remaining = deadline - (time.time() - start_time)
        if remaining < 20:
            break
        
        # Force relaxation
        fr_cents, fr_r = _force_relax(cents, r, time_limit=min(remaining - 15, 15.0), rng=rng)
        if fr_r > best_r:
            best_r = fr_r
            best_cents = fr_cents.copy()
        
        # Newton polish
        remaining = deadline - (time.time() - start_time)
        if remaining > 15:
            nt_cents, nt_r = _newton_polish(fr_cents, fr_r, time_limit=min(remaining - 10, 10.0))
            if nt_r > best_r:
                best_r = nt_r
                best_cents = nt_cents.copy()
    
    # 3. SA + polish on best solution
    remaining = deadline - (time.time() - start_time)
    if remaining > 30:
        sa_cents, sa_r = _simulated_annealing(best_cents, best_r, time_limit=min(remaining - 20, 20.0), rng=rng)
        if sa_r > best_r:
            best_r = sa_r
            best_cents = sa_cents.copy()
        
        remaining = deadline - (time.time() - start_time)
        if remaining > 15:
            # Force relax from SA result
            fr_cents, fr_r = _force_relax(sa_cents, sa_r, time_limit=min(remaining - 10, 10.0), rng=rng)
            if fr_r > best_r:
                best_r = fr_r
                best_cents = fr_cents.copy()
            
            remaining = deadline - (time.time() - start_time)
            if remaining > 10:
                nt_cents, nt_r = _newton_polish(best_cents, best_r, time_limit=min(remaining - 5, 8.0))
                if nt_r > best_r:
                    best_r = nt_r
                    best_cents = nt_cents.copy()
    
    # 4. Final L-BFGS polish
    remaining = deadline - (time.time() - start_time)
    if remaining > 10:
        for beta in [2000, 5000]:
            remaining = deadline - (time.time() - start_time)
            if remaining < 8:
                break
            pol_cents, pol_r = _lbfgs_polish(best_cents, best_r, time_limit=min(remaining - 4, 7.0), beta=beta)
            if pol_r > best_r:
                best_r = pol_r
                best_cents = pol_cents.copy()
    
    # 5. Final Newton polish
    remaining = deadline - (time.time() - start_time)
    if remaining > 5:
        nt_cents, nt_r = _newton_polish(best_cents, best_r, time_limit=min(remaining - 2, 4.0))
        if nt_r > best_r:
            best_r = nt_r
            best_cents = nt_cents.copy()
    
    # Ensure validity
    final_r = _compute_r(best_cents)
    if final_r <= 0 or not np.all(np.isfinite(best_cents)):
        best_cents, best_r = _hex_init()
    
    return best_cents

# EVOLVE-BLOCK-END
