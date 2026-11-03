"""Rewritten program for the 50-circle packing task."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50

# EVOLVE-BLOCK-START
def _score(c):
    """Compute the packing radius for centers c of shape (N, 2)."""
    c = np.asarray(c, dtype=float)
    if c.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(c)):
        return 0.0
    # Wall constraints
    wall = 0.5 - np.abs(c).max()
    # Pairwise distances
    diff = c[:, None, :] - c[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    min_pair = dist.min() / 2.0
    return float(min(wall, min_pair))

def _hard_constraint_obj(x):
    """
    Objective function for SLSQP.
    x = [c_1x, c_1y, ..., c_Nx, c_Ny, r]
    We want to maximize r, so we minimize -r.
    """
    c = x[:-1].reshape(N, 2)
    r = x[-1]
    return -r

def _hard_constraint_grad(x):
    """Gradient of the objective."""
    grad = np.zeros_like(x)
    grad[-1] = -1.0
    return grad

def _hard_constraints(x):
    """
    Inequality constraints for SLSQP.
    Returns a tuple of constraints.
    Constraint: g(x) >= 0
    1. Wall constraints: 0.5 - |c_ik| - r >= 0
    2. Pairwise constraints: |c_i - c_j|/2 - r >= 0
    """
    c = x[:-1].reshape(N, 2)
    r = x[-1]
    
    constraints = []
    
    # Wall constraints
    # For each coordinate k of each circle i: 0.5 - |c_ik| - r >= 0
    # We can vectorize this into a single constraint vector if needed, but SLSQP accepts a list or array.
    # Let's create an array of all wall violations.
    wall_violations = 0.5 - np.abs(c) - r
    # Flatten to a 1D array of constraints
    constraints.append(wall_violations.ravel())
    
    # Pairwise constraints
    # For each pair (i, j) with i < j: |c_i - c_j|/2 - r >= 0
    diff = c[:, None, :] - c[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    # We only need i < j to avoid duplicates, but SLSQP can handle the full matrix if we mask it.
    # It's more efficient to provide only the unique pairs.
    idx_i, idx_j = np.triu_indices(N, k=1)
    pair_violations = dist[idx_i, idx_j] / 2.0 - r
    constraints.append(pair_violations)
    
    # Combine into a single array for SLSQP
    all_constraints = np.concatenate(constraints)
    return all_constraints

def _hard_constraint_jac(x):
    """
    Jacobian of the constraints.
    Shape: (N_constraints, 2*N + 1)
    """
    c = x[:-1].reshape(N, 2)
    r = x[-1]
    
    # Number of wall constraints: 2*N
    # Number of pair constraints: N*(N-1)/2
    n_wall = 2 * N
    n_pairs = N * (N - 1) // 2
    n_cons = n_wall + n_pairs
    
    jac = np.zeros((n_cons, 2 * N + 1))
    
    # Wall constraints Jacobian
    # g_k = 0.5 - |c_ik| - r
    # d(g_k)/d(c_ik) = -sign(c_ik)
    # d(g_k)/d(r) = -1
    
    # We need to map the flat wall violations to the correct indices in the Jacobian.
    # The order in _hard_constraints is wall.ravel() then pairs.
    # wall.ravel() order: (c_0x, c_0y, c_1x, c_1y, ...)
    
    for i in range(N):
        for k in range(2):
            cons_idx = i * 2 + k
            # Variable index for c_ik is i*2 + k
            var_idx = i * 2 + k
            jac[cons_idx, var_idx] = -np.sign(c[i, k])
            jac[cons_idx, -1] = -1.0 # Derivative w.r.t r
    
    # Pairwise constraints Jacobian
    # g_p = |c_i - c_j|/2 - r
    # d(g_p)/d(c_i) = (c_i - c_j) / (2 * |c_i - c_j|)
    # d(g_p)/d(c_j) = (c_j - c_i) / (2 * |c_i - c_j|)
    # d(g_p)/d(r) = -1
    
    idx_i, idx_j = np.triu_indices(N, k=1)
    for m in range(len(idx_i)):
        i = idx_i[m]
        j = idx_j[m]
        cons_idx = n_wall + m
        
        d = c[i] - c[j]
        dist = np.linalg.norm(d)
        if dist < 1e-12:
            # If distance is 0, gradient is undefined. Use a small random direction or 0.
            # For SLSQP, 0 is safer to avoid NaNs, though it might stall.
            # We'll add a small epsilon to dist to avoid division by zero.
            dist = 1e-12
            d = np.array([1e-12, 0])
        
        grad_ci = d / (2.0 * dist)
        grad_cj = -d / (2.0 * dist)
        
        # Variable indices for c_i and c_j
        var_i = i * 2
        var_j = j * 2
        
        jac[cons_idx, var_i] = grad_ci[0]
        jac[cons_idx, var_i + 1] = grad_ci[1]
        jac[cons_idx, var_j] = grad_cj[0]
        jac[cons_idx, var_j + 1] = grad_cj[1]
        jac[cons_idx, -1] = -1.0 # Derivative w.r.t r
    
    return jac

def _make_hexagonal_grid():
    """Create a hexagonal packing initial configuration."""
    best_c = None
    best_score = -1
    
    for n_rows in range(5, 11):
        base = N // n_rows
        extra = N % n_rows
        row_sizes = [base + (1 if i < extra else 0) for i in range(n_rows)]
        
        c = np.zeros((N, 2))
        idx = 0
        for i, n_cols in enumerate(row_sizes):
            y = (i - (n_rows - 1) / 2.0) * 0.15
            x_offset = (i % 2) * 0.07
            for j in range(n_cols):
                x = (j - (n_cols - 1) / 2.0) * 0.14 + x_offset
                c[idx, 0] = x
                c[idx, 1] = y
                idx += 1
        
        c = np.clip(c, -0.45, 0.45)
        s = _score(c)
        if s > best_score:
            best_score = s
            best_c = c.copy()
    
    return best_c

def _make_grid_rect():
    """Create a rectangular grid initial configuration."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].copy()

def _force_directed_relax(c, max_iters=500, deadline=None):
    """
    Physics-based relaxation.
    Circles repel each other and walls.
    """
    c = c.copy().astype(float)
    for it in range(max_iters):
        if deadline and time.time() - deadline[0] > 120:
            break
        
        forces = np.zeros_like(c)
        
        # Pairwise repulsion
        diff = c[:, None, :] - c[None, :, :]
        dist = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist, np.inf)
        
        # Only consider pairs that are close to avoid O(N^2) full force calculation if possible?
        # N=50 is small, so O(N^2) is fine.
        # Force magnitude ~ 1/dist^2 or similar.
        # We want to push them apart.
        # F = k * (target_dist - dist) / dist * dir
        # Let's use a simple repulsive force if dist < some threshold.
        
        # Calculate forces for all pairs
        for i in range(N):
            for j in range(i + 1, N):
                d = c[i] - c[j]
                dist_ij = np.linalg.norm(d)
                if dist_ij < 1e-6:
                    d = np.random.randn(2) * 1e-6
                    dist_ij = np.linalg.norm(d)
                
                # Target distance is roughly the current minimum distance or a fixed value?
                # Better: repel strongly if close.
                # Let's use a spring force that pushes them to a distance of 2*r_current
                # But r_current is not known directly in the loop.
                # Instead, just repel: F = d / dist^3
                f = d / (dist_ij ** 3)
                forces[i] += f
                forces[j] -= f
        
        # Wall repulsion
        # Push away from walls if close
        wall_margin = 0.05
        for i in range(N):
            for k in range(2):
                if c[i, k] > 0.5 - wall_margin:
                    forces[i, k] -= (c[i, k] - (0.5 - wall_margin)) * 10.0
                elif c[i, k] < -0.5 + wall_margin:
                    forces[i, k] += ((-0.5 + wall_margin) - c[i, k]) * 10.0
        
        # Update positions
        # Damping
        c += forces * 0.01
        c = np.clip(c, -0.5, 0.5)
        
    return c

def _slsqp_optimize(c, deadline):
    """
    Optimize using SLSQP with hard constraints.
    """
    if deadline and time.time() - deadline[0] > 120:
        return c
    
    r_init = _score(c)
    x0 = np.concatenate([c.ravel(), [r_init]])
    
    bounds = [(-0.5, 0.5)] * (2 * N) + [(0, 0.5)]
    
    try:
        res = minimize(
            _hard_constraint_obj,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=_hard_constraints,
            jac=_hard_constraint_grad,
            options={'maxiter': 50, 'ftol': 1e-12}
        )
        
        if res.success:
            c_new = res.x[:-1].reshape(N, 2)
            # Check if it's valid and better
            s_new = _score(c_new)
            if s_new > r_init:
                return c_new
    except Exception:
        pass
    
    return c

def _simulated_annealing(c, deadline, max_iter=5000):
    """
    Simulated Annealing to escape local minima.
    """
    if deadline and time.time() - deadline[0] > 120:
        return c
    
    c_current = c.copy().astype(float)
    best_c = c_current.copy()
    best_score = _score(best_c)
    
    T = 0.05
    T_min = 1e-6
    alpha = 0.99
    
    for i in range(max_iter):
        if deadline and time.time() - deadline[0] > 120:
            break
        
        # Propose a move
        c_proposed = c_current.copy()
        idx = np.random.randint(N)
        c_proposed[idx] += np.random.randn(2) * T
        
        # Keep inside square
        c_proposed = np.clip(c_proposed, -0.5, 0.5)
        
        s_current = _score(c_current)
        s_proposed = _score(c_proposed)
        
        # Accept if better, or with probability exp((s_proposed - s_current)/T)
        if s_proposed > s_current:
            c_current = c_proposed
            if s_proposed > best_score:
                best_c = c_current.copy()
                best_score = s_proposed
        else:
            delta = s_proposed - s_current
            if delta < 0:
                prob = np.exp(delta / T)
                if np.random.rand() < prob:
                    c_current = c_proposed
        
        T *= alpha
        if T < T_min:
            break
    
    return best_c

def construct():
    deadline = [time.time()]
    
    # Phase 1: Initialization
    c_hex = _make_hexagonal_grid()
    c_grid = _make_grid_rect()
    
    best_c = c_hex.copy()
    best_score = _score(best_c)
    
    s_grid = _score(c_grid)
    if s_grid > best_score:
        best_c = c_grid.copy()
        best_score = s_grid
    
    # Phase 2: Force Directed Relaxation
    c = _force_directed_relax(best_c, max_iters=200, deadline=deadline)
    s = _score(c)
    if s > best_score:
        best_c = c.copy()
        best_score = s
    
    # Phase 3: SLSQP Optimization
    c = _slsqp_optimize(best_c, deadline)
    s = _score(c)
    if s > best_score:
        best_c = c.copy()
        best_score = s
    
    # Phase 4: Simulated Annealing
    c = _simulated_annealing(best_c, deadline, max_iter=2000)
    s = _score(c)
    if s > best_score:
        best_c = c.copy()
        best_score = s
    
    # Phase 5: Another SLSQP pass
    c = _slsqp_optimize(best_c, deadline)
    s = _score(c)
    if s > best_score:
        best_c = c.copy()
        best_score = s
        
    # Phase 6: Final SA
    c = _simulated_annealing(best_c, deadline, max_iter=1000)
    s = _score(c)
    if s > best_score:
        best_c = c.copy()
        best_score = s

    # Ensure validity
    best_c = np.clip(best_c, -0.5, 0.5)
    return best_c
# EVOLVE-BLOCK-END
