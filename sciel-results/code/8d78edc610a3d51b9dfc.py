"""Pack 50 equal circles in the unit disk via structured init + vectorized optimization + SA polish."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
DEADLINE_MARGIN = 5.0  # seconds of safety margin


def _compute_score(pts):
    """Compute the packing score: min(min pair dist/2, min (1-|c_i|))."""
    n = len(pts)
    if n == 0:
        return 0.0
    # Pairwise distances
    diff = pts[:, None, :] - pts[None, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    # Distance to boundary
    radii = np.sqrt(np.sum(pts * pts, axis=1))
    min_boundary = np.min(1.0 - radii)
    return min(min_pair, min_boundary)


def _hex_lattice_init():
    """Generate a hexagonal lattice initialization and select N best points."""
    # Start with a reasonable spacing
    best_pts = None
    best_score = 0.0
    
    for spacing in np.linspace(0.22, 0.30, 8):
        k = int(2.0 / spacing) + 2
        rows = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                r = np.sqrt(x * x + y * y)
                if r <= 1.0 - spacing / 2.0:
                    rows.append([x, y])
        pts = np.array(rows)
        if len(pts) >= N:
            # Sort by distance from origin, take N closest
            radii = np.sqrt(np.sum(pts * pts, axis=1))
            idx = np.argsort(radii)[:N]
            pts_sel = pts[idx]
            score = _compute_score(pts_sel)
            if score > best_score:
                best_score = score
                best_pts = pts_sel
    
    if best_pts is None:
        # Fallback: random
        best_pts = np.random.randn(N, 2) * 0.8
        best_score = _compute_score(best_pts)
    
    return best_pts, best_score


def _slsqp_optimize(pts, max_time=30.0):
    """Optimize using SLSQP with a smooth objective."""
    start_time = time.time()
    n = len(pts)
    
    def objective(x):
        pts = x.reshape(n, 2)
        # Smooth min using log-sum-exp trick for the pair distances
        diff = pts[:, None, :] - pts[None, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        # Boundary distances
        radii = np.sqrt(np.sum(pts * pts, axis=1))
        boundary_dists = 1.0 - radii
        
        # Combine all constraints into a single vector
        all_dists = np.concatenate([dists[~np.eye(n, dtype=bool)], boundary_dists])
        
        # Use a soft-min (log-sum-exp with negative) for smoothness
        beta = 200.0
        min_val = -np.log(np.sum(np.exp(-beta * all_dists))) / beta
        
        # We want to maximize the minimum, so minimize the negative
        return -min_val
    
    def jac(x):
        pts = x.reshape(n, 2)
        diff = pts[:, None, :] - pts[None, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        radii = np.sqrt(np.sum(pts * pts, axis=1))
        boundary_dists = 1.0 - radii
        
        all_dists = np.concatenate([dists[~np.eye(n, dtype=bool)], boundary_dists])
        
        beta = 200.0
        exp_vals = np.exp(-beta * all_dists)
        sum_exp = np.sum(exp_vals)
        
        # Gradient of soft-min w.r.t. each distance
        weights = -exp_vals / sum_exp  # d(-softmin)/d(dist_k) = -w_k where w_k = exp(-beta*d_k)/sum
        
        # Build gradient
        grad = np.zeros(2 * n)
        
        # Pair distances part
        pair_idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                d = dists[i, j]
                if d > 1e-10:
                    w = weights[pair_idx]
                    # d(dist_ij)/d(pts_i) = (pts_i - pts_j)/dist_ij
                    dir_vec = (pts[i] - pts[j]) / d
                    grad[2*i:2*i+2] += w * dir_vec
                    grad[2*j:2*j+2] -= w * dir_vec
                pair_idx += 1
        
        # Boundary distances part
        for i in range(n):
            w = weights[n*(n-1)//2 + i]
            # d(1 - |c_i|)/d(c_i) = -c_i/|c_i|
            r = radii[i]
            if r > 1e-10:
                grad[2*i:2*i+2] += w * (-pts[i] / r)
        
        return grad
    
    # Initial guess
    x0 = pts.flatten()
    
    # Constraints: keep points inside the unit disk (with small margin)
    def constraint(i):
        def c(x):
            pts = x.reshape(n, 2)
            return 1.0 - np.sqrt(np.sum(pts[i]**2)) - 0.001
        return c
    
    cons = [{'type': 'ineq', 'fun': constraint(i)} for i in range(n)]
    
    result = minimize(objective, x0, method='SLSQP', jac=jac,
                      constraints=cons, options={'maxiter': 200, 'ftol': 1e-12})
    
    if result.success or result.fun < objective(x0):
        return result.x.reshape(n, 2), result.fun
    return pts, objective(x0)


def _sa_polish(pts, max_time=60.0):
    """Simulated annealing polish to escape local minima."""
    start_time = time.time()
    n = len(pts)
    best_pts = pts.copy()
    best_score = _compute_score(pts)
    
    # Temperature schedule
    T = 0.05
    T_min = 0.001
    cooling_rate = 0.995
    
    current_pts = pts.copy()
    current_score = best_score
    
    while time.time() - start_time < max_time:
        if T < T_min:
            break
        
        # Pick a random circle to move
        i = np.random.randint(n)
        # Random perturbation
        delta = np.random.randn(2) * T
        new_pts = current_pts.copy()
        new_pts[i] += delta
        
        # Check boundary constraint
        r_new = np.sqrt(np.sum(new_pts[i]**2))
        if r_new > 1.0 - 0.001:
            # Project back
            new_pts[i] *= (1.0 - 0.001) / r_new
        
        new_score = _compute_score(new_pts)
        
        # Accept if better or with probability exp((new_score - current_score)/T)
        if new_score >= current_score:
            current_pts = new_pts
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_pts = new_pts.copy()
        else:
            delta_e = (new_score - current_score) / T
            if np.random.random() < np.exp(delta_e):
                current_pts = new_pts
                current_score = new_score
        
        T *= cooling_rate
    
    return best_pts, best_score


def _final_polish(pts, max_time=20.0):
    """High-precision final polish using tighter tolerances."""
    start_time = time.time()
    n = len(pts)
    
    def objective(x):
        pts = x.reshape(n, 2)
        diff = pts[:, None, :] - pts[None, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        radii = np.sqrt(np.sum(pts * pts, axis=1))
        boundary_dists = 1.0 - radii
        
        all_dists = np.concatenate([dists[~np.eye(n, dtype=bool)], boundary_dists])
        
        beta = 500.0
        min_val = -np.log(np.sum(np.exp(-beta * all_dists))) / beta
        return -min_val
    
    x0 = pts.flatten()
    
    def constraint(i):
        def c(x):
            pts = x.reshape(n, 2)
            return 1.0 - np.sqrt(np.sum(pts[i]**2)) - 0.0001
        return c
    
    cons = [{'type': 'ineq', 'fun': constraint(i)} for i in range(n)]
    
    result = minimize(objective, x0, method='SLSQP',
                      constraints=cons, options={'maxiter': 300, 'ftol': 1e-14})
    
    if result.success or result.fun < objective(x0):
        return result.x.reshape(n, 2)
    return pts


def construct():
    start_time = time.time()
    deadline = start_time + 120.0 - DEADLINE_MARGIN
    
    # Step 1: Structured initialization
    pts, score = _hex_lattice_init()
    
    # Step 2: SLSQP optimization with time check
    if time.time() < deadline - 80.0:
        pts, _ = _slsqp_optimize(pts, max_time=min(30.0, deadline - time.time() - 80.0))
    
    # Step 3: SA polish with time check
    if time.time() < deadline - 30.0:
        pts, _ = _sa_polish(pts, max_time=min(40.0, deadline - time.time() - 30.0))
    
    # Step 4: Final high-precision polish
    if time.time() < deadline - 5.0:
        pts = _final_polish(pts, max_time=min(15.0, deadline - time.time() - 5.0))
    
    # Ensure validity: project any points outside the disk
    radii = np.sqrt(np.sum(pts * pts, axis=1))
    for i in range(len(pts)):
        if radii[i] > 1.0:
            pts[i] *= (1.0 - 1e-9) / radii[i]
    
    return pts
