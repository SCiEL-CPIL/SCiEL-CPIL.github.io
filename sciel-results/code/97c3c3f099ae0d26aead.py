import numpy as np
import time
from scipy.optimize import minimize

N = 50
KNOWN_RADIUS = 0.1258254895304

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    def compute_radius(pts):
        """Compute the packing radius for given points."""
        if pts.shape != (N, 2):
            return 0.0
        if not np.all(np.isfinite(pts)):
            return 0.0
        norms = np.linalg.norm(pts, axis=1)
        if np.any(norms > 1.0 + 1e-10):
            return 0.0
        b_min = np.min(1.0 - norms)
        # Pairwise distances - vectorized
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        d_min = np.min(dists)
        r = min(d_min / 2.0, b_min)
        return max(r, 0.0)
    
    def is_feasible(pts, r, tol=1e-9):
        """Check if pts are feasible for radius r."""
        if pts.shape != (N, 2):
            return False
        if not np.all(np.isfinite(pts)):
            return False
        norms = np.linalg.norm(pts, axis=1)
        if np.any(norms > 1.0 - r + tol):
            return False
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        if np.min(dists) < 2.0 * r - tol:
            return False
        return True
    
    def smooth_objective(pts, r, soft_beta=30.0):
        """Smooth surrogate: minimize -soft_min of all constraint margins."""
        if pts.shape != (N, 2):
            return 1e10
        if not np.all(np.isfinite(pts)):
            return 1e10
        
        norms = np.linalg.norm(pts, axis=1)
        b_margins = 1.0 - norms - r
        
        # Pairwise margins
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        # Only take upper triangle
        iu = np.triu_indices(N, k=1)
        p_margins = dists[iu] / 2.0 - r
        
        all_margins = np.concatenate([b_margins, p_margins])
        
        neg_beta_x = -soft_beta * all_margins
        max_val = np.max(neg_beta_x)
        log_sum_exp = max_val + np.log(np.sum(np.exp(neg_beta_x - max_val)))
        soft_min_val = -log_sum_exp / soft_beta
        
        return -soft_min_val
    
    def smooth_grad(pts, r, soft_beta=30.0):
        """Gradient of smooth_objective."""
        if pts.shape != (N, 2):
            return np.zeros_like(pts)
        
        norms = np.linalg.norm(pts, axis=1)
        b_margins = 1.0 - norms - r
        
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        iu = np.triu_indices(N, k=1)
        p_margins = dists[iu] / 2.0 - r
        
        all_margins = np.concatenate([b_margins, p_margins])
        
        neg_beta_x = -soft_beta * all_margins
        max_val = np.max(neg_beta_x)
        exp_vals = np.exp(neg_beta_x - max_val)
        sum_exp = np.sum(exp_vals)
        weights = exp_vals / sum_exp
        
        grad = np.zeros_like(pts)
        
        # Boundary margins
        for i in range(N):
            if norms[i] > 1e-12:
                grad[i] += -weights[i] * (-pts[i] / norms[i])
        
        # Pairwise margins
        for idx in range(len(iu[0])):
            i, j = iu[0][idx], iu[1][idx]
            w = weights[N + idx]
            d = dists[i, j]
            if d > 1e-12:
                diff_ij = pts[i] - pts[j]
                grad[i] += -w * (diff_ij / (2.0 * d))
                grad[j] += -w * (-diff_ij / (2.0 * d))
        
        return grad
    
    def optimize_for_radius(r_target, initial_pts, max_iter=2000, soft_beta=30.0):
        """Optimize points to be feasible for radius r_target."""
        x0 = initial_pts.flatten().copy()
        
        def obj(x):
            pts = x.reshape(N, 2)
            return smooth_objective(pts, r_target, soft_beta)
        
        def jac(x):
            pts = x.reshape(N, 2)
            return smooth_grad(pts, r_target, soft_beta)
        
        try:
            result = minimize(obj, x0, method='L-BFGS-B', jac=jac,
                             options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-10})
            opt_pts = result.x.reshape(N, 2)
            return opt_pts
        except Exception:
            return initial_pts
    
    def dynamics_refine(pts, r_target, n_steps=5000, dt=0.001, damping=0.99):
        """Rigid-body dynamics to refine packing."""
        pts = pts.copy()
        velocities = np.zeros_like(pts)
        
        for step in range(n_steps):
            if time.time() > deadline - 5:
                break
            
            forces = np.zeros_like(pts)
            
            # Boundary force: push inward if too close to boundary
            norms = np.linalg.norm(pts, axis=1)
            for i in range(N):
                if norms[i] > 1e-12:
                    # Desired distance is 1 - r_target
                    target_r = 1.0 - r_target
                    if norms[i] > target_r:
                        # Push inward
                        direction = -pts[i] / norms[i]
                        force_mag = 100.0 * (norms[i] - target_r)
                        forces[i] += direction * force_mag
            
            # Pairwise repulsion
            for i in range(N):
                for j in range(i+1, N):
                    diff = pts[i] - pts[j]
                    d = np.linalg.norm(diff)
                    min_d = 2.0 * r_target
                    if d < min_d * 1.5 and d > 1e-12:
                        # Repulsive force
                        direction = diff / d
                        force_mag = 50.0 * (min_d - d) if d < min_d else 0
                        forces[i] += direction * force_mag
                        forces[j] -= direction * force_mag
            
            # Update velocities and positions
            velocities = velocities * damping + forces * dt
            pts += velocities * dt
            
            # Clamp to unit circle
            norms = np.linalg.norm(pts, axis=1)
            for i in range(N):
                if norms[i] > 1.0:
                    pts[i] = pts[i] / norms[i] * 0.999
            
            # Damp velocities
            velocities *= 0.95
        
        return pts
    
    def hexagonal_init(spacing):
        """Generate points on a hexagonal lattice."""
        points = []
        k = int(1.0 / spacing) + 3
        for i in range(-k, k + 1):
            for j in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                if x*x + y*y <= 1.0:
                    points.append([x, y])
        points = np.array(points)
        if len(points) < N:
            return None
        dists = np.linalg.norm(points, axis=1)
        indices = np.argsort(dists)[:N]
        return points[indices]
    
    def ring_init(n_rings, r_start, r_end):
        """Initialize with concentric rings."""
        pts = [[0.0, 0.0]]
        for ring in range(1, n_rings + 1):
            r = r_start + (r_end - r_start) * (ring - 1) / max(n_rings - 1, 1)
            if ring == 1:
                n_pts = 6
            else:
                n_pts = ring * 6
            n_pts = min(n_pts, N - len(pts))
            if n_pts <= 0:
                break
            angles = np.linspace(0, 2*np.pi, n_pts, endpoint=False)
            for a in angles:
                pts.append([r * np.cos(a), r * np.sin(a)])
        pts = np.array(pts[:N])
        if len(pts) < N:
            return None
        return pts
    
    # Generate initial candidates
    candidates = []
    
    # Hexagonal grids
    for s in np.linspace(0.12, 0.22, 15):
        pts = hexagonal_init(s)
        if pts is not None:
            candidates.append(pts)
    
    # Ring-based inits
    for n_rings in [5, 6, 7, 8]:
        for r_start in [0.05, 0.1, 0.15]:
            pts = ring_init(n_rings, r_start, 0.85)
            if pts is not None:
                candidates.append(pts)
    
    # Random inits
    for _ in range(10):
        pts = np.random.rand(N, 2) * 2 - 1
        norms = np.linalg.norm(pts, axis=1)
        pts = pts / norms[:, None] * 0.9
        candidates.append(pts)
    
    # Evaluate all candidates to find best starting point
    best_pts = None
    best_r = 0.0
    for c in candidates:
        r = compute_radius(c)
        if r > best_r:
            best_r = r
            best_pts = c.copy()
    
    if best_pts is None:
        best_pts = np.random.rand(N, 2) * 0.7
        best_r = 0.01
    
    # Binary search on radius
    r_low = best_r
    r_high = 0.135
    
    # Verify r_low is feasible
    if not is_feasible(best_pts, r_low):
        r_low = 0.05
    
    # Binary search iterations
    for iteration in range(15):
        if time.time() > deadline - 20:
            break
        
        r_mid = (r_low + r_high) / 2.0
        
        # Try multiple soft_beta values
        found_feasible = False
        for beta in [20.0, 30.0, 50.0, 80.0]:
            if time.time() > deadline - 10:
                break
            
            opt_pts = optimize_for_radius(r_mid, best_pts, max_iter=1500, soft_beta=beta)
            
            # Also try dynamics refinement
            dyn_pts = dynamics_refine(opt_pts, r_mid, n_steps=2000, dt=0.0005)
            
            if is_feasible(opt_pts, r_mid):
                r_low = r_mid
                best_pts = opt_pts.copy()
                best_r = compute_radius(opt_pts)
                found_feasible = True
                break
            elif is_feasible(dyn_pts, r_mid):
                r_low = r_mid
                best_pts = dyn_pts.copy()
                best_r = compute_radius(dyn_pts)
                found_feasible = True
                break
        
        if not found_feasible:
            r_high = r_mid
    
    # Final polish
    if time.time() < deadline - 10:
        r_final = r_low
        # Try to push slightly higher
        for dr in [0.00005, 0.00002, 0.00001]:
            if time.time() > deadline - 5:
                break
            r_try = r_final + dr
            opt_pts = optimize_for_radius(r_try, best_pts, max_iter=2000, soft_beta=100.0)
            dyn_pts = dynamics_refine(opt_pts, r_try, n_steps=3000, dt=0.0003)
            
            if is_feasible(opt_pts, r_try):
                best_pts = opt_pts
                r_final = r_try
                best_r = compute_radius(opt_pts)
            elif is_feasible(dyn_pts, r_try):
                best_pts = dyn_pts
                r_final = r_try
                best_r = compute_radius(dyn_pts)
        
        # One more polish at the final radius
        if time.time() < deadline - 3:
            opt_pts = optimize_for_radius(r_final, best_pts, max_iter=3000, soft_beta=150.0)
            dyn_pts = dynamics_refine(opt_pts, r_final, n_steps=5000, dt=0.0002)
            
            if compute_radius(opt_pts) >= best_r - 1e-10:
                best_pts = opt_pts
                best_r = compute_radius(opt_pts)
            if compute_radius(dyn_pts) >= best_r - 1e-10:
                best_pts = dyn_pts
                best_r = compute_radius(dyn_pts)
    
    # Ensure final result is valid
    final_r = compute_radius(best_pts)
    if final_r <= 0:
        best_pts = np.random.rand(N, 2) * 0.7
        best_pts[0] = [0.0, 0.0]
    
    return best_pts
