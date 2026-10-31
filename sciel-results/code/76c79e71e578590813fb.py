"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16
D = 2

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    # 1. Pre-compute diverse initial configurations to break symmetry
    def get_initial_configs():
        configs = []
        
        # Grid 4x4
        x = np.linspace(-1, 1, 4)
        xx, yy = np.meshgrid(x, x)
        configs.append(np.column_stack((xx.ravel(), yy.ravel())))
        
        # Hexagonal-ish lattice (subset)
        # Generate a larger hex lattice and pick the best 16
        pts = []
        r = 0.5
        for i in range(-4, 5):
            for j in range(-4, 5):
                x = i * r * np.sqrt(3)
                y = j * r * 1.5
                if i % 2 != 0:
                    y += r * 0.75
                pts.append([x, y])
        pts = np.array(pts)
        # Center and normalize
        pts = pts - pts.mean(axis=0)
        norms = np.linalg.norm(pts, axis=1)
        max_norm = norms.max()
        if max_norm > 0:
            pts = pts / max_norm
        # Select 16 points closest to center? Or just first 16? 
        # Better: select a compact subset. 
        # For now, just take a central crop if it exists, otherwise random.
        # Since we generated a grid, let's just take the first 16 sorted by radius
        idx = np.argsort(norms)[:N]
        configs.append(pts[idx])
        
        # Two concentric rings (8+8)
        theta1 = np.linspace(0, 2*np.pi, 8, endpoint=False)
        theta2 = np.linspace(0, 2*np.pi, 8, endpoint=False) + np.pi/8
        r1 = 0.5
        r2 = 1.0
        p1 = np.column_stack((r1*np.cos(theta1), r1*np.sin(theta1)))
        p2 = np.column_stack((r2*np.cos(theta2), r2*np.sin(theta2)))
        configs.append(np.vstack((p1, p2)))
        
        # Random
        configs.append(np.random.uniform(-1, 1, (N, 2)))
        
        return configs

    def score_raw(pts):
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        sq_dists = np.sum(diff**2, axis=2)
        iu = np.triu_indices(N, 1)
        d_sq = sq_dists[iu]
        d_min = np.sqrt(d_sq.min())
        d_max = np.sqrt(d_sq.max())
        if d_min < 1e-12:
            return 1e10
        return (d_max / d_min) ** 2

    def normalize(pts):
        pts = pts - np.mean(pts, axis=0)
        scale = np.sqrt(np.mean(np.sum(pts**2, axis=1)))
        if scale > 1e-9:
            pts = pts / scale
        return pts

    # 2. Objective function with penalty for min distance
    # We want to minimize (d_max / d_min)^2.
    # Equivalent to minimizing d_max^2 subject to d_min^2 >= 1 (after scaling) 
    # or minimizing ratio directly.
    # Direct ratio minimization is non-smooth.
    # Penalty method: Minimize (d_max^2) + M * max(0, 1 - d_min^2)^2
    # But we also need to handle the ratio.
    # Let's try minimizing: d_max^2 + Lambda * (1/d_min^2)
    # This is smooth if d_min > 0.
    
    def objective(x):
        pts = x.reshape(N, 2)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        sq_dists = np.sum(diff**2, axis=2)
        iu = np.triu_indices(N, 1)
        d_sq = sq_dists[iu]
        
        d_min_sq = d_sq.min()
        d_max_sq = d_sq.max()
        
        if d_min_sq < 1e-8:
            return 1e10
        
        # Ratio squared is d_max_sq / d_min_sq
        return d_max_sq / d_min_sq

    def gradient(x):
        # Finite difference gradient for robustness, or analytic?
        # Analytic is faster but tricky for min/max.
        # Given time constraints, finite difference on 32D is okay for a few iterations,
        # but SLSQP needs gradients for efficiency.
        # Let's use finite difference to avoid bugs in analytic min/max gradients.
        eps = 1e-6
        g = np.zeros_like(x)
        f0 = objective(x)
        for i in range(len(x)):
            x_old = x[i]
            x[i] += eps
            f1 = objective(x)
            x[i] = x_old - eps
            f2 = objective(x)
            x[i] = x_old
            # Central difference
            g[i] = (f1 - f2) / (2 * eps)
        return g

    # 3. Optimization Loop
    best_score = 1e10
    best_pts = None
    
    inits = get_initial_configs()
    
    for init in inits:
        if time.time() > deadline - 10:
            break
            
        init_norm = normalize(init).ravel()
        
        try:
            res = minimize(
                objective,
                init_norm,
                jac=gradient,
                method='SLSQP',
                options={'maxiter': 200, 'ftol': 1e-10, 'disp': False}
            )
            
            if res.success or res.fun < best_score:
                current_score = res.fun
                if current_score < best_score:
                    best_score = current_score
                    best_pts = res.x.reshape(N, 2)
        except Exception:
            pass

    # If SLSQP fails or doesn't find anything, fallback to a simple grid/random search
    if best_pts is None:
        # Fallback: return the best initial config
        for init in inits:
            s = score_raw(init)
            if s < best_score:
                best_score = s
                best_pts = init

    return best_pts

# EVOLVE-BLOCK-END
