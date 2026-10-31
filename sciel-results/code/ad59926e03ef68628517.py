"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16
D = 2

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 150s limit, leave margin for overhead and safety

    def score_points(pts):
        """
        Calculate (max_dist / min_dist)^2 for a set of points.
        pts: np.array of shape (N, 2)
        """
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        iu = np.triu_indices(N, 1)
        d_sq = dists_sq[iu]
        min_d_sq = np.min(d_sq)
        max_d_sq = np.max(d_sq)
        if min_d_sq < 1e-12:
            return 1e10
        return max_d_sq / min_d_sq

    def soft_score(pts, beta=50.0):
        """
        Smooth surrogate for the max/min ratio using log-sum-exp.
        This makes the objective differentiable for gradient-based optimizers.
        """
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        iu = np.triu_indices(N, 1)
        d_sq = dists_sq[iu]
        
        # Soft max: (1/beta) * log(sum(exp(beta * x)))
        # Soft min: -(1/beta) * log(sum(exp(-beta * x)))
        # We want to minimize max/min, so minimize soft_max - soft_min
        # But we need to be careful with numerical stability
        
        max_d_sq = np.max(d_sq)
        min_d_sq = np.min(d_sq)
        
        if min_d_sq < 1e-12:
            return 1e10
        
        # Use log-sum-exp for numerical stability
        # soft_max = max_d_sq + (1/beta) * log(sum(exp(beta * (d_sq - max_d_sq))))
        # soft_min = min_d_sq - (1/beta) * log(sum(exp(-beta * (d_sq - min_d_sq))))
        
        exp_max = np.exp(beta * (d_sq - max_d_sq))
        soft_max = max_d_sq + (1.0 / beta) * np.log(np.sum(exp_max))
        
        exp_min = np.exp(-beta * (d_sq - min_d_sq))
        soft_min = min_d_sq - (1.0 / beta) * np.log(np.sum(exp_min))
        
        return soft_max / soft_min

    def generate_hexagonal_seed():
        """Generate a hexagonal lattice seed and select 16 points."""
        v1 = np.array([1.0, 0.0])
        v2 = np.array([0.5, np.sqrt(3)/2])
        
        hex_points = []
        for i in range(-2, 3):
            for j in range(-2, 3):
                p = i * v1 + j * v2
                d_sq = np.sum(p**2)
                if d_sq <= 4.0 + 1e-9:
                    hex_points.append(p)
        
        hex_points = np.array(hex_points)
        
        if len(hex_points) != 19:
            return hex_points[:16]
        
        d_sqs = np.sum(hex_points**2, axis=1)
        outer_indices = np.where(np.isclose(d_sqs, 4.0))[0]
        
        best_combo = None
        best_score = float('inf')
        for remove_idx in itertools.combinations(outer_indices, 3):
            keep_idx = [i for i in range(19) if i not in remove_idx]
            pts = hex_points[keep_idx]
            score = score_points(pts)
            if score < best_score:
                best_score = score
                best_combo = pts
        
        return best_combo

    def generate_torus_seed():
        """
        Generate a seed using 4x4 grid on a torus with minimum-image convention.
        We try different aspect ratios (w/h) to find the best geometry.
        """
        best_pts = None
        best_score = float('inf')
        
        # Try a range of aspect ratios
        for aspect in [1.0, 0.8, 1.2, 0.7, 1.4, np.sqrt(2)/2, 2/np.sqrt(3)]:
            # Torus dimensions
            W = aspect
            H = 1.0
            
            # 4x4 grid points on the torus
            pts = []
            for i in range(4):
                for j in range(4):
                    x = (i + 0.5) * W / 4.0
                    y = (j + 0.5) * H / 4.0
                    pts.append(np.array([x, y]))
            
            pts = np.array(pts)
            
            # Compute distances using minimum-image convention
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            # Wrap to [-L/2, L/2]
            diff[0] -= W * np.round(diff[0] / W)
            diff[1] -= H * np.round(diff[1] / H)
            d_sq = np.sum(diff**2, axis=2)
            iu = np.triu_indices(N, 1)
            d_sq = d_sq[iu]
            
            min_d_sq = np.min(d_sq)
            max_d_sq = np.max(d_sq)
            if min_d_sq < 1e-12:
                continue
            score = max_d_sq / min_d_sq
            
            if score < best_score:
                best_score = score
                # Unwrap: use the actual positions (not wrapped)
                best_pts = pts.copy()
        
        if best_pts is None:
            # Fallback: unit square 4x4 grid
            best_pts = np.array([[i * 0.25, j * 0.25] for i in range(4) for j in range(4)])
        
        return best_pts

    def generate_concentric_circles_seed():
        """Generate a seed with points on concentric circles."""
        rng = np.random.default_rng(42)
        
        # 1 center + 5 inner + 10 outer
        pts = [np.array([0.0, 0.0])]
        
        # Inner ring: 5 points
        for i in range(5):
            angle = 2 * np.pi * i / 5
            pts.append(np.array([0.5 * np.cos(angle), 0.5 * np.sin(angle)]))
        
        # Outer ring: 10 points
        for i in range(10):
            angle = 2 * np.pi * i / 10
            pts.append(np.array([1.0 * np.cos(angle), 1.0 * np.sin(angle)]))
        
        return np.array(pts)

    def generate_perturbed_grid_seed():
        """Generate a perturbed grid seed."""
        rng = np.random.default_rng(123)
        
        # 4x4 grid with small perturbations
        pts = []
        for i in range(4):
            for j in range(4):
                x = i * 0.5 + rng.normal(0, 0.02)
                y = j * 0.5 + rng.normal(0, 0.02)
                pts.append(np.array([x, y]))
        
        return np.array(pts)

    def prepare_for_optimization(init_pts):
        """
        Normalize and reorder points for optimization.
        Fix p0 at origin and p1 on x-axis to remove translation/rotation/scale.
        """
        # Find the pair with minimum distance
        diff = init_pts[:, np.newaxis, :] - init_pts[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        iu = np.triu_indices(N, 1)
        d_sq = dists_sq[iu]
        
        min_idx = np.argmin(d_sq)
        p, q = np.unravel_index(min_idx, (N, N))
        
        if p > q:
            p, q = q, p
        
        p0 = init_pts[p]
        p1 = init_pts[q]
        
        v = p1 - p0
        norm_v = np.linalg.norm(v)
        
        if norm_v < 1e-8:
            return None, None
            
        # Rotation matrix to align v with x-axis
        cos_t = v[0] / norm_v
        sin_t = v[1] / norm_v
        R_mat = np.array([[cos_t, sin_t], [-sin_t, cos_t]])
        
        # Transform all points
        transformed = (init_pts - p0) @ R_mat.T / norm_v
        
        # Reorder points so that p is at index 0 and q is at index 1
        perm = list(range(N))
        perm[0] = p
        perm[1] = q
        remaining = [idx for idx in range(N) if idx != p and idx != q]
        perm[2:] = remaining
        
        transformed = transformed[perm]
        
        # init_params are the coordinates of points 2..15
        init_params = transformed[2:].flatten()
        
        return init_params, (p, q)

    def reconstruct_points(params, p, q):
        """Reconstruct full points array from optimized params."""
        pts = np.zeros((N, 2))
        pts[0] = [0.0, 0.0]
        pts[1] = [1.0, 0.0]
        pts[2:] = params.reshape(-1, 2)
        
        # Reorder back
        perm = list(range(N))
        perm[0] = p
        perm[1] = q
        remaining = [idx for idx in range(N) if idx != p and idx != q]
        perm[2:] = remaining
        
        inv_perm = np.argsort(perm)
        return pts[inv_perm]

    # Generate diverse seeds
    seeds = []
    
    # Hexagonal seed
    hex_seed = generate_hexagonal_seed()
    if hex_seed is not None:
        seeds.append(('hex', hex_seed))
    
    # Concentric circles seed
    circle_seed = generate_concentric_circles_seed()
    seeds.append(('circle', circle_seed))
    
    # Perturbed grid seed
    grid_seed = generate_perturbed_grid_seed()
    seeds.append(('grid', grid_seed))
    
    # Add some random seeds
    rng = np.random.default_rng(999)
    for _ in range(3):
        random_pts = rng.random((N, 2)) * 2 - 1  # Uniform in [-1, 1]^2
        seeds.append(('random', random_pts))
    
    # Torus seed
    torus_seed = generate_torus_seed()
    seeds.append(('torus', torus_seed))
    
    best_score = float('inf')
    best_params = None
    best_pq = None
    best_seed_name = None
    
    for seed_name, init_pts in seeds:
        if time.time() > deadline - 30:
            break
        
        init_params, pq = prepare_for_optimization(init_pts)
        if init_params is None:
            continue
        
        p, q = pq
        
        # Try multiple optimization strategies
        strategies = [
            ('Nelder-Mead', {'maxiter': 500, 'maxfev': 5000}),
            ('Powell', {'maxiter': 300, 'maxfev': 3000}),
        ]
        
        for method, options in strategies:
            if time.time() > deadline - 20:
                break
            
            # Define objective function
            def objective(params, method=method):
                if time.time() > deadline:
                    return 1e10
                pts = np.zeros((N, 2))
                pts[0] = [0.0, 0.0]
                pts[1] = [1.0, 0.0]
                pts[2:] = params.reshape(-1, 2)
                
                # Use soft score for gradient-based, exact for derivative-free
                if method in ['L-BFGS-B', 'BFGS', 'CG']:
                    return soft_score(pts, beta=100.0)
                else:
                    return score_points(pts)
            
            try:
                res = minimize(objective, init_params, method=method, options=options)
                
                if res.fun < best_score:
                    best_score = res.fun
                    best_params = res.x.copy()
                    best_pq = (p, q)
                    best_seed_name = f"{seed_name}_{method}"
            except Exception:
                pass
    
    # Final high-precision polish: first soft L-BFGS-B, then exact Nelder-Mead
    if best_params is not None and time.time() < deadline - 15:
        def soft_objective(params):
            pts = np.zeros((N, 2))
            pts[0] = [0.0, 0.0]
            pts[1] = [1.0, 0.0]
            pts[2:] = params.reshape(-1, 2)
            return soft_score(pts, beta=200.0)
        
        try:
            res = minimize(soft_objective, best_params, method='L-BFGS-B',
                          options={'maxiter': 200, 'maxfev': 2000, 'ftol': 1e-15, 'gtol': 1e-15})
            
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
        
        # Exact local polish with Nelder-Mead on the true objective
        if time.time() < deadline - 5:
            def exact_objective(params):
                pts = np.zeros((N, 2))
                pts[0] = [0.0, 0.0]
                pts[1] = [1.0, 0.0]
                pts[2:] = params.reshape(-1, 2)
                return score_points(pts)
            
            try:
                res2 = minimize(exact_objective, best_params, method='Nelder-Mead',
                               options={'maxiter': 1000, 'maxfev': 10000, 'xatol': 1e-14, 'fatol': 1e-14})
                
                # Verify with exact score
                pts_check = np.zeros((N, 2))
                pts_check[0] = [0.0, 0.0]
                pts_check[1] = [1.0, 0.0]
                pts_check[2:] = res2.x.reshape(-1, 2)
                exact_s = score_points(pts_check)
                
                if exact_s < best_score:
                    best_score = exact_s
                    best_params = res2.x.copy()
            except Exception:
                pass
    
    # Construct final points
    if best_params is not None:
        p, q = best_pq
        final_pts = reconstruct_points(best_params, p, q)
        
        # Ensure no points are exactly identical
        final_score = score_points(final_pts)
        if final_score < 1e8:  # Valid score
            return final_pts
    
    # Fallback: return the best seed
    return seeds[0][1]

# EVOLVE-BLOCK-END
