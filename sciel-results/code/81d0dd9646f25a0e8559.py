import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import time

N = 16
D = 2

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Tighter safety margin

    def score_points(pts):
        d = pdist(pts, 'euclidean')
        min_d = np.min(d)
        if min_d < 1e-8:
            return 1e10
        max_d = np.max(d)
        return (max_d / min_d) ** 2

    def soft_score(pts, beta):
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        iu = np.triu_indices(N, 1)
        d_sq = dists_sq[iu]
        d_sq = np.maximum(d_sq, 1e-12)
        
        max_d_sq = np.max(d_sq)
        soft_max = max_d_sq + (1.0 / beta) * np.log(np.sum(np.exp(beta * (d_sq - max_d_sq))))
        
        min_d_sq = np.min(d_sq)
        soft_min = min_d_sq - (1.0 / beta) * np.log(np.sum(np.exp(-beta * (d_sq - min_d_sq))))
        
        if soft_min < 1e-8:
            return 1e10
        return soft_max / soft_min

    def generate_hex_seed():
        pts = []
        for q in range(-3, 4):
            for r in range(-3, 4):
                x = 1.5 * q
                y = np.sqrt(3) / 2 * q + np.sqrt(3) * r
                pts.append([x, y])
        pts = np.array(pts)
        dists = np.sum(pts**2, axis=1)
        idx = np.argsort(dists)[:16]
        return pts[idx]

    def generate_ring_seed():
        angles1 = np.linspace(0, 2*np.pi, 8, endpoint=False)
        angles2 = np.linspace(0, 2*np.pi, 8, endpoint=False) + np.pi/8
        r1 = 1.0
        r2 = 2.0
        pts1 = np.column_stack([r1*np.cos(angles1), r1*np.sin(angles1)])
        pts2 = np.column_stack([r2*np.cos(angles2), r2*np.sin(angles2)])
        return np.vstack([pts1, pts2])

    def generate_polygon_seed():
        angles = np.linspace(0, 2*np.pi, 16, endpoint=False)
        r = 1.0
        return np.column_stack([r*np.cos(angles), r*np.sin(angles)])

    def generate_ring_seed_v2(r1, r2, offset):
        angles1 = np.linspace(0, 2*np.pi, 8, endpoint=False)
        angles2 = np.linspace(0, 2*np.pi, 8, endpoint=False) + offset
        pts1 = np.column_stack([r1*np.cos(angles1), r1*np.sin(angles1)])
        pts2 = np.column_stack([r2*np.cos(angles2), r2*np.sin(angles2)])
        return np.vstack([pts1, pts2])

    def generate_hex_perturbed(base, rng, scale):
        return base + rng.normal(0, scale, size=(16, 2))

    # Generate focused set of high-quality seeds
    rng = np.random.default_rng(42)
    hex_base = generate_hex_seed()
    ring_base = generate_ring_seed()
    poly_base = generate_polygon_seed()

    seeds = [
        hex_base,
        ring_base,
        poly_base,
        generate_ring_seed_v2(0.7, 1.8, np.pi/8),
        generate_ring_seed_v2(0.8, 2.0, np.pi/4),
        generate_ring_seed_v2(0.6, 1.6, np.pi/12),
        generate_hex_perturbed(hex_base, rng, 0.15),
        generate_hex_perturbed(hex_base, rng, 0.15),
        generate_ring_perturbed := ring_base + rng.normal(0, 0.15, size=(16, 2)),
        generate_ring_perturbed2 := ring_base + rng.normal(0, 0.15, size=(16, 2)),
        poly_base + rng.normal(0, 0.1, size=(16, 2)),
        poly_base + rng.normal(0, 0.1, size=(16, 2)),
    ]

    best_score = 1e10
    best_pts = None

    # Check all seeds first
    for seed in seeds:
        s = score_points(seed)
        if s < best_score:
            best_score = s
            best_pts = seed.copy()

    # Optimization: use fewer betas, focus on quality over quantity
    betas = [20, 80, 300]
    
    for seed in seeds:
        if time.time() > deadline - 15:
            break
            
        current_pts = seed.copy()
        
        # L-BFGS-B on soft objective
        for beta in betas:
            if time.time() > deadline - 15:
                break
            try:
                res = minimize(
                    lambda p, b=beta: soft_score(p.reshape(-1, 2), b),
                    current_pts.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                new_pts = res.x.reshape(-1, 2)
                new_score = score_points(new_pts)
                
                if new_score < best_score:
                    best_score = new_score
                    best_pts = new_pts.copy()
                
                if res.fun < 1e9:
                    current_pts = new_pts
            except Exception:
                pass
        
        # Nelder-Mead polish (better for nonsmooth than Powell)
        if time.time() < deadline - 10:
            try:
                res_nm = minimize(
                    lambda p: score_points(p.reshape(-1, 2)),
                    current_pts.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'maxfev': 5000, 'xatol': 1e-10, 'fatol': 1e-12}
                )
                new_pts = res_nm.x.reshape(-1, 2)
                new_score = score_points(new_pts)
                
                if new_score < best_score:
                    best_score = new_score
                    best_pts = new_pts.copy()
            except Exception:
                pass

        # Final Powell polish if time allows
        if time.time() < deadline - 5 and best_pts is not None:
            try:
                res_powell = minimize(
                    lambda p: score_points(p.reshape(-1, 2)),
                    best_pts.flatten(),
                    method='Powell',
                    options={'maxiter': 100, 'maxfev': 1000}
                )
                new_pts = res_powell.x.reshape(-1, 2)
                new_score = score_points(new_pts)
                
                if new_score < best_score:
                    best_score = new_score
                    best_pts = new_pts.copy()
            except Exception:
                pass

    if best_pts is None:
        return hex_base
        
    d_check = pdist(best_pts, 'euclidean')
    if np.min(d_check) < 1e-10:
        return hex_base
        
    return best_pts
