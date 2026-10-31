import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize, basinhopping
import time

N = 16
D = 2

def construct():
    start_time = time.time()
    deadline = start_time + 135

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

    def generate_random_seed(rng):
        return rng.uniform(-2, 2, size=(16, 2))

    seeds = []
    hex_seed = generate_hex_seed()
    seeds.append(hex_seed)
    ring_seed = generate_ring_seed()
    seeds.append(ring_seed)
    poly_seed = generate_polygon_seed()
    seeds.append(poly_seed)
    
    rng = np.random.default_rng(42)
    for s in [hex_seed, ring_seed, poly_seed]:
        for _ in range(3):
            seeds.append(s + rng.normal(0, 0.2, size=(16, 2)))
    for _ in range(5):
        seeds.append(generate_random_seed(rng))

    best_score = 1e10
    best_pts = None

    # Use Basin Hopping for global exploration
    # Objective for Basin Hopping: Soft score with high beta
    beta_bh = 200
    
    def bh_objective(x):
        pts = x.reshape(-1, 2)
        return soft_score(pts, beta_bh)

    def bh_minimizer(x):
        # Use L-BFGS-B for speed, it works well on the smooth soft objective
        res = minimize(
            bh_objective,
            x,
            method='L-BFGS-B',
            options={'maxiter': 200, 'ftol': 1e-10}
        )
        return res.x, res.fun, {'nfev': res.nfev}

    def bh_step(x, T):
        # Random kick
        step_size = 0.5 * T
        return x + rng.normal(0, step_size, size=x.shape)

    for i, seed in enumerate(seeds):
        if time.time() > deadline - 15:
            break
            
        current_pts = seed.copy()
        
        # Check initial score
        curr_score = score_points(current_pts)
        if curr_score < best_score:
            best_score = curr_score
            best_pts = current_pts.copy()
            
        # Run Basin Hopping
        try:
            bh_res = basinhopping(
                bh_objective,
                current_pts.flatten(),
                minimizer=bh_minimizer,
                step=bh_step,
                T=1.0,
                stepsize=0.5,
                maxfun=500,
                niter=20,
                interval=10,
                disp=False
            )
            new_pts = bh_res.x.reshape(-1, 2)
            new_score = score_points(new_pts)
            
            if new_score < best_score:
                best_score = new_score
                best_pts = new_pts.copy()
                current_pts = new_pts
        except Exception:
            pass

        # Final polish with Nelder-Mead on exact objective
        if time.time() < deadline - 5:
            try:
                res_nm = minimize(
                    lambda p: score_points(p.reshape(-1, 2)),
                    current_pts.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10}
                )
                new_pts = res_nm.x.reshape(-1, 2)
                new_score = score_points(new_pts)
                
                if new_score < best_score:
                    best_score = new_score
                    best_pts = new_pts.copy()
            except Exception:
                pass

    if best_pts is None:
        return hex_seed
        
    d_check = pdist(best_pts, 'euclidean')
    if np.min(d_check) < 1e-10:
        return hex_seed
        
    return best_pts
