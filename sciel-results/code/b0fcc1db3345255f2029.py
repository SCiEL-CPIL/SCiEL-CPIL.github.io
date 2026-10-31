import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import time
import random

N = 16
D = 2

def construct():
    start_time = time.time()
    deadline = start_time + 135  # Safety margin for 150s limit

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

    def generate_seed_rings():
        """Generate ring-based seeds with varied radii and angles."""
        seeds = []
        rng = np.random.default_rng(7)
        
        # 1. Two rings: 8 inner, 8 outer
        for r_in in [0.5, 0.6, 0.7]:
            for r_out in [1.2, 1.4, 1.6]:
                for offset in [0, np.pi/8, np.pi/4]:
                    angles_in = np.linspace(0, 2*np.pi, 8, endpoint=False)
                    angles_out = np.linspace(0, 2*np.pi, 8, endpoint=False) + offset
                    pts_in = np.column_stack([r_in*np.cos(angles_in), r_in*np.sin(angles_in)])
                    pts_out = np.column_stack([r_out*np.cos(angles_out), r_out*np.sin(angles_out)])
                    seeds.append(np.vstack([pts_in, pts_out]))
                    
        # 2. Three rings: 6, 5, 5 or 4, 6, 6
        configs = [(6, 5, 5), (4, 6, 6), (5, 6, 5)]
        radii_sets = [
            (0.4, 1.0, 1.6),
            (0.5, 1.1, 1.8),
            (0.3, 0.9, 1.5)
        ]
        for n1, n2, n3 in configs:
            for r1, r2, r3 in radii_sets:
                a1 = np.linspace(0, 2*np.pi, n1, endpoint=False)
                a2 = np.linspace(0, 2*np.pi, n2, endpoint=False) + np.pi/n2
                a3 = np.linspace(0, 2*np.pi, n3, endpoint=False) + np.pi/n3
                p1 = np.column_stack([r1*np.cos(a1), r1*np.sin(a1)])
                p2 = np.column_stack([r2*np.cos(a2), r2*np.sin(a2)])
                p3 = np.column_stack([r3*np.cos(a3), r3*np.sin(a3)])
                seeds.append(np.vstack([p1, p2, p3]))

        # 3. Random perturbations of the best ring seeds
        for s in seeds[:10]:
            for _ in range(3):
                perturbed = s + rng.normal(0, 0.1, size=(16, 2))
                seeds.append(perturbed)
        
        return seeds

    def generate_seed_hex():
        """Generate hexagonal lattice seeds."""
        seeds = []
        # Standard hex
        hex_pts = []
        R = 5
        for q in range(-R, R + 1):
            for r in range(-R, R + 1):
                x = q + 0.5 * r
                y = (np.sqrt(3) / 2) * r
                hex_pts.append((x, y))
        hex_pts = np.array(hex_pts)
        dists = np.sum(hex_pts**2, axis=1)
        idx = np.argsort(dists)[:16]
        seeds.append(hex_pts[idx])
        
        # Perturbed hex
        rng = np.random.default_rng(11)
        for _ in range(5):
            seeds.append(seeds[0] + rng.normal(0, 0.1, size=(16, 2)))
        return seeds

    def generate_seed_random():
        rng = np.random.default_rng(42)
        seeds = []
        for _ in range(5):
            seeds.append(rng.uniform(-1, 1, size=(16, 2)))
        return seeds

    all_seeds = generate_seed_rings() + generate_seed_hex() + generate_seed_random()

    best_score = 1e10
    best_pts = None

    # Optimization strategy:
    # 1. Score all seeds
    # 2. For top seeds, run Powell -> L-BFGS-B (soft) -> Powell -> L-BFGS-B (higher beta)
    
    scored_seeds = [(score_points(s), s) for s in all_seeds]
    scored_seeds.sort(key=lambda x: x[0])
    
    # Process top seeds
    num_to_optimize = 20
    
    for i in range(min(num_to_optimize, len(scored_seeds))):
        if time.time() > deadline - 20:
            break
        curr_score, pts = scored_seeds[i]
        if curr_score < best_score:
            best_score = curr_score
            best_pts = pts.copy()
        
        # Optimize this seed
        current_pts = pts.copy()
        
        # Phase 1: Powell on exact objective
        try:
            res1 = minimize(lambda p: score_points(p.reshape(-1, 2)), current_pts.flatten(), 
                            method='Powell', options={'maxiter': 100, 'maxfev': 1000})
            if res1.fun < best_score:
                best_score = res1.fun
                best_pts = res1.x.reshape(-1, 2).copy()
            current_pts = res1.x.reshape(-1, 2)
        except:
            pass
        
        # Phase 2: L-BFGS-B on soft objective (beta=50)
        if time.time() < deadline - 15:
            try:
                res2 = minimize(lambda p: soft_score(p.reshape(-1, 2), 50.0), current_pts.flatten(),
                                method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-12})
                exact_score = score_points(res2.x.reshape(-1, 2))
                if exact_score < best_score:
                    best_score = exact_score
                    best_pts = res2.x.reshape(-1, 2).copy()
                current_pts = res2.x.reshape(-1, 2)
            except:
                pass
                
        # Phase 3: Powell again on exact objective to refine
        if time.time() < deadline - 10:
            try:
                res3 = minimize(lambda p: score_points(p.reshape(-1, 2)), current_pts.flatten(), 
                                method='Powell', options={'maxiter': 200, 'maxfev': 2000})
                if res3.fun < best_score:
                    best_score = res3.fun
                    best_pts = res3.x.reshape(-1, 2).copy()
                current_pts = res3.x.reshape(-1, 2)
            except:
                pass
                
        # Phase 4: L-BFGS-B on higher beta soft objective (beta=200)
        if time.time() < deadline - 5:
            try:
                res4 = minimize(lambda p: soft_score(p.reshape(-1, 2), 200.0), current_pts.flatten(),
                                method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-14})
                exact_score = score_points(res4.x.reshape(-1, 2))
                if exact_score < best_score:
                    best_score = exact_score
                    best_pts = res4.x.reshape(-1, 2).copy()
            except:
                pass

    # Final polish if time permits
    if best_pts is not None and time.time() < deadline - 3:
        try:
            res_final = minimize(lambda p: soft_score(p.reshape(-1, 2), 500.0), best_pts.flatten(),
                                 method='L-BFGS-B', options={'maxiter': 2000, 'ftol': 1e-15})
            exact_score = score_points(res_final.x.reshape(-1, 2))
            if exact_score < best_score:
                best_score = exact_score
                best_pts = res_final.x.reshape(-1, 2).copy()
        except:
            pass

    if best_pts is None:
        # Fallback
        return all_seeds[0]
        
    # Verify distinctness
    d_check = pdist(best_pts, 'euclidean')
    if np.min(d_check) < 1e-10:
        return all_seeds[0]
        
    return best_pts
