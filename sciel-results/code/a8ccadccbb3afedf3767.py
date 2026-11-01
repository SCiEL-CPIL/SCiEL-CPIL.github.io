import numpy as np
from scipy.optimize import minimize
import time

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    def objective(x, r):
        x = x.reshape(-1, 2)
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        iu = np.triu_indices(N, 1)
        dists = np.sqrt(dists_sq[iu])
        viol_cc = np.maximum(0, 2*r - dists)
        val_cc = np.sum(viol_cc**2)
        
        norms = np.linalg.norm(x, axis=1)
        viol_cb = np.maximum(0, norms - (1-r))
        val_cb = np.sum(viol_cb**2)
        
        return val_cc + val_cb

    def gradient(x, r):
        x = x.reshape(-1, 2)
        grad = np.zeros_like(x)
        
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        dists = np.sqrt(np.maximum(dists_sq, 1e-12))
        
        mask = dists < 2*r
        np.fill_diagonal(mask, False)
        
        coeff = np.zeros_like(dists)
        coeff[mask] = 2 * (dists[mask] - 2*r) / dists[mask]
        
        sum_coeff = np.sum(coeff, axis=1)
        term2 = coeff @ x.T
        term1 = x * sum_coeff[:, np.newaxis]
        grad_cc = term1 - term2
        
        norms = np.linalg.norm(x, axis=1)
        R = 1 - r
        mask_cb = norms > R
        grad_cb = np.zeros_like(x)
        if np.any(mask_cb):
            f_prime = 2 * (norms[mask_cb] - R)
            grad_cb[mask_cb] = f_prime[:, np.newaxis] * x[mask_cb] / norms[mask_cb, np.newaxis]
            
        return (grad_cc + grad_cb).flatten()

    def init_positions(r):
        pts = []
        pts.append([0.0, 0.0])
        k = 1
        while len(pts) < N:
            radius = k * 2 * r
            n_pts = 6 * k
            angles = np.linspace(0, 2*np.pi, n_pts, endpoint=False)
            offset = 0 if k % 2 == 0 else np.pi / n_pts
            angles += offset
            ring_pts = np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])
            pts.extend(ring_pts)
            k += 1
        pts = np.array(pts[:N])
        noise = np.random.normal(0, r*0.1, pts.shape)
        pts += noise
        norms = np.linalg.norm(pts, axis=1)
        max_norm = 1 - r
        mask = norms > max_norm
        if np.any(mask):
            pts[mask] *= max_norm / norms[mask, np.newaxis]
        return pts.flatten()

    # Fallback: Hex lattice
    def hex_lattice():
        spacing = 2.0 / (np.sqrt(N) + 1.0)
        while True:
            k = int(1.0 / spacing) + 2
            rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                    for i in range(-k, k + 1) for j in range(-k, k + 1)]
            points = np.array(rows)
            radius = np.hypot(points[:, 0], points[:, 1])
            points, radius = points[radius <= 1.0 - spacing / 2.0], radius[radius <= 1.0 - spacing / 2.0]
            if len(points) >= N:
                return points[np.argsort(radius, kind="stable")[:N]]
            spacing *= 0.98

    best_x = hex_lattice().flatten()
    best_r = 0.0
    # Calculate score of fallback
    def calc_score(pts):
        pts = pts.reshape(-1, 2)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        min_cc = np.min(dists) / 2.0
        norms = np.linalg.norm(pts, axis=1)
        min_cb = np.min(1.0 - norms)
        return min(min_cc, min_cb)
        
    best_score = calc_score(best_x)
    
    r = 0.10
    x = init_positions(r)
    
    # Try to optimize
    for iter in range(100):
        if time.time() > deadline:
            break
            
        res = minimize(objective, x, args=(r,), method='L-BFGS-B', jac=gradient, options={'maxiter': 1000, 'ftol': 1e-10})
        
        if res.success and res.fun < 1e-6:
            # Check actual score
            score = calc_score(res.x)
            if score > best_score:
                best_x = res.x
                best_score = score
                best_r = r
                
            # Try to increase r
            r_new = r + 0.001
            if r_new > 0.13:
                break
            x = res.x
            r = r_new
        else:
            # Failed
            # Try kick
            kick = np.random.normal(0, 0.01, x.shape)
            x_kicked = x + kick
            x_kicked = x_kicked.reshape(-1, 2)
            norms = np.linalg.norm(x_kicked, axis=1)
            max_norm = 1 - r
            mask = norms > max_norm
            if np.any(mask):
                x_kicked[mask] *= max_norm / norms[mask, np.newaxis]
            x_kicked = x_kicked.flatten()
            
            res = minimize(objective, x_kicked, args=(r,), method='L-BFGS-B', jac=gradient, options={'maxiter': 1000, 'ftol': 1e-10})
            if res.success and res.fun < 1e-6:
                score = calc_score(res.x)
                if score > best_score:
                    best_x = res.x
                    best_score = score
                    best_r = r
                x = res.x
                # Don't increase r if we had to kick
            else:
                # Reduce step or break
                # Try smaller r? No, we want max.
                # Break if stuck
                break

    return best_x.reshape(N, 2)
