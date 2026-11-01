import numpy as np
import time
from scipy.optimize import minimize

N = 32

def construct():
    start_time = time.time()
    deadline = start_time + 140 # 10s safety margin
    
    best_score = 0.0
    best_solution = None

    def get_initial_guesses():
        guesses = []
        
        # 1. Grid 6x6
        side = 6
        xs = (np.arange(side) + 0.5) / side
        ys = (np.arange(side) + 0.5) / side
        gx, gy = np.meshgrid(xs, ys)
        gx = gx.flatten()[:N]
        gy = gy.flatten()[:N]
        guesses.append(np.column_stack((gx, gy)))
        
        # 2. Perturbed Grid
        np.random.seed(42)
        pg = np.column_stack((gx, gy)) + np.random.randn(N, 2) * 0.02
        pg = np.clip(pg, 0.05, 0.95)
        guesses.append(pg)
        
        # 3. Random
        np.random.seed(123)
        guesses.append(np.random.rand(N, 2) * 0.8 + 0.1)
        
        # 4. Another Random
        np.random.seed(456)
        guesses.append(np.random.rand(N, 2) * 0.8 + 0.1)
        
        # 5. Hexagonal-ish packing init
        # Just another perturbed grid
        np.random.seed(789)
        pg2 = np.column_stack((gx, gy)) + np.random.randn(N, 2) * 0.05
        pg2 = np.clip(pg2, 0.05, 0.95)
        guesses.append(pg2)

        return guesses

    def calculate_radii_and_grads(pts):
        """
        Calculate radii based on nearest neighbor and boundary.
        Also returns gradient of sum of radii w.r.t pts.
        """
        # Pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        np.fill_diagonal(dist_sq, np.inf)
        
        min_dist_sq = np.min(dist_sq, axis=1)
        min_dist = np.sqrt(min_dist_sq)
        nearest_idx = np.argmin(dist_sq, axis=1)
        
        # Distance to boundary
        db = np.minimum(pts[:, 0], 1 - pts[:, 0])
        db = np.minimum(db, np.minimum(pts[:, 1], 1 - pts[:, 1]))
        
        # Radius
        radii = np.minimum(min_dist / 2.0, db)
        
        # Active constraint: 0=neighbor, 1=boundary
        active = np.zeros(N, dtype=int)
        active[min_dist / 2.0 >= db] = 1
        
        # Boundary side: 0=left, 1=right, 2=bottom, 3=top
        bound_side = np.zeros(N, dtype=int)
        dl = pts[:, 0]
        dr = 1 - pts[:, 0]
        dbot = pts[:, 1]
        dtop = 1 - pts[:, 1]
        
        for i in range(N):
            if active[i] == 1:
                d_vals = [dl[i], dr[i], dbot[i], dtop[i]]
                bound_side[i] = np.argmin(d_vals)
        
        # Gradient
        grad = np.zeros_like(pts)
        
        for i in range(N):
            # Self constraint gradient
            if active[i] == 1:
                side = bound_side[i]
                if side == 0: grad[i, 0] += 1.0
                elif side == 1: grad[i, 0] -= 1.0
                elif side == 2: grad[i, 1] += 1.0
                elif side == 3: grad[i, 1] -= 1.0
            else:
                j = nearest_idx[i]
                d = min_dist[i]
                if d > 1e-10:
                    grad[i] += (pts[i] - pts[j]) / (2 * d)
            
            # Neighbor contributions (i is nn of k)
            # We can vectorize this slightly or just loop
            # For each k, if active[k]==0, add grad to nearest_idx[k]
        
        for k in range(N):
            if active[k] == 0:
                j = nearest_idx[k]
                d = min_dist[k]
                if d > 1e-10:
                    # grad of r_k w.r.t p_j
                    grad[j] += (pts[k] - pts[j]) / (2 * d)
                    
        return radii, grad

    def shrink_to_fit(pts, radii):
        """
        If any circles overlap, shrink them until they don't.
        """
        # Simple iterative shrink
        for _ in range(10):
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dist_sq = np.sum(diff**2, axis=2)
            np.fill_diagonal(dist_sq, np.inf)
            dist = np.sqrt(dist_sq)
            
            r_sum = radii[:, np.newaxis] + radii[np.newaxis, :]
            
            # Find overlaps
            overlap = r_sum - dist
            max_overlap = np.max(overlap)
            
            if max_overlap <= 1e-9:
                break
                
            # Shrink all radii by half the max overlap?
            # Or shrink only the involved ones?
            # Global shrink is safer but suboptimal.
            # Let's shrink all radii by 0.5 * max_overlap
            shrink = 0.5 * max_overlap
            radii = np.maximum(0, radii - shrink)
            
        # Also check boundaries
        db = np.minimum(pts[:, 0], 1 - pts[:, 0])
        db = np.minimum(db, np.minimum(pts[:, 1], 1 - pts[:, 1]))
        radii = np.minimum(radii, db)
        
        return radii

    def is_valid(pts, radii):
        if np.any(pts[:, 0] < radii - 1e-9) or np.any(1 - pts[:, 0] < radii - 1e-9):
            return False
        if np.any(pts[:, 1] < radii - 1e-9) or np.any(1 - pts[:, 1] < radii - 1e-9):
            return False
        
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        np.fill_diagonal(dist_sq, np.inf)
        dist = np.sqrt(dist_sq)
        
        r_sum = radii[:, np.newaxis] + radii[np.newaxis, :]
        
        if np.any(dist < r_sum - 1e-9):
            return False
        return True

    def relax(pts, steps=2000, alpha=0.05):
        pts = pts.copy()
        for _ in range(steps):
            if time.time() > deadline - 10:
                break
                
            radii, grad = calculate_radii_and_grads(pts)
            
            # Normalize gradient to prevent huge steps?
            # grad_norm = np.linalg.norm(grad)
            # if grad_norm > 0:
            #     grad = grad / grad_norm
            
            pts += alpha * grad
            pts = np.clip(pts, 0, 1)
            
        return pts

    guesses = get_initial_guesses()
    
    for v0 in guesses:
        if time.time() > deadline - 20:
            break
            
        # Run relaxation
        pts_opt = relax(v0, steps=3000, alpha=0.05)
        
        # Calculate radii
        radii, _ = calculate_radii_and_grads(pts_opt)
        
        # Shrink to fit to ensure validity
        radii_valid = shrink_to_fit(pts_opt, radii)
        
        if is_valid(pts_opt, radii_valid):
            score = np.sum(radii_valid)
            if score > best_score:
                best_score = score
                best_solution = np.column_stack((pts_opt, radii_valid))
        
        # Polish: smaller step size
        if time.time() < deadline - 10:
            pts_polish = relax(pts_opt, steps=2000, alpha=0.01)
            radii_polish, _ = calculate_radii_and_grads(pts_polish)
            radii_polish_valid = shrink_to_fit(pts_polish, radii_polish)
            
            if is_valid(pts_polish, radii_polish_valid):
                score_polish = np.sum(radii_polish_valid)
                if score_polish > best_score:
                    best_score = score_polish
                    best_solution = np.column_stack((pts_polish, radii_polish_valid))

    if best_solution is None:
        side = 6
        xs = (np.arange(side) + 0.5) / side
        ys = (np.arange(side) + 0.5) / side
        gx, gy = np.meshgrid(xs, ys)
        gx = gx.flatten()[:N]
        gy = gy.flatten()[:N]
        r = 0.5 / side * 0.9
        return np.column_stack((gx, gy, r))
        
    return best_solution
