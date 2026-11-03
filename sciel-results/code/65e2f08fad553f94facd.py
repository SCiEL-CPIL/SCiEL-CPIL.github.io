import numpy as np
import time
import random
from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint

N = 200
SQ = 0.5  # half side length

def construct():
    max_time = 118.0
    start_time = time.time()
    np.random.seed(42)
    
    def compute_radius(P):
        """Compute the achievable radius for configuration P."""
        diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=-1))
        np.fill_diagonal(dist, np.inf)
        min_pair = np.min(dist) / 2.0
        min_bound = np.min([
            np.min(P[:, 0] + SQ),
            np.min(SQ - P[:, 0]),
            np.min(P[:, 1] + SQ),
            np.min(SQ - P[:, 1])
        ])
        return min(min_pair, min_bound)
    
    def relax_fixed_r(P_init, r, max_iters=300, alpha=0.01):
        """Relax configuration to achieve radius r. Returns (P, achieved_r)."""
        P = P_init.copy()
        current_alpha = alpha
        best_achieved = compute_radius(P)
        
        for it in range(max_iters):
            if time.time() - start_time > max_time - 2:
                break
                
            diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
            dist = np.sqrt(np.sum(diff**2, axis=-1))
            np.fill_diagonal(dist, np.inf)
            
            forces = np.zeros_like(P)
            
            # Pairwise repulsion for pairs closer than 2r
            mask = dist < 2 * r
            dist_safe = np.where(dist > 1e-10, dist, 1.0)
            force_mag = np.where(mask, (2 * r - dist) / dist_safe, 0.0)
            coeff = force_mag / dist_safe
            coeff = np.where(mask, coeff, 0.0)
            forces += np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
            
            # Boundary forces
            # Left wall
            ml = P[:, 0] < -SQ + r
            if np.any(ml):
                forces[ml, 0] += (r - (P[ml, 0] + SQ))
            # Right wall
            mr = P[:, 0] > SQ - r
            if np.any(mr):
                forces[mr, 0] -= (P[mr, 0] - (SQ - r))
            # Bottom wall
            mb = P[:, 1] < -SQ + r
            if np.any(mb):
                forces[mb, 1] += (r - (P[mb, 1] + SQ))
            # Top wall
            mt = P[:, 1] > SQ - r
            if np.any(mt):
                forces[mt, 1] -= (P[mt, 1] - (SQ - r))
            
            P += current_alpha * forces
            P[:, 0] = np.clip(P[:, 0], -SQ + r, SQ - r)
            P[:, 1] = np.clip(P[:, 1], -SQ + r, SQ - r)
            
            # Check if achieved
            min_pair = np.min(dist)
            min_bound = min(
                np.min(P[:, 0] + SQ),
                np.min(SQ - P[:, 0]),
                np.min(P[:, 1] + SQ),
                np.min(SQ - P[:, 1])
            )
            current_achieved = min(min_pair / 2.0, min_bound)
            
            if current_achieved > best_achieved:
                best_achieved = current_achieved
                
            if min_pair >= 2 * r and min_bound >= r:
                break
                
            # Adaptive alpha
            if it > 0 and current_achieved < best_achieved - 1e-6:
                current_alpha *= 0.9
                if current_alpha < 0.0001:
                    break
                    
        return P, compute_radius(P)
    
    def generate_hex_grid(m, jitter_scale=0.0):
        """Generate a hexagonal grid with approximately N points."""
        pts = []
        dx = 1.0 / (m + 1)
        dy = dx * np.sqrt(3) / 2
        total_rows = int(1.0 / dy) + 1
        y_start = -total_rows * dy / 2 + dy / 2
        
        for row in range(total_rows):
            y = y_start + row * dy
            offset = 0 if row % 2 == 0 else dx / 2
            x_start = -0.5 + dx / 2 + offset
            n_in_row = int((1.0 - dx) / dx) + 1
            for col in range(n_in_row):
                x = x_start + col * dx
                if -SQ <= x <= SQ and -SQ <= y <= SQ:
                    pts.append([x, y])
        
        pts = np.array(pts[:N])
        if len(pts) < N:
            pad = np.random.uniform(-SQ, SQ, (N - len(pts), 2))
            pts = np.vstack([pts, pad])
        if jitter_scale > 0:
            pts = pts + np.random.normal(0, jitter_scale, pts.shape)
        return pts
    
    def generate_square_grid(m, jitter_scale=0.0):
        """Generate a square grid with approximately N points."""
        pts = []
        for i in range(m):
            for j in range(m):
                x = -SQ + (i + 0.5) / m
                y = -SQ + (j + 0.5) / m
                pts.append([x, y])
        pts = np.array(pts[:N])
        if len(pts) < N:
            pad = np.random.uniform(-SQ, SQ, (N - len(pts), 2))
            pts = np.vstack([pts, pad])
        if jitter_scale > 0:
            pts = pts + np.random.normal(0, jitter_scale, pts.shape)
        return pts
    
    best_P = None
    best_r = 0.0
    
    # Generate diverse initial configurations
    inits = []
    # Hex grids with different densities
    for m in [13, 14, 15, 16]:
        inits.append(generate_hex_grid(m, jitter_scale=0.005))
    # Square grids
    for m in [14, 15]:
        inits.append(generate_square_grid(m, jitter_scale=0.005))
    # Random
    inits.append(np.random.uniform(-SQ, SQ, (N, 2)))
    
    # Try multiple target radii
    target_radii = [0.036, 0.0362, 0.0364, 0.0365, 0.0366]
    
    for r_target in target_radii:
        if time.time() - start_time > max_time - 10:
            break
            
        for P_init in inits:
            if time.time() - start_time > max_time - 3:
                break
                
            P_relaxed, achieved_r = relax_fixed_r(P_init, r_target, max_iters=200, alpha=0.015)
            
            if achieved_r > best_r:
                best_r = achieved_r
                best_P = P_relaxed
                
            # If we achieved the target, try to push higher
            if achieved_r >= r_target:
                # Try slightly higher
                for extra in [0.0001, 0.0002, 0.0003]:
                    r_try = r_target + extra
                    if time.time() - start_time > max_time - 2:
                        break
                    P_extra, achieved_extra = relax_fixed_r(P_relaxed, r_try, max_iters=100, alpha=0.005)
                    if achieved_extra > best_r:
                        best_r = achieved_extra
                        best_P = P_extra
                break  # Move to next target radius
    
    # If best_P is still None or very low, use a fallback
    if best_P is None or best_r < 0.035:
        # Use a simple grid as fallback
        m = 14
        best_P = generate_square_grid(m, jitter_scale=0.0)
        best_r = compute_radius(best_P)
    
    # Final optimization: try to squeeze out more radius using scipy
    if time.time() - start_time < max_time - 5:
        def objective(x):
            P = x.reshape(-1, 2)
            return -compute_radius(P)
        
        # Try a few scipy optimizations
        for _ in range(3):
            if time.time() - start_time > max_time - 2:
                break
            x0 = best_P.flatten()
            try:
                result = minimize(objective, x0, method='L-BFGS-B', 
                                  bounds=[(-SQ, SQ)] * N * 2,
                                  options={'maxiter': 100, 'maxls': 5})
                if result.fun < -best_r:
                    P_opt = result.x.reshape(-1, 2)
                    r_opt = compute_radius(P_opt)
                    if r_opt > best_r:
                        best_r = r_opt
                        best_P = P_opt
            except:
                pass
    
    return best_P
