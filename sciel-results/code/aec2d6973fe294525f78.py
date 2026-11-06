import numpy as np
import time
from scipy.optimize import minimize

N = 100
MAX_TIME = 140.0

def _calc_score(pts):
    """Calculate the actual score (radius) for a set of points."""
    if pts.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    if np.any(pts < -0.5) or np.any(pts > 0.5):
        return 0.0
        
    # Pair distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dist_sq, 1e10)
    min_pair_dist = np.sqrt(np.min(dist_sq))
    
    # Wall distances
    min_wall_dist = np.min(0.5 - np.abs(pts))
    
    return min(min_pair_dist / 2.0, min_wall_dist)

def _penalty_energy_and_grad(x, r):
    """
    Compute penalty energy and gradient for a target radius r.
    Penalty is sum of squared violations of:
    1. Pairwise distance < 2r
    2. Wall distance < r
    """
    x = x.reshape(N, 3)
    
    # Pairwise distances
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    
    mask = ~np.eye(N, dtype=bool)
    dist_sq_masked = np.where(mask, dist_sq, 1e10)
    dists = np.sqrt(dist_sq_masked)
    dists = np.where(mask, dists, 1e10)
    
    # Violations: max(0, 2r - dist)
    min_dist_req = 2.0 * r
    violations = np.maximum(0.0, min_dist_req - dists)
    
    # Energy: sum of squared violations
    energy = 0.5 * np.sum(violations**2)
    
    # Gradient
    # d(energy)/dx_i = sum_j d(energy)/d(dist_ij) * d(dist_ij)/dx_i
    # d(energy)/d(violation_ij) = violation_ij
    # d(violation_ij)/d(dist_ij) = -1 if violation > 0, else 0
    # d(dist_ij)/dx_i = (x_i - x_j) / dist_ij
    
    grad = np.zeros_like(x)
    
    # Only compute gradient for pairs with violations
    viol_mask = violations > 1e-10
    if np.any(viol_mask):
        # factor[i,j] = -violation[i,j] / dist[i,j]
        # We need to be careful with division by zero, but dist > 0 if violation > 0 (unless 2r=0)
        factor = np.zeros_like(dists)
        factor[viol_mask] = -violations[viol_mask] / dists[viol_mask]
        
        # grad_x[i] += sum_j factor[i,j] * (x_i - x_j)
        weighted_diff = diff * factor[:, :, np.newaxis]
        grad -= np.sum(weighted_diff, axis=1)
        
    # Wall violations
    # d_left = x + 0.5, d_right = 0.5 - x
    d_left = x + 0.5
    d_right = 0.5 - x
    
    v_left = np.maximum(0.0, r - d_left)
    v_right = np.maximum(0.0, r - d_right)
    
    energy += 0.5 * np.sum(v_left**2) + 0.5 * np.sum(v_right**2)
    
    # Gradient from walls
    # d(energy)/dx = v_left - v_right  (since d(d_left)/dx=1, d(d_right)/dx=-1)
    grad += v_left - v_right
    
    return energy, grad.flatten()

def _is_feasible(x0, r, time_limit=5.0):
    """
    Try to find a configuration with minimum radius at least r.
    Returns (success, best_x, min_penalty).
    """
    start_time = time.time()
    x = x0.copy()
    bounds = [(-0.5, 0.5)] * 3 * N
    
    def objective(x_flat):
        e, g = _penalty_energy_and_grad(x_flat, r)
        return e, g
    
    try:
        res = minimize(objective, x.flatten(), method='L-BFGS-B', jac=True,
                       bounds=bounds,
                       options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-12})
        
        if res.success or res.fun < 1e-6:
            new_x = res.x.reshape(N, 3)
            new_x = np.clip(new_x, -0.5, 0.5)
            # Verify actual score
            actual_score = _calc_score(new_x)
            if actual_score >= r - 1e-5:
                return True, new_x, res.fun
            else:
                return False, new_x, res.fun
        else:
            return False, x, res.fun if hasattr(res, 'fun') else 1e10
    except Exception:
        return False, x, 1e10

def _generate_lattice_initialization():
    """
    Generate a high-density initial configuration using a perturbed FCC lattice.
    FCC has higher packing density than BCC or Simple Cubic.
    """
    # We need 100 points.
    # Let's use a grid of FCC unit cells.
    # FCC primitive vectors: a/2(1,1,0), a/2(1,0,1), a/2(0,1,1)
    # We'll generate a large FCC lattice and crop/center it, then jitter.
    
    # Approximate radius for 100 spheres in unit cube: ~0.11
    # Volume per sphere ~ 4/3 pi r^3 ~ 0.0018
    # Total volume 1. Density ~ 18%.
    # FCC density is ~ 74%. So we have room to expand.
    
    # Let's try a 4x4x4 grid of FCC cells?
    # FCC has 4 atoms per cell. 4*4*4 * 4 = 256 atoms. Too many.
    # 3x3x3 * 4 = 108 atoms. Close to 100.
    
    points = []
    a = 0.2 # lattice constant, will adjust
    
    # Generate FCC points in a 3x3x3 supercell
    # Basis: (0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)
    basis = [
        np.array([0.0, 0.0, 0.0]),
        np.array([0.5, 0.5, 0.0]),
        np.array([0.5, 0.0, 0.5]),
        np.array([0.0, 0.5, 0.5])
    ]
    
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for b in basis:
                    p = np.array([i, j, k]) * a + b * a
                    points.append(p)
    
    points = np.array(points)
    # We have 108 points. Need 100.
    # Center the cloud
    points -= np.mean(points, axis=0)
    
    # Scale to fit in cube [-0.5, 0.5] with some margin
    max_coord = np.max(np.abs(points))
    scale = 0.45 / max_coord
    points *= scale
    
    # Jitter to break symmetry
    np.random.seed(42)
    points += np.random.uniform(-0.01, 0.01, points.shape)
    
    # Take first 100
    return points[:N]

def construct():
    start_time = time.time()
    deadline = start_time + MAX_TIME
    
    # Initial guess
    x = _generate_lattice_initialization()
    x = np.clip(x, -0.49, 0.49)
    
    best_x = x.copy()
    best_score = _calc_score(best_x)
    
    # Bisection search on radius r
    # Lower bound: current score
    # Upper bound: theoretical max (sqrt(3)/6 ~ 0.288 for 1 sphere, much less for 100)
    # Let's estimate upper bound.
    # Volume argument: 100 * (4/3) pi r^3 <= 1 => r <= (3/400 pi)^(1/3) ~ 0.13
    # But packing efficiency limits it. 0.11 is known best. Let's search up to 0.12.
    
    r_low = best_score
    r_high = 0.12
    
    # Quick check: is r_high feasible? Probably not.
    # We will binary search between r_low and r_high.
    
    # To speed up, we can do a few linear probes first.
    # But bisection is robust.
    
    # We need to be careful with time.
    # Each feasibility check takes ~1-5s.
    # We can do ~20-30 checks in 120s.
    # log2((0.12 - 0.111) / 1e-5) ~ log2(900) ~ 10 checks for 5 digits precision.
    # So we have plenty of time for high precision.
    
    # Strategy:
    # 1. Try to increase r_low.
    # 2. If feasible, update best_x and r_low.
    # 3. If not, update r_high.
    
    # Let's use a slightly more aggressive initial r_high to see if we can jump.
    # But if r_high is infeasible, we waste time.
    # Let's stick to bisection.
    
    # Optimization: Use the result of the previous feasibility check as the starting point
    # for the next one (warm start).
    
    current_x = best_x.copy()
    
    while time.time() < deadline - 10:
        r_mid = (r_low + r_high) / 2.0
        
        # If the interval is very small, we are done
        if r_high - r_low < 1e-6:
            break
            
        feasible, new_x, penalty = _is_feasible(current_x, r_mid, time_limit=5.0)
        
        if feasible:
            r_low = r_mid
            current_x = new_x
            score = _calc_score(new_x)
            if score > best_score:
                best_score = score
                best_x = new_x
        else:
            r_high = r_mid
            # If infeasible, we might still have improved the configuration for the lower r
            # but we don't know the score. Let's keep current_x as the warm start.
            # Actually, if it's infeasible, the optimizer tried to satisfy r_mid and failed.
            # The resulting x might be better than current_x for r_low?
            # Let's check score of new_x.
            score_new = _calc_score(new_x)
            if score_new > best_score:
                best_score = score_new
                best_x = new_x
                current_x = new_x # Use this as warm start for next iteration
        
    # Final polish: Use the best_x found and try to maximize score directly with a high-beta soft-min
    # or just return best_x. The bisection should have found a very good local optimum.
    # Let's do one final L-BFGS run on the soft-min energy to squeeze out any last bits.
    
    if time.time() < deadline - 5:
        def objective_soft(x_flat):
            x_r = x_flat.reshape(N, 3)
            diff = x_r[:, np.newaxis, :] - x_r[np.newaxis, :, :]
            dist_sq = np.sum(diff**2, axis=2)
            mask = ~np.eye(N, dtype=bool)
            dist_sq_masked = np.where(mask, dist_sq, 1.0)
            dists = np.sqrt(dist_sq_masked) / 2.0
            dists = np.where(mask, dists, 1.0)
            
            d_walls = np.concatenate([
                (x_r + 0.5).flatten(),
                (0.5 - x_r).flatten()
            ])
            
            all_dists = np.concatenate([dists.flatten(), d_walls])
            
            beta = 100.0
            d_min = np.min(all_dists)
            exp_terms = np.exp(-beta * (all_dists - d_min))
            log_sum = np.log(np.sum(exp_terms))
            energy = d_min + log_sum / beta
            
            weights = exp_terms / np.sum(exp_terms)
            
            grad_x = np.zeros_like(x_r)
            
            pair_grad_weights = weights[:N*N].reshape(N, N)
            dists_raw = np.sqrt(dist_sq_masked)
            dists_raw = np.where(mask, dists_raw, 1.0)
            factor = np.where(mask, pair_grad_weights / dists_raw, 0.0)
            
            weighted_diff = diff * factor[:, :, np.newaxis]
            grad_x -= np.sum(weighted_diff, axis=1)
            
            wall_grad_weights = weights[N*N:]
            grad_x += wall_grad_weights[:N*3].reshape(N, 3)
            grad_x -= wall_grad_weights[N*3:].reshape(N, 3)
            
            return energy, grad_x.flatten()
        
        try:
            res = minimize(objective_soft, best_x.flatten(), method='L-BFGS-B', jac=True,
                           bounds=[(-0.5, 0.5)] * 3 * N,
                           options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-14})
            if res.success:
                final_x = res.x.reshape(N, 3)
                final_x = np.clip(final_x, -0.5, 0.5)
                final_score = _calc_score(final_x)
                if final_score > best_score:
                    best_score = final_score
                    best_x = final_x
        except Exception:
            pass
            
    return best_x

if __name__ == "__main__":
    # Quick self-test
    t0 = time.time()
    pts = construct()
    t1 = time.time()
    score = _calc_score(pts)
    print(f"Time: {t1-t0:.2f}s, Score: {score}")
