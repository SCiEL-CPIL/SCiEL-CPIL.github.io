# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def squares_in_square10() -> tuple:
    """
    Maximizes the sum of side lengths of 10 non-overlapping axis-aligned squares
    inside the unit square [0, 1]^2.
    
    Uses a penalty-based local search approach starting from a 3x3 grid.
    """
    start_time = time.time()
    timeout = 600.0
    n = 10
    
    # Initial 3x3 grid configuration
    k = 3
    side = 1.0 / k
    idx = np.arange(n)
    cols = idx % k
    rows = idx // k
    cx = (cols + 0.5) * side
    cy = (rows + 0.5) * side
    centers = np.stack([cx, cy], axis=1).astype(np.float64)
    sides = np.full(n, side, dtype=np.float64)
    
    best_centers = centers.copy()
    best_sides = sides.copy()
    best_score = np.sum(sides)
    
    def check_validity(c, s):
        """
        Checks if the packing is valid according to the task contract.
        c: (10, 2) centers
        s: (10,) sides
        """
        if np.any(~np.isfinite(c)) or np.any(~np.isfinite(s)):
            return False
        if np.any(s <= 0):
            return False
        tol = 1e-7
        # Boundary check
        # x in [0, 1], y in [0, 1]
        if np.any(c[:, 0] - s/2 < -tol) or np.any(c[:, 0] + s/2 > 1 + tol):
            return False
        if np.any(c[:, 1] - s/2 < -tol) or np.any(c[:, 1] + s/2 > 1 + tol):
            return False
        # Overlap check
        for i in range(n):
            for j in range(i+1, n):
                t = (s[i] + s[j]) / 2 - tol
                # If distance in x < t AND distance in y < t, they overlap
                if abs(c[i, 0] - c[j, 0]) < t and abs(c[i, 1] - c[j, 1]) < t:
                    return False
        return True

    def penalty(c, s):
        """
        Calculates a smooth penalty for boundary violations and overlaps.
        """
        p = 0.0
        # Boundary penalty
        for i in range(n):
            x, y = c[i]
            half = s[i] / 2.0
            # Left
            if x - half < 0:
                p += (x - half) ** 2
            # Right
            if x + half > 1:
                p += (x + half - 1) ** 2
            # Bottom
            if y - half < 0:
                p += (y - half) ** 2
            # Top
            if y + half > 1:
                p += (y + half - 1) ** 2
        
        # Overlap penalty
        for i in range(n):
            for j in range(i+1, n):
                dx = abs(c[i, 0] - c[j, 0])
                dy = abs(c[i, 1] - c[j, 1])
                req_x = (s[i] + s[j]) / 2.0
                req_y = (s[i] + s[j]) / 2.0
                ox = max(0.0, req_x - dx)
                oy = max(0.0, req_y - dy)
                p += ox**2 + oy**2
        return p

    def objective(z):
        """
        Objective function for the optimizer.
        Minimizes -sum(sides) + lambda * penalty.
        """
        c = z[:2*n].reshape(n, 2)
        s = z[2*n:]
        val = -np.sum(s)
        pen = penalty(c, s)
        return val + 100.0 * pen

    def pack(c, s):
        return np.concatenate([c.flatten(), s])

    def unpack(z):
        c = z[:2*n].reshape(n, 2)
        s = z[2*n:]
        return c, s

    bounds = [(0, 1)] * (2*n) + [(0, 1)] * n
    
    # 1. Optimize from the initial grid
    z0 = pack(centers, sides)
    res = minimize(objective, z0, method='L-BFGS-B', bounds=bounds, 
                   options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-8})
    c_opt, s_opt = unpack(res.x)
    if check_validity(c_opt, s_opt):
        score = np.sum(s_opt)
        if score > best_score:
            best_score = score
            best_centers = c_opt.copy()
            best_sides = s_opt.copy()

    # 2. Multi-start local search with perturbations
    rng = np.random.default_rng(42)
    
    # We will try 100 restarts
    for i in range(100):
        if time.time() - start_time > timeout - 30:
            break
            
        # Perturb centers and sides
        # Small perturbations to escape local minima
        c_pert = centers + rng.normal(0, 0.02, size=(n, 2))
        # Ensure centers stay roughly within bounds to help optimizer
        c_pert[:, 0] = np.clip(c_pert[:, 0], 0.05, 0.95)
        c_pert[:, 1] = np.clip(c_pert[:, 1], 0.05, 0.95)
        
        s_pert = sides * (1 + rng.normal(0, 0.1, size=n))
        s_pert = np.clip(s_pert, 0.05, 1.0)
        
        z0 = pack(c_pert, s_pert)
        
        # Use L-BFGS-B
        res = minimize(objective, z0, method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8})
        
        c_opt, s_opt = unpack(res.x)
        
        # Check validity
        if check_validity(c_opt, s_opt):
            score = np.sum(s_opt)
            if score > best_score:
                best_score = score
                best_centers = c_opt.copy()
                best_sides = s_opt.copy()

    # Final sanity check
    if not check_validity(best_centers, best_sides):
        # Fallback to grid if something went wrong
        return centers, sides

    return best_centers, best_sides

# EVOLVE-BLOCK-END
