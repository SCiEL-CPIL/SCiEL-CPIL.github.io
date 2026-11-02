"""
Circle packing in unit square: 26 circles, maximize sum of radii.
Uses SLSQP with hard non-overlap and boundary constraints.
"""
import numpy as np
from scipy.optimize import minimize
import time

N = 26

def construct():
    """
    Pack 26 circles in the unit square maximizing sum of radii.
    Returns an array of shape (26, 3) with rows (x, y, radius).
    """
    deadline = time.time() + 115  # Safety margin for 120s limit
    best_score = -1
    best_result = None

    # --- Helper Functions ---

    def unpack_params(p):
        """Unpack flat parameter vector into centers and radii."""
        centers = p[:2*N].reshape(N, 2)
        radii = p[2*N:]
        return centers, radii

    def objective(p):
        """Minimize negative sum of radii."""
        centers, radii = unpack_params(p)
        return -np.sum(radii)

    def constraints(p):
        """
        Define constraints for SLSQP.
        SLSQP expects constraints in the form f(x) >= 0 for inequalities.
        
        1. Boundary constraints:
           x_i - r_i >= 0
           1 - x_i - r_i >= 0
           y_i - r_i >= 0
           1 - y_i - r_i >= 0
           
        2. Non-overlap constraints:
           dist(i, j) - (r_i + r_j) >= 0
        """
        centers, radii = unpack_params(p)
        cons = []
        
        # Boundary constraints
        for i in range(N):
            x, y = centers[i]
            r = radii[i]
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i] - x[2*N+i]})
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: 1 - x[2*i] - x[2*N+i]})
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i+1] - x[2*N+i]})
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: 1 - x[2*i+1] - x[2*N+i]})
        
        # Non-overlap constraints
        for i in range(N):
            for j in range(i+1, N):
                cons.append({
                    'type': 'ineq',
                    'fun': lambda x, i=i, j=j: np.linalg.norm(x[2*i:2*i+2] - x[2*j:2*j+2]) - (x[2*N+i] + x[2*N+j])
                })
        
        return cons

    def is_valid(centers, radii, tol=1e-6):
        """Check if a configuration is strictly valid."""
        if np.any(radii < -tol) or np.any(radii > 0.5):
            return False
        # Boundary check
        if np.any(centers[:, 0] - radii < -tol) or np.any(centers[:, 0] + radii > 1 + tol):
            return False
        if np.any(centers[:, 1] - radii < -tol) or np.any(centers[:, 1] + radii > 1 + tol):
            return False
        
        # Overlap check
        for i in range(N):
            for j in range(i+1, N):
                dist = np.linalg.norm(centers[i] - centers[j])
                if dist < radii[i] + radii[j] - tol:
                    return False
        return True

    def run_optimization(p0, maxiter=200):
        """Run SLSQP optimization from initial guess p0."""
        try:
            res = minimize(
                objective,
                p0,
                method='SLSQP',
                constraints=constraints,
                options={'maxiter': maxiter, 'ftol': 1e-8, 'disp': False}
            )
            return res
        except Exception as e:
            return None

    # --- Initial Configuration Generators ---

    def init_grid():
        """5x5 grid + 1 extra circle in a gap."""
        side = 5
        centers = np.zeros((N, 2))
        radii = np.zeros(N)
        for i in range(side):
            for j in range(side):
                idx = i * side + j
                if idx < N:
                    centers[idx] = [(i + 0.5) / side, (j + 0.5) / side]
                    radii[idx] = 0.5 / side * 0.9
        # Add 26th circle in a corner or gap
        # Place it near (0.5, 0.5) but slightly offset, or just pick a random valid spot
        # For now, place it in the center of the 3x3 block? No, let's just perturb one.
        # Actually, a 5x5 grid has 25 spots. Let's put the 26th at (0.5, 0.5) with small radius?
        # Better: Use a 6x5 grid? No, 30 circles.
        # Let's use a 5x5 grid and squeeze the 26th into the middle of the square, shrinking others if needed.
        # Simpler: Just use a 5x5 grid for 25 circles and add one small circle in the middle.
        # But we want to maximize sum.
        # Let's try a different grid: 6 columns x 5 rows is 30.
        # Let's use a random initialization for the 26th or just perturb the grid.
        
        # Alternative: 6x5 grid is too many. 
        # Let's stick to 5x5 for 25, and place the 26th at (0.5, 0.5) with r=0.05, 
        # and let the optimizer fix it.
        centers[25] = [0.5, 0.5]
        radii[25] = 0.05
        
        p = np.concatenate([centers.flatten(), radii])
        return p

    def init_hex():
        """Hexagonal lattice approximation."""
        centers = np.zeros((N, 2))
        radii = np.zeros(N)
        # Hex packing: rows are offset
        # Approximate radius for hex packing in unit square for N=26
        # Side length approx sqrt(26) ~ 5.1
        # Let's generate a hex grid and pick the best 26
        pts = []
        r_est = 0.1
        dx = 2 * r_est
        dy = np.sqrt(3) * r_est
        for row in range(6):
            for col in range(6):
                x = col * dx + (row % 2) * (dx / 2)
                y = row * dy
                if 0 <= x <= 1 and 0 <= y <= 1:
                    pts.append([x, y])
        
        # If we have fewer than 26, add randoms. If more, pick closest to center?
        # For now, just take the first 26 valid points from a denser grid
        pts = []
        r_est = 0.08
        dx = 2 * r_est
        dy = np.sqrt(3) * r_est
        for row in range(8):
            for col in range(8):
                x = col * dx + (row % 2) * (dx / 2)
                y = row * dy
                if 0 <= x <= 1 and 0 <= y <= 1:
                    pts.append([x, y])
        
        if len(pts) < N:
            # Pad with random
            while len(pts) < N:
                pts.append([np.random.rand(), np.random.rand()])
        
        centers = np.array(pts[:N])
        radii = np.full(N, r_est * 0.9)
        
        # Ensure initial validity (might overlap slightly, SLSQP will fix)
        p = np.concatenate([centers.flatten(), radii])
        return p

    def init_random():
        """Random initial configuration."""
        centers = np.random.rand(N, 2)
        radii = np.full(N, 0.05)
        p = np.concatenate([centers.flatten(), radii])
        return p

    # --- Main Optimization Loop ---

    # 1. Try Grid
    if time.time() < deadline:
        p0 = init_grid()
        res = run_optimization(p0, maxiter=100)
        if res and res.success:
            centers, radii = unpack_params(res.x)
            if is_valid(centers, radii):
                score = np.sum(radii)
                if score > best_score:
                    best_score = score
                    best_result = np.column_stack([centers, radii])

    # 2. Try Hex
    if time.time() < deadline:
        p0 = init_hex()
        res = run_optimization(p0, maxiter=100)
        if res and res.success:
            centers, radii = unpack_params(res.x)
            if is_valid(centers, radii):
                score = np.sum(radii)
                if score > best_score:
                    best_score = score
                    best_result = np.column_stack([centers, radii])

    # 3. Multiple Random Restarts
    np.random.seed(42)
    max_restarts = 10
    for i in range(max_restarts):
        if time.time() > deadline - 5:
            break
            
        p0 = init_random()
        # Slight perturbation to avoid exact duplicates
        p0 += np.random.randn(len(p0)) * 0.01
        
        res = run_optimization(p0, maxiter=150)
        if res:
            # SLSQP might not declare success but still find a good point
            # Check validity of the result
            centers, radii = unpack_params(res.x)
            if is_valid(centers, radii):
                score = np.sum(radii)
                if score > best_score:
                    best_score = score
                    best_result = np.column_stack([centers, radii])

    # Fallback if no valid result found (should not happen with SLSQP)
    if best_result is None:
        # Return a valid grid configuration
        side = 5
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) < N:
                    cells.append([(i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 0.01])
        best_result = np.array(cells[:N])

    return best_result

