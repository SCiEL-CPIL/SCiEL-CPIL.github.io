# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def squares_in_square10() -> tuple:
    """
    Erdos "squares in a square" problem. Place n squares with pairwise disjoint
    interiors inside the unit square [0,1]^2 so as to MAXIMIZE the sum of their
    side lengths. Here n = 10.

    Returns:
        centers: np.ndarray of shape (10, 2) with the (x, y) center of each square.
        sides:   np.ndarray of shape (10,) with the side length of each square.
    The i-th axis-aligned square occupies
        [cx - s/2, cx + s/2] x [cy - s/2, cy + s/2].
    """
    n = 10
    deadline = time.time() + 570  # safety margin before 600s limit

    def _is_feasible(centers, sides, tol=1e-7):
        """Check if the configuration is feasible (vectorized)."""
        if centers.shape != (n, 2) or sides.shape != (n,):
            return False
        if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
            return False
        if np.any(sides <= 0):
            return False
        # Boundary check
        if np.any(centers[:, 0] - sides/2 < -tol) or np.any(centers[:, 0] + sides/2 > 1 + tol):
            return False
        if np.any(centers[:, 1] - sides/2 < -tol) or np.any(centers[:, 1] + sides/2 > 1 + tol):
            return False
        # Overlap check (vectorized)
        for i in range(n):
            for j in range(i + 1, n):
                t = (sides[i] + sides[j]) / 2 - tol
                if abs(centers[i, 0] - centers[j, 0]) < t and abs(centers[i, 1] - centers[j, 1]) < t:
                    return False
        return True

    def _sum_sides(sides):
        return np.sum(sides)

    def _objective(centers, sides):
        """Maximize sum of sides, penalize infeasibility."""
        if _is_feasible(centers, sides):
            return _sum_sides(sides)
        return -1e6

    def _penalty(x):
        """
        Penalty function for scipy minimize.
        x = [cx_0, cy_0, s_0, cx_1, cy_1, s_1, ..., cx_9, cy_9, s_9]
        Returns 0 if feasible, positive penalty otherwise.
        Uses smooth max for better gradient behavior.
        """
        centers = x[:2*n].reshape(n, 2)
        sides = x[2*n:]
        
        # Boundary violations - smooth
        penalty = 0.0
        for i in range(n):
            cx, cy, s = centers[i, 0], centers[i, 1], sides[i]
            if s <= 0.01:
                penalty += 10.0
            # Smooth boundary penalties
            b1 = cx - s/2 + 1e-7
            b2 = cx + s/2 - 1 - 1e-7
            b3 = cy - s/2 + 1e-7
            b4 = cy + s/2 - 1 - 1e-7
            penalty += np.maximum(-b1, 0)**2 * 100
            penalty += np.maximum(b2, 0)**2 * 100
            penalty += np.maximum(-b3, 0)**2 * 100
            penalty += np.maximum(b4, 0)**2 * 100
        
        # Overlap violations - smooth
        for i in range(n):
            for j in range(i + 1, n):
                t = (sides[i] + sides[j]) / 2 - 1e-7
                dx = abs(centers[i, 0] - centers[j, 0])
                dy = abs(centers[i, 1] - centers[j, 1])
                # Violation if both dx < t and dy < t
                vx = np.maximum(t - dx, 0)
                vy = np.maximum(t - dy, 0)
                # Penalty only when both are violated
                if vx > 0 and vy > 0:
                    penalty += (vx**2 + vy**2) * 100
        
        return penalty

    def _scipy_objective(x):
        """Combined objective: maximize sum of sides, penalize infeasibility."""
        centers = x[:2*n].reshape(n, 2)
        sides = x[2*n:]
        
        # Ensure sides are positive
        if np.any(sides <= 0.01):
            return 1e6
        
        sum_sides = np.sum(sides)
        pen = _penalty(x)
        
        if pen < 1e-6:
            return -sum_sides
        else:
            return -sum_sides + pen * 1000

    # Start with a better analytical construction with small gaps:
    # 2 large squares (side ~0.49) in top-left and bottom-right
    # 8 small squares (side ~0.245) filling the remaining space
    centers = np.zeros((n, 2), dtype=np.float64)
    sides = np.zeros(n, dtype=np.float64)
    
    # Square 0: top-left, side 0.49
    centers[0] = [0.245, 0.755]
    sides[0] = 0.49
    
    # Square 1: bottom-right, side 0.49
    centers[1] = [0.755, 0.245]
    sides[1] = 0.49
    
    # Squares 2-5: Top-right 2x2 grid (4 squares of side 0.245)
    for i in range(4):
        col = i % 2
        row = i // 2
        centers[2 + i] = [0.5 + (col + 0.5) * 0.245, 0.5 + (row + 0.5) * 0.245]
        sides[2 + i] = 0.245
    
    # Squares 6-9: Bottom-left 2x2 grid (4 squares of side 0.245)
    for i in range(4):
        col = i % 2
        row = i // 2
        centers[6 + i] = [(col + 0.5) * 0.245, (row + 0.5) * 0.245]
        sides[6 + i] = 0.245
    
    # Total: 2*0.49 + 8*0.245 = 0.98 + 1.96 = 2.94 (slightly less than 3.0 but feasible)
    
    best_centers = centers.copy()
    best_sides = sides.copy()
    best_obj = _objective(best_centers, best_sides)

    # Helper to convert between x vector and (centers, sides)
    def _to_x(centers, sides):
        x = np.empty(3 * n)
        x[:2*n] = centers.flatten()
        x[2*n:] = sides
        return x

    def _from_x(x):
        centers = x[:2*n].reshape(n, 2)
        sides = x[2*n:]
        return centers, sides

    # Local search: perturb centers and sides, accept if better
    rng = np.random.default_rng(42)
    step_size = 0.02
    max_iters = 80000
    
    current_x = _to_x(centers, sides)
    
    for it in range(max_iters):
        if time.time() > deadline - 90:  # Leave more time for scipy polish
            break
        
        # Random perturbation with adaptive step size
        x_p = current_x + rng.normal(0, step_size, 3 * n)
        centers_p, sides_p = _from_x(x_p)
        sides_p = np.maximum(sides_p, 0.01)  # keep sides positive
        
        # Simple repair: clip to boundaries
        for i in range(n):
            s = sides_p[i]
            centers_p[i, 0] = np.clip(centers_p[i, 0], s/2, 1 - s/2)
            centers_p[i, 1] = np.clip(centers_p[i, 1], s/2, 1 - s/2)
        
        x_p = _to_x(centers_p, sides_p)
        obj_p = _objective(centers_p, sides_p)
        if obj_p > best_obj:
            best_centers = centers_p.copy()
            best_sides = sides_p.copy()
            best_obj = obj_p
            current_x = x_p
            # Reduce step size on improvement
            step_size *= 0.98
        else:
            # Occasionally accept worse solutions to escape local optima
            if rng.random() < 0.1:
                current_x = x_p
                step_size = min(step_size * 1.02, 0.08)
        
        # Update current position occasionally
        if it % 100 == 0:
            current_x = _to_x(best_centers, best_sides)

    # Use scipy to polish from best solution found so far
    if time.time() < deadline - 60:
        x0 = _to_x(best_centers, best_sides)
        bounds = []
        for i in range(n):
            # cx: [0, 1], cy: [0, 1], s: [0.01, 1]
            bounds.append((0.0, 1.0))
            bounds.append((0.0, 1.0))
            bounds.append((0.01, 1.0))
        
        try:
            res = minimize(_scipy_objective, x0, method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': 3000, 'ftol': 1e-15, 'gtol': 1e-13})
            if res.success:
                c_res, s_res = _from_x(res.x)
                if _is_feasible(c_res, s_res):
                    obj_res = _sum_sides(s_res)
                    if obj_res > best_obj:
                        best_centers = c_res.copy()
                        best_sides = s_res.copy()
                        best_obj = obj_res
        except Exception:
            pass
        
        # Second scipy pass with different method if time allows
        if time.time() < deadline - 30:
            try:
                res2 = minimize(_scipy_objective, x0, method='Nelder-Mead', 
                              options={'maxiter': 8000, 'xatol': 1e-12, 'fatol': 1e-14})
                if res2.success:
                    c_res2, s_res2 = _from_x(res2.x)
                    if _is_feasible(c_res2, s_res2):
                        obj_res2 = _sum_sides(s_res2)
                        if obj_res2 > best_obj:
                            best_centers = c_res2.copy()
                            best_sides = s_res2.copy()
                            best_obj = obj_res2
            except Exception:
                pass

    # Final high-precision polish - much faster than before
    # Use coordinate descent with smaller steps
    for _ in range(200):
        if time.time() > deadline - 10:
            break
        improved = False
        for i in range(n):
            # Try small adjustments to each square's position and size
            for dx in [-0.0001, 0.0, 0.0001]:
                for dy in [-0.0001, 0.0, 0.0001]:
                    for ds in [-0.0001, 0.0, 0.0001]:
                        if dx == 0 and dy == 0 and ds == 0:
                            continue
                        c_new = best_centers.copy()
                        s_new = best_sides.copy()
                        c_new[i, 0] += dx
                        c_new[i, 1] += dy
                        s_new[i] = max(s_new[i] + ds, 0.01)
                        # Clip to boundaries
                        s = s_new[i]
                        c_new[i, 0] = np.clip(c_new[i, 0], s/2, 1 - s/2)
                        c_new[i, 1] = np.clip(c_new[i, 1], s/2, 1 - s/2)
                        
                        obj_new = _objective(c_new, s_new)
                        if obj_new > best_obj + 1e-12:
                            best_centers = c_new
                            best_sides = s_new
                            best_obj = obj_new
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break
    
    # One final scipy polish from the improved point
    if time.time() < deadline - 5:
        try:
            x_final = _to_x(best_centers, best_sides)
            res_final = minimize(_scipy_objective, x_final, method='L-BFGS-B', bounds=bounds,
                              options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-14})
            if res_final.success:
                c_final, s_final = _from_x(res_final.x)
                if _is_feasible(c_final, s_final):
                    obj_final = _sum_sides(s_final)
                    if obj_final > best_obj:
                        best_centers = c_final.copy()
                        best_sides = s_final.copy()
                        best_obj = obj_final
        except Exception:
            pass

    return best_centers, best_sides


# EVOLVE-BLOCK-END
