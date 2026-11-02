import numpy as np
import time
from scipy.optimize import minimize

N = 32
TIME_LIMIT = 145.0  # Evaluator kills at 150s, keep 5s margin

def construct():
    start_time = time.time()
    
    def _elapsed():
        return time.time() - start_time

    def _is_valid(x, y, r, eps=1e-7):
        if np.any(x < -eps) or np.any(x > 1.0 + eps):
            return False
        if np.any(y < -eps) or np.any(y > 1.0 + eps):
            return False
        if np.any(r < -eps):
            return False
        if np.any(x < r - eps) or np.any(x > 1.0 - r + eps):
            return False
        if np.any(y < r - eps) or np.any(y > 1.0 - r + eps):
            return False
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.hypot(dx, dy)
                if d < (r[i] + r[j]) - eps:
                    return False
        return True

    def _sum_r(r):
        return float(np.sum(r))

    def _maximize_radii(x, y, r, max_iter=100, tol=1e-12):
        """
        Iteratively update radii to be the maximum possible given fixed centers.
        This is a relaxation method: r_i = min( dist_to_bound_i, min_j (dist_ij - r_j) )
        We iterate until convergence.
        """
        r = r.copy()
        for _ in range(max_iter):
            if _elapsed() > TIME_LIMIT - 2.0:
                break
            changed = False
            for i in range(N):
                # Distance to boundaries
                min_bound = min(x[i], 1.0 - x[i], y[i], 1.0 - y[i])
                
                # Distance to other circles
                min_dist_other = np.inf
                for j in range(N):
                    if i == j:
                        continue
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    d = np.hypot(dx, dy)
                    # The radius i can have is limited by d - r_j
                    # If d < r_j, it's an overlap we can't resolve by shrinking r_i alone 
                    # without moving centers, so we set it to a small value or 0?
                    # Actually, if centers are fixed and overlap, we must shrink both.
                    # This iterative scheme handles that by reducing r_i based on current r_j.
                    candidate = d - r[j]
                    if candidate < min_dist_other:
                        min_dist_other = candidate
                
                new_r = max(1e-9, min(min_bound, min_dist_other) - 1e-9)
                
                if abs(new_r - r[i]) > tol:
                    changed = True
                r[i] = new_r
            if not changed:
                break
        return r

    def _push_apart_pbd(x, y, r, iterations=50):
        """
        Position-Based Dynamics style push apart.
        Moves centers to resolve overlaps, keeping radii fixed.
        """
        x = x.copy()
        y = y.copy()
        for _ in range(iterations):
            if _elapsed() > TIME_LIMIT - 2.0:
                break
            moved = False
            for i in range(N):
                for j in range(i + 1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    d = np.hypot(dx, dy)
                    min_d = r[i] + r[j]
                    if d < min_d - 1e-10:
                        if d < 1e-10:
                            dx, dy = 1e-3, 0.0
                            d = np.hypot(dx, dy)
                        # Push apart proportional to overlap
                        overlap = min_d - d
                        nx, ny = dx/d, dy/d
                        # Weight by inverse mass (equal mass here)
                        push = overlap * 0.5 * 0.9 # Damping factor
                        x[i] += nx * push
                        y[i] += ny * push
                        x[j] -= nx * push
                        y[j] -= ny * push
                        moved = True
            # Clamp to bounds
            for i in range(N):
                x[i] = np.clip(x[i], r[i], 1.0 - r[i])
                y[i] = np.clip(y[i], r[i], 1.0 - y[i])
            if not moved:
                break
        return x, y

    def _objective(params, soft_eps=1e-4, mu=100.0):
        """
        Objective for L-BFGS-B.
        We optimize centers (x, y) primarily. Radii are treated as fixed 
        in the gradient calculation for speed, but we will re-maximize them 
        after the step.
        However, to guide the optimizer, we include a penalty for overlaps 
        based on the *current* radii.
        """
        x = params[:N]
        y = params[N:2*N]
        r = params[2*N:3*N] # Fixed in this sub-step
        
        obj = 0.0
        # We want to maximize sum(r), but r is fixed here. 
        # So we just minimize violations.
        # If we want to encourage larger r, we could add -sum(r) but r is fixed.
        # Actually, let's just minimize the penalty. The radius maximization 
        # is handled by the outer loop.
        
        pen = 0.0
        # Pairwise overlap
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx*dx + dy*dy)
                min_d = r[i] + r[j]
                if d < min_d:
                    diff = min_d - d
                    pen += diff * diff # Quadratic penalty
                    # Gradient contribution would be handled by minimize if we provided jac,
                    # but we rely on numerical diff which is slow.
                    # Let's use a smooth max for better numerical behavior
                    # smooth_max = 0.5 * (diff + np.sqrt(diff*diff + soft_eps*soft_eps))
                    # pen += smooth_max
        # Boundary
        for i in range(N):
            # Left
            val = r[i] - x[i]
            if val > 0: pen += val*val
            # Right
            val = x[i] - (1.0 - r[i])
            if val > 0: pen += val*val
            # Bottom
            val = r[i] - y[i]
            if val > 0: pen += val*val
            # Top
            val = y[i] - (1.0 - r[i])
            if val > 0: pen += val*val
            
        return mu * pen

    def _hex_init():
        """
        Hexagonal packing initialization tuned for N=32.
        """
        rows = [6, 5, 6, 5, 6, 4]
        x = []
        y = []
        r = []
        total_rows = len(rows)
        row_h = 1.0 / total_rows
        
        for row_idx, count in enumerate(rows):
            ry = (row_idx + 0.5) * row_h
            offset = 0.0
            if row_idx % 2 == 1:
                offset = 0.5 / count
            
            for c in range(count):
                cx = (c + 0.5 + offset) / count
                cr = 0.48 * row_h
                cr = min(cr, cx * 0.99, (1.0 - cx) * 0.99, ry * 0.99, (1.0 - ry) * 0.99)
                x.append(cx)
                y.append(ry)
                r.append(cr)
        
        x = x[:N]
        y = y[:N]
        r = r[:N]
        while len(x) < N:
            x.append(0.5)
            y.append(0.5)
            r.append(0.02)
            
        return np.array(x[:N]), np.array(y[:N]), np.array(r[:N])

    def _rect_init():
        """Rectangular grid initialization."""
        side = int(np.ceil(np.sqrt(N)))
        x = []
        y = []
        r = []
        for i in range(side):
            for j in range(side):
                if len(x) < N:
                    cx = (i + 0.5) / side
                    cy = (j + 0.5) / side
                    cr = 0.48 / side
                    cr = min(cr, cx * 0.99, (1.0 - cx) * 0.99, cy * 0.99, (1.0 - cy) * 0.99)
                    x.append(cx)
                    y.append(cy)
                    r.append(cr)
        return np.array(x[:N]), np.array(y[:N]), np.array(r[:N])

    def _random_init(seed):
        """Random initialization with fixed seed for reproducibility."""
        rng = np.random.RandomState(seed)
        x = rng.uniform(0.1, 0.9, N)
        y = rng.uniform(0.1, 0.9, N)
        r = np.full(N, 0.06)
        return x, y, r

    def _optimize_centers(x, y, r, time_budget=10):
        """
        Optimize centers x, y to minimize overlap penalty given fixed radii r.
        """
        x0 = np.concatenate([x, y])
        
        # Use a wrapper that only takes x, y
        def obj_xy(xy):
            x_curr = xy[:N]
            y_curr = xy[N:2*N]
            # Use the main objective but with fixed r
            params = np.concatenate([x_curr, y_curr, r])
            return _objective(params, soft_eps=1e-4, mu=50.0)
            
        try:
            res = minimize(obj_xy, x0, method='L-BFGS-B',
                           options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8,
                                    'maxfun': 5000})
            return res.x[:N], res.x[N:2*N]
        except Exception:
            return x, y

    def _local_search(x, y, r, max_iter=100):
        """
        Local search with random perturbations of centers.
        1. Perturb centers.
        2. Push apart to resolve immediate overlaps.
        3. Optimize centers with L-BFGS-B.
        4. Maximize radii.
        5. Accept if valid and better.
        """
        best_x = x.copy()
        best_y = y.copy()
        best_r = r.copy()
        best_score = _sum_r(best_r)
        rng = np.random.RandomState(12345)
        
        for i in range(max_iter):
            if _elapsed() > TIME_LIMIT - 15:
                break
            
            # Annealing scale
            scale = 0.05 * (1 - i / max_iter) + 0.01
            perturbed_x = best_x + rng.normal(0, scale, size=N)
            perturbed_y = best_y + rng.normal(0, scale, size=N)
            
            # Clamp to [0,1] roughly
            perturbed_x = np.clip(perturbed_x, 0.0, 1.0)
            perturbed_y = np.clip(perturbed_y, 0.0, 1.0)
            
            # Use current radii for push apart
            px, py = _push_apart_pbd(perturbed_x, perturbed_y, best_r, iterations=20)
            
            # Optimize centers
            ox, oy = _optimize_centers(px, py, best_r, time_budget=5)
            
            # Maximize radii for new centers
            new_r = _maximize_radii(ox, oy, best_r, max_iter=50, tol=1e-10)
            
            # Check validity
            if _is_valid(ox, oy, new_r, eps=1e-7):
                score = _sum_r(new_r)
                if score > best_score:
                    best_score = score
                    best_x = ox
                    best_y = oy
                    best_r = new_r
                    
        return best_x, best_y, best_r

    # --- Main Execution ---
    
    # 1. Generate Initial Guesses
    x_hex, y_hex, r_hex = _hex_init()
    x_rect, y_rect, r_rect = _rect_init()
    
    guesses = []
    guesses.append((x_hex, y_hex, r_hex))
    guesses.append((x_rect, y_rect, r_rect))
    
    for s in [1, 2, 3]:
        x_r, y_r, r_r = _random_init(s)
        x_r, y_r = _push_apart_pbd(x_r, y_r, r_r, iterations=50)
        guesses.append((x_r, y_r, r_r))

    best_x = None
    best_y = None
    best_r = None
    best_score = 0.0

    # 2. Phase 1: Coarse Optimization from each start
    for x0, y0, r0 in guesses:
        if _elapsed() > TIME_LIMIT - 40:
            break
            
        # First, maximize radii for the initial guess
        r_opt = _maximize_radii(x0, y0, r0, max_iter=50)
        
        # Optimize centers
        x_opt, y_opt = _optimize_centers(x0, y0, r_opt, time_budget=10)
        
        # Maximize radii again
        r_final = _maximize_radii(x_opt, y_opt, r_opt, max_iter=100)
        
        # Validate
        if _is_valid(x_opt, y_opt, r_final):
            score = _sum_r(r_final)
            if score > best_score:
                best_score = score
                best_x = x_opt
                best_y = y_opt
                best_r = r_final

    if best_x is None:
        # Fallback
        x_f, y_f, r_f = _hex_init()
        r_f = _maximize_radii(x_f, y_f, r_f)
        return np.column_stack((x_f, y_f, r_f))

    # 3. Phase 2: Local Search
    if _elapsed() < TIME_LIMIT - 20:
        best_x, best_y, best_r = _local_search(best_x, best_y, best_r, max_iter=150)
        
        # Re-validate and update score
        if _is_valid(best_x, best_y, best_r):
             score = _sum_r(best_r)
             if score > best_score:
                 best_score = score

    # 3b. Phase 2b: Additional local search if time permits
    if _elapsed() < TIME_LIMIT - 15:
        best_x, best_y, best_r = _local_search(best_x, best_y, best_r, max_iter=100)
        if _is_valid(best_x, best_y, best_r):
            score = _sum_r(best_r)
            if score > best_score:
                best_score = score

    # 4. Phase 3: High-precision Polish
    # Do a final round of center optimization with very tight tolerance
    # and then a very long radius maximization
    if _elapsed() < TIME_LIMIT - 5:
        # Slight perturbation to escape potential flat spots? No, just optimize.
        x_pol, y_pol = _optimize_centers(best_x, best_y, best_r, time_budget=10)
        r_pol = _maximize_radii(x_pol, y_pol, best_r, max_iter=200, tol=1e-14)
        
        if _is_valid(x_pol, y_pol, r_pol):
            score = _sum_r(r_pol)
            if score > best_score:
                best_score = score
                best_x = x_pol
                best_y = y_pol
                best_r = r_pol

    # 4b. Phase 4b: Final polish if time permits
    if _elapsed() < TIME_LIMIT - 2:
        # One more pass of radius maximization with extreme tightness
        r_final2 = _maximize_radii(best_x, best_y, best_r, max_iter=500, tol=1e-15)
        
        if _is_valid(best_x, best_y, r_final2):
            score = _sum_r(r_final2)
            if score > best_score:
                best_score = score
                best_r = r_final2

    # Final Output
    return np.column_stack((best_x, best_y, best_r))

if __name__ == "__main__":
    # Quick test
    t0 = time.time()
    res = construct()
    print(f"Time: {time.time()-t0:.2f}s")
    print(f"Sum of radii: {np.sum(res[:, 2]):.6f}")
    print(f"Valid: {_is_valid(res[:, 0], res[:, 1], res[:, 2])}")
