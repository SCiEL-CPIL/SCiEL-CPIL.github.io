"""
Circle packing in perimeter-4 rectangle: 21 circles.
Uses a direct coordinate/radius optimization approach with a smooth penalty-based objective.
This allows circles to move freely, enabling high-density packings (e.g., hexagonal structures)
that are unreachable by rigid grid-based methods.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    def check_validity(x, y, r, w, tol=1e-9):
        h = 2.0 - w
        if w <= 0.1 or h <= 0.1:
            return False, 0.0
        if np.any(r <= 0):
            return False, 0.0
        
        # Boundary checks
        if np.any(x < r - tol) or np.any(x > w - r + tol):
            return False, 0.0
        if np.any(y < r - tol) or np.any(y > h - r + tol):
            return False, 0.0
            
        # Overlap checks
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d2 = dx * dx + dy * dy
                r_sum = r[i] + r[j]
                if d2 < r_sum * r_sum - tol:
                    return False, 0.0
        return True, float(np.sum(r))

    def objective(z, penalty_weight=1000.0, beta=50.0):
        """
        Hybrid objective: 
        1. Maximizes sum of radii.
        2. Penalizes overlaps and boundary violations using a combination of 
           smooth log-sum-exp (global) and quadratic (local) terms.
        """
        x = z[0::3][:N]
        y = z[1::3][:N]
        r = z[2::3][:N]
        w = z[-1]
        h = 2.0 - w

        if w < 0.2 or h < 0.2 or np.any(r < 0):
            return 1e10

        obj = -np.sum(r)

        # Pairwise overlaps
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx * dx + dy * dy
        r_sum = r[:, None] + r[None, :]

        iu = np.triu_indices(N, k=1)
        d_sq_pairs = d_sq[iu]
        r_sum_pairs = r_sum[iu]
        d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))

        overlap_vals = r_sum_pairs - d_pairs
        
        # Smooth max penalty
        shifted = beta * overlap_vals
        shifted = np.clip(shifted, -500, 500)
        max_shifted = np.max(shifted)
        lse = max_shifted + np.log(np.sum(np.exp(shifted - max_shifted)))
        smooth_max_overlap = lse / beta
        
        # Hard overlap penalty
        hard_overlap = np.maximum(overlap_vals, 0.0)
        
        obj += penalty_weight * (0.5 * smooth_max_overlap + 0.5 * np.sum(hard_overlap * hard_overlap))

        # Boundary violations
        left_viol = np.maximum(r - x, 0.0)
        right_viol = np.maximum(x + r - w, 0.0)
        bottom_viol = np.maximum(r - y, 0.0)
        top_viol = np.maximum(y + r - h, 0.0)
        
        all_viol = np.concatenate([left_viol, right_viol, bottom_viol, top_viol])
        
        shifted_b = beta * all_viol
        shifted_b = np.clip(shifted_b, -500, 500)
        max_b = np.max(shifted_b)
        lse_b = max_b + np.log(np.sum(np.exp(shifted_b - max_b)))
        smooth_max_viol = lse_b / beta
        
        hard_viol = np.maximum(all_viol, 0.0)
        
        obj += penalty_weight * (0.5 * smooth_max_viol + 0.5 * np.sum(hard_viol * hard_viol))

        # Aspect ratio penalty to keep w in a reasonable range
        if w < 0.5:
            obj += penalty_weight * (0.5 - w) ** 2
        elif w > 1.5:
            obj += penalty_weight * (w - 1.5) ** 2

        return obj

    def run_optimization(z_init, time_budget):
        """
        Run SLSQP or L-BFGS-B optimization from a given initial guess.
        """
        # Bounds: x, y, r > 0. w in [0.5, 1.5]
        bounds = []
        for i in range(N):
            bounds.append((1e-6, None)) # x
            bounds.append((1e-6, None)) # y
            bounds.append((1e-6, None)) # r
        bounds.append((0.5, 1.5)) # w
        
        # We need tighter bounds for x, y based on w, but w is a variable.
        # SLSQP handles inequality constraints better, but is slower.
        # L-BFGS-B with penalty is faster. Let's use L-BFGS-B with a large penalty.
        
        try:
            res = minimize(
                objective,
                z_init,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 2000}
            )
            
            if res.success or res.fun < 0:
                x = res.x[0::3][:N]
                y = res.x[1::3][:N]
                r = res.x[2::3][:N]
                w = res.x[-1]
                
                # Refine with a local polish using SLSQP if time permits
                if time.time() < deadline - 2.0:
                    # SLSQP constraints
                    cons = []
                    # Boundary constraints: x >= r, x <= w-r, etc.
                    # It's easier to stick to the penalty method for speed, 
                    # but SLSQP can be more precise on the boundary.
                    # Let's just return the L-BFGS-B result for now to save time.
                    pass
                
                valid, val = check_validity(x, y, r, w)
                if valid:
                    return x, y, r, w, val
                else:
                    return None
        except Exception:
            return None

    best_value = 0.0
    best_result = None

    # Generate initial guesses
    def get_initial_guesses():
        guesses = []
        side = int(np.ceil(np.sqrt(N)))
        
        # 1. Grid-based guesses for various aspect ratios
        aspect_ratios = [1.0, 0.85, 1.18, 0.75, 1.33, 0.67, 1.5, 0.6, 1.2, 0.8]
        for w0 in aspect_ratios:
            h0 = 2.0 - w0
            if h0 < 0.3:
                continue
            z = np.zeros(3 * N + 1)
            for i in range(N):
                row, col = divmod(i, side)
                cx = (col + 0.5) / side * w0
                cy = (row + 0.5) / side * h0
                r0 = min(w0 / side, h0 / side) * 0.45
                z[3 * i] = cx
                z[3 * i + 1] = cy
                z[3 * i + 2] = r0
            z[-1] = w0
            guesses.append(z)

        # 2. Hexagonal-like guesses
        w0 = 1.0
        h0 = 1.0
        z = np.zeros(3 * N + 1)
        rows = 5
        cols = 5
        spacing_x = w0 / (cols + 0.5)
        spacing_y = h0 / (rows + 0.5)
        for i in range(N):
            row, col = divmod(i, cols)
            cx = (col + 0.5 + (0.5 if row % 2 == 1 else 0)) * spacing_x
            cy = (row + 0.5) * spacing_y
            r0 = min(spacing_x, spacing_y) * 0.48
            cx = min(max(cx, r0), w0 - r0)
            cy = min(max(cy, r0), h0 - r0)
            z[3 * i] = cx
            z[3 * i + 1] = cy
            z[3 * i + 2] = r0
        z[-1] = w0
        guesses.append(z)

        # 3. Random guesses
        rng = np.random.default_rng(42)
        for _ in range(10):
            w0 = rng.uniform(0.7, 1.3)
            h0 = 2.0 - w0
            z = np.zeros(3 * N + 1)
            for i in range(N):
                r0 = rng.uniform(0.05, 0.12)
                cx = rng.uniform(r0, w0 - r0)
                cy = rng.uniform(r0, h0 - r0)
                z[3 * i] = cx
                z[3 * i + 1] = cy
                z[3 * i + 2] = r0
            z[-1] = w0
            guesses.append(z)

        return guesses

    guesses = get_initial_guesses()
    
    # Sort guesses by initial objective value to try best ones first?
    # Or just iterate.
    for z_init in guesses:
        if time.time() > deadline - 5.0:
            break
        
        time_budget = deadline - time.time() - 2.0
        if time_budget < 0.5:
            continue
            
        result = run_optimization(z_init, time_budget)
        if result is not None:
            x, y, r, w, val = result
            if val > best_value:
                best_value = val
                best_result = (x, y, r, w)

    if best_result is not None:
        x, y, r, w = best_result
        # Shrink slightly to ensure strict inequality
        r_final = r * (1 - 1e-9)
        return np.column_stack((x, y, r_final))

    # Fallback: Simple grid in unit square
    cols, rows = 3, 7
    w = 1.0
    h = 1.0
    x = np.zeros(N)
    y = np.zeros(N)
    r = np.zeros(N)
    
    for i in range(N):
        row, col = divmod(i, cols)
        cx = (col + 0.5) * (w / cols)
        cy = (row + 0.5) * (h / rows)
        x[i] = cx
        y[i] = cy
        cell_w = w / cols
        cell_h = h / rows
        r[i] = min(cell_w, cell_h) * 0.5 # Inradius of uniform cell
        
    return np.column_stack((x, y, r * (1 - 1e-9)))

if __name__ == "__main__":
    res = construct()
    print(res.shape)
