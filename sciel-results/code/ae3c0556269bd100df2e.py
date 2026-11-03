"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import linprog, minimize

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin for 150s limit

    # Perimeter 4 => W + H = 2
    # We optimize the aspect ratio w (width), so h = 2 - w.
    # Optimal aspect ratio for 21 circles is likely near 1.0 (square) or slightly rectangular.
    
    # Strategy:
    # 1. Initialize with a hexagonal lattice packing for a given aspect ratio.
    # 2. Use Successive Convex Relaxation (SCR) to optimize radii and positions.
    #    - Variables: centers (cx_i, cy_i) and radii (r_i).
    #    - Objective: Maximize sum(r_i).
    #    - Constraints:
    #       - 0 <= cx_i <= w, 0 <= cy_i <= h
    #       - r_i >= 0
    #       - ||c_i - c_j|| >= r_i + r_j (non-convex)
    #    - SCR Approach:
    #       - Linearize the non-convex constraints at the current point.
    #       - Solve a QP (or LP if linearized properly) to get a new point.
    #       - Iterate until convergence.
    #    
    #    Specifically, we can formulate a QP where we minimize the violation of constraints 
    #    or maximize the sum of radii subject to linearized non-overlap constraints.
    #    
    #    A more robust SCR for circle packing:
    #    Maximize sum(r_i)
    #    Subject to:
    #       c_i^T c_j - r_i^2 - r_j^2 - 2 r_i r_j >= 0 (This is non-convex)
    #       
    #    Alternative SCR formulation (common in literature):
    #    Fix the "active" constraints from the previous iteration.
    #    For each pair (i, j), if they were touching or close in the previous step, 
    #    enforce the linearized constraint:
    #    (c_i^k - c_j^k)^T (c_i - c_j) >= r_i + r_j + r_i^k + r_j^k - ||c_i^k - c_j^k||
    #    Wait, the standard linearization of ||c_i - c_j|| >= r_i + r_j is:
    #    Let d_i_j = c_i - c_j.
    #    Constraint: d_i_j^T d_i_j >= (r_i + r_j)^2.
    #    This is non-convex.
    #    
    #    However, we can use a simpler heuristic SCR:
    #    1. Start with a feasible packing (hexagonal).
    #    2. Identify "bottleneck" pairs (those with small gap).
    #    3. Solve a QP to maximize sum(r) while moving centers to increase gaps for bottleneck pairs.
    #    
    #    Let's implement a simpler, robust variant:
    #    We will use `scipy.optimize.minimize` with `trust-constr` or `SLSQP` on the non-convex problem directly, 
    #    but initialized well. The inspirations suggest SCR is better. 
    #    
    #    Let's implement a specific SCR loop:
    #    At iteration k, we have (c^k, r^k).
    #    We solve:
    #    Maximize sum(r_i)
    #    Subject to:
    #       c_i in [r_i, w-r_i] x [r_i, h-r_i]  (Linear if we fix r_i? No, r_i is variable. 
    #         Actually, c_i >= r_i is linear. c_i <= w - r_i is linear.)
    #       r_i >= 0
    #       For all pairs (i,j) where dist(c_i^k, c_j^k) < r_i^k + r_j^k + epsilon (active set):
    #           Linearized non-overlap:
    #           Let u = (c_i^k - c_j^k) / ||c_i^k - c_j^k||.
    #           Constraint: u^T (c_i - c_j) >= r_i + r_j.
    #           Note: This is a valid under-approximation of the true constraint if we are moving away?
    #           Actually, the tangent plane to the convex set { (c,r) | ||c_i-c_j|| >= r_i+r_j } is not straightforward because the set is non-convex.
    #           
    #    Let's stick to the "Penalty Method with Hexagonal Init" but optimized, 
    #    OR use the "QCP with Trust-Region" from inspiration 2, 
    #    BUT the directive asks for a STRUCTURAL change. 
    #    Inspiration 1 (SCR) scored 1.11. Let's try to implement a simplified SCR.
    #    
    #    Simplified SCR for Circle Packing:
    #    We maximize sum(r).
    #    We treat the non-overlap constraints as "soft" in a QP? No.
    #    
    #    Let's use a different structural change: 
    #    **Global Optimization via Multi-Start Local Search with Hexagonal Init.**
    #    The current program uses grid search. 
    #    A hexagonal lattice is a much better starting point.
    #    We will:
    #    1. Generate a hexagonal lattice packing for a range of aspect ratios.
    #    2. Use `scipy.optimize.minimize` (L-BFGS-B) to refine the packing, 
    #       maximizing sum(r) subject to constraints (using penalty or SLSQP).
    #    3. This is different from the grid search because it uses continuous local search from a smart start.
    
    def hexagonal_packing(w, h, n_circles):
        """
        Generate a hexagonal lattice packing of n_circles in w x h.
        Returns centers and radii.
        """
        # Estimate radius from area
        area = w * h
        # Hexagonal packing density ~ 0.9069
        # n * pi * r^2 * 0.9069 ~ area => r ~ sqrt(area / (n * pi * 0.9069))
        r_est = math.sqrt(area / (n_circles * math.pi * 0.9069))
        
        # Try to fit rows
        # Hexagonal lattice:
        # Row 0: y = r
        # Row 1: y = r + sqrt(3)*r, x offset by r
        # Spacing x: 2r
        
        # We will adjust r to fit.
        # Let's try a few radius scales.
        best_centers = None
        best_r = 0
        
        for scale in np.linspace(0.8, 1.2, 9):
            r = r_est * scale
            
            # Check if it fits roughly
            if 2*r > w or 2*r > h:
                continue
                
            # Generate points
            points = []
            y = r
            row = 0
            while y <= h - r:
                x_start = r
                if row % 2 == 1:
                    x_start = r + r # offset by r (half of 2r)
                
                x = x_start
                while x <= w - r:
                    points.append((x, y))
                    x += 2*r
                y += math.sqrt(3) * r
                row += 1
            
            # If we have more points than needed, we can trim or adjust.
            # If we have less, we need a smaller r.
            if len(points) >= n_circles:
                # Take the first n_circles? Or center them?
                # Better: Center the cluster.
                # For now, just take the first n_circles in a spiral order from center?
                # Simpler: Take a subset that is roughly centered.
                # Let's just take the first n_circles for now, but ensure they are valid.
                # Actually, if we have more, we can increase r.
                # If we have less, we decrease r.
                # This binary search on r is better.
                pass
        
        # Binary search for max r such that hexagonal lattice fits n_circles
        lo, hi = 0.0, min(w, h) / 2.0
        best_r = 0
        best_pts = None
        
        for _ in range(20):
            mid = (lo + hi) / 2.0
            r = mid
            if 2*r > w or 2*r > h:
                hi = mid
                continue
                
            points = []
            y = r
            row = 0
            while y <= h - r + 1e-9:
                x_start = r
                if row % 2 == 1:
                    x_start = r + r
                
                x = x_start
                while x <= w - r + 1e-9:
                    points.append((x, y))
                    x += 2*r
                y += math.sqrt(3) * r
                row += 1
            
            if len(points) >= n_circles:
                best_r = r
                # Select n_circles from points.
                # To be robust, select points closest to the center of the rectangle.
                cx_rect = w / 2.0
                cy_rect = h / 2.0
                points_with_dist = [(p[0], p[1], (p[0]-cx_rect)**2 + (p[1]-cy_rect)**2) for p in points]
                points_with_dist.sort(key=lambda x: x[2])
                best_pts = [(p[0], p[1]) for p in points_with_dist[:n_circles]]
                lo = mid
            else:
                hi = mid
        
        if best_pts is None:
            return None, 0
            
        return np.array(best_pts), best_r

    def refine_packing(w, h, centers, radii, time_limit=10):
        """
        Refine the packing using local search (L-BFGS-B with penalty).
        """
        start_time = time.time()
        
        x0 = np.concatenate([centers, radii])
        
        def objective(x):
            if time.time() > start_time + time_limit:
                return 1e10
            c = x[:2*N]
            r = x[2*N:]
            
            # Penalty for constraints
            penalty = 0.0
            
            # Bounds
            for i in range(N):
                cx, cy, ri = c[2*i], c[2*i+1], r[i]
                if cx < ri:
                    penalty += (ri - cx)**2
                if cx > w - ri:
                    penalty += (cx - (w - ri))**2
                if cy < ri:
                    penalty += (ri - cy)**2
                if cy > h - ri:
                    penalty += (cy - (h - ri))**2
                if ri < 0:
                    penalty += ri**2
            
            # Overlaps
            for i in range(N):
                for j in range(i+1, N):
                    dx = c[2*i] - c[2*j]
                    dy = c[2*i+1] - c[2*j+1]
                    dist = math.hypot(dx, dy)
                    min_dist = r[i] + r[j]
                    if dist < min_dist:
                        penalty += (min_dist - dist)**2
            
            # We want to maximize sum(r), so minimize -sum(r) + penalty
            return -np.sum(r) + penalty * 10.0
        
        # Bounds for L-BFGS-B
        # c_i in [0, w], r_i in [0, min(w,h)/2]
        bounds = []
        for i in range(N):
            bounds.append((0, w))
            bounds.append((0, h))
        for i in range(N):
            bounds.append((0, min(w, h)/2))
        
        try:
            res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, 
                           options={'maxiter': 200, 'ftol': 1e-9, 'gtol': 1e-6})
            
            if res.success or res.fun < 1e-5:
                c_opt = res.x[:2*N].reshape(N, 2)
                r_opt = res.x[2*N:]
                
                # Validate
                valid = True
                for i in range(N):
                    cx, cy, ri = c_opt[i,0], c_opt[i,1], r_opt[i]
                    if cx < ri - 1e-6 or cx > w - ri + 1e-6 or cy < ri - 1e-6 or cy > h - ri + 1e-6:
                        valid = False
                        break
                    if ri < 0:
                        valid = False
                        break
                
                if valid:
                    for i in range(N):
                        for j in range(i+1, N):
                            dist = math.hypot(c_opt[i,0]-c_opt[j,0], c_opt[i,1]-c_opt[j,1])
                            if dist < r_opt[i] + r_opt[j] - 1e-6:
                                valid = False
                                break
                        if not valid:
                            break
                
                if valid:
                    return c_opt, r_opt
        except Exception:
            pass
        
        return None, None

    best_score = -1
    best_result = None
    
    # Try a range of aspect ratios
    # Focus on near square
    aspect_ratios = [1.0, 0.95, 1.05, 0.9, 1.1, 0.85, 1.15]
    
    for w_ratio in aspect_ratios:
        if time.time() > deadline:
            break
            
        w = 2.0 * w_ratio / (1.0 + w_ratio)
        h = 2.0 - w
        
        if w <= 0 or h <= 0:
            continue
            
        # Get hexagonal init
        centers, r_init = hexagonal_packing(w, h, N)
        
        if centers is None:
            continue
            
        radii = np.full(N, r_init)
        
        # Refine
        c_opt, r_opt = refine_packing(w, h, centers, radii, time_limit=5)
        
        if c_opt is not None:
            score = np.sum(r_opt)
            if score > best_score:
                best_score = score
                best_result = np.column_stack([c_opt, r_opt])

    # Fallback if no refinement worked
    if best_result is None:
        # Simple grid fallback
        side = 5
        cells = []
        for i in range(side):
            for j in range(side):
                cx = (i + 0.5) / side
                cy = (j + 0.5) / side
                r = 0.5 / side - 1e-4
                cells.append((cx, cy, r))
        best_result = np.array(cells[:N])
        
    return best_result
# EVOLVE-BLOCK-END
