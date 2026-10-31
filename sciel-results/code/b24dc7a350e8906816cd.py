"""
Combined Strategy: Lattice-Initialized NLP Optimization for Circle Packing

This program combines the structural initialization of the current program 
(hexagonal and grid lattices) with the refinement machinery of the inspiration 
(penalty-based Non-Linear Programming optimization).

1. We generate multiple high-quality initial packings using hexagonal and grid 
   layouts for various aspect ratios.
2. Each packing is then refined using scipy.optimize.minimize. The optimization 
   variables include the x, y, r for each circle, and the width W of the rectangle.
   The height H is determined by H = 2 - W to satisfy the perimeter constraint.
3. The objective function minimizes -sum(r) + penalty(overlaps) + penalty(boundaries).
4. The best valid solution found across all initializations is returned.
"""
import numpy as np
import time
import random
from scipy.optimize import minimize

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    best_sum_r = 0.0
    best_config = None
    best_W = 1.0
    best_H = 1.0

    # --- Helper: Generate Hexagonal Layout ---
    def generate_hexagonal_layout(n_circles, aspect_ratio):
        """
        Generate a hexagonal packing layout for n_circles.
        Returns (xs, ys, r) normalized such that width of centers is 1.
        """
        if n_circles == 1:
            return np.array([0.5]), np.array([0.5]), np.array([0.5])
        
        best_layout = None
        best_score = -1
        
        max_rows = int(np.sqrt(n_circles) * 2) + 5
        for R in range(1, max_rows):
            base_c = n_circles // R
            rem = n_circles % R
            
            points = []
            r = 1.0 
            dx = 2.0 * r
            dy = np.sqrt(3) * r
            
            for row in range(R):
                offset = 0.0 if row % 2 == 0 else r
                current_c = base_c + (1 if row < rem else 0)
                
                for col in range(current_c):
                    x = offset + col * dx
                    y = row * dy
                    points.append((x, y))
                    
            if len(points) < n_circles:
                continue
            
            pts = np.array(points[:n_circles])
            
            w = pts[:, 0].max() - pts[:, 0].min()
            h = pts[:, 1].max() - pts[:, 1].min()
            
            if w == 0 or h == 0:
                continue
                
            pts[:, 0] = (pts[:, 0] - pts[:, 0].min()) / w
            pts[:, 1] = (pts[:, 1] - pts[:, 1].min()) / w
            
            # Calculate container aspect ratio for this layout
            # Centers span [0, 1] x [0, h_norm].
            # Circles span [-1, 2] x [-1, h_norm+1].
            # Width = 3, Height = h_norm + 2.
            container_aspect = 3.0 / (pts[:, 1].max() + 2.0)
            
            error = abs(container_aspect - aspect_ratio)
            
            if error < best_score:
                best_score = error
                best_layout = pts.copy()
                best_r_layout = 1.0

        if best_layout is None:
            return None, None, None
            
        return best_layout[:, 0], best_layout[:, 1], best_r_layout

    # --- Helper: Generate Grid Layout ---
    def generate_grid_layout(n_circles, aspect_ratio):
        best_layout = None
        best_score = -1
        
        max_cols = int(np.sqrt(n_circles) * 2) + 5
        for C in range(1, max_cols):
            R = int(np.ceil(n_circles / C))
            
            points = []
            r = 1.0
            dx = 2.0 * r
            dy = 2.0 * r
            
            for row in range(R):
                for col in range(C):
                    if len(points) >= n_circles:
                        break
                    x = col * dx
                    y = row * dy
                    points.append((x, y))
                if len(points) >= n_circles:
                    break
            
            if len(points) < n_circles:
                continue
                
            pts = np.array(points[:n_circles])
            
            w = pts[:, 0].max() - pts[:, 0].min()
            h = pts[:, 1].max() - pts[:, 1].min()
            
            if w == 0 or h == 0:
                continue
                
            pts[:, 0] = (pts[:, 0] - pts[:, 0].min()) / w
            pts[:, 1] = (pts[:, 1] - pts[:, 1].min()) / w
            
            container_aspect = 3.0 / (pts[:, 1].max() + 2.0)
            error = abs(container_aspect - aspect_ratio)
            
            if error < best_score:
                best_score = error
                best_layout = pts.copy()
                best_r_layout = 1.0

        if best_layout is None:
            return None, None, None
            
        return best_layout[:, 0], best_layout[:, 1], best_r_layout

    # --- NLP Objective and Constraints ---
    
    def objective(z):
        """
        Objective function for NLP.
        z = [x1, y1, r1, ..., xN, yN, rN, W]
        H = 2.0 - W
        Minimize -sum(r) + penalties.
        """
        x = z[0::3][:N]
        y = z[1::3][:N]
        r = z[2::3][:N]
        w = z[-1]
        h = 2.0 - w
        
        # Sanity checks
        if w < 0.2 or h < 0.2 or np.any(r < 0):
            return 1e6
            
        obj = -np.sum(r)
        
        # Penalty Weight
        penalty_weight = 500.0
        
        # Overlap Penalty
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d_sq = dx * dx + dy * dy
                dist_req = r[i] + r[j]
                if d_sq < dist_req * dist_req:
                    d = np.sqrt(d_sq)
                    overlap = dist_req - d
                    obj += penalty_weight * overlap * overlap
        
        # Boundary Penalty
        for i in range(N):
            # Left/Right
            if x[i] < r[i]:
                obj += penalty_weight * (r[i] - x[i]) ** 2
            elif x[i] > w - r[i]:
                obj += penalty_weight * (x[i] - (w - r[i])) ** 2
            # Bottom/Top
            if y[i] < r[i]:
                obj += penalty_weight * (r[i] - y[i]) ** 2
            elif y[i] > h - r[i]:
                obj += penalty_weight * (y[i] - (h - r[i])) ** 2
                
        return obj

    def check_validity(z, tol=1e-8):
        x = z[0::3][:N]
        y = z[1::3][:N]
        r = z[2::3][:N]
        w = z[-1]
        h = 2.0 - w
        
        if w <= 0.1 or h <= 0.1:
            return False, 0.0
            
        for i in range(N):
            if r[i] <= 0:
                return False, 0.0
            if x[i] < r[i] - tol or x[i] > w - r[i] + tol:
                return False, 0.0
            if y[i] < r[i] - tol or y[i] > h - r[i] + tol:
                return False, 0.0
        
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d_sq = dx * dx + dy * dy
                r_sum = r[i] + r[j]
                if d_sq < r_sum * r_sum - tol:
                    return False, 0.0
        
        val = float(np.sum(r))
        return True, val

    def force_relax(z, max_iter=200):
        """
        Simple force-directed relaxation to ensure validity before NLP.
        Helps NLP converge better.
        """
        z = z.copy().astype(float)
        n = N
        dt = 0.02
        
        for _ in range(max_iter):
            x = z[0::3][:n]
            y = z[1::3][:n]
            r = z[2::3][:n]
            w = z[-1]
            h = 2.0 - w
            
            if w < 0.1 or h < 0.1:
                break
                
            fx = np.zeros(n)
            fy = np.zeros(n)
            
            # Wall repulsion
            for i in range(n):
                if x[i] - r[i] < 0:
                    fx[i] += (x[i] - r[i]) * 100
                if x[i] + r[i] > w:
                    fx[i] -= (x[i] + r[i] - w) * 100
                if y[i] - r[i] < 0:
                    fy[i] += (y[i] - r[i]) * 100
                if y[i] + r[i] > h:
                    fy[i] -= (y[i] + r[i] - h) * 100
            
            # Pairwise repulsion
            for i in range(n):
                for j in range(i + 1, n):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    d_sq = dx * dx + dy * dy
                    min_d = r[i] + r[j]
                    if d_sq < min_d * min_d and d_sq > 1e-12:
                        d = np.sqrt(d_sq)
                        overlap = min_d - d
                        f = overlap * 10
                        fx[i] += (dx / d) * f
                        fy[i] += (dy / d) * f
                        fx[j] -= (dx / d) * f
                        fy[j] -= (dy / d) * f
            
            # Update
            z[0::3][:n] += dt * fx
            z[1::3][:n] += dt * fy
            
            # Clamp
            for i in range(n):
                z[3 * i] = np.clip(z[3 * i], r[i], w - r[i])
                z[3 * i + 1] = np.clip(z[3 * i + 1], r[i], h - r[i])
                
        return z

    def try_packing(z_init, time_budget):
        """
        Run NLP optimization starting from z_init.
        Returns best valid config found during optimization.
        """
        if time_budget < 5:
            return None, 0.0
            
        z = z_init.copy()
        
        # Define constraints for SLSQP
        # We use penalty in objective, but SLSQP benefits from explicit bounds.
        # Bounds: r > 0, W in [0.5, 1.5]
        bounds = []
        for i in range(N):
            bounds.append((0.0, 1.0)) # x
            bounds.append((0.0, 1.0)) # y
            bounds.append((0.01, 0.5)) # r (reasonable upper bound for 21 circles in P=4)
        bounds.append((0.5, 1.5)) # W
        
        # SLSQP is good for small N. 
        # We monitor the objective and validity.
        
        best_val = 0.0
        best_z = z.copy()
        
        # Check initial validity
        valid, val = check_validity(z)
        if valid:
            best_val = val
            best_z = z.copy()
            
        # Callback to track best valid solution
        def callback(zk):
            nonlocal best_val, best_z
            valid, val = check_validity(zk)
            if valid and val > best_val:
                best_val = val
                best_z = zk.copy()
            return False # Do not stop
                
        try:
            res = minimize(
                objective, 
                z, 
                method='SLSQP', 
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-12, 'disp': False},
                callback=callback
            )
            
            # Final check
            valid, val = check_validity(res.x)
            if valid and val > best_val:
                best_val = val
                best_z = res.x.copy()
                
        except Exception as e:
            pass
            
        # If we found a valid solution, try a second pass with COBYLA or just return
        # SLSQP should be sufficient for this size.
        
        if best_val > 0:
            # Format config
            config = np.zeros((N, 3))
            config[:, 0] = best_z[0::3][:N]
            config[:, 1] = best_z[1::3][:N]
            config[:, 2] = best_z[2::3][:N]
            W = best_z[-1]
            H = 2.0 - W
            return config, best_val
        else:
            return None, 0.0

    # --- Main Search ---
    np.random.seed(42)
    random.seed(42)
    
    # Generate candidate initializations
    # We use the lattice generators to create diverse starts
    candidates = []
    
    aspect_ratios = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4, 1.6]
    
    for ar in aspect_ratios:
        # Hex
        hx, hy, hr = generate_hexagonal_layout(N, ar)
        if hx is not None:
            # Scale layout to fit in a box roughly matching ar
            # The layout is normalized to width 1 (centers).
            # We need to place it in a W x H box.
            # Let's estimate W, H from the layout's natural aspect ratio.
            W_span = hx.max() - hx.min() + 2*hr
            H_span = hy.max() - hy.min() + 2*hr
            # Natural aspect ratio of the "blob"
            nat_ar = W_span / H_span
            
            # We want to map this to a specific W, H pair with W+H=2
            # Ideally W/H = nat_ar.
            H_est = 2.0 / (nat_ar + 1.0)
            W_est = 2.0 - H_est
            
            # Center and scale
            cx = hx.mean()
            cy = hy.mean()
            
            # Scale factor to fit in W_est, H_est with some margin
            # The centers span W_span * scale.
            # We want W_span * scale <= W_est - 2*hr*scale ? No, radius scales too.
            # Let's just use the rigid scaling logic from before to get a good start.
            s = 2.0 / (W_span + H_span)
            
            xs = s * (hx - cx) + W_est / 2.0
            ys = s * (hy - cy) + H_est / 2.0
            rs = s * hr * np.ones(N)
            
            z = np.zeros(3*N + 1)
            z[0::3][:N] = xs
            z[1::3][:N] = ys
            z[2::3][:N] = rs
            z[-1] = W_est
            
            # Relax slightly to ensure validity before NLP
            z_relaxed = force_relax(z, max_iter=50)
            candidates.append(z_relaxed)

        # Grid
        gx, gy, gr = generate_grid_layout(N, ar)
        if gx is not None:
            W_span = gx.max() - gx.min() + 2*gr
            H_span = gy.max() - gy.min() + 2*gr
            nat_ar = W_span / H_span
            
            H_est = 2.0 / (nat_ar + 1.0)
            W_est = 2.0 - H_est
            
            cx = gx.mean()
            cy = gy.mean()
            
            s = 2.0 / (W_span + H_span)
            
            xs = s * (gx - cx) + W_est / 2.0
            ys = s * (gy - cy) + H_est / 2.0
            rs = s * gr * np.ones(N)
            
            z = np.zeros(3*N + 1)
            z[0::3][:N] = xs
            z[1::3][:N] = ys
            z[2::3][:N] = rs
            z[-1] = W_est
            
            z_relaxed = force_relax(z, max_iter=50)
            candidates.append(z_relaxed)
            
        # Check time
        if time.time() > deadline - 30.0:
            break

    # Add a few random starts if time permits
    if time.time() < deadline - 30.0:
        for _ in range(5):
            w0 = np.random.uniform(0.6, 1.4)
            h0 = 2.0 - w0
            z = np.zeros(3*N + 1)
            for i in range(N):
                r0 = np.random.uniform(0.05, 0.1)
                cx = np.random.uniform(r0, w0 - r0)
                cy = np.random.uniform(r0, h0 - r0)
                z[3*i] = cx
                z[3*i+1] = cy
                z[3*i+2] = r0
            z[-1] = w0
            z_relaxed = force_relax(z, max_iter=50)
            candidates.append(z_relaxed)

    # Optimize each candidate
    for z_init in candidates:
        if time.time() > deadline - 15.0:
            break
            
        time_budget = deadline - time.time()
        config, val = try_packing(z_init, time_budget)
        
        if config is not None and val > best_sum_r:
            best_sum_r = val
            best_config = config
            # Extract W from config? No, W is in z_init or we can derive it.
            # Actually, try_packing returns config (x,y,r). We need W,H for validity check in evaluator?
            # The evaluator just takes the array. It checks if circles are in *some* rectangle of perimeter <= 4?
            # No, the task says: "n disjoint circles whose bounding rectangle has perimeter at most 4".
            # Wait, the evaluator measures "raw value".
            # The interface is: construct() -> array of shape (n, 3).
            # The evaluator likely computes the bounding box of the circles and checks its perimeter.
            # So we don't need to return W, H. The bounding box is implied by the circles.
            
            # However, we must ensure the circles *actually* fit in a rectangle of perimeter <= 4.
            # My NLP ensures they fit in W x H where W+H=2.
            # So the bounding box of the circles will be <= W x H, thus perimeter <= 4.
            # This is valid.
            
            # We don't store W, H separately, just the config.
            pass

    if best_config is not None:
        return best_config
    
    # Fallback
    side = int(np.ceil(np.sqrt(N)))
    W = 1.0
    H = 1.0
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) >= N:
                break
            x = (i + 0.5) * W / side
            y = (j + 0.5) * H / side
            r = 0.5 * min(W, H) / side - 1e-4
            cells.append([x, y, r])
    return np.array(cells[:N])

# EVOLVE-BLOCK-END
