"""Circle packing in perimeter-4 rectangle: maximize sum of radii for 21 circles."""
import numpy as np
from scipy.optimize import minimize
import time

N = 21
DEADLINE = 145.0  # safety margin before 150s kill

def construct():
    start_time = time.time()
    
    def elapsed():
        return time.time() - start_time
    
    def time_left():
        return DEADLINE - elapsed()
    
    def objective(x, w):
        """Negative sum of radii (we minimize negative)."""
        h = 2.0 - w
        centers = x[:2*N].reshape(N, 2)
        radii = x[2*N:]
        # Ensure positive radii
        radii = np.maximum(radii, 1e-6)
        return -np.sum(radii)
    
    def constraints(x, w):
        """List of constraint dicts."""
        h = 2.0 - w
        centers = x[:2*N].reshape(N, 2)
        radii = x[2*N:]
        radii = np.maximum(radii, 1e-6)
        
        cons = []
        
        # Non-overlap constraints
        for i in range(N):
            for j in range(i+1, N):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist_sq = dx*dx + dy*dy
                min_dist = radii[i] + radii[j]
                # Constraint: dist_sq >= min_dist^2
                cons.append({'type': 'ineq', 'fun': lambda di, dj, ri, rj: di*di + dj*dj - (ri + rj)**2,
                           'args': (dx, dy, radii[i], radii[j])})
        
        # Bounds: circles inside rectangle
        for i in range(N):
            # x_i >= r_i, x_i <= w - r_i
            cons.append({'type': 'ineq', 'fun': lambda xi, ri: xi - ri, 'args': (centers[i,0], radii[i])})
            cons.append({'type': 'ineq', 'fun': lambda xi, ri, w_: w_ - xi - ri, 'args': (centers[i,0], radii[i], w)})
            # y_i >= r_i, y_i <= h - r_i
            cons.append({'type': 'ineq', 'fun': lambda yi, ri: yi - ri, 'args': (centers[i,1], radii[i])})
            cons.append({'type': 'ineq', 'fun': lambda yi, ri, h_: h_ - yi - ri, 'args': (centers[i,1], radii[i], h)})
        
        # Positive radii
        for i in range(N):
            cons.append({'type': 'ineq', 'fun': lambda ri: ri - 1e-6, 'args': (radii[i],)})
        
        return cons
    
    def bounds_for(w):
        h = 2.0 - w
        bnds = []
        for i in range(N):
            bnds.append((1e-6, w - 1e-6))  # x_i
            bnds.append((1e-6, h - 1e-6))  # y_i
        bnds.append((1e-6, min(w, h)/2.0))  # r_i (each circle must fit)
        return bnds
    
    def make_initial(w, h, pattern='grid'):
        """Create initial configuration."""
        centers = np.zeros((N, 2))
        radii = np.zeros(N)
        
        if pattern == 'grid':
            # Grid arrangement
            cols = int(np.ceil(np.sqrt(N * w / h)))
            rows = int(np.ceil(N / cols))
            cell_w = w / cols
            cell_h = h / rows
            for idx in range(N):
                r_i = idx // cols
                c_i = idx % cols
                centers[idx, 0] = (c_i + 0.5) * cell_w
                centers[idx, 1] = (r_i + 0.5) * cell_h
                radii[idx] = min(cell_w, cell_h) * 0.45
        elif pattern == 'hex':
            # Hexagonal packing
            r0 = 0.1
            dy = r0 * np.sqrt(3)
            dx = 2 * r0
            row = 0
            col = 0
            for idx in range(N):
                x = (col + 0.5) * dx + (row % 2) * dx * 0.5
                y = (row + 0.5) * dy
                if x > w - r0:
                    col = 0
                    row += 1
                    x = (col + 0.5) * dx + (row % 2) * dx * 0.5
                    y = (row + 0.5) * dy
                centers[idx, 0] = min(max(x, r0), w - r0)
                centers[idx, 1] = min(max(y, r0), h - r0)
                radii[idx] = r0 * 0.9
        elif pattern == 'random':
            np.random.seed(42)
            for idx in range(N):
                centers[idx, 0] = np.random.uniform(0.1, w - 0.1)
                centers[idx, 1] = np.random.uniform(0.1, h - 0.1)
                radii[idx] = 0.08
        
        x = np.concatenate([centers.flatten(), radii])
        return x
    
    def try_optimize(w, x0, maxiter=300):
        """Run SLSQP optimization."""
        h = 2.0 - w
        bnds = bounds_for(w)
        
        # Build constraints once (they depend on x, so we use a closure)
        def cons_func(x):
            h_ = 2.0 - w
            centers = x[:2*N].reshape(N, 2)
            radii = x[2*N:]
            radii = np.maximum(radii, 1e-6)
            
            # Non-overlap
            vals = []
            for i in range(N):
                for j in range(i+1, N):
                    dx = centers[i, 0] - centers[j, 0]
                    dy = centers[i, 1] - centers[j, 1]
                    vals.append(dx*dx + dy*dy - (radii[i] + radii[j])**2)
            
            # Boundary constraints
            for i in range(N):
                vals.append(centers[i,0] - radii[i])
                vals.append(w - centers[i,0] - radii[i])
                vals.append(centers[i,1] - radii[i])
                vals.append(h_ - centers[i,1] - radii[i])
            
            # Positive radii
            for i in range(N):
                vals.append(radii[i] - 1e-6)
            
            return vals
        
        res = minimize(
            lambda x: objective(x, w),
            x0,
            method='SLSQP',
            bounds=bnds,
            constraints=[{'type': 'ineq', 'fun': cons_func}],
            options={'maxiter': maxiter, 'ftol': 1e-8}
        )
        return res
    
    # Try different aspect ratios
    best_result = None
    best_value = -np.inf
    
    # Aspect ratios to try: w from 0.6 to 1.4 (since h = 2-w, this covers reasonable ranges)
    w_values = np.linspace(0.7, 1.3, 7)
    patterns = ['grid', 'hex', 'random']
    
    for w in w_values:
        if time_left() < 10.0:
            break
        
        for pattern in patterns:
            if time_left() < 5.0:
                break
            
            x0 = make_initial(w, 2.0 - w, pattern)
            res = try_optimize(w, x0, maxiter=200)
            
            if res.success or res.fun < 0:
                value = -res.fun
                if value > best_value:
                    best_value = value
                    best_result = res.x.copy()
                    best_w = w
    
    # If we have a good result, do a polish pass
    if best_result is not None and time_left() > 15.0:
        h = 2.0 - best_w
        bnds = bounds_for(best_w)
        
        def cons_func(x):
            h_ = 2.0 - best_w
            centers = x[:2*N].reshape(N, 2)
            radii = x[2*N:]
            radii = np.maximum(radii, 1e-6)
            
            vals = []
            for i in range(N):
                for j in range(i+1, N):
                    dx = centers[i, 0] - centers[j, 0]
                    dy = centers[i, 1] - centers[j, 1]
                    vals.append(dx*dx + dy*dy - (radii[i] + radii[j])**2)
            
            for i in range(N):
                vals.append(centers[i,0] - radii[i])
                vals.append(best_w - centers[i,0] - radii[i])
                vals.append(centers[i,1] - radii[i])
                vals.append(h_ - centers[i,1] - radii[i])
            
            for i in range(N):
                vals.append(radii[i] - 1e-6)
            
            return vals
        
        res = minimize(
            lambda x: objective(x, best_w),
            best_result,
            method='SLSQP',
            bounds=bnds,
            constraints=[{'type': 'ineq', 'fun': cons_func}],
            options={'maxiter': 500, 'ftol': 1e-10}
        )
        
        if res.fun < best_value * -1:  # better
            best_result = res.x.copy()
            best_value = -res.fun
    
    # Construct final output
    if best_result is not None:
        centers = best_result[:2*N].reshape(N, 2)
        radii = best_result[2*N:]
        radii = np.maximum(radii, 1e-6)
        
        # Verify validity and adjust if needed
        h = 2.0 - best_w
        for i in range(N):
            # Clamp to ensure inside rectangle
            centers[i, 0] = np.clip(centers[i, 0], radii[i], best_w - radii[i])
            centers[i, 1] = np.clip(centers[i, 1], radii[i], h - radii[i])
        
        result = np.column_stack([centers, radii])
        
        # Final check: ensure no overlaps by slightly reducing radii if needed
        for iter_ in range(5):
            max_overlap = 0
            for i in range(N):
                for j in range(i+1, N):
                    dx = centers[i,0] - centers[j,0]
                    dy = centers[i,1] - centers[j,1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    min_dist = radii[i] + radii[j]
                    if dist < min_dist:
                        overlap = min_dist - dist
                        max_overlap = max(max_overlap, overlap)
            
            if max_overlap < 1e-6:
                break
            
            # Reduce all radii slightly
            scale = 1.0 - max_overlap / (2.0 * np.max(radii)) * 0.5
            radii *= scale
            centers = best_result[:2*N].reshape(N, 2)
            result = np.column_stack([centers, radii])
        
        return result
    else:
        # Fallback: simple grid
        w = 1.0
        h = 1.0
        cols = 5
        rows = 5
        cell_w = w / cols
        cell_h = h / rows
        cells = []
        for idx in range(N):
            r_i = idx // cols
            c_i = idx % cols
            cx = (c_i + 0.5) * cell_w
            cy = (r_i + 0.5) * cell_h
            r = min(cell_w, cell_h) * 0.4
            cells.append([cx, cy, r])
        return np.array(cells)
