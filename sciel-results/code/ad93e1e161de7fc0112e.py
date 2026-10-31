"""Circle packing in perimeter-4 rectangle using topological graph model with force-directed relaxation."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    best_value = 0.0
    best_result = None
    
    # Try multiple aspect ratios and random initializations
    aspect_ratios = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 2.0, 3.0]
    
    for attempt in range(80):
        if time.time() > deadline:
            break
        
        # Generate random initial positions and radii
        rng = np.random.default_rng(attempt * 42 + 7)
        
        # Choose aspect ratio
        ar = aspect_ratios[attempt % len(aspect_ratios)]
        
        # Rectangle with perimeter 4: 2*(w+h) = 4, so w+h = 2
        # If aspect ratio is w/h = ar, then w = ar*h, so h*(ar+1) = 2
        h = 2.0 / (ar + 1.0)
        w = ar * h
        
        # Start with a grid-like arrangement as initial guess
        side = 5  # 5x5 grid
        centers = []
        radii = []
        for i in range(side):
            for j in range(side):
                if len(centers) < N:
                    x = (i + 0.5) / side * w + rng.uniform(-0.05 * w, 0.05 * w)
                    y = (j + 0.5) / side * h + rng.uniform(-0.05 * h, 0.05 * h)
                    centers.append([x, y])
                    radii.append(0.1 * min(w, h))
        
        centers = np.array(centers[:N])
        radii = np.array(radii[:N])
        
        # Optimize using force-directed relaxation
        remaining_time = deadline - (time.time() - start_time)
        if remaining_time < 5:
            break
            
        result = _topological_optimize(centers, radii, w, h, remaining_time)
        if result is not None:
            value = np.sum(result[:, 2])
            if value > best_value:
                best_value = value
                best_result = result
        
        if time.time() > deadline:
            break
    
    if best_result is None:
        # Fallback to simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return best_result


def _topological_optimize(centers, radii, w_init, h_init, time_budget):
    """Topological graph model with force-directed relaxation."""
    start_time = time.time()
    
    n = len(centers)
    
    # Build initial contact graph based on proximity
    # Two circles are "connected" if they are close to each other
    contact_edges = _build_contact_graph(centers, radii, w_init, h_init)
    
    # Parameter vector: [w, h, x1, y1, r1, x2, y2, r2, ..., xn, yn, rn]
    def objective(params):
        w = params[0]
        h = params[1]
        if w <= 0 or h <= 0:
            return 1e6
        
        # Enforce perimeter constraint softly
        if w + h > 2.0:
            return 1e6 + (w + h - 2.0) * 100
        
        centers_opt = params[2:2+2*n].reshape(n, 2)
        radii_opt = params[2+2*n:2+3*n]
        
        # Penalty for circles outside rectangle
        penalty = 0.0
        for i in range(n):
            x, y = centers_opt[i]
            r = radii_opt[i]
            if x - r < 0:
                penalty += (x - r) * 100
            if x + r > w:
                penalty += (x + r - w) * 100
            if y - r < 0:
                penalty += (y - r) * 100
            if y + r > h:
                penalty += (y + r - h) * 100
        
        # Penalty for overlapping circles (stronger for non-contact pairs)
        for i in range(n):
            for j in range(i+1, n):
                dx = centers_opt[i, 0] - centers_opt[j, 0]
                dy = centers_opt[i, 1] - centers_opt[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = radii_opt[i] + radii_opt[j]
                if dist < min_dist:
                    # Stronger penalty for pairs not in contact graph
                    is_contact = (i, j) in contact_edges or (j, i) in contact_edges
                    weight = 20 if not is_contact else 50
                    penalty += (min_dist - dist) * weight
        
        # Topological constraint: for contact edges, encourage distance to be close to sum of radii
        # This helps maintain the rigid bar-joint framework structure
        for (i, j) in contact_edges:
            dx = centers_opt[i, 0] - centers_opt[j, 0]
            dy = centers_opt[i, 1] - centers_opt[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            target_dist = radii_opt[i] + radii_opt[j]
            # Soft constraint: distance should be close to target
            if dist > 0:
                penalty += (dist - target_dist) ** 2 * 0.1
        
        # Maximize sum of radii (negate for minimization)
        return -np.sum(radii_opt) + penalty
    
    # Initial parameters
    params_init = np.concatenate(([w_init, h_init], centers.flatten(), radii))
    
    # Bounds
    bounds = [(0.1, 1.9), (0.1, 1.9)]  # w, h
    for i in range(n):
        bounds.append((0, 2))  # x
        bounds.append((0, 2))  # y
        bounds.append((0.01, 0.5))  # r
    
    # Optimize with multiple methods and restarts
    best_obj = np.inf
    best_params = params_init.copy()
    
    time_per_opt = max(1.0, min(5.0, time_budget / 10))
    
    for method_idx, method in enumerate(['L-BFGS-B', 'SLSQP']):
        if time.time() - start_time > time_budget:
            break
        
        for restart in range(3):
            if time.time() - start_time > time_budget:
                break
            
            # Add small perturbation for restart
            if restart > 0:
                perturbation = np.random.default_rng(restart * 123 + 456).uniform(-0.01, 0.01, size=params_init.shape)
                perturbation[0] = 0  # Don't perturb w
                perturbation[1] = 0  # Don't perturb h
                params_current = params_init + perturbation
            else:
                params_current = params_init.copy()
            
            # Limit iterations based on time
            maxiter = int(max(100, min(2000, time_per_opt * 100)))
            
            try:
                res = minimize(objective, params_current, method=method, bounds=bounds,
                              options={'maxiter': maxiter, 'ftol': 1e-8, 'gtol': 1e-8})
                
                if res.fun < best_obj:
                    best_obj = res.fun
                    best_params = res.x.copy()
                    
            except Exception:
                pass
    
    if best_params is not None and best_obj < 1e4:
        w = best_params[0]
        h = best_params[1]
        if w + h <= 2.0 + 1e-6 and w > 0 and h > 0:
            centers_opt = best_params[2:2+2*n].reshape(n, 2)
            radii_opt = best_params[2+2*n:2+3*n]
            
            # Verify solution is valid
            valid = True
            for i in range(n):
                x, y = centers_opt[i]
                r = radii_opt[i]
                if x - r < -1e-6 or x + r > w + 1e-6 or y - r < -1e-6 or y + r > h + 1e-6:
                    valid = False
                    break
                if r < 0:
                    valid = False
                    break
            
            if valid:
                for i in range(n):
                    for j in range(i+1, n):
                        dx = centers_opt[i, 0] - centers_opt[j, 0]
                        dy = centers_opt[i, 1] - centers_opt[j, 1]
                        dist = np.sqrt(dx*dx + dy*dy)
                        if dist < radii_opt[i] + radii_opt[j] - 1e-6:
                            valid = False
                            break
                    if not valid:
                        break
            
            if valid:
                # Scale to ensure perimeter <= 4
                scale = min(1.0, 2.0 / (w + h))
                centers_final = centers_opt * scale
                radii_final = radii_opt * scale
                w_final = w * scale
                h_final = h * scale
                
                result = np.column_stack([centers_final, radii_final])
                return result
    
    return None


def _build_contact_graph(centers, radii, w, h):
    """Build a contact graph based on proximity of circles."""
    n = len(centers)
    contact_edges = set()
    
    # For each pair, check if they are close to each other
    # Use a threshold based on the average radius
    avg_r = np.mean(radii)
    threshold = 1.5 * avg_r
    
    for i in range(n):
        for j in range(i+1, n):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            
            # If they are close, add to contact graph
            if dist < threshold:
                contact_edges.add((i, j))
    
    return contact_edges
