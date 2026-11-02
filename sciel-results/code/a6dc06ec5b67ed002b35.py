"""Circle packing in unit square - optimized approach."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import itertools


def construct():
    """
    Pack N circles with disjoint interiors inside [0,1]^2 maximizing sum of radii.
    Uses a two-stage approach:
    1. Start from a reasonable initial configuration (grid-based with perturbation).
    2. Optimize positions and radii using gradient-based optimization with constraints.
    
    The objective is to maximize the sum of radii subject to:
    - Circles don't overlap: dist(i,j) >= r_i + r_j
    - Circles stay in unit square: r_i <= x_i <= 1-r_i, r_i <= y_i <= 1-r_i
    """
    n = N
    
    # Initial configuration: use a grid with some randomization to escape local optima
    # Try multiple initial configurations and keep the best
    
    best_result = None
    best_score = -np.inf
    
    n_attempts = 5
    
    for attempt in range(n_attempts):
        # Generate initial positions using a perturbed grid
        side = int(np.ceil(np.sqrt(n)))
        
        if attempt == 0:
            # Regular grid
            positions = []
            for i in range(side):
                for j in range(side):
                    if len(positions) < n:
                        x = (i + 0.5) / side
                        y = (j + 0.5) / side
                        positions.append([x, y])
            positions = np.array(positions[:n])
        else:
            # Perturbed grid
            rng = np.random.RandomState(attempt * 12345 + 67890)
            positions = []
            for i in range(side):
                for j in range(side):
                    if len(positions) < n:
                        x = (i + 0.5) / side + rng.uniform(-0.1, 0.1) / side
                        y = (j + 0.5) / side + rng.uniform(-0.1, 0.1) / side
                        positions.append([x, y])
            positions = np.array(positions[:n])
        
        # Initial radii: estimate based on grid spacing
        min_dist = np.min([
            np.linalg.norm(positions[i] - positions[j])
            for i in range(n) for j in range(i+1, n)
        ])
        base_radius = min_dist / 2 * 0.9  # 90% of half min distance
        
        # Also consider boundary
        for i in range(n):
            dist_to_boundary = min(positions[i, 0], 1 - positions[i, 0], 
                                    positions[i, 1], 1 - positions[i, 1])
            base_radius = min(base_radius, dist_to_boundary * 0.9)
        
        radii = np.full(n, base_radius)
        
        # Combined variables: [x_0, y_0, r_0, x_1, y_1, r_1, ..., x_{n-1}, y_{n-1}, r_{n-1}]
        x0 = np.zeros(3 * n)
        for i in range(n):
            x0[3*i] = positions[i, 0]
            x0[3*i+1] = positions[i, 1]
            x0[3*i+2] = radii[i]
        
        # Define objective: negative sum of radii (we minimize negative to maximize)
        def objective(x):
            r = x[2::3]
            return -np.sum(r)
        
        # Define constraints
        def constraints(x):
            cons = []
            xs = x[0::3]
            ys = x[1::3]
            rs = x[2::3]
            
            # Boundary constraints: r <= x <= 1-r, r <= y <= 1-r
            # x - r >= 0
            cons.append({'type': 'ineq', 'fun': lambda x, _xs=xs, _rs=rs: _xs - _rs, 'jac': None})
            # 1 - x - r >= 0
            cons.append({'type': 'ineq', 'fun': lambda x, _xs=xs, _rs=rs: 1 - _xs - _rs, 'jac': None})
            # y - r >= 0
            cons.append({'type': 'ineq', 'fun': lambda x, _ys=ys, _rs=rs: _ys - _rs, 'jac': None})
            # 1 - y - r >= 0
            cons.append({'type': 'ineq', 'fun': lambda x, _ys=ys, _rs=rs: 1 - _ys - _rs, 'jac': None})
            
            # Non-overlap constraints: dist(i,j) >= r_i + r_j
            # dist^2 - (r_i + r_j)^2 >= 0
            for i in range(n):
                for j in range(i+1, n):
                    dx = xs[i] - xs[j]
                    dy = ys[i] - ys[j]
                    cons.append({
                        'type': 'ineq',
                        'fun': lambda x, _i=i, _j=j, _xs=x[0::3], _ys=x[1::3], _rs=x[2::3]:
                            (_xs[_i] - _xs[_j])**2 + (_ys[_i] - _ys[_j])**2 - (_rs[_i] + _rs[_j])**2
                    })
            
            return cons
        
        # Use SLSQP with constraints
        # This might be slow, so let's try a simpler approach first
        
        # Alternative: iterative relaxation approach
        # Start with positions, iteratively adjust radii and positions
        
        # Let's use a different strategy: 
        # 1. Fix positions from a good initial guess
        # 2. Compute max possible radius for each circle given others
        # 3. Iterate
        
        # Better approach: use gradient ascent on radii with position adjustment
        
        # Let's try a direct optimization with fewer constraints for speed
        # Use penalty method instead
        
        def penalty_objective(x):
            xs = x[0::3]
            ys = x[1::3]
            rs = x[2::3]
            
            # Base objective: maximize sum of radii
            score = np.sum(rs)
            
            # Penalty for boundary violations
            penalty = 0
            penalty_weight = 100.0
            
            # x - r < 0
            viol = np.maximum(0, rs - xs)
            penalty += penalty_weight * np.sum(viol**2)
            # 1 - x - r < 0
            viol = np.maximum(0, rs + xs - 1)
            penalty += penalty_weight * np.sum(viol**2)
            # y - r < 0
            viol = np.maximum(0, rs - ys)
            penalty += penalty_weight * np.sum(viol**2)
            # 1 - y - r < 0
            viol = np.maximum(0, rs + ys - 1)
            penalty += penalty_weight * np.sum(viol**2)
            
            # Penalty for overlaps
            for i in range(n):
                for j in range(i+1, n):
                    dx = xs[i] - xs[j]
                    dy = ys[i] - ys[j]
                    dist = np.sqrt(dx*dx + dy*dy)
                    min_dist = rs[i] + rs[j]
                    if dist < min_dist:
                        viol = min_dist - dist
                        penalty += penalty_weight * viol**2
            
            return -score + penalty
        
        # Run optimization
        try:
            result = minimize(penalty_objective, x0, method='L-BFGS-B', 
                             options={'maxiter': 500, 'ftol': 1e-10})
            
            if result.success or result.fun < best_score:
                x_opt = result.x
                xs = x_opt[0::3]
                ys = x_opt[1::3]
                rs = x_opt[2::3]
                
                # Clamp to valid range
                for i in range(n):
                    rs[i] = max(0, min(rs[i], min(xs[i], 1-xs[i], ys[i], 1-ys[i])))
                
                # Check validity
                valid = True
                for i in range(n):
                    if xs[i] < rs[i] - 1e-6 or xs[i] > 1 - rs[i] + 1e-6:
                        valid = False
                        break
                    if ys[i] < rs[i] - 1e-6 or ys[i] > 1 - rs[i] + 1e-6:
                        valid = False
                        break
                if valid:
                    for i in range(n):
                        for j in range(i+1, n):
                            dx = xs[i] - xs[j]
                            dy = ys[i] - ys[j]
                            dist = np.sqrt(dx*dx + dy*dy)
                            if dist < rs[i] + rs[j] - 1e-6:
                                valid = False
                                break
                        if not valid:
                            break
                
                if valid:
                    score = np.sum(rs)
                    if score > best_score:
                        best_score = score
                        best_result = np.column_stack([xs, ys, rs])
                        
        except Exception:
            pass
    
    # Fallback to grid if optimization didn't work well
    if best_result is None:
        side = int(np.ceil(np.sqrt(n)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
                for i in range(side) for j in range(side)]
        best_result = np.array(cells[:n])
    
    return best_result

# EVOLVE-BLOCK-END
