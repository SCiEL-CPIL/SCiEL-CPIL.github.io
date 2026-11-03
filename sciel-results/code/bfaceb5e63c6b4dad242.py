import numpy as np
from scipy.optimize import minimize
import itertools
import time

N = 21

def construct():
    best_value = -np.inf
    best_solution = None
    
    # Define aspect ratios to try
    # w + h = 2
    # w from 0.5 to 1.5
    widths = np.linspace(0.6, 1.4, 9)
    
    # Grid resolution
    grid_res = 12
    
    # Local optimization function (same as before but reusable)
    def cost_and_grad(z, w_fixed=None):
        """
        z: [x1, y1, r1, ..., xN, yN, rN]
        w is fixed for this call to simplify gradients or included in z.
        Let's keep w in z for flexibility in polish phase, but fix it in discrete phase.
        """
        # If w is not in z, we need to pass it. 
        # Let's make a version that takes w as a parameter for the discrete polish.
        # But the main construct returns a full solution.
        # Let's stick to the previous structure for the polish step.
        pass

    def polish_solution(x, y, r, w):
        """
        Takes discrete solution and polishes it with L-BFGS-B.
        """
        h = 2.0 - w
        z0 = np.zeros(1 + 3*N)
        z0[0] = w
        for i in range(N):
            z0[1 + 3*i] = x[i]
            z0[2 + 3*i] = y[i]
            z0[3 + 3*i] = r[i]
        
        def cost_and_grad(z):
            w = z[0]
            h = 2.0 - w
            if w <= 0.1 or h <= 0.1:
                return 1e10, np.zeros_like(z)
            
            x = z[1::3]
            y = z[2::3]
            r = z[3::3]
            
            cost = -np.sum(r)
            grad = np.zeros_like(z)
            
            Lambda = 10.0
            
            for i in range(N):
                xi = x[i]
                yi = y[i]
                ri = r[i]
                
                viol_l = ri - xi
                if viol_l > 0:
                    cost += Lambda * viol_l**2
                    grad[1 + 3*i] += -2 * Lambda * viol_l
                    grad[3 + 3*i] += 2 * Lambda * viol_l
                
                viol_r = xi + ri - w
                if viol_r > 0:
                    cost += Lambda * viol_r**2
                    grad[1 + 3*i] += 2 * Lambda * viol_r
                    grad[3 + 3*i] += 2 * Lambda * viol_r
                    grad[0] += -2 * Lambda * viol_r
                
                viol_b = ri - yi
                if viol_b > 0:
                    cost += Lambda * viol_b**2
                    grad[2 + 3*i] += -2 * Lambda * viol_b
                    grad[3 + 3*i] += 2 * Lambda * viol_b
                
                viol_t = yi + ri - (2.0 - w)
                if viol_t > 0:
                    cost += Lambda * viol_t**2
                    grad[2 + 3*i] += 2 * Lambda * viol_t
                    grad[3 + 3*i] += 2 * Lambda * viol_t
                    grad[0] += 2 * Lambda * viol_t
                
                grad[3 + 3*i] += -1.0

            for i in range(N):
                for j in range(i + 1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dist_sq = dx*dx + dy*dy
                    dist = np.sqrt(dist_sq + 1e-12)
                    r_sum = r[i] + r[j]
                    
                    viol = r_sum - dist
                    if viol > 0:
                        cost += Lambda * viol**2
                        grad[1 + 3*i] += 2 * Lambda * viol * ( -dx / dist )
                        grad[2 + 3*i] += 2 * Lambda * viol * ( -dy / dist )
                        grad[1 + 3*j] += 2 * Lambda * viol * ( dx / dist )
                        grad[2 + 3*j] += 2 * Lambda * viol * ( dy / dist )
                        grad[3 + 3*i] += 2 * Lambda * viol
                        grad[3 + 3*j] += 2 * Lambda * viol

            return cost, grad

        try:
            res = minimize(cost_and_grad, z0, method='L-BFGS-B', jac=True, options={'maxiter': 200, 'ftol': 1e-8})
            z_opt = res.x
            w = z_opt[0]
            h = 2.0 - w
            if w <= 0.1 or h <= 0.1:
                return None, -np.inf
            
            x = z_opt[1::3]
            y = z_opt[2::3]
            r = z_opt[3::3]
            
            # Validate
            valid = True
            if np.any(r <= 0):
                valid = False
            
            if valid:
                for i in range(N):
                    if x[i] < r[i] - 1e-5 or x[i] > w - r[i] + 1e-5 or y[i] < r[i] - 1e-5 or y[i] > h - r[i] + 1e-5:
                        valid = False
                        break
            
            if valid:
                for i in range(N):
                    for j in range(i+1, N):
                        d = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                        if d < r[i] + r[j] - 1e-5:
                            valid = False
                            break
                    if not valid:
                        break
            
            if valid:
                return np.column_stack((x, y, r)), np.sum(r)
            else:
                return None, -np.inf
        except:
            return None, -np.inf

    # Discrete Grid Search
    for w in widths:
        h = 2.0 - w
        # Generate grid points
        # We want to place N circles.
        # Let's create a fine grid and pick the best N non-overlapping ones.
        
        # Grid step
        # If we have a 12x12 grid, we have 144 points.
        # We need to select 21.
        
        xs = np.linspace(w/24, w - w/24, 12)
        ys = np.linspace(h/24, h - h/24, 12)
        
        candidates = []
        for x in xs:
            for y in ys:
                # Max possible radius at this point
                # Limited by distance to walls and other points (approx)
                # For now, let's just use a base radius and adjust during polish.
                # Or, calculate max radius based on distance to nearest wall.
                r_max = min(x, w-x, y, h-y)
                # Also limited by grid spacing?
                # Let's just store the center and a tentative radius.
                # We will use a greedy selection based on potential radius.
                candidates.append((x, y, r_max))
        
        # Sort candidates by potential radius (descending)
        candidates.sort(key=lambda c: c[2], reverse=True)
        
        selected = []
        for c in candidates:
            if len(selected) == N:
                break
            x, y, r_tent = c
            # Check if this candidate overlaps with any selected
            conflict = False
            for s in selected:
                sx, sy, sr = s
                dist = np.sqrt((x-sx)**2 + (y-sy)**2)
                # We need to ensure they can fit.
                # If we pick a fixed radius for all, it's easier.
                # Let's assume a uniform radius for the discrete phase?
                # No, that's suboptimal.
                # Let's use a heuristic: if dist > r_tent + sr, it's ok?
                # This is tricky because r is variable.
                
                # Alternative: Pick centers first, then optimize radii.
                # If we pick 21 centers, we can run a local optimization to find optimal radii.
                pass
        
        # This "pick centers first" is a good strategy.
        # 1. Generate a grid of 21+ points.
        # 2. Select 21 points that are well-separated.
        # 3. Run L-BFGS-B to optimize radii and positions (starting from these centers).
        
        # Let's refine the selection of 21 centers.
        # Use a simple greedy: pick the point farthest from already picked points.
        # Start with a random point or corner.
        
        # Actually, a 5x5 grid in a square is a very good starting point.
        # The current code does this.
        
        # Let's try different grid shapes.
        # For w=1, h=1, 5x5 grid gives 25 points. Pick 21.
        # For w=0.8, h=1.2, maybe 4x6 grid?
        
        # Let's automate this:
        # Try different n_rows, n_cols such that n_rows * n_cols >= N
        for n_rows in range(3, 10):
            for n_cols in range(3, 10):
                if n_rows * n_cols < N:
                    continue
                if n_rows * n_cols > 30: # Too many, skip
                    continue
                
                # Generate grid
                xs = np.linspace(w / (2 * n_cols), w - w / (2 * n_cols), n_cols)
                ys = np.linspace(h / (2 * n_rows), h - h / (2 * n_rows), n_rows)
                
                centers = []
                for y in ys:
                    for x in xs:
                        centers.append((x, y))
                
                # We have more than N centers. We need to select N.
                # To maximize potential radii, we want them spread out.
                # A simple way is to take the first N (which is row-major).
                # Or, we can use a k-means like selection?
                # Let's just take the first N for now, or random subsets.
                
                # To be efficient, let's just take the first N.
                # This is similar to the current code's `get_initial_grid`.
                
                selected_centers = centers[:N]
                
                # Initial radii: small
                r_init = 0.05
                x_init = np.array([c[0] for c in selected_centers])
                y_init = np.array([c[1] for c in selected_centers])
                r_init_arr = np.full(N, r_init)
                
                # Polish
                sol, val = polish_solution(x_init, y_init, r_init_arr, w)
                if sol is not None:
                    if val > best_value:
                        best_value = val
                        best_solution = sol
    
    if best_solution is None:
        # Fallback
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return best_solution
