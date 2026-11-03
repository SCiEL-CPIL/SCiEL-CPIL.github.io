import numpy as np
from scipy.optimize import minimize
import itertools

N = 21

def construct():
    # We will try multiple initializations and aspect ratios
    best_value = -np.inf
    best_solution = None
    
    # Possible aspect ratios (width, height) such that w + h = 2
    # We'll optimize w as well, but providing good initial guesses helps.
    
    def get_initial_grid(w, h, n_rows, n_cols):
        """Generate initial positions on a grid."""
        xs = np.linspace(w / (2 * n_cols), w - w / (2 * n_cols), n_cols)
        ys = np.linspace(h / (2 * n_rows), h - h / (2 * n_rows), n_rows)
        centers = []
        for y in ys:
            for x in xs:
                centers.append((x, y))
                if len(centers) == N:
                    break
            if len(centers) == N:
                break
        # If grid is too small, fill remaining with random or repeat
        while len(centers) < N:
            centers.append((w/2, h/2))
        return centers[:N]

    def cost_and_grad(z):
        """
        z: [w, x1, y1, r1, ..., xN, yN, rN]
        Returns cost (negative sum of radii + penalties) and gradient.
        """
        w = z[0]
        h = 2.0 - w
        if w <= 0.1 or h <= 0.1:
            return 1e10, np.zeros_like(z)
        
        # Extract circle data
        # z[1:] is [x1, y1, r1, x2, y2, r2, ...]
        x = z[1::3]
        y = z[2::3]
        r = z[3::3]
        
        cost = -np.sum(r)
        grad = np.zeros_like(z)
        grad[0] = 0.0 # Gradient for w will be accumulated
        
        # Boundary penalties
        # x_i >= r_i, x_i <= w - r_i
        # y_i >= r_i, y_i <= h - r_i
        
        # We use a smooth penalty function: max(0, violation)^2
        # For gradient, we need derivatives.
        
        # Let's use a simpler penalty for gradient-free methods or careful analytic gradients.
        # Given the complexity, let's use Nelder-Mead or Powell which don't require gradients, 
        # or implement gradients carefully.
        # For robustness and speed in 120s, Nelder-Mead on 68 dims might be slow.
        # Let's use L-BFGS-B with analytic gradients.
        
        # Penalties:
        # 1. Boundary:
        # P_b = sum_i [ max(0, r_i - x_i)^2 + max(0, x_i - (w - r_i))^2 + max(0, r_i - y_i)^2 + max(0, y_i - (h - r_i))^2 ]
        # 2. Overlap:
        # P_o = sum_{i<j} max(0, r_i + r_j - dist_ij)^2
        
        # Total Cost = -sum(r_i) + Lambda_b * P_b + Lambda_o * P_o
        # We want the cost to be dominated by penalties if invalid, but we want to maximize sum(r).
        # A common trick is to minimize P_b + P_o - sum(r_i) * alpha.
        # If we set alpha=1, we are maximizing sum(r) subject to constraints.
        
        Lambda = 10.0 # Penalty weight
        
        # Boundary Violations
        # x < r -> viol = r - x
        # x > w-r -> viol = x - (w-r)
        
        # We will compute the gradient analytically.
        
        # --- Boundary Terms ---
        for i in range(N):
            xi = x[i]
            yi = y[i]
            ri = r[i]
            
            # Left: x_i >= r_i
            viol_l = ri - xi
            if viol_l > 0:
                cost += Lambda * viol_l**2
                # d/dxi: -2*Lambda*viol_l
                grad[1 + 3*i] += -2 * Lambda * viol_l
                # d/dri: 2*Lambda*viol_l
                grad[3 + 3*i] += 2 * Lambda * viol_l
                
            # Right: x_i <= w - r_i  =>  x_i + r_i <= w
            viol_r = xi + ri - w
            if viol_r > 0:
                cost += Lambda * viol_r**2
                # d/dxi: 2*Lambda*viol_r
                grad[1 + 3*i] += 2 * Lambda * viol_r
                # d/dri: 2*Lambda*viol_r
                grad[3 + 3*i] += 2 * Lambda * viol_r
                # d/dw: -2*Lambda*viol_r
                grad[0] += -2 * Lambda * viol_r
                
            # Bottom: y_i >= r_i
            viol_b = ri - yi
            if viol_b > 0:
                cost += Lambda * viol_b**2
                grad[2 + 3*i] += -2 * Lambda * viol_b
                grad[3 + 3*i] += 2 * Lambda * viol_b
                
            # Top: y_i <= h - r_i => y_i + r_i <= h = 2 - w
            viol_t = yi + ri - (2.0 - w)
            if viol_t > 0:
                cost += Lambda * viol_t**2
                grad[2 + 3*i] += 2 * Lambda * viol_t
                grad[3 + 3*i] += 2 * Lambda * viol_t
                # d/dw: derivative of (2-w) is -1, so d/dw (y+r-(2-w)) = +1
                # Term is (y+r-2+w)^2. d/dw is 2*(...)*1
                grad[0] += 2 * Lambda * viol_t
                
            # Objective: -sum(r_i)
            grad[3 + 3*i] += -1.0

        # --- Overlap Terms ---
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist_sq = dx*dx + dy*dy
                dist = np.sqrt(dist_sq + 1e-12) # avoid div by zero
                r_sum = r[i] + r[j]
                
                viol = r_sum - dist
                if viol > 0:
                    cost += Lambda * viol**2
                    
                    # Gradient with respect to x_i
                    # d/dxi (r_i+r_j - sqrt((xi-xj)^2 + (yi-yj)^2))^2
                    # = 2 * viol * ( - (xi-xj)/dist )
                    grad[1 + 3*i] += 2 * Lambda * viol * ( -dx / dist )
                    grad[2 + 3*i] += 2 * Lambda * viol * ( -dy / dist )
                    
                    # Gradient with respect to x_j
                    grad[1 + 3*j] += 2 * Lambda * viol * ( dx / dist )
                    grad[2 + 3*j] += 2 * Lambda * viol * ( dy / dist )
                    
                    # Gradient with respect to r_i
                    grad[3 + 3*i] += 2 * Lambda * viol
                    # Gradient with respect to r_j
                    grad[3 + 3*j] += 2 * Lambda * viol

        return cost, grad

    # Initializations
    inits = []
    
    # 1. Square 1x1, 5x5 grid
    w0 = 1.0
    centers = get_initial_grid(w0, 1.0, 5, 5)
    r0 = 0.1
    z0 = np.zeros(1 + 3*N)
    z0[0] = w0
    for i in range(N):
        z0[1 + 3*i] = centers[i][0]
        z0[2 + 3*i] = centers[i][1]
        z0[3 + 3*i] = r0
    inits.append(z0)
    
    # 2. Rectangle 0.8 x 1.2
    w0 = 0.8
    h0 = 1.2
    centers = get_initial_grid(w0, h0, 4, 6) # 24 slots, take 21
    z0 = np.zeros(1 + 3*N)
    z0[0] = w0
    for i in range(N):
        z0[1 + 3*i] = centers[i][0]
        z0[2 + 3*i] = centers[i][1]
        z0[3 + 3*i] = r0
    inits.append(z0)
    
    # 3. Rectangle 1.2 x 0.8
    w0 = 1.2
    h0 = 0.8
    centers = get_initial_grid(w0, h0, 6, 4)
    z0 = np.zeros(1 + 3*N)
    z0[0] = w0
    for i in range(N):
        z0[1 + 3*i] = centers[i][0]
        z0[2 + 3*i] = centers[i][1]
        z0[3 + 3*i] = r0
    inits.append(z0)
    
    # 4. Random jittered square
    w0 = 1.0
    np.random.seed(42)
    centers = [(np.random.uniform(0.2, 0.8), np.random.uniform(0.2, 0.8)) for _ in range(N)]
    z0 = np.zeros(1 + 3*N)
    z0[0] = w0
    for i in range(N):
        z0[1 + 3*i] = centers[i][0]
        z0[2 + 3*i] = centers[i][1]
        z0[3 + 3*i] = r0
    inits.append(z0)

    # Run optimization for each initialization
    for z_init in inits:
        try:
            res = minimize(cost_and_grad, z_init, method='L-BFGS-B', 
                           jac=True, options={'maxiter': 500, 'ftol': 1e-10})
            
            z_opt = res.x
            # Validate and extract
            w = z_opt[0]
            h = 2.0 - w
            if w <= 0.1 or h <= 0.1:
                continue
                
            x = z_opt[1::3]
            y = z_opt[2::3]
            r = z_opt[3::3]
            
            # Check validity: 
            # 1. All r > 0
            # 2. Inside box
            # 3. No overlaps
            
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
                val = np.sum(r)
                if val > best_value:
                    best_value = val
                    best_solution = np.column_stack((x, y, r))
        except Exception as e:
            pass

    if best_solution is None:
        # Fallback to seed solution
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return best_solution
