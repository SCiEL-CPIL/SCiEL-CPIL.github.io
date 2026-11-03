import numpy as np
from scipy.optimize import minimize
import itertools

N = 21

def construct():
    # We will try multiple initializations and aspect ratios
    best_value = -np.inf
    best_solution = None
    
    def get_hexagonal_packing(w, h, target_n):
        """Generate hexagonal packing positions."""
        centers = []
        # Hexagonal packing: rows offset
        row_spacing = 0.15
        col_spacing = 0.15
        max_x = w - row_spacing
        max_y = h - row_spacing
        
        row = 0
        while len(centers) < target_n:
            y = row_spacing + row * row_spacing * np.sqrt(3) / 2
            if y > max_y:
                break
            offset = (row % 2) * col_spacing / 2
            col = 0
            while len(centers) < target_n:
                x = row_spacing + offset + col * col_spacing
                if x > max_x:
                    break
                centers.append((x, y))
                col += 1
            row += 1
        
        # If not enough, add remaining randomly
        np.random.seed(123)
        while len(centers) < target_n:
            centers.append((np.random.uniform(row_spacing, max_x), 
                           np.random.uniform(row_spacing, max_y)))
        return centers[:target_n]
    
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
        while len(centers) < N:
            centers.append((w/2, h/2))
        return centers[:N]
    
    def cost_and_grad(z, Lambda=50.0):
        """
        z: [w, x1, y1, r1, ..., xN, yN, rN]
        Returns cost (negative sum of radii + penalties) and gradient.
        """
        w = z[0]
        h = 2.0 - w
        if w <= 0.05 or h <= 0.05:
            return 1e10, np.zeros_like(z)
        
        x = z[1::3]
        y = z[2::3]
        r = z[3::3]
        
        cost = -np.sum(r)
        grad = np.zeros_like(z)
        
        # Boundary penalties
        for i in range(N):
            xi = x[i]
            yi = y[i]
            ri = r[i]
            
            # Left: x_i >= r_i
            viol_l = ri - xi
            if viol_l > 0:
                cost += Lambda * viol_l**2
                grad[1 + 3*i] += -2 * Lambda * viol_l
                grad[3 + 3*i] += 2 * Lambda * viol_l
                
            # Right: x_i <= w - r_i
            viol_r = xi + ri - w
            if viol_r > 0:
                cost += Lambda * viol_r**2
                grad[1 + 3*i] += 2 * Lambda * viol_r
                grad[3 + 3*i] += 2 * Lambda * viol_r
                grad[0] += -2 * Lambda * viol_r
                
            # Bottom: y_i >= r_i
            viol_b = ri - yi
            if viol_b > 0:
                cost += Lambda * viol_b**2
                grad[2 + 3*i] += -2 * Lambda * viol_b
                grad[3 + 3*i] += 2 * Lambda * viol_b
                
            # Top: y_i <= h - r_i
            viol_t = yi + ri - (2.0 - w)
            if viol_t > 0:
                cost += Lambda * viol_t**2
                grad[2 + 3*i] += 2 * Lambda * viol_t
                grad[3 + 3*i] += 2 * Lambda * viol_t
                grad[0] += 2 * Lambda * viol_t
            
            grad[3 + 3*i] += -1.0

        # Overlap penalties
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
                    grad[1 + 3*i] += 2 * Lambda * viol * (-dx / dist)
                    grad[2 + 3*i] += 2 * Lambda * viol * (-dy / dist)
                    grad[1 + 3*j] += 2 * Lambda * viol * (dx / dist)
                    grad[2 + 3*j] += 2 * Lambda * viol * (dy / dist)
                    grad[3 + 3*i] += 2 * Lambda * viol
                    grad[3 + 3*j] += 2 * Lambda * viol

        return cost, grad
    
    def validate_solution(w, x, y, r, tol=1e-4):
        """Check if solution is valid."""
        h = 2.0 - w
        if w <= 0.05 or h <= 0.05:
            return False
        if np.any(r <= 0):
            return False
        for i in range(N):
            if x[i] < r[i] - tol or x[i] > w - r[i] + tol:
                return False
            if y[i] < r[i] - tol or y[i] > h - r[i] + tol:
                return False
        for i in range(N):
            for j in range(i+1, N):
                d = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                if d < r[i] + r[j] - tol:
                    return False
        return True
    
    # Generate diverse initializations
    inits = []
    
    # 1. Hexagonal packing in square
    w0 = 1.0
    centers = get_hexagonal_packing(w0, 1.0, N)
    r0 = 0.08
    z0 = np.zeros(1 + 3*N)
    z0[0] = w0
    for i in range(N):
        z0[1 + 3*i] = centers[i][0]
        z0[2 + 3*i] = centers[i][1]
        z0[3 + 3*i] = r0
    inits.append(z0)
    
    # 2. Hexagonal packing in wide rectangle
    w0 = 1.4
    h0 = 0.6
    centers = get_hexagonal_packing(w0, h0, N)
    z0 = np.zeros(1 + 3*N)
    z0[0] = w0
    for i in range(N):
        z0[1 + 3*i] = centers[i][0]
        z0[2 + 3*i] = centers[i][1]
        z0[3 + 3*i] = r0
    inits.append(z0)
    
    # 3. Hexagonal packing in tall rectangle
    w0 = 0.6
    h0 = 1.4
    centers = get_hexagonal_packing(w0, h0, N)
    z0 = np.zeros(1 + 3*N)
    z0[0] = w0
    for i in range(N):
        z0[1 + 3*i] = centers[i][0]
        z0[2 + 3*i] = centers[i][1]
        z0[3 + 3*i] = r0
    inits.append(z0)
    
    # 4. Grid in square
    w0 = 1.0
    centers = get_initial_grid(w0, 1.0, 5, 5)
    z0 = np.zeros(1 + 3*N)
    z0[0] = w0
    for i in range(N):
        z0[1 + 3*i] = centers[i][0]
        z0[2 + 3*i] = centers[i][1]
        z0[3 + 3*i] = r0
    inits.append(z0)
    
    # 5. Grid in wide rectangle
    w0 = 1.3
    h0 = 0.7
    centers = get_initial_grid(w0, h0, 4, 6)
    z0 = np.zeros(1 + 3*N)
    z0[0] = w0
    for i in range(N):
        z0[1 + 3*i] = centers[i][0]
        z0[2 + 3*i] = centers[i][1]
        z0[3 + 3*i] = r0
    inits.append(z0)
    
    # 6. Grid in tall rectangle
    w0 = 0.7
    h0 = 1.3
    centers = get_initial_grid(w0, h0, 6, 4)
    z0 = np.zeros(1 + 3*N)
    z0[0] = w0
    for i in range(N):
        z0[1 + 3*i] = centers[i][0]
        z0[2 + 3*i] = centers[i][1]
        z0[3 + 3*i] = r0
    inits.append(z0)
    
    # 7-12. Random initializations with different seeds
    for seed in [42, 123, 456, 789, 101, 202]:
        np.random.seed(seed)
        w0 = np.random.uniform(0.7, 1.3)
        h0 = 2.0 - w0
        centers = [(np.random.uniform(0.1, w0-0.1), 
                   np.random.uniform(0.1, h0-0.1)) for _ in range(N)]
        z0 = np.zeros(1 + 3*N)
        z0[0] = w0
        for i in range(N):
            z0[1 + 3*i] = centers[i][0]
            z0[2 + 3*i] = centers[i][1]
            z0[3 + 3*i] = 0.07
        inits.append(z0)
    
    # Run optimization for each initialization
    for z_init in inits:
        try:
            # First pass with moderate penalty
            res = minimize(cost_and_grad, z_init, method='L-BFGS-B', 
                           jac=True, options={'maxiter': 300, 'ftol': 1e-12})
            
            z_opt = res.x
            w = z_opt[0]
            h = 2.0 - w
            if w <= 0.05 or h <= 0.05:
                continue
                
            x = z_opt[1::3]
            y = z_opt[2::3]
            r = z_opt[3::3]
            
            # Second pass with higher penalty for refinement
            res2 = minimize(cost_and_grad, z_opt, method='L-BFGS-B', 
                           jac=True, options={'maxiter': 500, 'ftol': 1e-14},
                           args=(100.0,))
            z_opt2 = res2.x
            w2 = z_opt2[0]
            h2 = 2.0 - w2
            if w2 <= 0.05 or h2 <= 0.05:
                continue
                
            x2 = z_opt2[1::3]
            y2 = z_opt2[2::3]
            r2 = z_opt2[3::3]
            
            # Try both solutions
            for (wv, xv, yv, rv) in [(w, x, y, r), (w2, x2, y2, r2)]:
                if validate_solution(wv, xv, yv, rv):
                    val = np.sum(rv)
                    if val > best_value:
                        best_value = val
                        best_solution = np.column_stack((xv, yv, rv))
        except Exception as e:
            pass

    if best_solution is None:
        # Fallback: simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return best_solution
