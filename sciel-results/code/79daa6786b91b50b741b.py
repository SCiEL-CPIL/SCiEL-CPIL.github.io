import numpy as np
from scipy.optimize import minimize
import time

N = 21

def construct():
    start_time = time.time()
    deadline = 100.0  # Safety margin before 120s
    
    best_solution = None
    best_score = -np.inf
    
    # Helper to extract W, H, centers, radii from flat vector
    def unpack(x):
        W = x[0]
        H = x[1]
        # x[2:2+3*N] contains x_i, y_i, r_i for i in 0..N-1
        # Actually, let's structure it as:
        # x[0]=W, x[1]=H
        # x[2+3*i] = x_i
        # x[3+3*i] = y_i
        # x[4+3*i] = r_i
        centers = np.array([(x[2 + 3*i], x[3 + 3*i]) for i in range(N)])
        radii = np.array([x[4 + 3*i] for i in range(N)])
        return W, H, centers, radii

    def objective(x):
        # Maximize sum of radii -> Minimize negative sum
        W, H, centers, radii = unpack(x)
        return -np.sum(radii)

    def constraints(x):
        W, H, centers, radii = unpack(x)
        cons = []
        
        # Perimeter constraint: 2*(W+H) <= 4  => W+H <= 2
        cons.append({'type': 'ineq', 'fun': lambda x: 2.0 - (x[0] + x[1])})
        
        # Boundary constraints:
        # -W/2 <= x_i - r_i
        # x_i + r_i <= W/2
        # -H/2 <= y_i - r_i
        # y_i + r_i <= H/2
        
        for i in range(N):
            xi = x[2 + 3*i]
            yi = x[3 + 3*i]
            ri = x[4 + 3*i]
            
            # Left: xi - ri >= -W/2  => xi - ri + W/2 >= 0
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[2+3*i] - x[4+3*i] + x[0]/2.0})
            # Right: W/2 - (xi + ri) >= 0
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[0]/2.0 - (x[2+3*i] + x[4+3*i])})
            # Bottom: yi - ri >= -H/2 => yi - ri + H/2 >= 0
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[3+3*i] - x[4+3*i] + x[1]/2.0})
            # Top: H/2 - (yi + ri) >= 0
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[1]/2.0 - (x[3+3*i] + x[4+3*i])})
            
        # Overlap constraints:
        # (xi - xj)^2 + (yi - yj)^2 >= (ri + rj)^2
        # This is non-linear. SLSQP handles it.
        for i in range(N):
            for j in range(i + 1, N):
                xi = x[2 + 3*i]
                yi = x[3 + 3*i]
                ri = x[4 + 3*i]
                
                xj = x[2 + 3*j]
                yj = x[3 + 3*j]
                rj = x[4 + 3*j]
                
                # dist_sq - (ri+rj)^2 >= 0
                cons.append({'type': 'ineq', 'fun': lambda x, i=i, j=j: 
                             (x[2+3*i] - x[2+3*j])**2 + (x[3+3*i] - x[3+3*j])**2 - (x[4+3*i] + x[4+3*j])**2})
                             
        return cons

    def bounds(x):
        # W, H
        b = [(0.1, 1.9), (0.1, 1.9)]
        for i in range(N):
            b.append((-1.0, 1.0)) # xi
            b.append((-1.0, 1.0)) # yi
            b.append((0.001, 0.5)) # ri
        return b

    # Initial guesses: Try a few aspect ratios
    # 1. Square 1x1
    # 2. Rectangle 1.2 x 0.8
    # 3. Rectangle 0.8 x 1.2
    
    aspect_ratios = [
        (1.0, 1.0),
        (1.2, 0.8),
        (0.8, 1.2),
        (1.4, 0.6),
        (0.6, 1.4)
    ]
    
    # Create initial grid layouts for each aspect ratio
    for W_init, H_init in aspect_ratios:
        if time.time() - start_time > deadline:
            break
            
        # Create a grid of points
        # Determine grid dimensions roughly proportional to W/H
        # For 21 circles, maybe 5 cols, 5 rows? Or 4x6?
        # Let's use a 5x5 grid and pick 21 points, or just 4x6 and pick 21.
        # 4x6 = 24 points.
        cols = 4
        rows = 6
        
        # Scale grid to fit inside W, H with some margin
        margin = 0.1
        w_usable = W_init - 2*margin
        h_usable = H_init - 2*margin
        
        # If aspect ratio is very different, adjust cols/rows
        if W_init > H_init:
             cols = 5
             rows = 5 # 25 slots, use 21
        else:
             cols = 4
             rows = 6 # 24 slots, use 21

        # Generate grid points
        x_grid = np.linspace(-w_usable/2, w_usable/2, cols)
        y_grid = np.linspace(-h_usable/2, h_usable/2, rows)
        
        points = []
        for xi in x_grid:
            for yi in y_grid:
                points.append((xi, yi))
                
        # Take first N points
        points = points[:N]
        
        # Initial radii: small, to ensure no overlap initially
        # Estimate max radius based on grid spacing
        dx = w_usable / (cols - 1) if cols > 1 else w_usable
        dy = h_usable / (rows - 1) if rows > 1 else h_usable
        r_init = 0.4 * min(dx, dy)
        
        # Construct initial vector
        x0 = [W_init, H_init]
        for i in range(N):
            if i < len(points):
                x0.append(points[i][0])
                x0.append(points[i][1])
            else:
                x0.append(0.0)
                x0.append(0.0)
            x0.append(r_init)
            
        # Add small random noise to break symmetry
        np.random.seed(42) # Reproducibility for the same start, but different starts
        x0[2:] += np.random.uniform(-0.01, 0.01, size=len(x0)-2)
        
        # Optimize
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds(x0), constraints=constraints(x0), options={'maxiter': 1000, 'ftol': 1e-8})
            if res.success:
                score = -res.fun
                if score > best_score:
                    best_score = score
                    best_solution = res.x
        except Exception:
            pass

    # If optimization failed or didn't find a valid solution, fallback to seed
    if best_solution is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    # Extract best solution
    W, H, centers, radii = unpack(best_solution)
    
    # The interface expects (n, 3) array: x, y, r
    # The coordinates in our optimization are centered at (0,0).
    # The seed program used (0,1) or similar? 
    # "construct() -> array of shape (n, 3): n disjoint circles whose bounding rectangle has perimeter at most 4"
    # The evaluator likely checks the bounding box of the returned points.
    # If we return centered coordinates, the bounding box will be from -W/2 to W/2.
    # This is a valid rectangle of width W and height H.
    # We just need to return x, y, r.
    
    result = np.zeros((N, 3))
    for i in range(N):
        result[i, 0] = centers[i, 0]
        result[i, 1] = centers[i, 1]
        result[i, 2] = radii[i]
        
    return result

