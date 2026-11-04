import numpy as np
from scipy.optimize import minimize
from scipy.spatial import Delaunay

N = 26

def construct():
    # We will use a hybrid approach:
    # 1. Greedy initialization using Delaunay triangulation to place circles.
    # 2. Local optimization using L-BFGS-B to maximize sum of radii.
    # 3. Multi-start to escape local minima.
    
    def greedy_initialize(n, seed=None):
        if seed is not None:
            np.random.seed(seed)
        
        # Start with a few large circles in the corners/center to seed the Delaunay
        # This helps avoid degenerate triangulations initially.
        # A robust seed: 4 corners and 1 center? Or just a small grid.
        # Let's start with 4 circles in the corners.
        # Add some randomness to the initial positions to allow multi-start diversity
        centers = np.array([
            [0.1 + np.random.uniform(-0.02, 0.02), 0.1 + np.random.uniform(-0.02, 0.02)],
            [0.9 + np.random.uniform(-0.02, 0.02), 0.1 + np.random.uniform(-0.02, 0.02)],
            [0.1 + np.random.uniform(-0.02, 0.02), 0.9 + np.random.uniform(-0.02, 0.02)],
            [0.9 + np.random.uniform(-0.02, 0.02), 0.9 + np.random.uniform(-0.02, 0.02)]
        ])
        radii = np.array([0.1, 0.1, 0.1, 0.1])
        
        circles = []
        for i in range(len(centers)):
            circles.append((centers[i][0], centers[i][1], radii[i]))
            
        for _ in range(n - len(circles)):
            if len(circles) < 3:
                pass
                
            # Convert to numpy arrays for Delaunay
            pts = np.array([c[:2] for c in circles])
            rad = np.array([c[2] for c in circles])
            
            # Compute Delaunay triangulation
            try:
                tri = Delaunay(pts)
            except Exception:
                # Fallback: place a circle in the center if triangulation fails
                circles.append((0.5, 0.5, 0.1))
                continue

            best_circle = None
            best_radius = 0.0
            
            # Check circumcenters of all triangles
            for simplex in tri.simplices:
                p1 = pts[simplex[0]]
                p2 = pts[simplex[1]]
                p3 = pts[simplex[2]]
                
                # Circumcenter calculation
                ax, ay = p1
                bx, by = p2
                cx, cy = p3
                
                d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
                if abs(d) < 1e-10:
                    continue # Collinear or degenerate
                    
                ux = ((ax**2 + ay**2) * (by - cy) + (bx**2 + by**2) * (cy - ay) + (cx**2 + cy**2) * (ay - by)) / d
                uy = ((ax**2 + ay**2) * (cx - bx) + (bx**2 + by**2) * (ax - cx) + (cx**2 + cy**2) * (bx - ax)) / d
                
                # Distance to vertices (circumradius)
                r_circum = np.sqrt((ux - ax)**2 + (uy - ay)**2)
                
                min_allowed = r_circum
                for idx in simplex:
                    min_allowed = min(min_allowed, r_circum - rad[idx])
                    
                if min_allowed > 0:
                    # Check boundary constraints
                    r_bound = min(ux, 1 - ux, uy, 1 - uy)
                    r_new = min(min_allowed, r_bound)
                    
                    if r_new > best_radius:
                        best_radius = r_new
                        best_circle = (ux, uy, r_new)
                        
            if best_circle is None:
                # Fallback: place a small circle in the center of the square
                cx, cy = 0.5, 0.5
                r_bound = min(cx, 1-cx, cy, 1-cy)
                r_max = r_bound
                for i in range(len(circles)):
                    dx = cx - circles[i][0]
                    dy = cy - circles[i][1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    r_max = min(r_max, dist - circles[i][2])
                
                if r_max > 1e-6:
                    best_circle = (cx, cy, r_max)
                else:
                    # Try a random location
                    for _ in range(100):
                        cx = np.random.uniform(0.1, 0.9)
                        cy = np.random.uniform(0.1, 0.9)
                        r_bound = min(cx, 1-cx, cy, 1-cy)
                        r_max = r_bound
                        for i in range(len(circles)):
                            dx = cx - circles[i][0]
                            dy = cy - circles[i][1]
                            dist = np.sqrt(dx*dx + dy*dy)
                            r_max = min(r_max, dist - circles[i][2])
                        if r_max > best_radius:
                            best_radius = r_max
                            best_circle = (cx, cy, r_max)
            
            if best_circle is not None and best_radius > 1e-6:
                circles.append(best_circle)
            else:
                circles.append((0.5, 0.5, 0.01))

        return np.array(circles)

    def run_optimization(init_circles):
        x0 = init_circles[:, 0]
        y0 = init_circles[:, 1]
        r0 = init_circles[:, 2]
        
        def energy(params):
            x_ = params[0::3]
            y_ = params[1::3]
            r_ = params[2::3]
            
            # 1. Boundary violations
            pen_bound = 0.0
            for i in range(N):
                pen_bound += max(0, r_[i] - x_[i])**2
                pen_bound += max(0, r_[i] - (1 - x_[i]))**2
                pen_bound += max(0, r_[i] - y_[i])**2
                pen_bound += max(0, r_[i] - (1 - y_[i]))**2
                
            # 2. Overlap violations
            pen_overlap = 0.0
            for i in range(N):
                for j in range(i+1, N):
                    dist_sq = (x_[i] - x_[j])**2 + (y_[i] - y_[j])**2
                    min_dist = r_[i] + r_[j]
                    if dist_sq < min_dist**2:
                        overlap = min_dist - np.sqrt(dist_sq)
                        pen_overlap += overlap**2
                        
            lam = 1000.0
            return -np.sum(r_) + lam * (pen_bound + pen_overlap)
            
        params0 = np.zeros(3*N)
        params0[0::3] = x0
        params0[1::3] = y0
        params0[2::3] = r0
        
        bounds = []
        for i in range(N):
            bounds.append((0, 1))
            bounds.append((0, 1))
            bounds.append((0, 0.5))
            
        # Run optimization
        res = minimize(energy, params0, method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8})
        
        params_opt = res.x
        x_opt = params_opt[0::3]
        y_opt = params_opt[1::3]
        r_opt = params_opt[2::3]
        
        # Post-processing: Ensure strict validity
        for _ in range(20):
            changed = False
            for i in range(N):
                r_max_bound = min(x_opt[i], 1-x_opt[i], y_opt[i], 1-y_opt[i])
                if r_opt[i] > r_max_bound:
                    r_opt[i] = r_max_bound
                    changed = True
                    
            for i in range(N):
                for j in range(i+1, N):
                    dist = np.sqrt((x_opt[i]-x_opt[j])**2 + (y_opt[i]-y_opt[j])**2)
                    min_dist = r_opt[i] + r_opt[j]
                    if dist < min_dist:
                        overlap = min_dist - dist
                        total_r = r_opt[i] + r_opt[j]
                        if total_r > 0:
                            scale = dist / total_r
                            r_opt[i] *= scale
                            r_opt[j] *= scale
                            changed = True
            if not changed:
                break
                
        for i in range(N):
            r_opt[i] = min(r_opt[i], x_opt[i], 1-x_opt[i], y_opt[i], 1-y_opt[i])
            
        result = np.zeros((N, 3))
        result[:, 0] = x_opt
        result[:, 1] = y_opt
        result[:, 2] = r_opt
        
        return result, np.sum(r_opt)

    best_result = None
    best_score = -np.inf
    
    # Multi-start
    num_trials = 15
    for i in range(num_trials):
        init_circles = greedy_initialize(N, seed=i)
        result, score = run_optimization(init_circles)
        
        if score > best_score:
            best_score = score
            best_result = result
            
    return best_result
