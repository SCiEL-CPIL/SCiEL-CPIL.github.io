import numpy as np
from scipy.optimize import minimize
from scipy.spatial import Delaunay

N = 26

def construct():
    def greedy_initialize(n, seed=0):
        np.random.seed(seed)
        # Start with 4 corners
        centers = np.array([
            [0.05, 0.05],
            [0.95, 0.05],
            [0.05, 0.95],
            [0.95, 0.95]
        ])
        radii = np.array([0.05, 0.05, 0.05, 0.05])
        
        circles = []
        for i in range(len(centers)):
            circles.append((centers[i][0], centers[i][1], radii[i]))
            
        for _ in range(n - len(circles)):
            if len(circles) < 3:
                circles.append((0.5, 0.5, 0.1))
                continue

            pts = np.array([c[:2] for c in circles])
            rad = np.array([c[2] for c in circles])
            
            try:
                tri = Delaunay(pts)
            except Exception:
                circles.append((0.5, 0.5, 0.01))
                continue

            best_circle = None
            best_radius = 0.0
            
            # Candidates: Circumcenters of triangles
            for simplex in tri.simplices:
                p1 = pts[simplex[0]]
                p2 = pts[simplex[1]]
                p3 = pts[simplex[2]]
                
                ax, ay = p1
                bx, by = p2
                cx, cy = p3
                
                d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
                if abs(d) < 1e-10:
                    continue
                    
                ux = ((ax**2 + ay**2) * (by - cy) + (bx**2 + by**2) * (cy - ay) + (cx**2 + cy**2) * (ay - by)) / d
                uy = ((ax**2 + ay**2) * (cx - bx) + (bx**2 + by**2) * (ax - cx) + (cx**2 + cy**2) * (bx - ax)) / d
                
                # Check if circumcenter is inside the square
                if ux < 0 or ux > 1 or uy < 0 or uy > 1:
                    continue

                r_circum = np.sqrt((ux - ax)**2 + (uy - ay)**2)
                
                min_allowed = r_circum
                for idx in simplex:
                    min_allowed = min(min_allowed, r_circum - rad[idx])
                    
                if min_allowed > 0:
                    r_bound = min(ux, 1 - ux, uy, 1 - uy)
                    r_new = min(min_allowed, r_bound)
                    
                    if r_new > best_radius:
                        best_radius = r_new
                        best_circle = (ux, uy, r_new)
                        
            # Also check edge midpoints and other points if no good triangle found
            if best_circle is None or best_radius < 0.01:
                # Sample random points
                for _ in range(50):
                    cx = np.random.uniform(0.05, 0.95)
                    cy = np.random.uniform(0.05, 0.95)
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

    def grid_initialize(n):
        # Simple grid initialization
        side = int(np.ceil(np.sqrt(n)))
        step = 1.0 / side
        centers = []
        for i in range(side):
            for j in range(side):
                if len(centers) < n:
                    cx = (i + 0.5) * step
                    cy = (j + 0.5) * step
                    centers.append([cx, cy])
        
        centers = np.array(centers)
        radii = np.full(n, step * 0.45)
        
        init = np.zeros((n, 3))
        init[:, 0] = centers[:, 0]
        init[:, 1] = centers[:, 1]
        init[:, 2] = radii
        return init

    def random_initialize(n, seed=0):
        np.random.seed(seed)
        centers = np.random.uniform(0.1, 0.9, size=(n, 2))
        radii = np.full(n, 0.05)
        init = np.zeros((n, 3))
        init[:, 0] = centers[:, 0]
        init[:, 1] = centers[:, 1]
        init[:, 2] = radii
        return init

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
                    
        lam = 500.0
        return -np.sum(r_) + lam * (pen_bound + pen_overlap)
        
    def run_optimization(init_circles):
        x0 = init_circles[:, 0]
        y0 = init_circles[:, 1]
        r0 = init_circles[:, 2]
        
        params0 = np.zeros(3*N)
        params0[0::3] = x0
        params0[1::3] = y0
        params0[2::3] = r0
        
        bounds = []
        for i in range(N):
            bounds.append((0, 1))
            bounds.append((0, 1))
            bounds.append((0, 0.5))
            
        try:
            res = minimize(energy, params0, method='L-BFGS-B', bounds=bounds, 
                           options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-6})
            params_opt = res.x
        except Exception:
            return init_circles, -np.inf

        x_opt = params_opt[0::3]
        y_opt = params_opt[1::3]
        r_opt = params_opt[2::3]
        
        # Post-processing: Ensure strict validity
        for _ in range(20):
            changed = False
            # Boundary fix
            for i in range(N):
                r_max_bound = min(x_opt[i], 1-x_opt[i], y_opt[i], 1-y_opt[i])
                if r_opt[i] > r_max_bound:
                    r_opt[i] = r_max_bound
                    changed = True
                    
            # Overlap fix
            for i in range(N):
                for j in range(i+1, N):
                    dist = np.sqrt((x_opt[i]-x_opt[j])**2 + (y_opt[i]-y_opt[j])**2)
                    min_dist = r_opt[i] + r_opt[j]
                    if dist < min_dist:
                        overlap = min_dist - dist
                        # Shrink both proportionally
                        total_r = r_opt[i] + r_opt[j]
                        if total_r > 1e-9:
                            scale = dist / total_r
                            r_opt[i] *= scale
                            r_opt[j] *= scale
                            changed = True
            if not changed:
                break
                
        # Final boundary clamp
        for i in range(N):
            r_opt[i] = min(r_opt[i], x_opt[i], 1-x_opt[i], y_opt[i], 1-y_opt[i])
            
        result = np.zeros((N, 3))
        result[:, 0] = x_opt
        result[:, 1] = y_opt
        result[:, 2] = r_opt
        
        # Validate
        valid = True
        for i in range(N):
            if result[i, 2] < 0 or result[i, 0] < 0 or result[i, 0] > 1 or result[i, 1] < 0 or result[i, 1] > 1:
                valid = False
                break
            if result[i, 2] > result[i, 0] + 1e-9 or result[i, 2] > 1 - result[i, 0] + 1e-9 or \
               result[i, 2] > result[i, 1] + 1e-9 or result[i, 2] > 1 - result[i, 1] + 1e-9:
                valid = False
                break
        if valid:
            for i in range(N):
                for j in range(i+1, N):
                    dist = np.sqrt((result[i,0]-result[j,0])**2 + (result[i,1]-result[j,1])**2)
                    if dist < result[i,2] + result[j,2] - 1e-9:
                        valid = False
                        break
                if not valid: break
                
        score = np.sum(result[:, 2]) if valid else -np.inf
        return result, score

    best_result = None
    best_score = -np.inf

    # Strategy 1: Greedy
    init1 = greedy_initialize(N, seed=42)
    res1, score1 = run_optimization(init1)
    if score1 > best_score:
        best_score = score1
        best_result = res1

    # Strategy 2: Grid
    init2 = grid_initialize(N)
    res2, score2 = run_optimization(init2)
    if score2 > best_score:
        best_score = score2
        best_result = res2

    # Strategy 3: Random
    init3 = random_initialize(N, seed=123)
    res3, score3 = run_optimization(init3)
    if score3 > best_score:
        best_score = score3
        best_result = res3
        
    # Strategy 4: Greedy with different seed
    init4 = greedy_initialize(N, seed=100)
    res4, score4 = run_optimization(init4)
    if score4 > best_score:
        best_score = score4
        best_result = res4

    if best_result is None:
        # Fallback
        best_result = grid_initialize(N)

    return best_result
