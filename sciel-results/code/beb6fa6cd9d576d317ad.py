import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import Delaunay, Voronoi

N = 26

def construct():
    
    def largest_empty_circle(centers, radii):
        """
        Finds the center and radius of the largest circle that can be added
        to the given set of circles without overlapping, within [0,1]x[0,1].
        """
        if len(centers) == 0:
            return (0.5, 0.5, 0.5)
        
        centers = np.array(centers)
        radii = np.array(radii)
        
        best_r = 0
        best_c = None
        
        # 1. Candidate centers from Voronoi vertices
        if len(centers) >= 3:
            try:
                vor = Voronoi(centers)
                
                # Voronoi vertices
                for v in vor.vertices:
                    if np.any(np.isnan(v)):
                        continue
                    if v[0] < 0 or v[0] > 1 or v[1] < 0 or v[1] > 1:
                        continue
                    
                    r = min(v[0], 1-v[0], v[1], 1-v[1])
                    for i in range(len(centers)):
                        d = np.sqrt((v[0]-centers[i][0])**2 + (v[1]-centers[i][1])**2)
                        r = min(r, d - radii[i])
                    
                    if r > best_r:
                        best_r = r
                        best_c = v
                        
            except Exception:
                pass
        
        # 2. Candidate centers on the boundary of the square
        num_samples = 500
        for side in range(4):
            for t in np.linspace(0, 1, num_samples):
                if side == 0: # y=0
                    x, y = t, 0.0
                elif side == 1: # y=1
                    x, y = t, 1.0
                elif side == 2: # x=0
                    x, y = 0.0, t
                else: # x=1
                    x, y = 1.0, t
                
                r = min(x, 1-x, y, 1-y)
                for i in range(len(centers)):
                    d = np.sqrt((x-centers[i][0])**2 + (y-centers[i][1])**2)
                    r = min(r, d - radii[i])
                
                if r > best_r:
                    best_r = r
                    best_c = np.array([x, y])
        
        # 3. Circle-Circle intersection points
        n_centers = len(centers)
        for i in range(n_centers):
            for j in range(i+1, n_centers):
                c1 = centers[i]
                c2 = centers[j]
                r1 = radii[i]
                r2 = radii[j]
                
                d = np.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)
                
                if d < 1e-6 or d > r1 + r2 or d < abs(r1 - r2):
                    continue
                
                a = (d**2 + r1**2 - r2**2) / (2*d)
                h = np.sqrt(max(0, r1**2 - a**2))
                
                mid_x = c1[0] + a * (c2[0] - c1[0]) / d
                mid_y = c1[1] + a * (c2[1] - c1[1]) / d
                
                off_x = h * (c2[1] - c1[1]) / d
                off_y = h * (c1[0] - c2[0]) / d
                
                p1 = np.array([mid_x + off_x, mid_y + off_y])
                p2 = np.array([mid_x - off_x, mid_y - off_y])
                
                for p in [p1, p2]:
                    if p[0] < 0 or p[0] > 1 or p[1] < 0 or p[1] > 1:
                        continue
                    
                    r = min(p[0], 1-p[0], p[1], 1-p[1])
                    for k in range(n_centers):
                        d_k = np.sqrt((p[0]-centers[k][0])**2 + (p[1]-centers[k][1])**2)
                        r = min(r, d_k - radii[k])
                    
                    if r > best_r:
                        best_r = r
                        best_c = p
        
        if best_c is None:
            best_c = np.array([0.5, 0.5])
            best_r = 0.01
            
        return best_c[0], best_c[1], max(best_r, 1e-6)

    def greedy_initialize(n):
        centers = []
        radii = []
        
        for i in range(4):
            x = 0.1 if i % 2 == 0 else 0.9
            y = 0.1 if i < 2 else 0.9
            centers.append([x, y])
            radii.append(0.1)
            
        for _ in range(n - 4):
            c_x, c_y, r = largest_empty_circle(centers, radii)
            centers.append([c_x, c_y])
            radii.append(r)
            
        return np.array(centers), np.array(radii)

    def relax_positions(centers, iterations=50):
        """
        Pushes overlapping points apart to maximize the minimum distance.
        This helps create a better initial spread for the optimizer.
        """
        centers = np.array(centers, dtype=float)
        n = len(centers)
        
        for _ in range(iterations):
            for i in range(n):
                for j in range(i+1, n):
                    d_vec = centers[i] - centers[j]
                    d = np.linalg.norm(d_vec)
                    min_dist = 0.15 # Approximate radius for 26 circles in unit square
                    if d < min_dist:
                        if d < 1e-6:
                            d_vec = np.array([np.random.uniform(-0.01, 0.01), np.random.uniform(-0.01, 0.01)])
                            d = np.linalg.norm(d_vec)
                        push = (min_dist - d) / 2.0
                        centers[i] += (d_vec / d) * push
                        centers[j] -= (d_vec / d) * push
            
            # Keep inside bounds
            centers = np.clip(centers, 0.05, 0.95)
            
        return centers

    def energy(params):
        x_ = params[0::3]
        y_ = params[1::3]
        r_ = params[2::3]
        
        # Boundary violations
        pen_bound = 0.0
        for i in range(N):
            pen_bound += max(0, r_[i] - x_[i])**2
            pen_bound += max(0, r_[i] - (1 - x_[i]))**2
            pen_bound += max(0, r_[i] - y_[i])**2
            pen_bound += max(0, r_[i] - (1 - y_[i]))**2
            
        # Overlap violations
        pen_overlap = 0.0
        for i in range(N):
            for j in range(i+1, N):
                dist_sq = (x_[i] - x_[j])**2 + (y_[i] - y_[j])**2
                min_dist = r_[i] + r_[j]
                if dist_sq < min_dist**2:
                    overlap = min_dist - np.sqrt(max(dist_sq, 1e-12))
                    pen_overlap += overlap**2
                    
        # Adaptive penalty: increase if constraints are violated
        total_pen = pen_bound + pen_overlap
        lam = 10.0 + 100.0 * total_pen
        return -np.sum(r_) + lam * total_pen

    def optimize(c_init, r_init, max_iter=500):
        params0 = np.zeros(3*N)
        params0[0::3] = c_init[:, 0]
        params0[1::3] = c_init[:, 1]
        params0[2::3] = r_init
        
        bounds = []
        for i in range(N):
            bounds.append((0.0, 1.0))
            bounds.append((0.0, 1.0))
            bounds.append((0.0, 0.45))
            
        # Multi-method optimization
        best_params = params0.copy()
        best_score = energy(params0)
        
        # Try L-BFGS-B
        try:
            res = minimize(energy, params0, method='L-BFGS-B', bounds=bounds, 
                           options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-10})
            if energy(res.x) < best_score:
                best_score = energy(res.x)
                best_params = res.x.copy()
        except:
            pass
        
        # Try Nelder-Mead on perturbed start
        try:
            params_nm = best_params + np.random.normal(0, 0.01, 3*N)
            params_nm = np.clip(params_nm, [b[0] for b in bounds], [b[1] for b in bounds])
            res = minimize(energy, params_nm, method='Nelder-Mead', 
                           options={'maxiter': max_iter, 'fatol': 1e-12, 'xatol': 1e-10})
            if energy(res.x) < best_score:
                best_score = energy(res.x)
                best_params = res.x.copy()
        except:
            pass
        
        params_opt = best_params
        x_opt = params_opt[0::3].copy()
        y_opt = params_opt[1::3].copy()
        r_opt = params_opt[2::3].copy()
        
        # Post-processing: Ensure strict validity
        for _ in range(30):
            changed = False
            # Enforce boundary constraints
            for i in range(N):
                r_max_bound = min(x_opt[i], 1-x_opt[i], y_opt[i], 1-y_opt[i])
                if r_opt[i] > r_max_bound:
                    r_opt[i] = r_max_bound
                    changed = True
                    
            # Enforce non-overlap
            for i in range(N):
                for j in range(i+1, N):
                    dist = np.sqrt((x_opt[i]-x_opt[j])**2 + (y_opt[i]-y_opt[j])**2)
                    min_dist = r_opt[i] + r_opt[j]
                    if dist < min_dist and dist > 1e-12:
                        scale = dist / min_dist
                        r_opt[i] *= scale
                        r_opt[j] *= scale
                        changed = True
            if not changed:
                break
                
        # Local refinement: try to increase radii slightly
        for _ in range(50):
            improved = False
            for i in range(N):
                # Max radius at current position
                r_max = min(x_opt[i], 1-x_opt[i], y_opt[i], 1-y_opt[i])
                for j in range(N):
                    if i != j:
                        d = np.sqrt((x_opt[i]-x_opt[j])**2 + (y_opt[i]-y_opt[j])**2)
                        r_max = min(r_max, d - r_opt[j])
                r_max = max(r_max, 0.001)
                if r_max > r_opt[i] * 1.001:
                    r_opt[i] = r_max
                    improved = True
            if not improved:
                break
                
        return x_opt, y_opt, r_opt

    # 1. Generate initial configuration using Greedy Method
    c_init, r_init = greedy_initialize(N)
    
    # 2. Optimize
    x_opt, y_opt, r_opt = optimize(c_init, r_init)
    best_score = np.sum(r_opt)
    best_x, best_y, best_r = x_opt.copy(), y_opt.copy(), r_opt.copy()
    
    # 3. Multi-start: Try diverse strategies
    np.random.seed(42)
    
    # Strategy A: Perturb greedy start
    for i in range(15):
        c_pert = c_init + np.random.normal(0, 0.03, (N, 2))
        c_pert = np.clip(c_pert, 0.01, 0.99)
        r_pert = np.clip(r_init * (0.7 + 0.3*np.random.random()), 0.01, 0.4)
        
        x_p, y_p, r_p = optimize(c_pert, r_pert)
        score = np.sum(r_p)
        
        if score > best_score:
            best_score = score
            best_x, best_y, best_r = x_p.copy(), y_p.copy(), r_p.copy()
            
    # Strategy B: Random init + Relaxation
    for i in range(15):
        c_rand = np.random.uniform(0.05, 0.95, (N, 2))
        c_relaxed = relax_positions(c_rand, iterations=50)
        
        # Estimate radii based on relaxed positions
        r_est = np.zeros(N)
        for i in range(N):
            min_dist = min(c_relaxed[i][0], 1-c_relaxed[i][0], c_relaxed[i][1], 1-c_relaxed[i][1])
            for j in range(N):
                if i != j:
                    d = np.linalg.norm(c_relaxed[i] - c_relaxed[j])
                    min_dist = min(min_dist, d/2.0)
            r_est[i] = max(min_dist * 0.85, 0.01)
            
        x_p, y_p, r_p = optimize(c_relaxed, r_est)
        score = np.sum(r_p)
        
        if score > best_score:
            best_score = score
            best_x, best_y, best_r = x_p.copy(), y_p.copy(), r_p.copy()
            
    # Strategy C: Grid-based initialization
    for i in range(5):
        # Create a rough grid and add random perturbation
        cols = 5
        rows = 6
        c_grid = []
        r_grid = []
        for row in range(rows):
            for col in range(cols):
                if len(c_grid) >= N:
                    break
                x = (col + 0.5) / cols + np.random.normal(0, 0.02)
                y = (row + 0.5) / rows + np.random.normal(0, 0.02)
                c_grid.append([x, y])
                r_grid.append(0.08)
        c_grid = np.array(c_grid[:N])
        r_grid = np.array(r_grid[:N])
        
        x_p, y_p, r_p = optimize(c_grid, r_grid)
        score = np.sum(r_p)
        
        if score > best_score:
            best_score = score
            best_x, best_y, best_r = x_p.copy(), y_p.copy(), r_p.copy()

    # 4. Final validation and formatting
    result = np.zeros((N, 3))
    result[:, 0] = best_x
    result[:, 1] = best_y
    result[:, 2] = best_r
    
    # Final safety check: ensure all circles are valid
    for i in range(N):
        # Clamp to boundary
        result[i, 2] = min(result[i, 2], result[i, 0], 1-result[i, 0], result[i, 1], 1-result[i, 1])
        result[i, 2] = max(result[i, 2], 0.001)
    
    # Fix any remaining overlaps
    for _ in range(10):
        changed = False
        for i in range(N):
            for j in range(i+1, N):
                dist = np.sqrt((result[i,0]-result[j,0])**2 + (result[i,1]-result[j,1])**2)
                min_dist = result[i,2] + result[j,2]
                if dist < min_dist and dist > 1e-12:
                    scale = dist / min_dist * 0.999
                    result[i,2] *= scale
                    result[j,2] *= scale
                    changed = True
        if not changed:
            break
    
    return result
