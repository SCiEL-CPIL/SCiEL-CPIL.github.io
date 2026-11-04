import numpy as np
from scipy.optimize import minimize
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
        # Sample the boundary to catch maxima that lie on the box edges
        num_samples = 200
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
        
        if best_c is None:
            best_c = np.array([0.5, 0.5])
            best_r = 0.01
            
        return best_c[0], best_c[1], max(best_r, 1e-6)

    def greedy_initialize(n):
        centers = []
        radii = []
        
        # Start with 4 corners
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

    def energy(params):
        x_ = params[0::3]
        y_ = params[1::3]
        r_ = params[2::3]
        
        pen_bound = 0.0
        for i in range(N):
            pen_bound += max(0, r_[i] - x_[i])**2
            pen_bound += max(0, r_[i] - (1 - x_[i]))**2
            pen_bound += max(0, r_[i] - y_[i])**2
            pen_bound += max(0, r_[i] - (1 - y_[i]))**2
            
        pen_overlap = 0.0
        for i in range(N):
            for j in range(i+1, N):
                dist_sq = (x_[i] - x_[j])**2 + (y_[i] - y_[j])**2
                min_dist = r_[i] + r_[j]
                if dist_sq < min_dist**2:
                    overlap = min_dist - np.sqrt(dist_sq)
                    pen_overlap += overlap**2
                    
        lam = 50.0
        return -np.sum(r_) + lam * (pen_bound + pen_overlap)

    def optimize(c_init, r_init):
        params0 = np.zeros(3*N)
        params0[0::3] = c_init[:, 0]
        params0[1::3] = c_init[:, 1]
        params0[2::3] = r_init
        
        bounds = []
        for i in range(N):
            bounds.append((0, 1))
            bounds.append((0, 1))
            bounds.append((0, 0.5))
            
        res = minimize(energy, params0, method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8})
        
        params_opt = res.x
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
                        # Scale down both radii
                        if min_dist > 1e-9:
                            scale = dist / min_dist
                            r_opt[i] *= scale
                            r_opt[j] *= scale
                            changed = True
            if not changed:
                break
                
        return x_opt, y_opt, r_opt

    # 1. Generate initial configuration using Greedy Method
    c_init, r_init = greedy_initialize(N)
    
    # 2. Optimize
    x_opt, y_opt, r_opt = optimize(c_init, r_init)
    best_score = np.sum(r_opt)
    best_x, best_y, best_r = x_opt, y_opt, r_opt
    
    # 3. Multi-start: Try a few random perturbations of the greedy start
    for _ in range(5):
        c_pert = c_init + np.random.normal(0, 0.02, (N, 2))
        c_pert = np.clip(c_pert, 0.01, 0.99)
        r_pert = np.clip(r_init * 0.8, 0.01, 0.4)
        
        x_p, y_p, r_p = optimize(c_pert, r_pert)
        score = np.sum(r_p)
        
        if score > best_score:
            best_score = score
            best_x, best_y, best_r = x_p, y_p, r_p

    # 4. Final validation and formatting
    result = np.zeros((N, 3))
    result[:, 0] = best_x
    result[:, 1] = best_y
    result[:, 2] = best_r
    
    return result
