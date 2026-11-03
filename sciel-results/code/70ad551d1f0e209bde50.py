import numpy as np
from scipy.optimize import minimize
import time

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin before 150s kill

    # Structure: 1 center, 6 ring 1, 12 ring 2, 18 ring 3, 13 ring 4
    # We will parameterize the points as:
    # p_i = R_{ring(i)} * (cos(theta_i + delta_i), sin(theta_i + delta_i))
    # where theta_i is the ideal angle for index i in its ring.
    # Parameters: [R1, R2, R3, R4, delta_0, ..., delta_49]
    # Total params: 54
    
    # Ideal angles for each ring
    n_rings = [1, 6, 12, 18, 13]
    ideal_angles = np.zeros(N)
    current_idx = 0
    for r_idx, n_points in enumerate(n_rings):
        if r_idx == 0:
            # Center
            ideal_angles[current_idx] = 0
            current_idx += 1
        else:
            angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
            ideal_angles[current_idx:current_idx + n_points] = angles
            current_idx += n_points

    ring_indices = np.zeros(N, dtype=int)
    current_idx = 0
    for r_idx, n_points in enumerate(n_rings):
        ring_indices[current_idx:current_idx + n_points] = r_idx
        current_idx += n_points

    def generate_points(params):
        """Generate 50 points from ring radii and angular deviations."""
        R = params[:4]
        deltas = params[4:]
        
        # Map ring index to radius param index (0->none, 1->R[0], 2->R[1], etc.)
        # R[0] is R1, R[1] is R2, R[2] is R3, R[3] is R4
        radii = np.zeros(N)
        for i in range(N):
            ring = ring_indices[i]
            if ring == 0:
                radii[i] = 0.0
            else:
                radii[i] = R[ring - 1]
                
        points = np.zeros((N, 2))
        for i in range(N):
            theta = ideal_angles[i] + deltas[i]
            points[i, 0] = radii[i] * np.cos(theta)
            points[i, 1] = radii[i] * np.sin(theta)
            
        return points

    def compute_score(points):
        """Compute the exact maximum radius."""
        # Distance to boundary
        dist_to_center = np.linalg.norm(points, axis=1)
        max_dist = np.max(dist_to_center)
        if max_dist >= 1.0:
            return 0.0
            
        # Minimum distance between any two centers
        # Using broadcasting for speed
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        # Set diagonal to inf
        np.fill_diagonal(dist, np.inf)
        min_dist = np.min(dist)
        
        r1 = min_dist / 2.0
        r2 = 1.0 - max_dist
        return min(r1, r2)

    def objective(params, k_soft=50.0):
        """Negative of the smoothed radius."""
        points = generate_points(params)
        
        # Boundary constraint
        dist_to_center = np.linalg.norm(points, axis=1)
        max_dist = np.max(dist_to_center)
        
        # Soft min of pairwise distances
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist, np.inf)
        
        # soft_min = -1/k * log(sum(exp(-k * dist)))
        # To avoid overflow/underflow, shift by min
        d_min = np.min(dist)
        if not np.isfinite(d_min):
             return 0.0
        exp_term = np.exp(-k_soft * (dist - d_min))
        sum_exp = np.sum(exp_term)
        soft_min_dist = d_min - (1.0 / k_soft) * np.log(sum_exp)
        
        # The true objective is min(soft_min_dist/2, 1 - max_dist)
        # We want to maximize this.
        # Smooth approximation: soft_min(soft_min_dist/2, 1 - max_dist)
        # Or just return -min(soft_min_dist/2, 1 - max_dist) and rely on the optimizer?
        # Using a soft-min for the final bottleneck makes it smoother.
        
        term1 = soft_min_dist / 2.0
        term2 = 1.0 - max_dist
        
        # Soft min of term1 and term2
        # m = min(a, b) ~ a - (a-b)/(1+exp(k(a-b))) ... actually simpler to use log-sum-exp on -terms
        # log-sum-exp(-k*a, -k*b) = -k * min(a,b) + log(1+exp(-k|a-b|))
        # min(a,b) = (a + b - |a-b|)/2
        # Let's just use a very sharp soft-min on the final result.
        if term1 > term2:
            # term2 is bottleneck
            return -term2 - 0.01 * np.exp(-k_soft * (term1 - term2)) # Small penalty if term1 is much larger? No.
            # Better: return -soft_min(term1, term2)
            # soft_min(x, y) = -1/k * log(exp(-k*x) + exp(-k*y))
            x = term1
            y = term2
            m = np.min([x, y])
            if not np.isfinite(m): return 0.0
            exp_x = np.exp(-k_soft * (x - m))
            exp_y = np.exp(-k_soft * (y - m))
            soft_min_final = m - (1.0 / k_soft) * np.log(exp_x + exp_y)
            return -soft_min_final
        else:
            x = term1
            y = term2
            m = np.min([x, y])
            if not np.isfinite(m): return 0.0
            exp_x = np.exp(-k_soft * (x - m))
            exp_y = np.exp(-k_soft * (y - m))
            soft_min_final = m - (1.0 / k_soft) * np.log(exp_x + exp_y)
            return -soft_min_final

    def run_optimization(initial_params, method='L-BFGS-B', maxiter=1000, k_soft=50.0):
        """Run optimization with given method and initial params."""
        # Bounds: Radii (0.01, 0.99), Deltas (-pi, pi)
        bounds = [(0.01, 0.99), (0.01, 0.99), (0.01, 0.99), (0.01, 0.99)]
        bounds += [(-np.pi, np.pi)] * N
        
        # Wrap the objective to pass k_soft
        def obj_wrapper(p):
            return objective(p, k_soft)
            
        if method == 'L-BFGS-B':
            result = minimize(obj_wrapper, initial_params, method='L-BFGS-B', bounds=bounds, 
                              options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-14, 'maxfun': maxiter*20})
        elif method == 'Powell':
            result = minimize(obj_wrapper, initial_params, method='Powell',
                              options={'maxiter': maxiter, 'ftol': 1e-14, 'xtol': 1e-14})
        else:
            result = minimize(obj_wrapper, initial_params, method='Nelder-Mead', 
                              options={'maxiter': maxiter, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True})
        
        return result

    # Initial guesses
    # Base radii for rings 1-4
    base_R = [0.25, 0.50, 0.75, 0.95]
    base_deltas = np.zeros(N)
    
    best_score = 0.0
    best_points = None
    best_params = None
    
    # Multi-start
    starts = []
    # Start 1: Symmetric
    starts.append(np.concatenate([base_R, base_deltas]))
    
    # Start 2: Slightly shifted angles
    for shift in [np.pi/12, np.pi/6, np.pi/4]:
        d = base_deltas.copy()
        # Shift all points in rings
        for i in range(N):
            if ring_indices[i] > 0:
                d[i] = shift
        starts.append(np.concatenate([base_R, d]))
        
    # Start 3: Different radii
    alt_R = [0.24, 0.49, 0.74, 0.94]
    starts.append(np.concatenate([alt_R, base_deltas]))
    
    # Start 4: More irregular
    rng = np.random.default_rng(42)
    d_irr = rng.uniform(-0.1, 0.1, size=N)
    d_irr[0] = 0 # Center
    starts.append(np.concatenate([base_R, d_irr]))

    methods = ['L-BFGS-B', 'Powell']
    
    for i, start in enumerate(starts):
        if time.time() > deadline - 40:
            break
            
        for method in methods:
            if time.time() > deadline - 20:
                break
                
            try:
                maxiter = 200 if method == 'L-BFGS-B' else 500
                # Use lower k_soft initially for smoother landscape, then higher?
                # Let's stick to a moderate k_soft
                result = run_optimization(start, method=method, maxiter=maxiter, k_soft=30.0)
                
                if result.fun < -best_score:  # result.fun is -radius
                    params = result.x
                    params = np.array(params)
                    params[:4] = np.clip(params[:4], 0.01, 0.99)
                    params[4:] = np.mod(params[4:] + np.pi, 2*np.pi) - np.pi # Normalize to -pi, pi
                    
                    points = generate_points(params)
                    score = compute_score(points)
                    
                    if score > best_score:
                        best_score = score
                        best_points = points
                        best_params = params
            except Exception as e:
                pass

    # Final polish with high k_soft (sharp) to refine exact min
    if best_params is not None and time.time() < deadline - 10:
        try:
            result = run_optimization(best_params, method='L-BFGS-B', maxiter=500, k_soft=100.0)
            if result.fun < -best_score:
                params = result.x
                params = np.array(params)
                params[:4] = np.clip(params[:4], 0.01, 0.99)
                params[4:] = np.mod(params[4:] + np.pi, 2*np.pi) - np.pi
                
                points = generate_points(params)
                score = compute_score(points)
                
                if score > best_score:
                    best_score = score
                    best_points = points
                    best_params = params
        except Exception as e:
            pass

    # Fallback
    if best_points is None or len(best_points) != N or not np.all(np.isfinite(best_points)):
        # Simple hexagonal fallback
        spacing = 2.0 / (np.sqrt(N) + 1.0)
        while True:
            k = int(1.0 / spacing) + 2
            rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                    for i in range(-k, k + 1) for j in range(-k, k + 1)]
            points = np.array(rows)
            radius = np.hypot(points[:, 0], points[:, 1])
            points, radius = points[radius <= 1.0 - spacing / 2.0], radius[radius <= 1.0 - spacing / 2.0]
            if len(points) >= N:
                best_points = points[np.argsort(radius, kind="stable")[:N]]
                break
            spacing *= 0.98
            
    return best_points
