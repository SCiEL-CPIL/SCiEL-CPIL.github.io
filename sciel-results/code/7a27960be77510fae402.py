import numpy as np
import scipy.spatial
import scipy.optimize
import time

N = 14

# EVOLVE-BLOCK-START

def objective(x):
    """
    Calculate the score: min triangle area / convex hull area.
    x: flattened array of shape (2*N,)
    """
    points = x.reshape(-1, 2)
    try:
        hull = scipy.spatial.ConvexHull(points)
        hull_area = hull.volume
        if hull_area < 1e-10:
            return 0.0
        
        min_area = float('inf')
        n = len(points)
        # Vectorized triangle area calculation is possible but N=14 is small
        for i in range(n-2):
            pi = points[i]
            for j in range(i+1, n-1):
                v1 = points[j] - pi
                for k in range(j+1, n):
                    v2 = points[k] - pi
                    cross = v1[0]*v2[1] - v1[1]*v2[0]
                    area = 0.5 * abs(cross)
                    if area < min_area:
                        min_area = area
        return min_area / hull_area
    except:
        return 0.0

def _expand_ring_params(params, ring_counts):
    """
    Expand ring parameters (radii and angles) to a list of points.
    params: array of shape (num_rings * 2,). 
            For each ring r, params[2*r] is radius, params[2*r+1] is angle offset.
    ring_counts: list of integers, number of points in each ring.
    """
    points = []
    num_rings = len(ring_counts)
    for r in range(num_rings):
        radius = params[2*r]
        angle_offset = params[2*r+1]
        count = ring_counts[r]
        if count == 1:
            # Place single point at angle_offset
            points.append([radius * np.cos(angle_offset), radius * np.sin(angle_offset)])
        else:
            angles = np.linspace(0, 2*np.pi, count, endpoint=False) + angle_offset
            for a in angles:
                points.append([radius * np.cos(a), radius * np.sin(a)])
    return np.array(points)

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    best_score = 0.0
    best_points = None
    
    # Define a few ring topologies to explore
    # (counts per ring)
    topologies = [
        [1, 3, 4, 6],
        [1, 4, 4, 5],
        [1, 2, 5, 6],
        [2, 3, 4, 5],
        [1, 3, 3, 7],
        [1, 4, 3, 6]
    ]
    
    def neg_objective(x):
        return -objective(x)
        
    for topo in topologies:
        if time.time() > deadline - 20:
            break
            
        num_rings = len(topo)
        # Parameters: radius and angle offset for each ring
        # Radii should be positive. We can optimize in log-space or just bound them.
        # Let's bound radii between 0.1 and 1.5 (normalized roughly to unit area hull)
        # Angles between 0 and 2*pi
        
        bounds = []
        for r in range(num_rings):
            bounds.append((0.1, 1.5)) # radius
            bounds.append((0, 2*np.pi)) # angle
        
        def ring_objective(params):
            pts = _expand_ring_params(params, topo)
            return -objective(pts.flatten())
            
        # Initial guess: regular polygon-like distribution
        x0 = []
        for i in range(num_rings):
            x0.append(0.5 + 0.3 * i) # radii
            x0.append(0.0) # angles
            
        try:
            # Use Differential Evolution to find good ring parameters
            result = scipy.optimize.differential_evolution(
                ring_objective, 
                bounds, 
                maxiter=50, 
                tol=1e-4, 
                seed=42 + i,
                popsize=10
            )
            
            if not result.success:
                # Even if not successful, check the best found
                pass
                
            best_params = result.x
            best_pts = _expand_ring_params(best_params, topo)
            current_score = objective(best_pts.flatten())
            
            if current_score > best_score:
                best_score = current_score
                best_points = best_pts.copy()
                
        except Exception as e:
            pass

    # Local refinement on the best point set found
    if best_points is not None and time.time() < deadline - 10:
        try:
            result = scipy.optimize.minimize(
                neg_objective, 
                best_points.flatten(), 
                method='Nelder-Mead', 
                options={'maxiter': 5000, 'xatol': 1e-9, 'fatol': 1e-9}
            )
            if result.fun < -best_score:
                best_score = -result.fun
                best_points = result.x.reshape(-1, 2)
        except Exception:
            pass
            
    # If we still have time, try a few random restarts with local refinement
    if time.time() < deadline - 20:
        for _ in range(5):
            if time.time() > deadline - 5:
                break
            # Random initial points within unit circle
            pts = np.random.uniform(-1, 1, (N, 2))
            # Scale to roughly unit area hull
            hull = scipy.spatial.ConvexHull(pts)
            area = hull.volume
            if area > 1e-6:
                pts = pts * np.sqrt(1.0/area)
                
            try:
                result = scipy.optimize.minimize(
                    neg_objective, 
                    pts.flatten(), 
                    method='Nelder-Mead', 
                    options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8}
                )
                score = -result.fun
                if score > best_score:
                    best_score = score
                    best_points = result.x.reshape(-1, 2)
            except Exception:
                pass

    if best_points is None:
        # Fallback to regular polygon
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
        
    return best_points

# EVOLVE-BLOCK-END
