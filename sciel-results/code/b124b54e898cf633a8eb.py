import numpy as np
import scipy.optimize as so
import time
import itertools

N = 14
D = 3

def _pairwise_distances_squared(X):
    """Compute squared pairwise distances for points X (N x D)."""
    # X is N x D
    # dist2[i, j] = ||X[i] - X[j]||^2
    # Using broadcasting: (X[:, np.newaxis, :] - X[np.newaxis, :, :])**2
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    dist2 = np.sum(diff**2, axis=2)
    return dist2

def _objective_ratio(X):
    """
    Objective function: (max dist / min dist)^2.
    X is flattened array of length N*D.
    We assume x[0] and x[1] are fixed in the caller or handled here?
    Better to handle fixed points in the wrapper.
    """
    X = X.reshape(N, D)
    dist2 = _pairwise_distances_squared(X)
    # Mask out diagonal
    mask = ~np.eye(N, dtype=bool)
    d2 = dist2[mask]
    if len(d2) == 0:
        return 1e10
    d_min = np.min(d2)
    d_max = np.max(d2)
    if d_min < 1e-12:
        return 1e10
    return (d_max / d_min)

def _objective_and_fixed(X_vars):
    """
    X_vars: array of length 3*(N-2).
    Constructs full X by prepending fixed points.
    """
    X = np.zeros((N, D))
    # Fix x0 at (0,0,0)
    # Fix x1 at (1,0,0)
    X[0] = [0, 0, 0]
    X[1] = [1, 0, 0]
    X[2:] = X_vars.reshape(N-2, D)
    return _objective_ratio(X)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_score = 1e10
    best_X = None
    
    # Initial guesses
    # 1. Random points in a sphere
    # 2. Points on a sphere (Fibonacci)
    # 3. Lattice points
    
    def generate_random_init():
        X = np.random.randn(N, D)
        # Center
        X -= np.mean(X, axis=0)
        # Scale
        norms = np.linalg.norm(X, axis=1)
        if np.min(norms) > 0:
            X /= np.min(norms) # This doesn't fix min dist to 1, but provides a shape
        return X

    def generate_fibonacci_sphere():
        # Fibonacci lattice on sphere
        N_pts = N
        golden_angle = np.pi * (3. - np.sqrt(5.))
        X = np.zeros((N_pts, 3))
        for i in range(N_pts):
            z = 1. - (i / (N_pts - 1.)) * 2.
            radius = np.sqrt(1. - z**2)
            theta = golden_angle * i
            x = np.cos(theta) * radius
            y = np.sin(theta) * radius
            X[i] = [x, y, z]
        # Center
        X -= np.mean(X, axis=0)
        # Scale to unit min distance? 
        # Just return as is, optimizer will scale/shape
        return X

    def generate_lattice():
        # Simple cubic lattice points, take first N
        side = int(np.ceil(N ** (1/3))) + 1
        grid = list(itertools.product(range(side), repeat=3))[:N]
        X = np.array(grid, dtype=float)
        X -= np.mean(X, axis=0)
        return X

    inits = [
        generate_fibonacci_sphere(),
        generate_lattice(),
        generate_random_init(),
        generate_random_init(),
        generate_random_init()
    ]
    
    # Add some perturbed versions of the best found so far
    
    for init_idx, X_init in enumerate(inits):
        if time.time() > deadline:
            break
            
        # Prepare variables for optimization
        # Fix x0=(0,0,0), x1=(1,0,0)
        # We need to rotate/translate X_init such that x0=0, x1=1?
        # Simpler: Just take X_init, center it, and take the first two points?
        # No, fixing specific points to (0,0,0) and (1,0,0) is a gauge fixing.
        # We can apply this gauge to any configuration.
        
        # 1. Center X_init
        X_c = X_init - np.mean(X_init, axis=0)
        
        # 2. Find two points with max distance? Or just first two?
        # To make the gauge fixing valid and non-degenerate, pick two points with non-zero distance.
        # Let's pick point 0 and point 1 from X_c.
        # If they are close, pick another pair.
        # For simplicity, let's just use the first two points of the centered cloud.
        # If their distance is too small, the optimization might struggle.
        # Let's find the pair with maximum distance in X_c to define the axis.
        dist2 = _pairwise_distances_squared(X_c)
        max_idx = np.unravel_index(np.argmax(dist2), dist2.shape)
        p_a, p_b = max_idx
        
        # Rotate X_c such that p_a is at origin and p_b is on x-axis?
        # This is complex. 
        # Alternative: Don't fix points to specific coords. 
        # Instead, fix the center of mass to 0 (3 constraints) and 
        # fix the orientation by aligning the principal axes?
        # Or just use Nelder-Mead on the full 42D space with the ratio objective.
        # The ratio is scale invariant, so Nelder-Mead might drift in scale.
        # To prevent scale drift, we can normalize the points in the objective?
        # No, that makes it non-differentiable if not careful.
        
        # Let's try the fixed-point approach properly.
        # Given X_init, we construct a candidate X where:
        # x[0] = 0
        # x[1] = 1
        # x[i] = (X_init[i] - X_init[0]) * scale_factor
        # where scale_factor = 1 / ||X_init[1] - X_init[0]||
        # This maps X_init[0] to 0 and X_init[1] to (1,0,0) ONLY if X_init[1]-X_init[0] is along x-axis.
        # We need to rotate X_init so that the vector X_init[1]-X_init[0] aligns with x-axis.
        
        v = X_c[1] - X_c[0]
        norm_v = np.linalg.norm(v)
        if norm_v < 1e-6:
            # Pick another point
            continue
        # Rotation to align v with x-axis
        # Simple rotation:
        # e_x = [1,0,0]
        # v_hat = v / norm_v
        # axis = cross(e_x, v_hat)
        # angle = dot(e_x, v_hat)
        # This is getting complicated.
        
        # Easier: Just use the first two points of the centered array, 
        # and let the optimizer figure out the rest. 
        # The gauge fixing x0=0, x1=1 restricts the search space to 
        # configurations where the first two points are at those locations.
        # This is a valid slice of the configuration space (modulo symmetry).
        # We just need to initialize the other points reasonably.
        
        # Let's initialize the other points as X_c[2:] - X_c[0], scaled by 1/norm(X_c[1]-X_c[0])
        # This preserves the relative geometry roughly.
        
        scale = 1.0 / norm_v
        X_opt_init = np.zeros((N, D))
        X_opt_init[0] = [0, 0, 0]
        X_opt_init[1] = [1, 0, 0]
        # Project X_c[2:] onto the plane/3D space relative to 0, scaled
        # Actually, just taking the vector differences is fine.
        X_opt_init[2:] = (X_c[2:] - X_c[0]) * scale
        
        X_vars_init = X_opt_init[2:].flatten()
        
        # Run Nelder-Mead
        # Max iterations to avoid hanging
        max_iter = 5000
        
        try:
            res = so.minimize(
                _objective_and_fixed, 
                X_vars_init, 
                method='Nelder-Mead',
                options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            
            if res.success or res.fun < best_score:
                X_best_candidate = np.zeros((N, D))
                X_best_candidate[0] = [0, 0, 0]
                X_best_candidate[1] = [1, 0, 0]
                X_best_candidate[2:] = res.x.reshape(N-2, D)
                
                # Calculate actual score
                dist2 = _pairwise_distances_squared(X_best_candidate)
                mask = ~np.eye(N, dtype=bool)
                d2 = dist2[mask]
                d_min = np.min(d2)
                d_max = np.max(d2)
                score = (d_max / d_min)
                
                if score < best_score:
                    best_score = score
                    best_X = X_best_candidate
                    
        except Exception:
            pass

    # If we have time, do a second pass with the best X found, 
    # trying to polish it with a different method or restart.
    
    if best_X is None:
        # Fallback: return a grid
        side = int(np.ceil(N ** (1 / D)))
        grid = list(itertools.product(range(side), repeat=D))[:N]
        return np.array(grid, dtype=float)
        
    # Optional: Refine best_X with Powell
    if time.time() < deadline - 10:
        X_vars_best = best_X[2:].flatten()
        try:
            res2 = so.minimize(
                _objective_and_fixed, 
                X_vars_best, 
                method='Powell',
                options={'maxiter': 200, 'xtol': 1e-9, 'ftol': 1e-9}
            )
            if res2.fun < best_score:
                X_best_candidate = np.zeros((N, D))
                X_best_candidate[0] = [0, 0, 0]
                X_best_candidate[1] = [1, 0, 0]
                X_best_candidate[2:] = res2.x.reshape(N-2, D)
                
                dist2 = _pairwise_distances_squared(X_best_candidate)
                mask = ~np.eye(N, dtype=bool)
                d2 = dist2[mask]
                d_min = np.min(d2)
                d_max = np.max(d2)
                score = (d_max / d_min)
                
                if score < best_score:
                    best_score = score
                    best_X = X_best_candidate
        except Exception:
            pass

    return best_X
