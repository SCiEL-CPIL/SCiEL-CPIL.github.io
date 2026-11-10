import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize, basinhopping

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Finds 14 points in 3D that maximize (d_min / d_max)^2.
    Uses Basin Hopping to navigate the non-smooth objective landscape.
    """
    n = 14
    d = 3
    start_time = time.time()
    # Evaluator kills at 630s. Task says 600s. Use 580s as hard stop for computation.
    deadline = start_time + 580.0

    def objective(x):
        """
        Objective function to minimize: - (d_min / d_max)^2
        x: 1D array of length n*d
        """
        if time.time() > deadline:
            raise TimeoutError("Time limit exceeded")
        
        pts = x.reshape(n, d)
        
        # Normalize points to unit sphere to fix scale and reduce search space
        # This is valid because the metric is scale-invariant.
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        # Avoid division by zero
        norms[norms == 0] = 1.0
        pts = pts / norms
        
        # Calculate pairwise distances
        dists = pdist(pts)
        if len(dists) == 0:
            return 0.0
            
        d_min = np.min(dists)
        d_max = np.max(dists)
        
        if d_max == 0 or d_min == 0:
            return 0.0
            
        score = (d_min / d_max) ** 2
        return -score  # We minimize, so negate

    def step(x, stepsize):
        """
        Random perturbation step for basin hopping.
        """
        new_x = x + stepsize * np.random.randn(*x.shape)
        # We don't normalize here because the objective function does it.
        # However, keeping the magnitude roughly stable can help.
        # The objective function projects to sphere, so the raw magnitude 
        # doesn't matter for the value, but for the gradient/step, it might.
        # Since objective projects, the "effective" position is on the sphere.
        # We can just return new_x.
        return new_x

    def accept_test(x_new, x_old, f_new, f_old, T):
        """
        Acceptance criterion.
        Accept if f_new < f_old (improvement) or with probability exp((f_old - f_new)/T)
        Standard Metropolis-Hastings.
        """
        if f_new < f_old:
            return True
        # f_new > f_old
        # prob = exp((f_old - f_new) / T)
        # Since we are minimizing, f_old - f_new is negative.
        # Wait, standard SA: accept if f_new <= f_old OR with prob exp(-(f_new - f_old)/T)
        # Here f_new > f_old. So exp(-(positive)/T).
        return np.random.random() < np.exp(-(f_new - f_old) / T)

    # Initial Point: Fibonacci Sphere is a good deterministic start
    def fib_sphere(n):
        points = []
        phi = np.pi * (3.0 - np.sqrt(5.0))
        for i in range(n):
            y = 1 - (i / float(n - 1)) * 2
            radius = np.sqrt(1 - y * y)
            theta = phi * i
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            points.append([x, y, z])
        return np.array(points)

    # We will run several independent basin hopping searches or one long one.
    # Given the time budget, one long, adaptive search is often better than many short ones.
    # However, basin hopping with a fixed step size might get stuck if the step is too small or too large.
    # Let's use a callback to monitor time and potentially adjust or stop.
    
    best_x = None
    best_f = np.inf
    
    # Start with Fibonacci sphere
    x0 = fib_sphere(n).flatten()
    
    # Initial local optimization to get into a good basin
    # Use Powell as it handles non-smoothness well
    try:
        res0 = minimize(objective, x0, method='Powell', options={'maxiter': 200, 'maxfev': 1000})
        if time.time() < deadline - 10:
            best_x = res0.x
            best_f = res0.fun
        else:
            best_x = x0
            best_f = objective(x0)
    except Exception:
        best_x = x0
        best_f = objective(x0)

    # Basin Hopping
    # stepsize: 0.2 is a reasonable guess for sphere coords (radians roughly)
    # T: Temperature. Starts high, cools down.
    
    # We need a custom callback to check time and update best
    # scipy's basinhopping callback signature: callback(state) where state has x, f, nfev, etc.
    
    def bh_callback(state):
        nonlocal best_x, best_f
        if state.f < best_f:
            best_f = state.f
            best_x = state.x.copy()
            
        # Check time
        if time.time() > deadline - 5:
            # Raise exception to stop the optimizer? 
            # Or just let it finish the current iteration.
            # Raising an exception in callback might be messy.
            # Better to rely on the objective function raising TimeoutError.
            pass

    # Set up basin hopping
    # maxiter: number of outer iterations. We want as many as possible.
    # We can set a large number and let the timeout handle it.
    # However, if each iteration takes 1s, 580s = 580 iterations.
    
    try:
        # Use a moderate step size. 
        # The coordinates are in [-1, 1]. A step of 0.1-0.5 is reasonable.
        # Let's try 0.2.
        res = basinhopping(
            objective, 
            best_x, 
            stepsize=0.2, 
            minimizer_kwargs={'method': 'Powell', 'options': {'maxiter': 100, 'maxfev': 500}},
            niter=1000, # Large number, will be cut by timeout
            T=1000.0, 
            accept_test=accept_test,
            callback=bh_callback
        )
        # If it finished naturally (unlikely), update best
        if res.f < best_f:
            best_f = res.f
            best_x = res.x
    except TimeoutError:
        pass
    except Exception:
        pass

    # If we still have time, do a final polish on the best point found
    if time.time() < deadline - 10 and best_x is not None:
        try:
            res_final = minimize(objective, best_x, method='Powell', options={'maxiter': 500, 'maxfev': 5000})
            if res_final.f < best_f:
                best_f = res_final.f
                best_x = res_final.x
        except Exception:
            pass

    # Convert best_x back to points and normalize
    if best_x is None:
        # Fallback
        best_x = fib_sphere(n).flatten()
        
    points = best_x.reshape(n, d)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    points = points / norms
    
    # Final sanity check
    if not np.all(np.isfinite(points)):
        points = fib_sphere(n)
        
    return points
