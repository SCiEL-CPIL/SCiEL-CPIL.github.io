import numpy as np
from scipy.optimize import minimize
import time

N = 50

def _hexagonal_seed(n):
    """Generate a hexagonal lattice seed for n points."""
    # Estimate spacing
    # Area of disk = pi. n circles of radius r. r ~ 1/sqrt(n).
    # Spacing s ~ 2r.
    s = 2.0 / (np.sqrt(n) + 1.0)
    
    points = []
    # Generate points in a square grid large enough to cover the disk
    # Hex grid:
    # x = i * s + (j % 2) * s / 2
    # y = j * s * sqrt(3) / 2
    max_k = int(1.0 / s) + 2
    for j in range(-max_k, max_k + 1):
        for i in range(-max_k, max_k + 1):
            x = (i + 0.5 * (j % 2)) * s
            y = j * s * np.sqrt(3.0) / 2.0
            if x*x + y*y <= 1.0:
                points.append((x, y))
    
    points = np.array(points)
    # Sort by distance to center, take closest N
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)
    return points[idx[:n]]

def _soft_min(vals, beta):
    """Compute soft minimum of a set of values."""
    # To avoid overflow in exp, shift by min
    m = np.min(vals)
    return m - (1.0 / beta) * np.log(np.sum(np.exp(beta * (vals - m))))

def _objective_and_grad(x, beta):
    """
    x: flattened array of shape (2N,)
    Returns: objective value, gradient
    Objective: - soft_min( distances )
    """
    c = x.reshape(N, 2)
    
    # Distances between centers
    # d_ij
    dists = []
    grad_dists = [] # gradients of d_ij wrt c_i, c_j
    
    # Container distances: 1 - |c_i|
    container_dists = []
    grad_container = []
    
    for i in range(N):
        r_i = np.linalg.norm(c[i])
        if r_i > 0:
            d_cont = 1.0 - r_i
        else:
            d_cont = 1.0
        
        container_dists.append(d_cont)
        # grad of (1 - r) wrt c_i = - c_i / r_i
        if r_i > 1e-8:
            grad_container.append(-c[i] / r_i)
        else:
            grad_container.append(np.zeros(2))

    for i in range(N):
        for j in range(i+1, N):
            diff = c[i] - c[j]
            d = np.linalg.norm(diff)
            if d > 1e-8:
                dists.append(d / 2.0) # radius is half distance
                # grad of d/2 wrt c_i = diff / (2d)
                # grad of d/2 wrt c_j = -diff / (2d)
                grad_i = diff / (2.0 * d)
                grad_j = -diff / (2.0 * d)
            else:
                dists.append(0.0)
                grad_i = np.zeros(2)
                grad_j = np.zeros(2)
            
            # Store gradients for later assembly
            # We need to sum gradients weighted by soft-min coefficients
            # This is getting complicated to do analytically in one pass without storing all diffs.
            # Let's just use numerical gradient or a simpler objective?
            # No, numerical gradient is too slow for 100 vars * many iters.
            # Let's restructure to compute the soft-min gradient properly.

    # Let's use a simpler differentiable objective:
    # Sum of exp(-beta * d) ? No, that minimizes sum, not min.
    # Soft min: S = (1/beta) log sum exp(-beta * x_k)
    # dS/dx_k = - exp(-beta * x_k) / sum exp(-beta * x_m)
    
    # Let's collect all "constraints" values and their gradients.
    vals = []
    grads = [] # list of arrays of shape (2N,)
    
    # 1. Pairwise distances
    for i in range(N):
        for j in range(i+1, N):
            diff = c[i] - c[j]
            d = np.linalg.norm(diff)
            val = d / 2.0
            vals.append(val)
            
            g = np.zeros(2*N)
            if d > 1e-8:
                grad_i = diff / (2.0 * d)
                grad_j = -diff / (2.0 * d)
                g[2*i:2*i+2] = grad_i
                g[2*j:2*j+2] = grad_j
            grads.append(g)
            
    # 2. Container distances
    for i in range(N):
        r_i = np.linalg.norm(c[i])
        val = 1.0 - r_i
        vals.append(val)
        
        g = np.zeros(2*N)
        if r_i > 1e-8:
            g[2*i:2*i+2] = -c[i] / r_i
        grads.append(g)
        
    vals = np.array(vals)
    # Compute soft min
    m = np.min(vals)
    # Shift to prevent overflow
    shifted = vals - m
    exps = np.exp(-beta * shifted)
    sum_exps = np.sum(exps)
    
    # Soft min value
    s_min = m - (1.0/beta) * np.log(sum_exps)
    
    # Gradient of soft min wrt each val_k
    # d(s_min)/d(val_k) = - (1/beta) * (1/sum_exps) * d(sum_exps)/d(val_k)
    # d(sum_exps)/d(val_k) = -beta * exps[k]
    # So d(s_min)/d(val_k) = exps[k] / sum_exps
    
    weights = exps / sum_exps
    
    # Total gradient wrt x
    total_grad = np.zeros(2*N)
    for k in range(len(vals)):
        total_grad += weights[k] * grads[k]
        
    # We want to maximize s_min, so minimize -s_min
    obj_val = -s_min
    grad_val = -total_grad
    
    return obj_val, grad_val

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    # 1. Initialize
    x0 = _hexagonal_seed(N).flatten()
    
    # 2. Optimize
    # We use L-BFGS-B as it handles large N well
    # beta controls the sharpness of the soft-min
    beta = 10.0
    
    best_x = x0
    best_score = -1e9
    
    # We can try a few different betas or just one
    # Let's run one optimization
    try:
        res = minimize(
            _objective_and_grad,
            x0,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        
        if res.success or res.fun < _objective_and_grad(x0, beta)[0]:
            best_x = res.x
            # Calculate actual score
            c = best_x.reshape(N, 2)
            min_dist = np.inf
            for i in range(N):
                for j in range(i+1, N):
                    d = np.linalg.norm(c[i] - c[j])
                    if d < min_dist:
                        min_dist = d
                d_cont = 1.0 - np.linalg.norm(c[i])
                if d_cont < min_dist:
                    min_dist = d_cont
            actual_r = min_dist / 2.0
            best_score = actual_r
            
    except Exception as e:
        print(f"Optimization error: {e}")
        # Fallback to seed
        best_x = x0
        c = best_x.reshape(N, 2)
        min_dist = np.inf
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(c[i] - c[j])
                if d < min_dist:
                    min_dist = d
            d_cont = 1.0 - np.linalg.norm(c[i])
            if d_cont < min_dist:
                min_dist = d_cont
        best_score = min_dist / 2.0

    # Check time
    if time.time() > deadline:
        # Return best found so far
        pass

    # If we have time, we could try to polish with a higher beta or local search
    # But L-BFGS-B should be sufficient for a jump.
    
    # Ensure validity
    c = best_x.reshape(N, 2)
    # Clip to disk if slightly outside due to numerical error?
    # The objective penalizes going outside, so it should be inside.
    
    return c
