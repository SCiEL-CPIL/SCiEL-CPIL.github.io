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
    
    # We want to maximize the minimum radius r.
    # r = min( min_{i<j} |c_i - c_j| / 2, min_i (1 - |c_i|) )
    # Let's define constraints:
    # g_k(x) = constraint_value_k
    # We want to maximize min_k g_k(x).
    # Using soft-min: S(x) = (1/beta) log sum exp(-beta * g_k(x)) is a soft-min of g_k?
    # No, standard soft-min is: m_beta(x) = -(1/beta) log sum exp(-beta * x_k).
    # This approximates min(x_k) as beta -> inf.
    # We want to maximize m_beta(g(x)).
    # So we minimize -m_beta(g(x)).
    
    # Let's compute all g_k and their gradients.
    # g_k for pairs: d_ij / 2
    # g_k for boundaries: 1 - |c_i|
    
    vals = []
    grads = [] # list of gradients of each g_k wrt x (shape 2N)
    
    # 1. Pairwise distances
    for i in range(N):
        for j in range(i+1, N):
            diff = c[i] - c[j]
            d = np.linalg.norm(diff)
            val = d / 2.0
            vals.append(val)
            
            g = np.zeros(2*N)
            if d > 1e-8:
                # grad of d/2 wrt c_i = (c_i - c_j) / (2d)
                # grad of d/2 wrt c_j = (c_j - c_i) / (2d)
                grad_i = diff / (2.0 * d)
                grad_j = -grad_i
                g[2*i:2*i+2] = grad_i
                g[2*j:2*j+2] = grad_j
            else:
                # If d is 0, gradient is undefined, but we can set to 0 or random
                # Usually happens at start if seed is bad, but hex seed should be fine.
                pass
            grads.append(g)
            
    # 2. Boundary distances
    for i in range(N):
        r_i = np.linalg.norm(c[i])
        val = 1.0 - r_i
        vals.append(val)
        
        g = np.zeros(2*N)
        if r_i > 1e-8:
            # grad of (1 - r) wrt c_i = -c_i / r_i
            g[2*i:2*i+2] = -c[i] / r_i
        else:
            # At origin, gradient is 0 (or undefined, but 0 is safe)
            pass
        grads.append(g)
        
    vals = np.array(vals)
    # Compute soft min
    # m_beta = -(1/beta) log sum exp(-beta * vals)
    # To avoid underflow/overflow, shift by min(vals)
    # Let v_min = min(vals)
    # sum exp(-beta * (vals - v_min))
    # log_sum = v_min * beta + log(sum exp(-beta * (vals - v_min))) ? No.
    # log sum exp(-beta * vals) = -beta * v_min + log sum exp(-beta * (vals - v_min))
    # So m_beta = -(1/beta) * (-beta * v_min + log(sum exp(...)))
    #           = v_min - (1/beta) * log(sum exp(-beta * (vals - v_min)))
    
    v_min = np.min(vals)
    shifted = vals - v_min
    exps = np.exp(-beta * shifted)
    sum_exps = np.sum(exps)
    
    m_beta = v_min - (1.0/beta) * np.log(sum_exps)
    
    # Gradient of m_beta wrt x
    # d(m_beta)/dx = d(m_beta)/d(vals) * d(vals)/dx
    # d(m_beta)/d(vals_k) = ?
    # m_beta = v_min - (1/beta) log(sum_exps)
    # Note v_min is one of the vals, so its derivative is tricky if we treat it separately.
    # Better to use the chain rule on the definition:
    # m_beta = -(1/beta) log( sum_k exp(-beta * vals_k) )
    # d(m_beta)/d(vals_k) = -(1/beta) * (1/sum_exps) * d(sum_exps)/d(vals_k)
    # d(sum_exps)/d(vals_k) = sum_m d(exp(-beta*vals_m))/d(vals_k)
    # Only m=k term is non-zero: exp(-beta*vals_k) * (-beta)
    # So d(m_beta)/d(vals_k) = -(1/beta) * (1/sum_exps) * (-beta * exp(-beta*vals_k))
    #                         = exp(-beta*vals_k) / sum_exps
    
    # Let weights_k = exp(-beta * vals_k) / sum_exps
    # Note: this is a probability distribution over the constraints.
    # The constraint with the smallest value (largest -beta*val, so largest exp) gets the highest weight.
    
    # We computed exps = exp(-beta * (vals - v_min)) = exp(-beta*vals) / exp(-beta*v_min)
    # sum_exps = sum exps
    # weights_k = exp(-beta*vals_k) / sum(exp(-beta*vals_m))
    #           = (exps_k * exp(-beta*v_min)) / (sum_exps * exp(-beta*v_min))
    #           = exps_k / sum_exps
    
    weights = exps / sum_exps
    
    # Total gradient wrt x
    total_grad = np.zeros(2*N)
    for k in range(len(vals)):
        total_grad += weights[k] * grads[k]
        
    # We want to maximize m_beta, so minimize -m_beta
    obj_val = -m_beta
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
    # Start with a moderate beta and increase it in stages to refine the solution
    # This "annealing" of the soft-min parameter helps escape local minima of the smooth approximation
    # and converge to the true min.
    
    best_x = x0
    best_score = -1e9
    
    # Calculate initial score
    c = x0.reshape(N, 2)
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

    # Try a sequence of betas
    # Low beta: smooth, easy to optimize, but poor approximation of min
    # High beta: sharp, good approximation, but hard to optimize (narrow valleys)
    betas = [5.0, 10.0, 20.0, 40.0, 80.0]
    
    current_x = x0.copy()
    
    for beta in betas:
        if time.time() > deadline:
            break
            
        try:
            res = minimize(
                _objective_and_grad,
                current_x,
                method='L-BFGS-B',
                jac=True,
                args=(beta,),
                options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            # Check if the new solution is valid and better
            new_x = res.x
            c_new = new_x.reshape(N, 2)
            
            # Check validity
            valid = True
            for i in range(N):
                if np.linalg.norm(c_new[i]) > 1.0:
                    valid = False
                    break
            
            if not valid:
                continue
                
            # Calculate actual score
            min_dist = np.inf
            for i in range(N):
                for j in range(i+1, N):
                    d = np.linalg.norm(c_new[i] - c_new[j])
                    if d < min_dist:
                        min_dist = d
                d_cont = 1.0 - np.linalg.norm(c_new[i])
                if d_cont < min_dist:
                    min_dist = d_cont
            actual_r = min_dist / 2.0
            
            if actual_r > best_score:
                best_score = actual_r
                best_x = new_x
                current_x = new_x.copy() # Use the best found so far as the starting point for the next beta
                
        except Exception as e:
            # If optimization fails, keep the current best
            pass

    # If we have time, we could try a few random restarts with the best beta?
    # But the budget is tight. The annealing approach above should be sufficient.
    
    # Ensure validity of the final result
    c = best_x.reshape(N, 2)
    
    return c
