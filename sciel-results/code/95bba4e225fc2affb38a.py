import numpy as np
import time
from scipy.optimize import minimize

N = 50

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    # The objective is to maximize value = l2 / (norm_1 * norm_inf)
    # where:
    # g = np.convolve(h, h)
    # m = len(g) = 2*N - 1
    # y = [0, g[0], ..., g[m-1], 0] (length m+2)
    # points = np.linspace(-0.5, 0.5, m+2)
    # w = points[1] - points[0]
    # l2 = sum_{i=0}^{m} (w/3) * (y[i]^2 + y[i]*y[i+1] + y[i+1]^2)
    # norm_1 = sum(abs(g)) / (m+1)
    # norm_inf = max(abs(g))
    # value = l2 / (norm_1 * norm_inf)

    # Note: h are nonnegative, so g = h*h (convolution) is nonnegative.
    # Thus norm_1 = sum(g)/(m+1) and norm_inf = max(g).
    
    # Let's analyze the structure.
    # If we scale h by c, g scales by c^2.
    # l2 scales by c^4 (since y^2 terms are squared in the sum? No, y is linear in g, so y^2 is quadratic in g, which is quadratic in h. So y^2 is quartic in h? Wait.
    # g ~ h^2.
    # y ~ g ~ h^2.
    # y^2 ~ h^4.
    # l2 ~ h^4.
    # norm_1 ~ h^2.
    # norm_inf ~ h^2.
    # value ~ h^4 / (h^2 * h^2) = constant.
    # So the value is scale-invariant!
    # This means we can fix a normalization, e.g., norm_inf = 1 or sum(h) = 1.
    # Let's fix norm_inf = 1, i.e., max(g) = 1.
    # Or simpler, fix sum(h) = 1.
    
    # Let's use a log-sum-exp trick or just standard optimization.
    # Since it's scale invariant, we can maximize log(value) or just value with a normalization constraint.
    # Constraint: h >= 0.
    
    # Let's try to maximize value directly, but with a normalization to avoid trivial scaling issues.
    # We can set max(h) = 1 or sum(h) = 1.
    # Let's use sum(h) = 1.
    
    # Variables: h (length 50), h >= 0, sum(h) = 1.
    # Objective: -value(h)
    
    # To handle the non-smoothness of max and abs (though g is nonnegative so abs is not an issue, but max is),
    # we can use a smooth approximation for max if needed, but scipy's L-BFGS-B might handle it okay if we are careful.
    # Actually, max(g) is differentiable almost everywhere.
    
    # Let's implement the objective function.
    
    def objective(h):
        # h is a vector of length N, nonnegative.
        # Check constraints (handled by bounds and penalty if needed, but L-BFGS-B handles bounds)
        # Normalization: we will enforce sum(h) = 1 in the optimizer or project.
        # For now, let's assume h is normalized.
        
        g = np.convolve(h, h)
        m = len(g)
        y = np.zeros(m + 2)
        y[1:m+1] = g
        
        points = np.linspace(-0.5, 0.5, m + 2)
        w = points[1] - points[0]
        
        # l2 = sum_{i=0}^{m} (w/3) * (y[i]^2 + y[i]*y[i+1] + y[i+1]^2)
        # This can be written as (w/3) * sum_{i=0}^{m} (y[i]^2 + y[i]*y[i+1] + y[i+1]^2)
        # Note: y[0]=0, y[m+1]=0.
        # Let's compute efficiently.
        # sum y[i]^2 + y[i]*y[i+1] + y[i+1]^2
        # = sum y[i]^2 + sum y[i]*y[i+1] + sum y[i+1]^2
        # = 2 * sum y[i]^2 - y[0]^2 - y[m+1]^2 + sum y[i]*y[i+1]  (since y[0]=y[m+1]=0, it's just 2*sum y^2 + sum y[i]y[i+1])
        # Wait, indices:
        # i=0..m
        # y[0]^2 + y[0]y[1] + y[1]^2
        # y[1]^2 + y[1]y[2] + y[2]^2
        # ...
        # y[m]^2 + y[m]y[m+1] + y[m+1]^2
        # Sum = y[0]^2 + 2*sum_{i=1}^{m} y[i]^2 + y[m+1]^2 + sum_{i=0}^{m} y[i]y[i+1]
        # Since y[0]=y[m+1]=0:
        # Sum = 2*sum_{i=1}^{m} y[i]^2 + sum_{i=0}^{m} y[i]y[i+1]
        # = 2*sum(g^2) + sum(g[i]*g[i+1])  (since y[1..m] = g[0..m-1])
        # Wait, y[1]=g[0], ..., y[m]=g[m-1].
        # sum_{i=1}^{m} y[i]^2 = sum_{j=0}^{m-1} g[j]^2 = sum(g^2)
        # sum_{i=0}^{m} y[i]y[i+1] = y[0]y[1] + ... + y[m]y[m+1] = 0 + sum_{j=0}^{m-2} g[j]g[j+1] + 0 = sum(g[:-1]*g[1:])
        
        sum_g2 = np.sum(g**2)
        sum_g_g1 = np.sum(g[:-1] * g[1:])
        
        l2 = (w / 3.0) * (2.0 * sum_g2 + sum_g_g1)
        
        norm_1 = np.sum(g) / (m + 1.0)
        norm_inf = np.max(g)
        
        if norm_1 == 0 or norm_inf == 0:
            return 0.0
            
        value = l2 / (norm_1 * norm_inf)
        return -value  # Minimize negative value

    # We need to handle the constraint sum(h) = 1.
    # L-BFGS-B doesn't handle equality constraints directly.
    # We can use SLSQP or trust-constr.
    # Or we can reparameterize h = exp(x) / sum(exp(x)) to ensure non-negativity and normalization.
    # Let's use the softmax parameterization.
    
    def objective_softmax(x):
        # x is unconstrained
        h = np.exp(x - np.max(x))  # for stability
        h = h / np.sum(h)
        return objective(h)

    # Initial guess: uniform h
    x0 = np.zeros(N)
    
    # Try a few different starting points
    best_val = 0.0
    best_h = np.ones(N) / N
    
    # Seed value
    h_seed = np.ones(N) / N
    g_seed = np.convolve(h_seed, h_seed)
    m_seed = len(g_seed)
    y_seed = np.zeros(m_seed + 2)
    y_seed[1:m_seed+1] = g_seed
    points_seed = np.linspace(-0.5, 0.5, m_seed + 2)
    w_seed = points_seed[1] - points_seed[0]
    sum_g2_seed = np.sum(g_seed**2)
    sum_g_g1_seed = np.sum(g_seed[:-1] * g_seed[1:])
    l2_seed = (w_seed / 3.0) * (2.0 * sum_g2_seed + sum_g_g1_seed)
    norm_1_seed = np.sum(g_seed) / (m_seed + 1.0)
    norm_inf_seed = np.max(g_seed)
    value_seed = l2_seed / (norm_1_seed * norm_inf_seed)
    
    best_val = value_seed
    best_h = h_seed.copy()
    
    # Optimization
    # Use SLSQP with bounds on h, and equality constraint sum(h)=1
    # Or use the softmax trick with L-BFGS-B.
    
    # Let's try SLSQP first, it's more robust for this kind of problem.
    
    constraints = [{'type': 'eq', 'fun': lambda h: np.sum(h) - 1.0}]
    bounds = [(0.0, 1.0) for _ in range(N)]
    
    # Start with uniform
    h0 = np.ones(N) / N
    
    # Run optimization
    try:
        res = minimize(objective, h0, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 1000, 'ftol': 1e-8})
        if res.success and -res.fun > best_val:
            best_val = -res.fun
            best_h = res.x
    except Exception as e:
        pass
    
    # Try some random initializations
    np.random.seed(42)
    for i in range(10):
        if time.time() > deadline:
            break
        h0 = np.random.dirichlet(np.ones(N))
        try:
            res = minimize(objective, h0, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 500, 'ftol': 1e-6})
            if res.success and -res.fun > best_val:
                best_val = -res.fun
                best_h = res.x
        except Exception as e:
            pass
    
    # Refine with Nelder-Mead if possible? Or just return best_h.
    # The problem is small (50 variables), so SLSQP should be fine.
    
    # Return the best h found
    return list(best_h)
# EVOLVE-BLOCK-END
