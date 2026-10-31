import numpy as np
from scipy.optimize import minimize
import time

N = 21

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    best_sum_r = 0.0
    best_params = None
    
    # Number of restarts
    n_restarts = 20
    
    # Objective and constraints for SLSQP
    # Variables: x[0:N], y[0:N], r[0:N], w (width)
    # h = 2 - w
    # Total variables: 3*N + 1 = 64
    
    def objective(v):
        x = v[:N]
        y = v[N:2*N]
        r = v[2*N:3*N]
        w = v[3*N]
        h = 2.0 - w
        # We want to maximize sum(r), so minimize -sum(r)
        return -np.sum(r)
    
    def constraints(v):
        x = v[:N]
        y = v[N:2*N]
        r = v[2*N:3*N]
        w = v[3*N]
        h = 2.0 - w
        
        cons = []
        
        # Pairwise constraints: r_i + r_j <= dist_ij
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                # Constraint: r_i + r_j - dist <= 0
                cons.append({'type': 'ineq', 'fun': lambda v, i=i, j=j: 
                             (v[i] + v[N+i] + v[2*N+i] + v[2*N+j] - np.sqrt((v[i]-v[N+j])**2 + (v[N+i]-v[N+j])**2)) if False else 
                             v[2*N+i] + v[2*N+j] - np.sqrt((v[i]-v[N+j])**2 + (v[N+i]-v[N+j])**2)
                             # Note: SLSQP fun returns g(v) >= 0 for ineq? 
                             # No, scipy minimize ineq is g(v) >= 0.
                             # So we need dist - (r_i + r_j) >= 0.
                             # Let's correct: fun returns dist - (r_i + r_j)
                             })
        
        # This lambda closure in loop is slow and error-prone. Let's use a vectorized approach or pre-allocate.
        # Actually, defining constraints as a list of dicts with lambdas is standard but can be slow if not careful.
        # Let's rewrite constraints properly.
        pass

    # Better approach: define constraint functions that take v and return array or scalar.
    # SLSQP accepts a list of constraint dicts.
    
    def make_constraints():
        cons = []
        # Pairwise
        for i in range(N):
            for j in range(i+1, N):
                def f(v, i=i, j=j):
                    dx = v[i] - v[N+j]
                    dy = v[N+i] - v[N+j]
                    dist = np.sqrt(dx*dx + dy*dy)
                    return dist - (v[2*N+i] + v[2*N+j])
                cons.append({'type': 'ineq', 'fun': f})
        
        # Boundary
        for i in range(N):
            # x_i >= r_i  => x_i - r_i >= 0
            def f_left(v, i=i):
                return v[i] - v[2*N+i]
            cons.append({'type': 'ineq', 'fun': f_left})
            
            # y_i >= r_i
            def f_bottom(v, i=i):
                return v[N+i] - v[2*N+i]
            cons.append({'type': 'ineq', 'fun': f_bottom})
            
            # w - x_i >= r_i => w - x_i - r_i >= 0
            def f_right(v, i=i):
                return v[3*N] - v[i] - v[2*N+i]
            cons.append({'type': 'ineq', 'fun': f_right})
            
            # h - y_i >= r_i => (2-w) - y_i - r_i >= 0
            def f_top(v, i=i):
                return (2.0 - v[3*N]) - v[N+i] - v[2*N+i]
            cons.append({'type': 'ineq', 'fun': f_top})
            
            # r_i >= 0.01
            def f_min_r(v, i=i):
                return v[2*N+i] - 0.01
            cons.append({'type': 'ineq', 'fun': f_min_r})
            
            # w >= 0.5, w <= 1.5 (reasonable bounds)
            # We will handle w bounds via 'bounds' in minimize
            
        return cons

    constraints = make_constraints()
    
    # Bounds for variables
    # x, y, r >= 0. w in [0.5, 1.5]
    bounds = [(0.0, 2.0) for _ in range(2*N)] + [(0.01, 1.0) for _ in range(N)] + [(0.5, 1.5)]
    
    # Initial guesses
    results = []
    
    for k in range(n_restarts):
        if time.time() > deadline:
            break
            
        # Generate initial guess
        # Grid based
        side = 5
        x0 = np.zeros(N)
        y0 = np.zeros(N)
        r0 = np.zeros(N)
        
        # Place in 5x5 grid in a 1x1 square initially
        for i in range(N):
            xi = i % side
            yi = i // side
            x0[i] = (xi + 0.5) / side
            y0[i] = (yi + 0.5) / side
            r0[i] = 0.5 / side - 0.01
        
        w0 = 1.0
        
        # Add random jitter
        if k > 0:
            jitter_scale = 0.1
            x0 += np.random.uniform(-jitter_scale, jitter_scale, N)
            y0 += np.random.uniform(-jitter_scale, jitter_scale, N)
            # Ensure they stay roughly in bounds
            x0 = np.clip(x0, 0.05, 1.95)
            y0 = np.clip(y0, 0.05, 1.95)
            
            # Random width
            w0 = np.random.uniform(0.7, 1.3)
        
        v0 = np.concatenate([x0, y0, r0, [w0]])
        
        try:
            res = minimize(objective, v0, method='SLSQP', bounds=bounds, constraints=constraints, 
                           options={'maxiter': 1000, 'ftol': 1e-6})
            
            if res.success:
                sum_r = -res.fun
                if sum_r > best_sum_r:
                    best_sum_r = sum_r
                    best_params = res.x
        except Exception as e:
            pass

    # If no valid result found, fallback to seed
    if best_params is None:
        side = 5
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    # Extract best result
    x = best_params[:N]
    y = best_params[N:2*N]
    r = best_params[2*N:3*N]
    
    # Final check and cleanup
    # Ensure radii are positive and circles are inside
    # The optimizer should have handled this, but let's be safe.
    # If any constraint is violated slightly due to numerical error, we might need to shrink radii.
    # However, SLSQP is quite accurate.
    
    out = np.column_stack((x, y, r))
    
    # One final polish pass with tighter tolerance if time permits
    if time.time() < deadline - 10:
        # Use the best result as start for a tighter optimization
        try:
            res2 = minimize(objective, best_params, method='SLSQP', bounds=bounds, constraints=constraints, 
                           options={'maxiter': 2000, 'ftol': 1e-8})
            if res2.success and -res2.fun > best_sum_r:
                x = res2.x[:N]
                y = res2.x[N:2*N]
                r = res2.x[2*N:3*N]
                out = np.column_stack((x, y, r))
        except:
            pass

    return out
