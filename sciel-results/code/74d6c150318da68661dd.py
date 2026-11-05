import numpy as np
import time
import sys

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s margin
    
    # We want to minimize V = 2*N*max(c)/S^2
    # Scale invariant, so fix S = 1.
    # Minimize t
    # s.t. c_k(h) <= t for all k
    #      sum(h) = 1
    
    # c_k(h) is a quadratic form.
    # We use scipy.optimize.minimize
    
    # Initial guess: Uniform
    h0 = np.ones(N) / N
    
    # Variables: x = [h, t]
    # x[0:N] = h
    # x[N] = t
    
    def objective(x):
        return x[N]
    
    def constraints(x):
        h = x[:N]
        t = x[N]
        # Compute autocorrelation
        c = np.convolve(h, h)
        # Inequality constraints: c_k - t <= 0
        # scipy expects f(x) <= 0
        return c - t
    
    # Equality constraint: sum(h) - 1 = 0
    def eq_constraint(x):
        return np.sum(x[:N]) - 1.0

    # Use trust-constr for general constraints
    # Note: trust-constr might be slow for 799 constraints.
    # Alternative: SLSQP.
    
    # Let's try SLSQP first, it's often faster for smaller N.
    # But 799 constraints might be too many for SLSQP's default settings.
    
    # We can reduce the number of constraints.
    # The max is likely near k=0 or symmetric.
    # But we don't know for sure.
    
    # Let's try a heuristic:
    # The optimal h is often related to the eigenvector of the circulant matrix?
    # No, it's a convex problem.
    
    # If time allows, run minimize.
    # Otherwise, return seed.
    
    try:
        # Reduce constraints to speed up?
        # Only check k=0 to N-1? No, full conv is 2N-1.
        # But c_{2N-1-k} = c_k.
        # So we only need k=0 to N-1.
        # Constraints: c_k - t <= 0 for k in 0..N-1.
        
        def constraints_reduced(x):
            h = x[:N]
            t = x[N]
            c = np.convolve(h, h)
            # c has length 2N-1
            # Symmetric: c[0:N] covers the unique values
            return c[:N] - t
        
        res = None
        # Try SLSQP
        try:
            res = scipy_optimize_minimize(objective, x0=np.append(h0, 1.0/N), 
                                          method='SLSQP', 
                                          constraints=[{'type': 'eq', 'fun': eq_constraint},
                                                       {'type': 'ineq', 'fun': lambda x: -constraints_reduced(x)}],
                                          options={'maxiter': 100, 'ftol': 1e-8})
            if time.time() > deadline:
                return h0
        except Exception:
            pass
        
        if res is None or not res.success:
            # Try trust-constr
            try:
                from scipy.optimize import LinearConstraint, NonlinearConstraint
                # Define constraint bounds
                # c_k - t <= 0 => -c_k + t >= 0
                # This is non-linear.
                
                # trust-constr with NonlinearConstraint
                # fun(x) >= lb, fun(x) <= ub
                # We want c_k - t <= 0 => t - c_k >= 0
                def t_minus_c(x):
                    h = x[:N]
                    t = x[N]
                    c = np.convolve(h, h)
                    return t - c[:N]
                
                nl_con = NonlinearConstraint(t_minus_c, lb=0, ub=np.inf)
                lin_con = LinearConstraint(np.eye(N), lb=1, ub=1) # sum(h)=1
                
                res = scipy_optimize_minimize(objective, x0=np.append(h0, 1.0/N),
                                              method='trust-constr',
                                              constraints=[lin_con, nl_con],
                                              options={'maxiter': 100, 'gtol': 1e-8})
            except Exception:
                pass
        
        if res is not None and res.success:
            h_opt = res.x[:N]
            # Normalize to ensure sum is 1 (safety)
            s = np.sum(h_opt)
            if abs(s) > 1e-10:
                h_opt = h_opt / s
            # Check value
            c = np.convolve(h_opt, h_opt)
            max_c = np.max(c)
            val = 2 * N * max_c / (np.sum(h_opt)**2)
            # If val < 2.0, we improved.
            # If val >= 2.0, return seed (or h_opt if it's valid).
            if val < 2.0:
                return h_opt
            else:
                return h0
        else:
            return h0
            
    except Exception:
        return h0

# Need to import scipy.optimize
import scipy.optimize as scipy_optimize_minimize

# Re-define construct to use the imported module properly
def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    h0 = np.ones(N) / N
    
    def objective(x):
        return x[N]
    
    def eq_constraint(x):
        return np.sum(x[:N]) - 1.0
        
    def ineq_constraint(x):
        h = x[:N]
        t = x[N]
        c = np.convolve(h, h)
        return -(c[:N] - t) # We want c_k - t <= 0, so t - c_k >= 0
        
    try:
        x0 = np.append(h0, 1.0/N)
        cons = [
            {'type': 'eq', 'fun': eq_constraint},
            {'type': 'ineq', 'fun': ineq_constraint}
        ]
        
        res = scipy_optimize_minimize.minimize(objective, x0, method='SLSQP', constraints=cons, options={'maxiter': 200, 'ftol': 1e-8})
        
        if res.success:
            h_opt = res.x[:N]
            s = np.sum(h_opt)
            if abs(s) > 1e-10:
                h_opt = h_opt / s
            c = np.convolve(h_opt, h_opt)
            val = 2 * N * np.max(c) / (s**2)
            if val < 2.0:
                return h_opt
        return h0
    except Exception:
        return h0
