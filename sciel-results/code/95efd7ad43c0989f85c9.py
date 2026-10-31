import numpy as np
from scipy.optimize import minimize

N = 100

def construct():
    # Initial guess: alternating 0.5? Or the current golden ratio block?
    # Let's use the current block as a good start, but perturb it slightly or use it directly.
    # The current score is 1.0590. The seed is 1.0.
    # If we just optimize, we might find a better one.
    
    # Let's try to optimize the full vector.
    # Objective: Minimize M
    # Variables: h (N), M (1)
    
    def objective(x):
        h = x[:N]
        M = x[N]
        return M

    def constraints(x):
        h = x[:N]
        M = x[N]
        cons = []
        # Equality constraint: sum(h) = N/2
        cons.append({
            'type': 'eq',
            'fun': lambda x: np.sum(x[:N]) - N/2
        })
        # Inequality constraints: O_k(h) <= M for all k
        # O_k(h) = sum(h_i * (1 - h_{(i+k)%N}))
        # We can vectorize this or loop. N=100, so loop is fine.
        for k in range(1, N):
            shifted_h = np.roll(h, -k)
            # O_k = sum(h * (1 - shifted_h))
            # Constraint: sum(h * (1 - shifted_h)) - M <= 0
            # Since M is a variable, this is linear in M, quadratic in h.
            # scipy minimize handles this.
            cons.append({
                'type': 'ineq',
                'fun': lambda x, k=k: M - np.sum(x[:N] * (1 - np.roll(x[:N], -k)))
            })
        return cons

    # Initial guess
    # Use the golden ratio block as a strong initial guess
    c = (np.sqrt(5) - 1) / 2
    h0 = np.zeros(N)
    h0[:N//2] = c
    h0[N//2:] = 1 - c
    
    # Calculate initial M
    max_overlap = 0
    for k in range(1, N):
        o_k = np.sum(h0 * (1 - np.roll(h0, -k)))
        if o_k > max_overlap:
            max_overlap = o_k
            
    x0 = np.concatenate([h0, [max_overlap + 0.1]]) # Add buffer to M
    
    # Bounds
    bounds = [(0, 1)] * N + [(0, 1)] # M is bounded by 1 (max overlap is 1)
    
    # Options
    options = {
        'maxiter': 1000,
        'ftol': 1e-9
    }
    
    # Run optimization
    try:
        res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints, options=options)
        if res.success:
            h_opt = res.x[:N]
            # Clip to ensure bounds are respected due to numerical errors
            h_opt = np.clip(h_opt, 0, 1)
            # Re-normalize? The constraint should handle it, but let's check sum.
            # If sum is not exactly N/2, we might have issues.
            # The equality constraint should force it.
            return h_opt
        else:
            # If optimization fails, return the initial guess or a fallback
            return h0
    except Exception:
        return h0
