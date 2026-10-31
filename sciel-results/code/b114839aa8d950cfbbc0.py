import numpy as np
from scipy.optimize import minimize

N = 100

def construct():
    # Use Fourier series parameterization
    # h_i = 0.5 + sum_{j=1}^{J} A_j * cos(2*pi*j*i/N)
    # This ensures sum(h) = 0.5 * N automatically (if no DC term added)
    # We need to ensure h_i in [0, 1]
    
    J = 10
    # Initial guess: small coefficients
    A0 = np.zeros(J)
    
    def get_h(A):
        i = np.arange(N)
        h = np.full(N, 0.5)
        for j in range(1, J+1):
            h += A[j-1] * np.cos(2 * np.pi * j * i / N)
        return h

    def calc_c(h):
        cs = np.zeros(N)
        for k in range(N):
            if k == 0:
                r = np.sum(h * h)
            else:
                r = np.sum(h[k:] * h[:-k])
            cs[k] = (2.0 / N) * (N / 2.0 - r)
        return cs

    def objective(A):
        h = get_h(A)
        # Check bounds
        if np.any(h < -1e-6) or np.any(h > 1 + 1e-6):
            # Penalty for violation
            viol = np.sum(np.maximum(0, -h)**2) + np.sum(np.maximum(0, h-1)**2)
            return 1e6 * viol
        cs = calc_c(h)
        return np.max(cs)

    # Constraints: h_i in [0, 1]
    # We can use bounds on A? Hard to map directly.
    # Use penalty method or constraints on A?
    # Let's use SLSQP with constraints on h_i?
    # SLSQP can handle equality/inequality constraints.
    
    def cons_lower(A):
        h = get_h(A)
        return h - 1e-9 # h >= 0

    def cons_upper(A):
        h = get_h(A)
        return 1 - h - 1e-9 # h <= 1

    constraints = []
    # Adding N constraints might be slow but N=100 is small.
    # However, SLSQP with 100 constraints might be slow.
    # Let's try to bound A directly?
    # If we keep A small, h stays in [0,1].
    # Max amplitude of sum of cosines is sum |A_j|.
    # So if sum |A_j| <= 0.5, then h in [0,1].
    # This is a sufficient condition.
    # Let's optimize A with bounds |A_j| <= 0.5/J?
    
    # Let's use a simpler objective that includes the constraint check
    # and just minimize max C.
    
    # Better: Use differential evolution or just a simple grid search?
    # No, let's use minimize with penalty.
    
    res = minimize(objective, A0, method='Nelder-Mead', options={'maxiter': 2000, 'xatol': 1e-8})
    
    if res.success:
        A_opt = res.x
    else:
        A_opt = A0
        
    h_opt = get_h(A_opt)
    
    # Clip to ensure validity
    h_opt = np.clip(h_opt, 0, 1)
    
    # Adjust sum to exactly N/2
    diff = N / 2.0 - np.sum(h_opt)
    if abs(diff) > 1e-9:
        # Distribute difference evenly
        h_opt += diff / N
        h_opt = np.clip(h_opt, 0, 1)
        
    return h_opt
