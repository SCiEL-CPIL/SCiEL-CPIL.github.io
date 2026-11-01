import numpy as np
from scipy.optimize import minimize
import time

def construct():
    start_time = time.time()
    deadline = start_time + 170.0  # 10s safety margin

    # Hermite Polynomials (Physicists')
    # H0(x) = 1
    # H4(x) = x^4 - 6x^2 + 3
    # H8(x) = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
    # H12(x) = x^12 - 84x^10 + 2520x^8 - 30240x^6 + 166320x^4 - 302400x^2 + 10395
    # Note: H12(0) = 10395.
    # Wait, standard H_n(0):
    # H0(0)=1
    # H2(0)=-1
    # H4(0)=3
    # H6(0)=-15
    # H8(0)=105
    # H10(0)=-945
    # H12(0)=10395
    
    # Coefficients c0, c4, c8 are free.
    # c12 is derived from g(0)=0.
    # g(0) = c0*1 + c4*3 + c8*105 + c12*10395 = 0
    # c12 = -(c0 + 3*c4 + 105*c8) / 10395
    
    def g(x, c0, c4, c8):
        c12 = -(c0 + 3.0*c4 + 105.0*c8) / 10395.0
        H0 = 1.0
        H4 = x**4 - 6.0*x**2 + 3.0
        H8 = x**8 - 28.0*x**6 + 210.0*x**4 - 420.0*x**2 + 105.0
        H12 = x**12 - 84.0*x**10 + 2520.0*x**8 - 30240.0*x**6 + 166320.0*x**4 - 302400.0*x**2 + 10395.0
        P = c0*H0 + c4*H4 + c8*H8 + c12*H12
        return P * np.exp(-x**2 / 2.0)

    # We want to maximize c0 (to minimize 1/c0 score)
    # Subject to g(x) <= 1 for all x.
    # And presumably g(x) should be "well-behaved".
    
    # Optimization Variables: [c0, c4, c8]
    # Objective: -c0
    # Constraint: max(g(x)) - 1 <= 0
    
    # Sample x for max calculation
    # The function decays fast. Range [-10, 10] is safe.
    x_samples = np.linspace(-10, 10, 1000)
    
    def objective(vars):
        c0, c4, c8 = vars
        return -c0

    def constraint(vars):
        c0, c4, c8 = vars
        vals = g(x_samples, c0, c4, c8)
        return 1.0 - np.max(vals)

    # Initial guess: use seed program values
    c0_init = -113.0 / 100.0
    c4_init = 1.0 / 25.0
    c8_init = 1.0 / 3240.0
    
    # The seed has negative c0. We want positive c0 to maximize it?
    # If c0 is negative, 1/c0 is negative. Score "lower is better".
    # If score is 1/c0, and c0 is negative, score is negative.
    # Is negative score better?
    # "reduce the upper bound".
    # If the bound is C, and we want to reduce C, we want C to be small.
    # If C = 1/c0, and c0 is large positive, C is small positive.
    # If c0 is large negative, C is small negative?
    # Uncertainty constants are positive.
    # So we likely want c0 > 0.
    
    # Let's try to find a solution with c0 > 0.
    # If the seed has c0 < 0, maybe the score is |1/c0|? Or the seed is just a starting point.
    
    # Let's assume we want to maximize c0 > 0.
    
    # Try a few starting points
    best_result = None
    best_score = np.inf
    
    # Starting points
    starts = [
        [1.0, 0.0, 0.0],
        [0.5, 0.1, 0.0],
        [1.0, -0.1, 0.0],
        [0.8, 0.0, 0.1],
        [1.2, 0.0, -0.1],
        [c0_init, c4_init, c8_init] # Seed
    ]
    
    for x0 in starts:
        if time.time() > deadline:
            break
            
        try:
            res = minimize(objective, x0, method='SLSQP', 
                           constraints=[{'type': 'ineq', 'fun': constraint}],
                           options={'maxiter': 100, 'ftol': 1e-8})
            
            if res.success:
                c0_opt, c4_opt, c8_opt = res.x
                # Verify constraint
                if constraint(res.x) >= -1e-5:
                    score = 1.0 / c0_opt
                    if score < best_score:
                        best_score = score
                        best_result = res.x
        except Exception as e:
            pass

    if best_result is None:
        # Fallback to seed if optimization fails
        return np.array([-113 / 100, 1 / 25, 1 / 3240])
    
    return np.array(best_result)
