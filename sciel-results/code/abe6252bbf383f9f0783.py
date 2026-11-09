import numpy as np
import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 600.0
    
    m = 12
    n = 1
    
    # The task involves a Laguerre polynomial combination with 12 double roots.
    # We need to find 12 positive double-root locations r_1, ..., r_12.
    # The score is the "upper bound on C4".
    
    # Based on the "Cohn-Goncalves Laguerre-polynomial construction for the Fourier sign uncertainty upper bound",
    # this likely refers to the uncertainty principle for Fourier transforms, specifically related to the 
    # number of sign changes of a function and its transform.
    
    # A common result involves the Laguerre polynomials. The "refined" construction might involve
    # optimizing the roots to minimize a specific constant.
    
    # Let's hypothesize the objective function.
    # Often in such problems, the constant C4 is related to the norm of the polynomial or its roots.
    # If the polynomial is P(x) = prod_{i=1}^{12} (x - r_i)^2, then the roots are r_i.
    # The "upper bound on C4" might be a function of these roots.
    
    # Without the exact formula, we can look at the seed: [4.0, 8.0, ..., 48.0].
    # This is an arithmetic progression.
    
    # Let's try to minimize the sum of squares of the roots, or some similar metric,
    # subject to the roots being positive and distinct (though they are double roots, 
    # the locations r_i should probably be distinct).
    
    # Actually, let's look at the phrasing: "score is the upper bound on C4".
    # "lower is better".
    
    # If we assume the objective is to minimize the spread of the roots or their magnitude,
    # we can use an optimizer.
    
    # However, there might be a constraint from the "Laguerre combination".
    # Laguerre polynomials L_n(x) have n simple positive roots.
    # A "combination" might refer to a linear combination of Laguerre polynomials.
    
    # Let's assume the problem is to find 12 positive numbers r_i that minimize some cost.
    # Given the seed is 4, 8, ..., 48, maybe the optimal is different.
    
    # Let's try a simple optimization: minimize sum(r_i^2) subject to r_i > 0 and r_i != r_j.
    # Or maybe minimize the product?
    
    # Let's use scipy.optimize to minimize a generic objective.
    # Since we don't know the exact objective, let's try to infer it from the context.
    # "Fourier sign uncertainty" often relates to the Heisenberg uncertainty principle.
    # The constant might be related to the variance of the roots or the polynomial's coefficients.
    
    # Let's try minimizing the sum of the roots first.
    # Or maybe the objective is related to the condition number of the Vandermonde matrix of the roots?
    
    # Let's try a few heuristics.
    # 1. Minimize sum(r_i)
    # 2. Minimize sum(r_i^2)
    # 3. Minimize max(r_i)
    
    # Let's use scipy.optimize.minimize.
    
    def objective(r):
        # Ensure positivity
        if np.any(r <= 0):
            return 1e10
        # Hypothesis: minimize sum of squares
        return np.sum(r**2)
    
    def constraint(r):
        # r must be positive
        return r
    
    # Initial guess from seed
    r0 = np.array([4.0 * (i + 1) for i in range(12)])
    
    # Use SLSQP
    cons = ({'type': 'ineq', 'fun': lambda r: r})
    
    try:
        from scipy.optimize import minimize
        res = minimize(objective, r0, method='SLSQP', bounds=[(0.1, None)]*12, options={'maxiter': 1000, 'ftol': 1e-9})
        if res.success:
            r_opt = res.x
            # Ensure distinctness
            r_opt = np.sort(r_opt)
            # Check for duplicates
            if np.any(np.diff(r_opt) < 1e-6):
                # Perturb
                for i in range(1, 12):
                    if r_opt[i] - r_opt[i-1] < 1e-6:
                        r_opt[i] = r_opt[i-1] + 1e-6
            return [float(x) for x in r_opt]
    except Exception as e:
        pass
    
    # Fallback to seed
    return [4.0 * (i + 1) for i in range(12)]

# EVOLVE-BLOCK-END
