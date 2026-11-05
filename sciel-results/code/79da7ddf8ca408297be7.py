"""Bermudan put exercise policy via LSMC with polynomial regression."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

def fit_exercise_policy(training_paths, contract):
    """
    Fit a Bermudan put exercise policy using LSMC.
    
    Parameters:
    - training_paths: array of shape (n_paths, n_dates+1) with S[0]=S0, S[1..12] at monthly dates
    - contract: dict with keys like 'K', 'r', 'sigma', 'T', 'n_dates' etc.
    
    Returns:
    - An object with .exercise(current_state, time) method returning boolean vector
    """
    t0 = _time.time()
    DEADLINE = 145.0  # safety margin under 150s
    
    # Extract parameters from contract
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_dates = contract.get('n_dates', 12)
    
    # training_paths shape: (n_paths, n_dates + 1)
    # Path[0] = S0, Path[1..n_dates] = prices at exercise dates
    n_paths, n_cols = training_paths.shape
    # n_cols should be n_dates + 1
    
    # Discount factors for each exercise date
    # Exercise dates are at t_j = j * T / n_dates for j = 1, ..., n_dates
    # No exercise at time 0
    dt = T / n_dates
    discount = np.exp(-r * dt)
    
    # We work backward from j = n_dates down to j = 1
    # At maturity (j = n_dates), the payoff is (K - S)
    # For j < n_dates, continuation = discount * E[max(K-S_{j+1}, cont_{j+1}) | S_j]
    
    # We'll store the continuation value regression coefficients for each date
    # reg_coeffs[j] = coefficients for date j (j from 1 to n_dates-1)
    # For date n_dates, continuation is just the payoff
    
    # Use polynomial regression of degree 4
    poly_degree = 4
    
    # Store continuation values at each path for each date
    # cont[j] = continuation value at date j for each path (for j = 1, ..., n_dates)
    # At maturity, cont[n_dates] = max(K - S[n_dates], 0)
    
    cont = np.zeros((n_dates + 1, n_paths))  # index 0 unused, 1..n_dates
    S = training_paths  # S[:, j] is price at date j
    
    # Maturity: exercise date n_dates
    cont[n_dates] = np.maximum(K - S[:, n_dates], 0.0)
    
    # Work backward
    # reg_coeffs[j] stores the polynomial coefficients for estimating E[cont[j+1] | S_j]
    reg_coeffs = [None] * (n_dates + 1)  # index 1..n_dates-1
    
    for j in range(n_dates - 1, 0, -1):
        # Check deadline
        if _time.time() - t0 > DEADLINE - 10:
            # Use simple threshold as fallback
            break
        
        # Target: discounted continuation value at next date
        # For each path, the "optimal" value at j+1 is max(K - S_{j+1}, cont[j+1])
        # But in LSMC, we use the actual realized value assuming optimal exercise going forward
        # Actually, in standard LSMC:
        #   continuation_j = discount * E[ max(K - S_{j+1}, cont_{j+1}) | S_j ]
        # where cont_{j+1} is the value function at j+1 (not the pathwise value)
        # 
        # Standard approach:
        #   At each step, we have "target" = discount * max(K - S_{j+1}, cont_{j+1})
        #   where cont_{j+1} is the estimated value from the regression at j+1
        #   Then we regress target on S_j to get continuation at j
        
        # The target for regression at date j:
        # For each path, compute the value at j+1 assuming optimal exercise
        # value_{j+1} = max(K - S_{j+1}, cont_{j+1})
        # where cont_{j+1} is the continuation value (from regression or payoff at maturity)
        
        # For j+1 = n_dates: value = max(K - S_{n_dates}, 0) = cont[n_dates]
        # For j+1 < n_dates: value = max(K - S_{j+1}, reg_{j+1}(S_{j+1}))
        
        # Compute the "realized value" at j+1 for each path
        if j + 1 == n_dates:
            value_next = cont[j + 1]  # = max(K - S_{j+1}, 0)
        else:
            # value_next = max(K - S_{j+1}, cont[j+1])
            # cont[j+1] should already be the estimated value function evaluated at S_{j+1}
            # But in our backward pass, cont[j+1] is the pathwise continuation estimate
            # Actually, let me reconsider...
            
            # In LSMC, the standard approach is:
            # 1. At maturity, V_N = max(K - S_N, 0)
            # 2. For j = N-1, ..., 1:
            #    a. Compute "exercise value" at j+1: E_{j+1} = max(K - S_{j+1}, 0)
            #    b. The "continuation" target uses the value function, not pathwise
            #    
            #    Actually, the correct LSMC procedure:
            #    - We maintain a "value" array that represents the estimated V_j(S_j) for each path
            #    - At maturity: value = max(K - S_N, 0)
            #    - Going backward:
            #      * The target for regression at j is: discount * value_{j+1}
            #        where value_{j+1} = max(K - S_{j+1}, V_{j+1}(S_{j+1}))
            #      * But V_{j+1}(S_{j+1}) is what we're trying to estimate...
            #
            #    The standard practical LSMC:
            #    - Start from maturity with V_N(S) = max(K-S, 0)
            #    - For j = N-1 down to 1:
            #      * For each path, compute "exercise at j+1" value: max(K - S_{j+1}, 0)
            #      * But we need the continuation value at j+1, which is V_{j+1}(S_{j+1})
            #      * Since we've already fit V_{j+1}, we evaluate it: V_{j+1}(S_{j+1})
            #      * Then "optimal value at j+1" = max(K - S_{j+1}, V_{j+1}(S_{j+1}))
            #      * Target for regression at j: discount * optimal_value_at_{j+1}
            #      * Regress target on S_j to get V_j(S_j)
            #      * Exercise at j if K - S_j > V_j(S_j)
            
            # So I need to evaluate the fitted regression at j+1 for each path
            # Let me restructure: store the regression objects/coefficients
            
            # Evaluate V_{j+1} at S_{j+1} using the fitted polynomial
            # reg_coeffs[j+1] should be the polynomial coefficients
            if reg_coeffs[j + 1] is not None:
                # Evaluate polynomial at S[:, j+1]
                S_next = S[:, j + 1]
                V_next = np.polyval(reg_coeffs[j + 1], S_next)
                value_next = np.maximum(K - S_next, V_next)
            else:
                value_next = cont[j + 1]
        
        # Target for regression at date j
        target = discount * value_next
        
        # Fit polynomial regression: target ~ S_j
        S_j = S[:, j]
        
        # Use np.polyfit
        try:
            coeffs = np.polyfit(S_j, target, poly_degree)
        except:
            # Fallback to linear
            try:
                coeffs = np.polyfit(S_j, target, 1)
            except:
                coeffs = np.array([0.0, K / 2.0])  # constant
        
        reg_coeffs[j] = coeffs
        
        # Store the continuation value for this date (for potential use)
        cont[j] = np.polyval(coeffs, S_j)
    
    # Now create the exercise policy
    # At exercise date j (1-indexed, j=1..12):
    #   Exercise if K - S_j > V_j(S_j)
    #   where V_j(S_j) is the continuation value from the regression
    #   For j = n_dates, V_j = max(K - S_j, 0), so always exercise (it's the last chance)
    
    class ExercisePolicy:
        def __init__(self, coeffs_list, K_val, n_d):
            self.coeffs = coeffs_list  # list of length n_dates+1, index 1..n_dates-1 have polynomials
            self.K = K_val
            self.n_dates = n_d
        
        def exercise(self, current_state, time):
            """
            current_state: array of spot prices at the current time
            time: the current time (or date index)
            Returns: boolean array, True if exercise
            """
            # Determine which exercise date we're at
            # time could be a float or an index
            # Based on the task: 12 monthly exercise dates, no time-0 exercise
            # Exercise dates are at t = 1/12, 2/12, ..., 12/12
            
            # Convert time to date index
            if isinstance(time, (int, np.integer)):
                j = int(time)
            else:
                # time is a float, convert to nearest date index
                j = int(round(time * self.n_dates))
            
            # Clamp j to valid range
            j = max(1, min(j, self.n_dates))
            
            S_current = np.asarray(current_state, dtype=float)
            
            if j == self.n_dates:
                # Last exercise date: always exercise if in the money
                return (self.K - S_current) > 0
            
            # Get the regression coefficients for this date
            if j < len(self.coeffs) and self.coeffs[j] is not None:
                # Continuation value
                cont_val = np.polyval(self.coeffs[j], S_current)
                # Exercise if intrinsic > continuation
                intrinsic = np.maximum(self.K - S_current, 0.0)
                return intrinsic > cont_val
            else:
                # Fallback: exercise if in the money
                return (self.K - S_current) > 0
    
    # Check deadline before finalizing
    if _time.time() - t0 > DEADLINE - 5:
        # Return a simple policy as fallback
        class SimplePolicy:
            def __init__(self, K_val):
                self.K = K_val
            def exercise(self, current_state, time):
                S = np.asarray(current_state, dtype=float)
                return (self.K - S) > 0
        return SimplePolicy(K)
    
    return ExercisePolicy(reg_coeffs, K, n_dates)

# EVOLVE-BLOCK-END
