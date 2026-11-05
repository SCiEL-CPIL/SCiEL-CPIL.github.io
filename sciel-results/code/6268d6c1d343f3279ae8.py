"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
from scipy.optimize import milp, LinearConstraint, Bounds
import numpy as np

def construct():
    start_time = time.time()
    
    # Denominators are 2 to 48 inclusive
    denominators = list(range(2, 49))
    n = len(denominators)
    
    # Calculate LCM of denominators to convert to integer arithmetic
    from math import gcd
    lcm_val = 1
    for d in denominators:
        lcm_val = lcm_val * d // gcd(lcm_val, d)
        
    # Weights w_i = LCM / d_i
    # Constraint: sum(w_i * x_i) <= LCM - 1
    # Objective: Maximize sum(w_i * x_i)  => Minimize -sum(w_i * x_i)
    
    w = np.array([lcm_val // d for d in denominators], dtype=np.int64)
    
    # Objective coefficients for minimization: -w
    c = -w
    
    # Constraint: w @ x <= LCM - 1
    A = w.reshape(1, -1)
    b_ub = lcm_val - 1
    b_lb = 0 # sum is non-negative
    
    # Variables x_i are binary
    integrality = np.ones(n)
    
    # Bounds for x_i: 0 <= x_i <= 1
    lb = np.zeros(n)
    ub = np.ones(n)
    
    # Solve ILP
    try:
        res = milp(
            c=c,
            constraints=LinearConstraint(A, lb, b_ub),
            integrality=integrality,
            bounds=Bounds(lb, ub)
        )
        
        if res.status == 0 and res.x is not None:
            # res.x contains the optimal values for x_i
            # Since integrality is 1, they should be 0 or 1
            # We select indices where x_i > 0.5
            selected_indices = [i for i, val in enumerate(res.x) if val > 0.5]
            
            # Verify strict inequality and non-empty
            if selected_indices:
                total = sum(Fraction(1, denominators[i]) for i in selected_indices)
                if total < 1:
                    return selected_indices
            
            # Fallback if ILP failed or returned empty (should not happen)
            # Use greedy as fallback
            selected, total = [], Fraction()
            for i, d in enumerate(denominators):
                if total + Fraction(1, d) < 1:
                    selected.append(i)
                    total += Fraction(1, d)
            return selected
            
        else:
            # Fallback to greedy if solver fails
            selected, total = [], Fraction()
            for i, d in enumerate(denominators):
                if total + Fraction(1, d) < 1:
                    selected.append(i)
                    total += Fraction(1, d)
            return selected
            
    except Exception:
        # Fallback to greedy if exception occurs
        selected, total = [], Fraction()
        for i, d in enumerate(denominators):
            if total + Fraction(1, d) < 1:
                selected.append(i)
                total += Fraction(1, d)
        return selected

# EVOLVE-BLOCK-END
