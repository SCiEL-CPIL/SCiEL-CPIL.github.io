import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from fractions import Fraction
from itertools import combinations

def construct():
    n = 24
    elements = list(range(2, 25))
    valid_constraints_data = []
    seen = set()
    
    # Determine max_size based on min sum
    # Min sum of k largest denominators (smallest reciprocals)
    # If min sum > 0.5, no conflict for a>=2.
    # We know k=9 is ~0.47, k=10 is ~0.52.
    # So max_size = 9 is safe and complete.
    max_size = 9
    
    for size in range(1, max_size + 1):
        for J in combinations(elements, size):
            s = Fraction(0, 1)
            for j in J:
                s += Fraction(1, j)
            
            # Optimization: if s > 1/2, we can't have a conflict for a>=2?
            # Actually, if s > 1/2, 1/a = s implies a < 2.
            # a=1 -> s=1.
            # So if s > 1/2 and s != 1, no conflict.
            # If s == 1, a=1.
            # If s < 1/2, a >= 2.
            
            if s.numerator == 1:
                a = s.denominator
                if 1 <= a <= 24:
                    if a not in J:
                        coeffs = [0] * 24
                        coeffs[a-1] = 1
                        for j in J:
                            coeffs[j-1] += 1
                        key = tuple(coeffs)
                        if key not in seen:
                            seen.add(key)
                            valid_constraints_data.append((coeffs, size))
    
    m = len(valid_constraints_data)
    if m == 0:
        return list(range(1, 25))
        
    A_ub = np.array([c for c, _ in valid_constraints_data], dtype=float)
    b_ub = np.array([s for _, s in valid_constraints_data], dtype=float)
    
    c = np.ones(n) * -1
    constraints = [LinearConstraint(A_ub, -np.inf, b_ub)]
    integrality = np.ones(n)
    bounds = Bounds(0, 1)
    
    res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
    
    if res.status == 0:
        x = res.x
        selected = [i+1 for i in range(n) if x[i] > 0.5]
        if selected:
            return selected
        else:
            return [1]
    else:
        return list(range(13, 25))
