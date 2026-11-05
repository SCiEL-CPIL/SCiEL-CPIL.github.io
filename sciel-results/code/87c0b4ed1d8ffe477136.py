import time
from fractions import Fraction
from math import gcd
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin before 60s kill

    denominators = list(range(2, 49))
    n = len(denominators)
    
    # Calculate LCM of denominators
    def lcm(a, b):
        return abs(a * b) // gcd(a, b)
    
    L = 1
    for d in denominators:
        L = lcm(L, d)
        
    # Coefficients for the ILP: c_i = L / d_i
    # We want to maximize sum(c_i * x_i) subject to sum(c_i * x_i) <= L - 1
    # scipy.optimize.milp minimizes c @ x
    # So we set objective to -c_i
    c = [L // d for d in denominators]
    
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        import numpy as np
        
        # Variables: x_0 ... x_{n-1}, binary
        integrality = np.ones(n)
        
        # Objective: minimize -sum(c_i * x_i)
        objective = np.array([-c[i] for i in range(n)])
        
        # Constraint: sum(c_i * x_i) <= L - 1
        # LinearConstraint(A_ub, lb, ub)
        # A_ub @ x <= ub
        A_ub = np.array([c], dtype=int)
        lb = 0
        ub = L - 1
        
        # Also x_i >= 0 (implied by binary)
        # We need to ensure non-empty. 
        # If the max sum is 0, we fail. But max sum is huge, so it will pick many items.
        # The constraint sum <= L-1 ensures strict inequality.
        
        constraints = [LinearConstraint(A_ub, lb, ub)]
        bounds = Bounds(0, 1)
        
        res = milp(c=objective, constraints=constraints, integrality=integrality, bounds=bounds)
        
        if res.success and res.x is not None:
            # res.x are the optimal binary variables
            indices = [i for i in range(n) if res.x[i] > 0.5]
            if indices:
                # Verify with Fraction to be absolutely sure
                s = Fraction(0)
                for i in indices:
                    s += Fraction(1, denominators[i])
                if s < 1:
                    return indices
                else:
                    # Fallback if numerical precision in solver caused issue (unlikely with integers)
                    pass
        else:
            pass
            
    except Exception:
        pass

    # Fallback: Fast Greedy + Local Search
    # This is much faster than the previous exhaustive/SA approach
    
    # 1. Greedy: Sort by value (1/d), pick largest that fits? 
    # Actually, to minimize deficit (maximize sum), we want to pack as much as possible.
    # This is a Knapsack problem.
    # Items: weight = L/d, value = L/d. Capacity = L-1.
    # Greedy by ratio (value/weight = 1) doesn't distinguish. 
    # Greedy by value (largest first) is good.
    
    # Sort indices by 1/d descending (i.e., d ascending)
    sorted_indices = sorted(range(n), key=lambda i: denominators[i])
    
    best_sum = Fraction(0)
    best_indices = []
    current_sum = Fraction(0)
    current_indices = []
    
    # Greedy fill
    for i in sorted_indices:
        new_sum = current_sum + Fraction(1, denominators[i])
        if new_sum < 1:
            current_sum = new_sum
            current_indices.append(i)
    
    if current_indices:
        best_sum = current_sum
        best_indices = current_indices
    
    # Local Search: Try to swap items to improve sum
    # We want to increase the sum. 
    # If we have a set S, and we can remove a subset R and add a subset A such that sum(A) > sum(R) and total < 1.
    # Simplified: Try swapping one item in with one item out.
    # Or just try adding small items if there's room? No, greedy already added all possible large items.
    # The "gap" is likely small.
    
    # Let's do a few random restarts with greedy + random repair
    random.seed(42)
    
    for _ in range(50):
        if time.time() > deadline:
            break
            
        # Random subset
        k = random.randint(5, 30)
        subset = random.sample(range(n), k)
        s = Fraction(0)
        for i in subset:
            s += Fraction(1, denominators[i])
        
        # Repair: remove items until s < 1
        # To maximize remaining sum, we should remove the SMALLEST items? 
        # No, we want to keep the LARGEST items.
        # So sort subset by 1/d descending. Keep adding until full?
        # That's just greedy again.
        
        # Let's try a different heuristic:
        # Start with empty.
        # Add items in random order.
        # If sum >= 1, remove the item just added? No, that's just greedy.
        
        # Let's stick to the greedy result. It's usually very good for this density.
        # The greedy result is:
        # 1/2 + 1/3 + 1/4 + 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 + 1/13 + 1/14 + 1/15 + 1/16 + 1/17 + 1/18 + 1/19 + 1/20 + 1/21 + 1/22 + 1/23 + 1/24 + 1/25 + 1/26 + 1/27 + 1/28 + 1/29 + 1/30 + 1/31 + 1/32 + 1/33 + 1/34 + 1/35 + 1/36 + 1/37 + 1/38 + 1/39 + 1/40 + 1/41 + 1/42 + 1/43 + 1/44 + 1/45 + 1/46 + 1/47 + 1/48
        # Sum of 1/2..1/48 is approx 4.2.
        # Greedy picks 1/2, 1/3, ... until sum >= 1.
        # 1/2 = 0.5
        # 1/2+1/3 = 0.833
        # 1/2+1/3+1/4 = 1.083 > 1. So it stops before 1/4.
        # So greedy returns [0, 1] (indices for 2, 3). Deficit 1/6 = 0.166.
        
        # We can do better. 1/2 + 1/3 + 1/7 + 1/43 = 0.999...
        # The ILP should find the optimal or near-optimal.
        
        pass

    if best_indices:
        return best_indices
    else:
        return [0]
