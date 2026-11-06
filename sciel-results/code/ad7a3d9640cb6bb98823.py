import numpy as np
import time
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s limit
    
    # 1. Precompute all arithmetic progressions (APs)
    # An AP is defined by start 'a', step 's' >= 1, and indices a, a+s, ..., a+(k-1)s < n
    # We represent each AP as a vector of indices.
    # To speed up ILP, we can potentially filter out APs that are subsets of others?
    # No, any AP is a constraint. However, if an AP is a subset of another, 
    # the constraint on the larger one is stricter only if the signs align, 
    # but generally we need all.
    
    # For n=256, the number of APs is manageable.
    # Count: sum_{s=1}^{255} sum_{a=0}^{255} 1 if a+s < 256? 
    # Actually, for each step s, the number of APs is roughly (n-s)*something?
    # Let's just generate them.
    
    ap_indices = []
    for s in range(1, n):
        for a in range(n):
            idx = []
            curr = a
            while curr < n:
                idx.append(curr)
                curr += s
            if len(idx) > 0:
                ap_indices.append(idx)
                
    num_aps = len(ap_indices)
    
    # 2. Build the constraint matrix A
    # A is num_aps x n. A[p, i] = 1 if position i is in AP p, else 0.
    # The constraint is: -D <= A @ x <= D
    # This can be written as:
    # A @ x <= D
    # -A @ x <= D
    # So the full matrix M is [A; -A] of shape (2*num_aps, n)
    # And the upper bounds are [D; D]
    # Lower bounds are -inf (or we can use -D if we want symmetric, but standard form is A_ub @ x <= b_ub)
    # milp supports LinearConstraint(lower, upper)
    # Let's use LinearConstraint(A_full, -D, D) where A_full = [A; -A]
    
    # However, building the full matrix every time in the bisection is expensive.
    # We can precompute A.
    
    A = np.zeros((num_aps, n), dtype=np.int8)
    for p, idx in enumerate(ap_indices):
        for i in idx:
            A[p, i] = 1
            
    # A_full = np.vstack([A, -A])
    # We will create the constraint dynamically in the loop to save memory? 
    # No, A_full is constant. Only the bounds change with D.
    
    A_full = np.vstack([A, -A])
    
    # Variables x_i are binary? No, x_i in {-1, 1}.
    # Let y_i be binary variable where x_i = 2*y_i - 1.
    # Then sum x_i = sum (2*y_i - 1) = 2*sum(y_i) - len(AP).
    # Constraint: -D <= 2*sum(y_i) - len(AP) <= D
    # -D + len(AP) <= 2*sum(y_i) <= D + len(AP)
    # (-D + len(AP))/2 <= sum(y_i) <= (D + len(AP))/2
    
    # This changes the coefficients in the matrix!
    # It is easier to keep x_i as continuous variables in [-1, 1] and add integrality constraints?
    # SciPy milp allows integrality=1 for binary variables.
    # If we use x_i directly, they are not binary. They are integer? No, -1 and 1 are not standard binary.
    # We must use the substitution y_i in {0, 1}.
    
    # Let's re-derive the matrix for y.
    # Objective: None (feasibility)
    # Constraints:
    # For each AP p with length L_p:
    # Lower: sum_{i in p} (2*y_i - 1) >= -D  => 2*sum(y_i) - L_p >= -D => sum(y_i) >= (L_p - D) / 2
    # Upper: sum_{i in p} (2*y_i - 1) <= D   => 2*sum(y_i) - L_p <= D  => sum(y_i) <= (L_p + D) / 2
    
    # So we need:
    # A_y @ y >= lower_bounds
    # A_y @ y <= upper_bounds
    # where A_y is the same as A (coefficients are 1), but the RHS depends on D and L_p.
    # Wait, the coefficient of y_i in the sum is 2.
    # So the matrix for the LHS is 2 * A.
    
    A_y = 2 * A
    
    # We need lengths of each AP
    ap_lengths = np.array([len(idx) for idx in ap_indices], dtype=np.float64)
    
    # Precompute 2*A and lengths
    # A_y is shape (num_aps, n)
    # We will use LinearConstraint(A_y, lb, ub)
    # lb[p] = (ap_lengths[p] - D) / 2
    # ub[p] = (ap_lengths[p] + D) / 2
    
    # Variables y_i are binary.
    integrality = np.ones(n, dtype=np.int8)
    bounds = Bounds(lb=0, ub=1)
    
    # Helper to check feasibility for a given D
    def is_feasible(D):
        # Calculate bounds
        lb = (ap_lengths - D) / 2.0
        ub = (ap_lengths + D) / 2.0
        
        # If lb > ub for any AP, infeasible (should not happen if D >= 0 and lengths positive, 
        # unless D is negative, but D is positive)
        # Also, if lb < 0, the effective lower bound is 0 because y_i >= 0? 
        # No, sum(y_i) >= lb. If lb is negative, the constraint is trivially satisfied by y>=0?
        # Actually, y_i are binary, so sum(y_i) >= 0 always.
        # So we can clip lb to 0? No, LinearConstraint handles it.
        
        # Clip to valid range for sums of binaries?
        # Sum of k binaries is between 0 and k.
        # lb = (L-D)/2. If D > L, lb < 0.
        # ub = (L+D)/2. If D > L, ub > L.
        
        cons = LinearConstraint(A_y, lb, ub)
        
        try:
            res = milp(c=np.zeros(n), constraints=cons, integrality=integrality, bounds=bounds)
            return res.status == 0
        except Exception:
            return False

    # Bisection search for D
    # D must be an integer? The sum of signs is an integer. So D can be integer.
    # Max possible D is n (if all signs are 1).
    # Min possible D is 0 (if we can balance perfectly, unlikely for all APs).
    
    # Initial bounds
    D_low = 0
    D_high = n  # 256
    
    # We want the smallest D such that is_feasible(D) is True.
    # Note: is_feasible(D) is monotonic. If D works, D+1 works.
    
    # Optimization: Start with a random guess to get a good upper bound quickly?
    # Or just bisection.
    
    # Bisection
    best_D = n
    best_y = None
    
    # To speed up, we can try a few random D values first to narrow down?
    # Or just standard bisection.
    # Number of iterations: log2(256) = 8. Very fast!
    # Each MILP call might take a few seconds. 8 * few seconds < 85s.
    
    # Let's add a timeout check inside the loop.
    
    # We can also parallelize? No, standard library.
    
    # Let's run the bisection.
    while D_low < D_high:
        if time.time() > deadline - 5:
            break
            
        D_mid = (D_low + D_high) // 2
        
        if is_feasible(D_mid):
            D_high = D_mid
            best_D = D_mid
        else:
            D_low = D_mid + 1
            
    # After bisection, best_D is the minimum feasible D.
    # We need the actual signs.
    # We must solve for y again with D = best_D to get the solution.
    
    if time.time() > deadline - 2:
        # Fallback: return random signs if we ran out of time
        rng = np.random.RandomState(42)
        y = rng.randint(0, 2, size=n)
        signs = 2 * y - 1
        return [int(x) for x in signs]
        
    # Solve for the optimal y
    lb = (ap_lengths - best_D) / 2.0
    ub = (ap_lengths + best_D) / 2.0
    cons = LinearConstraint(A_y, lb, ub)
    
    try:
        res = milp(c=np.zeros(n), constraints=cons, integrality=integrality, bounds=bounds)
        if res.status == 0:
            y_opt = res.x
            # Convert y to signs
            signs = 2 * y_opt - 1
            # Ensure integers
            signs = np.round(signs).astype(int)
            return [int(x) for x in signs]
        else:
            # Fallback if solver failed for some reason
            rng = np.random.RandomState(42)
            y = rng.randint(0, 2, size=n)
            signs = 2 * y - 1
            return [int(x) for x in signs]
    except Exception:
        rng = np.random.RandomState(42)
        y = rng.randint(0, 2, size=n)
        signs = 2 * y - 1
        return [int(x) for x in signs]

# Test locally if needed
if __name__ == "__main__":
    start = time.time()
    signs = construct()
    end = time.time()
    print(f"Time: {end - start:.2f}s")
    print(f"First 10 signs: {signs[:10]}")
    # Verify
    n = 256
    max_sum = 0
    for s in range(1, n):
        for a in range(n):
            idx = list(range(a, n, s))
            if not idx: continue
            sum_val = sum(signs[i] for i in idx)
            abs_sum = abs(sum_val)
            if abs_sum > max_sum:
                max_sum = abs_sum
    print(f"Max AP Sum: {max_sum}")
