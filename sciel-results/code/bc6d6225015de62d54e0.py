   import numpy as np
   import math
   from scipy.optimize import milp, LinearConstraint, Bounds
   
   def construct():
       # Calculate LCM of 1 to 30
       def lcm(a, b):
           return a * b // math.gcd(a, b)
       
       D = 1
       for i in range(1, 31):
           D = lcm(D, i)
           
       weights = np.array([D // i for i in range(1, 31)], dtype=np.int64)
       n = 30
       
       # Variables: c_1 ... c_n, M
       # Total n+1 variables.
       # c_i in {-1, 0, 1}
       # M >= 0
       
       # Objective: Minimize M
       c_obj = np.zeros(n + 1)
       c_obj[n] = 1.0
       
       # Bounds
       # c_i: [-1, 1]
       # M: [0, inf]
       lb = np.concatenate([np.full(n, -1), [0]])
       ub = np.concatenate([np.full(n, 1), [np.inf]])
       
       # Integrality
       # c_i must be integer
       # M can be continuous? No, M is sum of integers, so M is integer.
       # But if we minimize M, and constraints are tight, it will be integer.
       # Let's make M integer too to be safe?
       # Actually, M is just a bound. If S is integer, M will be integer in the optimal solution.
       # Let's set integrality for c_i as 1, and M as 0 (continuous).
       integrality = np.concatenate([np.ones(n, dtype=int), [0]])
       
       # Constraints
       # S = sum(c_i * w_i)
       # 1) S - M <= 0  =>  sum(w_i*c_i) - M <= 0
       # 2) -S - M <= 0 => -sum(w_i*c_i) - M <= 0
       # 3) S >= 1 (for positive case) => sum(w_i*c_i) >= 1
       
       # Matrix A for linear constraints: A @ x <= b
       # x = [c_1, ..., c_n, M]
       
       # Constraint 1: w_1*c_1 + ... + w_n*c_n - M <= 0
       A1 = np.zeros((1, n + 1))
       A1[0, :n] = weights
       A1[0, n] = -1
       b1 = np.array([0])
       
       # Constraint 2: -w_1*c_1 - ... - w_n*c_n - M <= 0
       A2 = np.zeros((1, n + 1))
       A2[0, :n] = -weights
       A2[0, n] = -1
       b2 = np.array([0])
       
       # Constraint 3 (Positive): w_1*c_1 + ... + w_n*c_n >= 1
       # => -w_1*c_1 - ... - w_n*c_n <= -1
       A3 = np.zeros((1, n + 1))
       A3[0, :n] = -weights
       A3[0, n] = 0
       b3 = np.array([-1])
       
       # Combine constraints
       A = np.vstack([A1, A2, A3])
       b = np.concatenate([b1, b2, b3])
       
       constraints = LinearConstraint(A, ub=np.concatenate([b, b, b])) # Wait, LinearConstraint(A, lb, ub)
       # The form is lb <= A @ x <= ub
       # My A @ x <= b is equivalent to -inf <= A @ x <= b
       # So lb = -inf, ub = b
       
       lb_con = np.full(3, -np.inf)
       ub_con = b
       
       constraints = LinearConstraint(A, lb=lb_con, ub=ub_con)
       
       bounds = Bounds(lb, ub)
       
       try:
           res = milp(c_obj, constraints=constraints, integrality=integrality, bounds=bounds)
           if res.success:
               coeffs = res.x[:n].astype(int)
               # Verify sum is nonzero
               s = np.dot(coeffs, weights)
               if s != 0:
                   return list(coeffs)
       except Exception:
           pass
       
       # Try Negative Case
       # Constraint 3 (Negative): S <= -1 => S + 1 <= 0 => sum(w_i*c_i) <= -1
       # => sum(w_i*c_i) <= -1
       # In form A @ x <= b:
       A3_neg = np.zeros((1, n + 1))
       A3_neg[0, :n] = weights
       A3_neg[0, n] = 0
       b3_neg = np.array([-1])
       
       A_neg = np.vstack([A1, A2, A3_neg])
       b_neg = np.concatenate([b1, b2, b3_neg])
       lb_con_neg = np.full(3, -np.inf)
       ub_con_neg = b_neg
       
       constraints_neg = LinearConstraint(A_neg, lb=lb_con_neg, ub=ub_con_neg)
       
       try:
           res = milp(c_obj, constraints=constraints_neg, integrality=integrality, bounds=bounds)
           if res.success:
               coeffs = res.x[:n].astype(int)
               s = np.dot(coeffs, weights)
               if s != 0:
                   return list(coeffs)
       except Exception:
           pass
       
       # Fallback
       return [0] * 28 + [1, -1]
   