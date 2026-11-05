import numpy as np
import time
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    start_time = time.time()
    deadline = 80.0  # Safety margin for 90s limit
    
    n = 256
    # Identify vertices that cannot be in the set (2*v is a square)
    # Squares up to 2*256 = 512 are 4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144, 169, 196, 225, 256, 289, 324, 361, 400, 441, 484.
    # 2*v = k^2 => v = k^2 / 2. So k must be even. k=2,4,6,8,10,12,14,16,18,20,22.
    # v = 2, 8, 18, 32, 50, 72, 98, 128, 162, 200, 242.
    
    # Build conflict graph edges for u < v where u+v is a square
    # Squares up to 256+255 = 511.
    squares = set()
    for k in range(1, 23): # 22^2 = 484, 23^2 = 529 > 511
        squares.add(k*k)
        
    # We need x_u + x_v <= 1 for each edge.
    # We also need x_v = 0 for v in V_bad.
    
    # Let's filter out V_bad first to reduce problem size?
    # Or just set bounds.
    
    # Number of variables = 256
    # c = -1 for all (maximize sum, so minimize -sum)
    c = -np.ones(n)
    
    # Constraints:
    # 1. x_v = 0 for v in V_bad. This can be handled by bounds or constraints.
    #    Using bounds is more efficient.
    
    # 2. x_u + x_v <= 1 for each edge (u, v).
    
    # Let's collect edges
    edges = []
    for u in range(1, n + 1):
        for v in range(u + 1, n + 1):
            if (u + v) in squares:
                edges.append((u - 1, v - 1)) # 0-indexed
    
    # Also, if u+v is a square, we add constraint.
    # What about u=v? We already handled via bounds (x_u=0 if 2u is square).
    # But if 2u is NOT a square, is there a constraint x_u + x_u <= 1? No, because a+b with a=b is just checking if 2a is square.
    # The condition is "a+b is never a perfect square".
    # If a is in set, 2a must not be square.
    # If a, b in set, a!=b, a+b must not be square.
    
    # So we only need pairwise constraints for u < v.
    
    A = []
    b_ub = []
    b_lb = []
    
    for u_idx, v_idx in edges:
        row = np.zeros(n)
        row[u_idx] = 1
        row[v_idx] = 1
        A.append(row)
        b_ub.append(1)
        b_lb.append(0) # x_u + x_v >= 0 is trivial, but milp requires linear constraints. 
                       # Actually, we can just use A_ub, b_ub if using linprog, but milp uses LinearConstraint.
                       # Let's use LinearConstraint(A, lb, ub).
    
    if A:
        A = np.array(A)
        b_ub = np.array(b_ub)
        b_lb = np.array(b_lb)
        constraints = LinearConstraint(A, b_lb, b_ub)
    else:
        constraints = LinearConstraint(np.zeros((0, n)), np.zeros(0), np.zeros(0))
    
    # Bounds:
    # x_v = 0 if 2v is square
    # 0 <= x_v <= 1 otherwise
    lb = np.zeros(n)
    ub = np.ones(n)
    
    for v in range(1, n + 1):
        if (2 * v) in squares:
            ub[v - 1] = 0
            lb[v - 1] = 0
    
    bounds = Bounds(lb, ub)
    
    try:
        res = milp(c, constraints=constraints, integrality=np.ones(n), bounds=bounds)
        if res.success:
            # res.x is the solution
            x = res.x
            # Convert to list of indices (1-based)
            result = [int(i + 1) for i in range(n) if x[i] > 0.5]
            if result:
                return result
            else:
                # Fallback if empty (shouldn't happen)
                return [1]
        else:
            # Fallback to seed
            return list(range(1, 257, 3))
    except Exception:
        # Fallback to seed if MILP fails or is not available
        return list(range(1, 257, 3))
