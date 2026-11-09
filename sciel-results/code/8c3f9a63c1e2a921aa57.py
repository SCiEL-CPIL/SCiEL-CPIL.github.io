# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds


def _build_data():
    n = 10
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2),
    ]
    D = len(demands)
    cw = []
    ccw = []
    for (s, t, w) in demands:
        if s == t:
            cw_edges = []
        else:
            cw_edges = []
            e = s
            while e != t:
                cw_edges.append(e)
                e = (e + 1) % n
        ccw_edges = [e for e in range(n) if e not in set(cw_edges)]
        cw.append((cw_edges, w))
        ccw.append((ccw_edges, w))
    return demands, cw, ccw, n, D


def ring_routing() -> np.ndarray:
    start = time.time()
    deadline = start + 600.0

    demands, cw, ccw, n, D = _build_data()
    
    # Precompute weights
    weights = np.array([demands[i][2] for i in range(D)], dtype=float)
    
    # MILP formulation:
    # Variables: x_0..x_{D-1} (binary, 1=CCW, 0=CW), T (continuous, >=0)
    # Minimize T
    # Subject to: Load_e <= T for all edges e
    
    # Coefficient vector c
    # x has D binary vars, then T
    c = np.zeros(D + 1)
    c[D] = 1.0 # Minimize T
    
    # Integrality: first D are integers (binary), T is continuous
    integrality = np.ones(D + 1, dtype=int)
    integrality[D] = 0
    
    # Bounds: x_i in [0, 1], T in [0, inf)
    lb = np.zeros(D + 1)
    ub = np.ones(D + 1)
    ub[D] = np.inf
    bounds = Bounds(lb, ub)
    
    # Constraints:
    # For each edge e:
    # Sum( w_i * I[e in CW_i] * (1-x_i) + w_i * I[e in CCW_i] * x_i ) <= T
    # Sum( w_i * I[e in CW_i] ) - Sum( w_i * I[e in CW_i] * x_i ) + Sum( w_i * I[e in CCW_i] * x_i ) <= T
    # Constant - Sum( (w_i * I[e in CW_i] - w_i * I[e in CCW_i]) * x_i ) <= T
    # Sum( (w_i * I[e in CW_i] - w_i * I[e in CCW_i]) * x_i ) + T >= Constant
    # Or: -Sum( (w_i * I[e in CW_i] - w_i * I[e in CCW_i]) * x_i ) - T <= -Constant
    
    # Let's use the form: A @ x <= b
    # Row for edge e:
    # Coeff for x_i: -(w_i * I[e in CW_i]) + (w_i * I[e in CCW_i])
    # Coeff for T: -1
    # RHS: -Constant_e
    
    A = np.zeros((n, D + 1))
    b = np.zeros(n)
    
    for e in range(n):
        constant_e = 0.0
        for i in range(D):
            if e in cw[i][0]:
                # If x_i=0 (CW), load is w_i. If x_i=1 (CCW), load is 0.
                # Term in load: w_i * (1 - x_i) = w_i - w_i * x_i
                constant_e += weights[i]
                A[e, i] = -weights[i]
            else:
                # If x_i=0 (CW), load is 0. If x_i=1 (CCW), load is w_i.
                # Term in load: w_i * x_i
                # Note: e is NOT in cw[i][0], so it is in ccw[i][0] (since s!=t usually, and ring covers all)
                # Wait, if s==t, cw is empty, ccw is all edges? 
                # If s==t, demand doesn't traverse any edge? 
                # "Clockwise from s to t uses edges s...t-1". If s==t, empty set.
                # "Counter-clockwise uses all other ring edges". If s==t, all edges? 
                # Usually s!=t in such problems. Here s!=t for all demands.
                # So if e not in cw, it is in ccw.
                A[e, i] = weights[i]
        
        # Constraint: Constant_e - Sum(A_pos * x) <= T
        # Sum(A_pos * x) + T >= Constant_e
        # -Sum(A_pos * x) - T <= -Constant_e
        # A[e, i] currently stores -w_i if CW, +w_i if CCW.
        # Load = Constant_e + Sum( A[e,i] * x_i ) ?
        # If CW: x=0. Load = Constant_e. Correct.
        # If CCW: x=1. Load = Constant_e + A[e,i].
        # If e in CW, A[e,i] = -w_i. Load = Constant_e - w_i.
        # Constant_e included w_i. So Load = (Const + w_i) - w_i = Const. 
        # Wait.
        # Let's re-verify.
        # If e in CW:
        #   x=0 (CW): Load += w_i.
        #   x=1 (CCW): Load += 0.
        #   Formula: w_i * (1-x_i).
        #   Constant part (x=0): w_i.
        #   Variable part: -w_i * x_i.
        # If e not in CW (i.e., in CCW):
        #   x=0 (CW): Load += 0.
        #   x=1 (CCW): Load += w_i.
        #   Formula: w_i * x_i.
        #   Constant part (x=0): 0.
        #   Variable part: w_i * x_i.
        
        # So Load_e = Constant_e + Sum( A[e, i] * x_i )
        # Where A[e, i] = -w_i if e in CW, else w_i.
        # And Constant_e = Sum( w_i for i where e in CW ).
        
        # We require Load_e <= T
        # Constant_e + Sum( A[e, i] * x_i ) - T <= 0
        # Sum( A[e, i] * x_i ) - T <= -Constant_e
        
        A[e, D] = -1.0
        b[e] = -constant_e
        
    constraints = LinearConstraint(A, -np.inf, b)
    
    # Solve
    try:
        res = milp(c=c, integrality=integrality, bounds=bounds, constraints=constraints)
        if res.status == 0: # Optimal
            routing = res.x[:D]
            # Ensure binary output
            # res.x should be close to 0 or 1
            routing = (routing >= 0.5).astype(float)
            return routing
        else:
            # Fallback to exhaustive search if MILP fails
            pass
    except Exception:
        pass

    # Fallback: Exhaustive Search (Original Logic)
    cw_edges = [cw[i][0] for i in range(D)]
    ccw_edges = [ccw[i][0] for i in range(D)]
    
    best_load = float('inf')
    best_mask = 0
    total = 1 << D
    
    loads = np.zeros(n)
    prev_gray = 0
    for i in range(D):
        for e in cw_edges[i]:
            loads[e] += weights[i]
    best_load = float(np.max(loads))
    best_mask = 0

    for step in range(1, total):
        gray = step ^ (step >> 1)
        diff = gray ^ prev_gray
        bit = (diff & -diff).bit_length() - 1
        is_ccw = (gray >> bit) & 1
        if is_ccw:
            for e in cw_edges[bit]:
                loads[e] -= weights[bit]
            for e in ccw_edges[bit]:
                loads[e] += weights[bit]
        else:
            for e in ccw_edges[bit]:
                loads[e] -= weights[bit]
            for e in cw_edges[bit]:
                loads[e] += weights[bit]
        prev_gray = gray
        cur = float(np.max(loads))
        if cur < best_load:
            best_load = cur
            best_mask = gray
        if time.time() > deadline - 30.0:
            break

    routing = np.zeros(D)
    for i in range(D):
        routing[i] = 1.0 if (best_mask >> i) & 1 else 0.0
    return routing


# EVOLVE-BLOCK-END
