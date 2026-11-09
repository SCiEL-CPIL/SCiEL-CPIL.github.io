"""Direct convex LP over allocation and payment variables for optimal finite-support mechanism."""
import numpy as np
from scipy.optimize import linprog
import time
from fractions import Fraction


def construct_auction(distribution_spec, support, constraints):
    """
    Construct an incentive-compatible finite auction mechanism.
    
    Args:
        distribution_spec: specification of the value distribution
        support: support of the value space
        constraints: additional constraints
    
    Returns:
        dict with 'allocations': (16, 2) array and 'payments': (16,) array
    """
    start_time = time.time()
    deadline = start_time + 85.0

    # Parse distribution
    values = [0, 1, 2, 3]
    
    if distribution_spec is not None:
        if isinstance(distribution_spec, dict):
            if 'probabilities' in distribution_spec:
                probs = np.array(distribution_spec['probabilities'], dtype=float)
            elif 'dist' in distribution_spec:
                probs = np.array(distribution_spec['dist'], dtype=float)
            elif 'p' in distribution_spec:
                probs = np.array(distribution_spec['p'], dtype=float)
            else:
                probs = None
        elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
            probs = np.array(distribution_spec, dtype=float)
        else:
            probs = None
    else:
        probs = None
    
    if probs is None or len(probs) != 16:
        probs = np.ones(16) / 16.0
    else:
        probs = probs / probs.sum()
    
    # Parse support
    if support is not None:
        if isinstance(support, (list, tuple)):
            if len(support) == 16:
                types = [tuple(t) for t in support]
            elif len(support) == 2:
                v1_vals = list(support[0])
                v2_vals = list(support[1])
                types = [(v1, v2) for v1 in v1_vals for v2 in v2_vals]
            else:
                types = [(i, j) for i in values for j in values]
        else:
            types = [(i, j) for i in values for j in values]
    else:
        types = [(i, j) for i in values for j in values]
    
    n = 16
    assert len(types) == n, f"Expected 16 types, got {len(types)}"
    
    val_matrix = np.array(types, dtype=float)  # shape (16, 2)
    
    # LP formulation:
    # Variables ordered: [x1_0..x1_15, x2_0..x2_15, p_0..p_15] = 48 vars
    # Maximize: sum_t probs[t] * p_t
    # IC: p_t - p_t' <= v_t . x_t - v_t . x_t'
    # IR: p_t <= v_t . x_t
    # 0 <= x <= 1, p >= 0
    
    n_vars = 48
    c = np.zeros(n_vars)
    c[32:] = -probs  # minimize -revenue
    
    A_rows = []
    b_rows = []
    
    # IC constraints
    for t in range(n):
        for t2 in range(n):
            if t == t2:
                continue
            row = np.zeros(n_vars)
            row[32 + t] = 1.0       # p_t
            row[32 + t2] = -1.0      # -p_t'
            row[t] = -val_matrix[t, 0]      # -v_t1 * x1_t
            row[t2] = val_matrix[t, 0]      # +v_t1 * x1_t'
            row[16 + t] = -val_matrix[t, 1] # -v_t2 * x2_t
            row[16 + t2] = val_matrix[t, 1] # +v_t2 * x2_t'
            A_rows.append(row)
            b_rows.append(0.0)
    
    # IR constraints
    for t in range(n):
        row = np.zeros(n_vars)
        row[32 + t] = 1.0
        row[t] = -val_matrix[t, 0]
        row[16 + t] = -val_matrix[t, 1]
        A_rows.append(row)
        b_rows.append(0.0)
    
    A_ub = np.array(A_rows) if A_rows else np.zeros((0, n_vars))
    b_ub = np.array(b_rows) if b_rows else np.zeros(0)
    
    bounds = [(0, 1)] * 32 + [(0, None)] * 16
    
    best_alloc = None
    best_pay = None
    best_rev = -np.inf
    
    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs',
                         options={'presolve': True})
        
        if result.success:
            x1 = result.x[:16]
            x2 = result.x[16:32]
            p = result.x[32:48]
            allocations = np.column_stack((x1, x2))
            payments = p
            
            allocations = np.clip(allocations, 0, 1)
            payments = np.clip(payments, 0, None)
            
            if _verify_constraints_exact(payments, allocations, val_matrix):
                best_alloc = allocations.copy()
                best_pay = payments.copy()
                best_rev = float(probs @ payments)
            
            # Polish pass with tighter tolerances if time permits
            elapsed = time.time() - start_time
            if elapsed < 70.0:
                try:
                    result2 = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs',
                                      options={'presolve': False,
                                               'dual_feasibility_tolerance': 1e-13,
                                               'primal_feasibility_tolerance': 1e-13,
                                               'dual_solution_tolerance': 1e-13,
                                               'primal_solution_tolerance': 1e-13,
                                               'crossover_presolve': False})
                    if result2.success:
                        x1p = result2.x[:16]
                        x2p = result2.x[16:32]
                        pp = result2.x[32:48]
                        alloc_p = np.column_stack((x1p, x2p))
                        alloc_p = np.clip(alloc_p, 0, 1)
                        pay_p = np.clip(pp, 0, None)
                        
                        if _verify_constraints_exact(pay_p, alloc_p, val_matrix):
                            rev_p = float(probs @ pay_p)
                            if rev_p > best_rev:
                                best_alloc = alloc_p
                                best_pay = pay_p
                                best_rev = rev_p
                except Exception:
                    pass
        
        if best_alloc is not None:
            return {'allocations': best_alloc, 'payments': best_pay}
            
    except Exception:
        pass
    
    # Fallback: Myerson-style closed form
    allocs, pays = _myerson_fallback(val_matrix, probs)
    if _verify_constraints_exact(pays, allocs, val_matrix):
        return {'allocations': allocs, 'payments': pays}
    
    # Last resort: zero mechanism
    return {'allocations': np.zeros((16, 2)), 'payments': np.zeros(16)}


def _verify_constraints_exact(payments, allocations, val_matrix, tol=1e-8):
    """Verify IC, IR, and non-negativity with exact rational arithmetic for robustness."""
    n = len(payments)
    
    if np.any(payments < -tol):
        return False
    if np.any(allocations < -tol) or np.any(allocations > 1 + tol):
        return False
    
    # Use exact rational arithmetic for verification
    p_frac = [Fraction(str(float(payments[t]))) for t in range(n)]
    a1_frac = [Fraction(str(float(allocations[t, 0]))) for t in range(n)]
    a2_frac = [Fraction(str(float(allocations[t, 1]))) for t in range(n)]
    v1_frac = [Fraction(int(val_matrix[t, 0])) for t in range(n)]
    v2_frac = [Fraction(int(val_matrix[t, 1])) for t in range(n)]
    
    # IR: p_t <= v_t1 * a1_t + v_t2 * a2_t
    for t in range(n):
        surplus = v1_frac[t] * a1_frac[t] + v2_frac[t] * a2_frac[t] - p_frac[t]
        if surplus < -Fraction(1, 10**8):
            return False
    
    # IC: p_t - p_t' <= v_t . a_t - v_t . a_t'
    for t in range(n):
        for t2 in range(n):
            if t == t2:
                continue
            lhs = p_frac[t] - p_frac[t2]
            rhs = (v1_frac[t] * a1_frac[t] + v2_frac[t] * a2_frac[t]) - \
                  (v1_frac[t] * a1_frac[t2] + v2_frac[t] * a2_frac[t2])
            if lhs - rhs > Fraction(1, 10**8):
                return False
    
    return True


def _myerson_fallback(val_matrix, probs):
    """Closed-form Myerson mechanism as fallback."""
    n = 16
    # For single bidder with two goods, the optimal mechanism allocates each good
    # based on the virtual valuation. With uniform distribution on {0,1,2,3}^2,
    # we use a simple threshold rule.
    allocations = np.zeros((n, 2))
    payments = np.zeros(n)
    
    # For each good, determine allocation based on marginal value
    # Simple approach: allocate good k to type t if v_ik > threshold
    # The threshold depends on the distribution
    
    # For uniform distribution, the virtual value for value v in {0,1,2,3} with prob 1/4 each:
    # phi(v) = v - (1-F(v))/f(v)
    # F(0)=0.25, F(1)=0.5, F(2)=0.75, F(3)=1.0
    # f(v) = 0.25 for all v
    # phi(0) = 0 - 1/0.25 = -4  (wait, 1-F(0) = 0.75, so phi(0) = 0 - 0.75/0.25 = -3)
    # phi(1) = 1 - 0.5/0.25 = 1 - 2 = -1
    # phi(2) = 2 - 0.25/0.25 = 2 - 1 = 1
    # phi(3) = 3 - 0/0.25 = 3
    
    # Allocate good k if virtual value > 0, i.e., v_k >= 2
    for t in range(n):
        v1, v2 = val_matrix[t]
        if v1 >= 2:
            allocations[t, 0] = 1.0
        if v2 >= 2:
            allocations[t, 1] = 1.0
    
    # Payment via Myerson identity: p_t = v_t . a_t - integral of a(s) ds
    # For discrete case: p_t = v_t . a_t - sum_{s < t} a(s) * (v_t - v_s) * delta
    # Simplified: for step function, payment is the area under the allocation curve
    # p_t = sum_k [v_ik * a_ik - (1/4) * sum_{v_k' < v_ik} a_k(v_k')]
    # For our step: a_k(v) = 1 if v >= 2 else 0
    # For good 1: if v1 >= 2, p1 = v1 - (1/4)(a(0)+a(1)) = v1 - 0 = v1... 
    # Actually for discrete Myerson: p(v) = v*a(v) - sum_{v' < v} a(v') * (v - v') * f(v')
    # This is getting complex; let's just use a simple payment that satisfies IC/IR
    
    # Simple payment: p_t = max(0, v1*a1 + v2*a2 - expected_surplus)
    # For the step mechanism, payment = sum of values of allocated goods minus a constant
    # To ensure IR: p_t <= v_t . a_t
    # To ensure IC: use the envelope theorem
    
    # For this specific mechanism, payment is:
    # p_t = (v1 if v1>=2 else 0) + (v2 if v2>=2 else 0) - C
    # where C is chosen for IR. Let's set C so that IR binds at the lowest allocated type.
    
    # Types with both goods allocated: v1>=2, v2>=2 -> (2,2),(2,3),(3,2),(3,3)
    # Min value for both: (2,2) with value 4
    # Types with one good: e.g., (2,0),(2,1),(3,0),(3,1) -> value 2 or 3
    # Min value for one good at threshold: (2,0) or (0,2) with value 2
    
    # Set payment so that IR binds at (2,2): p = 4 - 0 = 4? No, that's too high.
    # Actually for the step mechanism, the payment should be the integral of allocation.
    # p(v) = v1*I(v1>=2) + v2*I(v2>=2) - [I(v1>=2)*2 + I(v2>=2)*2]
    # = 0 for v1<2,v2<2; v1-2 for v1>=2,v2<2; v2-2 for v1<2,v2>=2; v1+v2-4 for both>=2
    
    for t in range(n):
        v1, v2 = val_matrix[t]
        p1 = max(0, v1 - 2) if v1 >= 2 else 0
        p2 = max(0, v2 - 2) if v2 >= 2 else 0
        payments[t] = p1 + p2
    
    return allocations, payments


def _verify_constraints(payments, allocations, val_matrix, probs, tol=1e-8):
    """Verify IC, IR, and non-negativity constraints (floating point version)."""
    n = len(payments)
    
    if np.any(payments < -tol):
        return False
    if np.any(allocations < -tol) or np.any(allocations > 1 + tol):
        return False
    
    for t in range(n):
        if payments[t] > val_matrix[t] @ allocations[t] + tol:
            return False
    
    for t in range(n):
        for t2 in range(n):
            if t == t2:
                continue
            lhs = payments[t] - payments[t2]
            rhs = val_matrix[t] @ allocations[t] - val_matrix[t] @ allocations[t2]
            if lhs > rhs + tol:
                return False
    
    return True
