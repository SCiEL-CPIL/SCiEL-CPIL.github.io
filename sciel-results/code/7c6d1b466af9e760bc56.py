# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linprog
from scipy.optimize import milp, LinearConstraint, Bounds

def construct_auction(distribution_spec, support, constraints):
    t_start = time.time()
    deadline = t_start + 75.0  # 15s safety margin before 90s kill

    # 1. Parse input to get types and probabilities
    # Assume support is a list of (v1, v2) tuples, length 16
    # distribution_spec is likely a dict {type: prob} or list of probs aligned with support
    # constraints may contain additional info but we mainly need support and probs
    
    types = []
    probs = []
    
    if isinstance(support, list) and len(support) > 0:
        # Check if support elements are tuples/lists of 2 values
        if isinstance(support[0], (tuple, list, np.ndarray)):
            types = [tuple(t) for t in support]
        else:
            # Fallback: assume standard grid
            types = [(v1, v2) for v1 in range(4) for v2 in range(4)]
    
    if isinstance(distribution_spec, dict):
        for t in types:
            # Try to find prob for this type
            prob = distribution_spec.get(t, 0.0)
            if prob is None:
                prob = 0.0
            probs.append(float(prob))
    elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
        if len(distribution_spec) == len(types):
            probs = [float(p) for p in distribution_spec]
        else:
            # Fallback: uniform
            probs = [1.0 / len(types)] * len(types)
    else:
        probs = [1.0 / len(types)] * len(types)
    
    n = len(types)
    if n == 0:
        n = 16
        types = [(v1, v2) for v1 in range(4) for v2 in range(4)]
        probs = [1.0 / 16.0] * 16

    # Normalize probabilities to ensure they sum to 1
    p_sum = sum(probs)
    if p_sum > 0:
        probs = [p / p_sum for p in probs]
    else:
        probs = [1.0 / n] * n

    p = np.array(probs)
    v = np.array(types, dtype=float)  # shape (n, 2)

    # 2. Solve the Mechanism Design LP
    # Variables: x_{t,t'} (allocation for report t' when true type is t) -> n*n vars
    #             q_{t,t'} (payment for report t' when true type is t) -> n*n vars
    # Total vars: 2 * n * n = 512 for n=16
    
    # Objective: Maximize expected revenue = sum_t p_t * (sum_{t'} x_{t,t'} * v_{t'} - q_{t,t'})
    # Wait, revenue is payment - cost? No, single bidder, no cost. Revenue = Payment.
    # Maximize sum_t p_t * q_{t,t}  (diagonal payments)
    # But we can also think of it as maximizing sum_t p_t * (sum_{t'} x_{t,t'} * v_t - q_{t,t'}) + sum_t p_t * sum_{t'} x_{t,t'} * v_t? No.
    # Standard formulation:
    # Maximize sum_t p_t * q_{t,t}
    # Subject to:
    # 1. IC: sum_{t'} x_{t,t'} * v_t' >= sum_{t'} x_{t',t} * v_t' + q_{t',t} - q_{t,t}  for all t, t'
    #    Rearranged: sum_{t'} x_{t,t'} * v_t' - q_{t',t} >= sum_{t'} x_{t',t} * v_t' - q_{t,t}
    #    Let U_t = max_{t'} (sum_{t'} x_{t,t'} * v_t' - q_{t,t'})
    #    IC: U_t >= sum_{t'} x_{t',t} * v_t' - q_{t,t} for all t, t'
    # 2. IR: U_t >= 0 for all t
    # 3. Feasibility: 0 <= x_{t,t'} <= 1
    # 4. Payments non-negative? The task says "nonnegative payments". 
    #    Usually q >= 0.
    # 5. Regret <= 1e-8. This is handled by the IC constraints if solved exactly.

    # Let's use the standard LP formulation with variables x_{t,t'} and q_{t,t}.
    # Actually, it's easier to use the "allocation rule" and "payment rule" directly.
    # Variables:
    # x_{t,t'} for t in 0..n-1, t' in 0..n-1 (allocation of good j when true type t, report t')
    # But wait, the allocation is a vector in {0,1}^2? Or probability?
    # "require feasible probabilities" -> x is probability of allocation.
    # So x_{t,t'} is a 2-vector? Or is it a scalar per good?
    # "allocations:(16,2)" -> output is a matrix of shape (16, 2).
    # This implies x_{t,t} is the allocation vector when reporting truthfully?
    # No, the output is the mechanism. The mechanism is a mapping from report to allocation/payment.
    # So we need x_{t'} (allocation when report is t') and q_{t'} (payment when report is t').
    # The output format `allocations:(16,2)` suggests `allocations[i]` is the allocation vector for report `i`.
    # So we need to find x_{t'} in [0,1]^2 and q_{t'} >= 0 for all t'.
    
    # Let's reformulate:
    # Variables: x_{t', j} for t' in 0..n-1, j in 0..1. (2n variables)
    #            q_{t'} for t' in 0..n-1. (n variables)
    # Total vars: 3n = 48.
    
    # Objective: Maximize sum_t p_t * q_t
    # Subject to:
    # 1. IC: For all true type t and report t':
    #    sum_j v_{t,j} * x_{t,j} - q_t >= sum_j v_{t,j} * x_{t',j} - q_{t'}
    #    (Utility of truthful report >= Utility of misreport)
    # 2. IR: For all t:
    #    sum_j v_{t,j} * x_{t,j} - q_t >= 0
    # 3. 0 <= x_{t', j} <= 1
    # 4. q_{t'} >= 0
    
    # This is a standard LP.
    # Number of constraints:
    # IC: n * n = 256
    # IR: n = 16
    # Bounds handled by Bounds object.
    
    # Variables ordering:
    # 0 .. n-1: q_t
    # n .. 2n-1: x_{t,0}
    # 2n .. 3n-1: x_{t,1}
    
    # c (objective coefficients):
    # Maximize sum p_t q_t -> Minimize -sum p_t q_t
    c = np.zeros(3 * n)
    c[:n] = -p  # Minimize -revenue
    
    # Inequality constraints: A_ub @ var <= b_ub
    # IC: sum_j v_{t,j} x_{t,j} - q_t - (sum_j v_{t,j} x_{t',j} - q_{t'}) >= 0
    # => -sum_j v_{t,j} x_{t,j} + q_t + sum_j v_{t,j} x_{t',j} - q_{t'} <= 0
    # Let's construct A_ub and b_ub.
    
    A_rows = []
    b_rows = []
    
    # IC Constraints
    for t in range(n):
        for t_rep in range(n):
            row = np.zeros(3 * n)
            # q_t coefficient: +1
            row[t] = 1.0
            # q_{t_rep} coefficient: -1
            row[t_rep] = -1.0
            # x_{t,j} coefficient: -v_{t,j}
            for j in range(2):
                row[n + t * 2 + j] = -v[t, j]
            # x_{t_rep,j} coefficient: +v_{t,j}
            for j in range(2):
                row[n + t_rep * 2 + j] = v[t, j]
            A_rows.append(row)
            b_rows.append(0.0)
            
    # IR Constraints
    # sum_j v_{t,j} x_{t,j} - q_t >= 0
    # => -sum_j v_{t,j} x_{t,j} + q_t <= 0
    for t in range(n):
        row = np.zeros(3 * n)
        row[t] = 1.0
        for j in range(2):
            row[n + t * 2 + j] = -v[t, j]
        A_rows.append(row)
        b_rows.append(0.0)
        
    A_ub = np.array(A_rows)
    b_ub = np.array(b_rows)
    
    # Bounds
    # q_t >= 0
    # 0 <= x <= 1
    bounds = []
    for i in range(n):
        bounds.append((0.0, np.inf))  # q
    for i in range(2 * n):
        bounds.append((0.0, 1.0))     # x
        
    # Solve LP
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if res.success:
        q_opt = res.x[:n]
        x_opt = res.x[n:]
        x_opt = x_opt.reshape(n, 2)
    else:
        # Fallback: VCG or simple heuristic
        # For single bidder, VCG is just allocating if value > price?
        # Let's try a simple heuristic: allocate good j if v_{t,j} > 1.5, price = 1.5?
        # Or just return zeros to be safe? No, we need valid output.
        # Let's try to recover from infeasibility by relaxing or using a default.
        # Default: No allocation, no payment.
        q_opt = np.zeros(n)
        x_opt = np.zeros((n, 2))
        
    # Clip small numerical errors
    x_opt = np.clip(x_opt, 0.0, 1.0)
    q_opt = np.clip(q_opt, 0.0, np.inf)
    
    # Verify constraints and regret
    # Calculate utility for each true type t when reporting t'
    # U(t, t') = sum_j v_{t,j} x_{t',j} - q_{t'}
    
    # We need to return allocations and payments.
    # The interface expects:
    # allocations: (16, 2) -> x_opt
    # payments: (16,) -> q_opt
    
    # Check time
    if time.time() > deadline:
        # Return best found so far (which is the LP solution)
        pass

    return {
        "allocations": x_opt,
        "payments": q_opt
    }
# EVOLVE-BLOCK-END
