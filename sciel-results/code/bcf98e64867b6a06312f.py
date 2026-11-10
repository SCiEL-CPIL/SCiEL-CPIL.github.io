"""Rewritten auction design from scratch."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product

# Deadline guard: keep a margin below the 90s wall-clock limit
_WALL_LIMIT = 85.0

def _solve_auction(distribution_spec, support, constraints):
    """Solve the finite auction design problem.
    
    One additive bidder, two goods, values in {0,1,2,3}.
    Returns (allocations: (16,2), payments: (16,)) maximizing expected revenue
    subject to feasibility, nonneg payments, IR, and truthfulness on the support.
    """
    start_time = time.time()
    
    # Values per good: 4 possibilities each
    vals = [0, 1, 2, 3]
    n_reports = 16  # 4 * 4
    
    # Distribution spec: likely a 16-element vector of probabilities
    if isinstance(distribution_spec, (list, tuple, np.ndarray)):
        p = np.array(distribution_spec, dtype=float)
    else:
        # Default uniform
        p = np.ones(n_reports) / n_reports
    
    # Ensure it sums to 1
    if p.sum() > 0:
        p = p / p.sum()
    else:
        p = np.ones(n_reports) / n_reports
    
    # Map report index -> (v1, v2)
    report_to_vals = {}
    for idx in range(n_reports):
        v1 = idx % 4
        v2 = idx // 4
        report_to_vals[idx] = (vals[v1], vals[v2])
    
    # We want to maximize expected revenue = sum_i p_i * payment_i
    # Subject to:
    # 1. Feasibility: allocation probabilities in [0,1] (but with one bidder,
    #    allocation is deterministic or probabilistic per report)
    # 2. Nonnegative payments
    # 3. Individual rationality: for each true value v, expected utility >= 0
    # 4. Incentive compatibility: for each true value v, reporting v is optimal
    # 5. Regret constraint: regret <= 1e-8
    
    # With one bidder and additive values, the mechanism is:
    # For each report r, allocate goods with some probability (x1_r, x2_r) and charge payment pmt_r.
    # Utility for reporting r' when true value is v: u(r', v) = v1*x1_r' + v2*x2_r' - pmt_r'
    
    # By Myerson, for single-parameter, the optimal mechanism allocates when v >= threshold.
    # For two goods with additive values, we can use LP.
    
    # Variables: x1_r, x2_r in [0,1] for each report r (allocation probability for each good)
    #           pmt_r >= 0 for each report r (payment)
    
    # Objective: maximize sum_r p_r * pmt_r
    
    # Constraints:
    # For each report r: 0 <= x1_r <= 1, 0 <= x2_r <= 1, pmt_r >= 0
    # 
    # IC constraints: For each true value v (indexed by report r_true) and each report r':
    #   u(r_true, v_true) >= u(r', v_true)
    #   i.e., v1_true * x1_r_true + v2_true * x2_r_true - pmt_r_true >= 
    #          v1_true * x1_r' + v2_true * x2_r' - pmt_r'
    #   => pmt_r' - pmt_r_true >= v1_true * (x1_r' - x1_r_true) + v2_true * (x2_r' - x2_r_true)
    #
    # IR constraints: For each report r_true:
    #   u(r_true, v_true) >= 0
    #   i.e., v1_true * x1_r_true + v2_true * x2_r_true - pmt_r_true >= 0
    #   => pmt_r_true <= v1_true * x1_r_true + v2_true * x2_r_true
    
    # This is a linear program. Let's set it up with scipy.
    
    from scipy.optimize import linprog
    
    # Decision variables: [x1_0, x2_0, pmt_0, x1_1, x2_1, pmt_1, ..., x1_15, x2_15, pmt_15]
    # 3 * 16 = 48 variables
    
    n_vars = 3 * n_reports
    
    # Objective: maximize sum p_r * pmt_r => minimize -sum p_r * pmt_r
    c = np.zeros(n_vars)
    for r in range(n_reports):
        c[3 * r + 2] = -p[r]  # coefficient for pmt_r
    
    # Inequality constraints A_ub @ x <= b_ub
    # 1. x1_r >= 0, x2_r >= 0, pmt_r >= 0 => -x1_r <= 0, -x2_r <= 0, -pmt_r <= 0
    A_ub = []
    b_ub = []
    
    for r in range(n_reports):
        # -x1_r <= 0
        row = np.zeros(n_vars)
        row[3 * r + 0] = -1
        A_ub.append(row)
        b_ub.append(0)
        
        # -x2_r <= 0
        row = np.zeros(n_vars)
        row[3 * r + 1] = -1
        A_ub.append(row)
        b_ub.append(0)
        
        # -pmt_r <= 0
        row = np.zeros(n_vars)
        row[3 * r + 2] = -1
        A_ub.append(row)
        b_ub.append(0)
    
    # 2. x1_r <= 1, x2_r <= 1
    for r in range(n_reports):
        row = np.zeros(n_vars)
        row[3 * r + 0] = 1
        A_ub.append(row)
        b_ub.append(1)
        
        row = np.zeros(n_vars)
        row[3 * r + 1] = 1
        A_ub.append(row)
        b_ub.append(1)
    
    # 3. IR constraints: pmt_r <= v1_r * x1_r + v2_r * x2_r
    #    => pmt_r - v1_r * x1_r - v2_r * x2_r <= 0
    for r in range(n_reports):
        v1, v2 = report_to_vals[r]
        row = np.zeros(n_vars)
        row[3 * r + 0] = -v1
        row[3 * r + 1] = -v2
        row[3 * r + 2] = 1
        A_ub.append(row)
        b_ub.append(0)
    
    # 4. IC constraints: for each true report r_t and each report r':
    #    pmt_r' - pmt_r_t >= v1_t * (x1_r' - x1_r_t) + v2_t * (x2_r' - x2_r_t)
    #    => pmt_r' - pmt_r_t - v1_t * x1_r' + v1_t * x1_r_t - v2_t * x2_r' + v2_t * x2_r_t >= 0
    #    => -pmt_r' + pmt_r_t + v1_t * x1_r' - v1_t * x1_r_t + v2_t * x2_r' - v2_t * x2_r_t <= 0
    for r_t in range(n_reports):
        v1_t, v2_t = report_to_vals[r_t]
        for r_prime in range(n_reports):
            row = np.zeros(n_vars)
            # -pmt_r_prime
            row[3 * r_prime + 2] = -1
            # +pmt_r_t
            row[3 * r_t + 2] = 1
            # +v1_t * x1_r_prime
            row[3 * r_prime + 0] = v1_t
            # -v1_t * x1_r_t
            row[3 * r_t + 0] = -v1_t
            # +v2_t * x2_r_prime
            row[3 * r_prime + 1] = v2_t
            # -v2_t * x2_r_t
            row[3 * r_t + 1] = -v2_t
            A_ub.append(row)
            b_ub.append(0)
    
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    # Bounds: all variables have lower bound 0, no upper bound (except x1,x2 which we handled)
    bounds = [(0, None)] * n_vars
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > _WALL_LIMIT - 5:
        # Not enough time, return a trivial feasible solution
        allocations = np.zeros((n_reports, 2))
        payments = np.zeros(n_reports)
        return {'allocations': allocations, 'payments': payments}
    
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    # Extract solution
    allocations = np.zeros((n_reports, 2))
    payments = np.zeros(n_reports)
    
    if result.status == 0:  # Optimal
        for r in range(n_reports):
            allocations[r, 0] = max(0, result.x[3 * r + 0])
            allocations[r, 1] = max(0, result.x[3 * r + 1])
            payments[r] = max(0, result.x[3 * r + 2])
    else:
        # Fallback: simple mechanism - allocate both goods when value high enough, charge VCG
        # For single bidder, VCG payment is 0. But IR requires nonnegative utility.
        # Let's use a simple threshold mechanism:
        # Allocate good 1 if v1 >= 2, good 2 if v2 >= 2. Payment = 0.
        for r in range(n_reports):
            v1, v2 = report_to_vals[r]
            allocations[r, 0] = 1.0 if v1 >= 2 else 0.0
            allocations[r, 1] = 1.0 if v2 >= 2 else 0.0
            payments[r] = 0.0
    
    # Verify constraints and fix any numerical issues
    for r in range(n_reports):
        v1, v2 = report_to_vals[r]
        # Clamp allocations to [0,1]
        allocations[r, 0] = np.clip(allocations[r, 0], 0, 1)
        allocations[r, 1] = np.clip(allocations[r, 1], 0, 1)
        payments[r] = max(0, payments[r])
        
        # Ensure IR: payment <= value
        max_payment = v1 * allocations[r, 0] + v2 * allocations[r, 1]
        payments[r] = min(payments[r], max_payment)
    
    # Final verification of IC (with tolerance)
    for r_t in range(n_reports):
        v1_t, v2_t = report_to_vals[r_t]
        # Utility of reporting truthfully
        util_true = v1_t * allocations[r_t, 0] + v2_t * allocations[r_t, 1] - payments[r_t]
        for r_p in range(n_reports):
            # Utility of misreporting
            util_misrep = v1_t * allocations[r_p, 0] + v2_t * allocations[r_p, 1] - payments[r_p]
            if util_misrep - util_true > 1e-8:
                # IC violated; adjust payment for r_t to fix
                payments[r_t] -= (util_misrep - util_true) + 1e-9
                payments[r_t] = max(0, payments[r_t])
                # Re-check IR
                max_payment = v1_t * allocations[r_t, 0] + v2_t * allocations[r_t, 1]
                payments[r_t] = min(payments[r_t], max_payment)
    
    # Ensure payments are nonnegative
    payments = np.maximum(payments, 0)
    
    # Final IR check
    for r in range(n_reports):
        v1, v2 = report_to_vals[r]
        max_payment = v1 * allocations[r, 0] + v2 * allocations[r, 1]
        payments[r] = min(payments[r], max_payment)
        payments[r] = max(0, payments[r])
    
    return {'allocations': allocations, 'payments': payments}


def construct_auction(distribution_spec, support, constraints):
    """Entry point: construct an incentive-compatible finite auction.
    
    Args:
        distribution_spec: specification of the value distribution
        support: the support of the value space
        constraints: additional constraints
        
    Returns:
        dict with 'allocations' (16,2) array and 'payments' (16,) array
    """
    start_time = time.time()
    
    try:
        result = _solve_auction(distribution_spec, support, constraints)
        return result
    except Exception as e:
        # Fallback: return a trivially valid mechanism
        n_reports = 16
        allocations = np.zeros((n_reports, 2))
        payments = np.zeros(n_reports)
        return {'allocations': allocations, 'payments': payments}
# EVOLVE-BLOCK-END
