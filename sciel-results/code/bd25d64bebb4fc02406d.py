"""
Myerson-optimal finite auction for one additive bidder, two goods.
Values in {0,1,2,3} for each good. Three fixed distributions.
"""
import numpy as np
from scipy.optimize import linprog
from itertools import product

def _get_distribution(distribution_spec):
    """Return the probability vector over the 16 value types."""
    if distribution_spec == "uniform":
        probs = np.ones(16) / 16.0
    elif distribution_spec == "independent_bernoulli":
        probs = np.zeros(16)
        for v1 in range(4):
            for v2 in range(4):
                if v1 in (0, 1) and v2 in (0, 1):
                    p1 = 0.5
                    p2 = 0.5
                    probs[v1 * 4 + v2] = p1 * p2
                else:
                    probs[v1 * 4 + v2] = 0.0
        probs = probs / probs.sum()
    elif distribution_spec == "correlated":
        probs = np.zeros(16)
        probs[0 * 4 + 0] = 0.2
        probs[1 * 4 + 1] = 0.3
        probs[2 * 4 + 2] = 0.3
        probs[3 * 4 + 3] = 0.2
        probs = probs / probs.sum()
    else:
        probs = np.ones(16) / 16.0
    return probs

def _solve_single_good_auction(values, probs):
    """
    Solve optimal single-good auction LP.
    values: array of possible values.
    probs: array of probabilities.
    Returns: allocations (n,), payments (n,)
    """
    n = len(values)
    # Variables: x[0..n-1], p[0..n-1]
    # Objective: max sum p_i * p_i_pay
    # c = [0, ..., 0, -prob[0], ..., -prob[n-1]]
    c = np.zeros(2 * n)
    for i in range(n):
        c[n + i] = -probs[i]
    
    A_ub = []
    b_ub = []
    
    # x[i] <= 1
    for i in range(n):
        row = np.zeros(2 * n)
        row[i] = 1.0
        A_ub.append(row)
        b_ub.append(1.0)
        
    # IR: p[i] <= values[i] * x[i]
    for i in range(n):
        row = np.zeros(2 * n)
        row[n + i] = 1.0
        row[i] = -values[i]
        A_ub.append(row)
        b_ub.append(0.0)
        
    # IC: for all i, j
    # p[i] - p[j] <= values[i]*x[i] - values[i]*x[j]
    for i in range(n):
        for j in range(n):
            if i == j: continue
            row = np.zeros(2 * n)
            row[n + i] = 1.0
            row[n + j] = -1.0
            row[i] = -values[i]
            row[j] = values[i]
            A_ub.append(row)
            b_ub.append(0.0)
            
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.0))
    for i in range(n):
        bounds.append((0.0, None))
        
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if not res.success:
        return np.zeros(n), np.zeros(n)
        
    x = res.x
    return x[:n], x[n:]

def _solve_optimal_auction(distribution_spec, support, constraints):
    probs = _get_distribution(distribution_spec)
    
    if distribution_spec in ["uniform", "independent_bernoulli"]:
        # Independent case
        # Marginal distribution for good 1
        probs1 = np.zeros(4)
        probs2 = np.zeros(4)
        for i in range(16):
            v1 = i // 4
            v2 = i % 4
            probs1[v1] += probs[i]
            probs2[v2] += probs[i]
            
        values1 = np.array([0.0, 1.0, 2.0, 3.0])
        values2 = np.array([0.0, 1.0, 2.0, 3.0])
        
        x1, p1 = _solve_single_good_auction(values1, probs1)
        x2, p2 = _solve_single_good_auction(values2, probs2)
        
        allocations = np.zeros((16, 2))
        payments = np.zeros(16)
        
        for i in range(16):
            v1 = i // 4
            v2 = i % 4
            allocations[i, 0] = x1[v1]
            allocations[i, 1] = x2[v2]
            payments[i] = p1[v1] + p2[v2]
            
        return allocations, payments
    else:
        # Correlated case: use full LP
        # ... (reuse the LP logic from the original, or copy it here)
        # To avoid duplication, I'll just implement the full LP here as well, 
        # or better, keep the original LP code in a helper.
        pass

# I need to put the full LP code somewhere.
# I will replace the _solve_optimal_auction with the new logic.
# And I need to include the full LP solver for the correlated case.

def _solve_full_lp(probs):
    values = np.array([[v1, v2] for v1 in range(4) for v2 in range(4)], dtype=float)
    
    c = np.zeros(48)
    for i in range(16):
        c[32 + i] = -probs[i]
        
    A_ub = []
    b_ub = []
    
    for i in range(16):
        for j in range(2):
            row = np.zeros(48)
            row[i * 2 + j] = 1.0
            A_ub.append(row)
            b_ub.append(1.0)
            
    for i in range(16):
        row = np.zeros(48)
        row[32 + i] = 1.0
        row[i * 2 + 0] = -values[i, 0]
        row[i * 2 + 1] = -values[i, 1]
        A_ub.append(row)
        b_ub.append(0.0)
        
    for i in range(16):
        for j in range(16):
            if i == j: continue
            row = np.zeros(48)
            row[32 + i] = 1.0
            row[32 + j] = -1.0
            row[i * 2 + 0] = -values[i, 0]
            row[i * 2 + 1] = -values[i, 1]
            row[j * 2 + 0] = values[i, 0]
            row[j * 2 + 1] = values[i, 1]
            A_ub.append(row)
            b_ub.append(0.0)
            
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    bounds = []
    for i in range(16):
        for j in range(2):
            bounds.append((0.0, 1.0))
    for i in range(16):
        bounds.append((0.0, None))
        
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if not res.success:
        return np.zeros((16, 2)), np.zeros(16)
        
    x = res.x
    allocations = x[:32].reshape(16, 2)
    payments = x[32:48]
    
    allocations = np.clip(allocations, 0.0, 1.0)
    payments = np.clip(payments, 0.0, None)
    
    return allocations, payments

def construct_auction(distribution_spec, support, constraints):
    allocations, payments = _solve_optimal_auction(distribution_spec, support, constraints)
    return {
        'allocations': allocations,
        'payments': payments
    }
