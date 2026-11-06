"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    # Define the type space: 16 types, values in {0,1,2,3} for 2 goods
    values = [0, 1, 2, 3]
    types = [(v1, v2) for v1 in values for v2 in values]
    n_types = len(types)  # 16
    
    # Parse distribution probabilities
    if isinstance(distribution_spec, dict):
        if 'probs' in distribution_spec:
            probs = np.array(distribution_spec['probs'], dtype=float)
        elif 'distribution' in distribution_spec:
            probs = np.array(distribution_spec['distribution'], dtype=float)
        elif 'probabilities' in distribution_spec:
            probs = np.array(distribution_spec['probabilities'], dtype=float)
        else:
            # Try to find any key that looks like probabilities
            for key in distribution_spec:
                if isinstance(distribution_spec[key], (list, np.ndarray)):
                    arr = np.array(distribution_spec[key], dtype=float)
                    if len(arr) == n_types:
                        probs = arr
                        break
            else:
                # Default: uniform distribution
                probs = np.ones(n_types) / n_types
    elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
        probs = np.array(distribution_spec, dtype=float)
    else:
        probs = np.ones(n_types) / n_types
    
    # Ensure probabilities are valid
    if len(probs) != n_types:
        # Pad or truncate if needed
        if len(probs) < n_types:
            new_probs = np.zeros(n_types)
            new_probs[:len(probs)] = probs
            probs = new_probs
        else:
            probs = probs[:n_types]
    
    probs = np.maximum(probs, 0)
    prob_sum = probs.sum()
    if prob_sum > 0:
        probs = probs / prob_sum
    else:
        probs = np.ones(n_types) / n_types
    
    # For single bidder with two goods, we can decouple into two independent single-item auctions
    # Each good has values in {0,1,2,3}
    # The allocation rule x_i(v) must be non-decreasing in v for incentive compatibility
    # Payment is determined by: p_i(v) = v * x_i(v) - integral_0^v x_i(t) dt
    
    # Since values are discrete {0,1,2,3}, we can enumerate all monotone allocation rules
    # For a single item with values {0,1,2,3}, a monotone allocation rule is determined by
    # x(0), x(1), x(2), x(3) where 0 <= x(0) <= x(1) <= x(2) <= x(3) <= 1
    
    # We need to find the optimal monotone allocation for each good independently
    # Revenue from good i is: sum_v p(v) * [v * x(v) - sum_{u<v} x(u) * (v-u) * delta]
    # For discrete case: payment at value v is v*x(v) - sum_{u=0}^{v-1} x(u)
    
    def solve_single_item(probs_1d):
        """Solve single-item auction for values {0,1,2,3}"""
        # probs_1d: array of 4 probabilities for values 0,1,2,3
        # Find monotone x = [x0, x1, x2, x3] with 0 <= x0 <= x1 <= x2 <= x3 <= 1
        # to maximize expected revenue
        
        # Expected revenue = sum_v probs[v] * (v * x[v] - sum_{u=0}^{v-1} x[u])
        # = probs[0] * (0*x0 - 0) + probs[1] * (1*x1 - x0) + probs[2] * (2*x2 - x0 - x1) + probs[3] * (3*x3 - x0 - x1 - x2)
        # = -probs[1]*x0 - probs[2]*x0 - probs[3]*x0 + probs[1]*x1 - probs[2]*x1 - probs[3]*x1 + 2*probs[2]*x2 - probs[3]*x2 + 3*probs[3]*x3
        # = -(probs[1]+probs[2]+probs[3])*x0 + (probs[1]-probs[2]-probs[3])*x1 + (2*probs[2]-probs[3])*x2 + 3*probs[3]*x3
        
        c0 = -(probs[1] + probs[2] + probs[3])
        c1 = (probs[1] - probs[2] - probs[3])
        c2 = (2 * probs[2] - probs[3])
        c3 = 3 * probs[3]
        
        # Maximize c0*x0 + c1*x1 + c2*x2 + c3*x3
        # Subject to: 0 <= x0 <= x1 <= x2 <= x3 <= 1
        # This is a linear program
        
        # Variables: x0, x1, x2, x3
        # Objective: maximize c^T x, so minimize -c^T x
        c_obj = -np.array([c0, c1, c2, c3])
        
        # Constraints:
        # x0 <= x1 -> x0 - x1 <= 0
        # x1 <= x2 -> x1 - x2 <= 0
        # x2 <= x3 -> x2 - x3 <= 0
        # x0 >= 0 -> -x0 <= 0
        # x1 >= 0 -> -x1 <= 0
        # x2 >= 0 -> -x2 <= 0
        # x3 >= 0 -> -x3 <= 0
        # x0 <= 1 -> x0 <= 1
        # x1 <= 1 -> x1 <= 1
        # x2 <= 1 -> x2 <= 1
        # x3 <= 1 -> x3 <= 1
        
        A_ub = np.array([
            [1, -1, 0, 0],   # x0 <= x1
            [0, 1, -1, 0],   # x1 <= x2
            [0, 0, 1, -1],   # x2 <= x3
            [-1, 0, 0, 0],   # -x0 <= 0
            [0, -1, 0, 0],   # -x1 <= 0
            [0, 0, -1, 0],   # -x2 <= 0
            [0, 0, 0, -1],   # -x3 <= 0
            [1, 0, 0, 0],    # x0 <= 1
            [0, 1, 0, 0],    # x1 <= 1
            [0, 0, 1, 0],    # x2 <= 1
            [0, 0, 0, 1],    # x3 <= 1
        ])
        b_ub = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1])
        
        # Bounds: 0 <= x_i <= 1
        bounds = [(0, 1)] * 4
        
        result = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if result.success:
            x = result.x
        else:
            # Fallback: x = 0 for all (no allocation)
            x = np.zeros(4)
        
        return x
    
    # For each good, compute marginal probabilities
    # Good 1: probability of each value v1, marginalized over v2
    probs_good1 = np.zeros(4)
    probs_good2 = np.zeros(4)
    
    for i, (v1, v2) in enumerate(types):
        probs_good1[v1] += probs[i]
        probs_good2[v2] += probs[i]
    
    # Solve for each good independently
    x_good1 = solve_single_item(probs_good1)
    x_good2 = solve_single_item(probs_good2)
    
    # Construct allocations and payments for all 16 types
    allocations = np.zeros((16, 2))
    payments = np.zeros(16)
    
    for i, (v1, v2) in enumerate(types):
        allocations[i, 0] = x_good1[v1]
        allocations[i, 1] = x_good2[v2]
        
        # Payment for good 1: p1(v1) = v1 * x1(v1) - sum_{u=0}^{v1-1} x1(u)
        p1 = v1 * x_good1[v1] - np.sum(x_good1[:v1])
        # Payment for good 2: p2(v2) = v2 * x2(v2) - sum_{u=0}^{v2-1} x2(u)
        p2 = v2 * x_good2[v2] - np.sum(x_good2[:v2])
        
        payments[i] = p1 + p2
    
    # Ensure non-negative payments (clip to 0)
    payments = np.maximum(payments, 0)
    
    # Check if we're still within time limit
    if time.time() > deadline:
        # Return what we have
        return {'allocations': allocations, 'payments': payments}
    
    # Verify incentive compatibility and individual rationality
    # IC: For each type, reporting truthfully gives max utility
    # Utility of type (v1,v2) reporting (v1',v2') = v1*x1(v1') + v2*x2(v2') - p1(v1') - p2(v2')
    # For IC, we need: v1*x1(v1) + v2*x2(v2) - p1(v1) - p2(v2) >= v1*x1(v1') + v2*x2(v2') - p1(v1') - p2(v2') for all v1',v2'
    
    # Since we decoupled, IC holds if each single-item mechanism is IC
    # For single item with monotone allocation, the payment rule p(v) = v*x(v) - sum_{u<v} x(u) ensures IC
    
    # Check IR: utility at truth must be >= 0
    # Utility at truth: v1*x1(v1) + v2*x2(v2) - p1(v1) - p2(v2) = sum_{u<v1} x1(u) + sum_{u<v2} x2(u) >= 0
    # This is automatically satisfied since x >= 0
    
    return {'allocations': allocations, 'payments': payments}
# EVOLVE-BLOCK-END
