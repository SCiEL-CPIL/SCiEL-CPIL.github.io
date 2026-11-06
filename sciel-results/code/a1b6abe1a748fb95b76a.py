# EVOLVE-BLOCK-START
import time
import numpy as np

def construct_auction(distribution_spec, support, constraints):
    """
    Construct an incentive-compatible auction for one additive bidder, two goods,
    values in {0,1,2,3}, with 16 types.
    
    Uses the single-bidder structure to decouple into two independent single-item auctions.
    For each item g, we choose an allocation rule x_g(v_g) that is non-decreasing in v_g.
    The payment for a type (v1, v2) is p(v1, v2) = v1*x1(v1) + v2*x2(v2) - u(v1, v2).
    To ensure IC and IR, we can set the utility u(v1, v2) such that it satisfies the envelope theorem.
    For a single item, if x(v) is monotone, the payment can be defined as:
    p(v) = v*x(v) - integral_0^v x(s) ds
    This ensures IR (p(0)=0, u(0)=0) and IC.
    Since the bidder is additive, the total payment is the sum of payments for each item.
    Total Revenue = E[v1*x1(v1) + v2*x2(v2) - (integral_0^v1 x1(s)ds + integral_0^v2 x2(s)ds)]
    = E[sum_g (v_g * x_g(v_g) - integral_0^v_g x_g(s) ds)]
    
    We can optimize x_g(v) for each g independently.
    The objective for item g is:
    Maximize sum_v prob_g(v) * (v * x_g(v) - integral_0^v x_g(s) ds)
    
    Let's use the discrete version.
    Values are 0, 1, 2, 3.
    integral_0^v x(s) ds approximated by sum_{s < v} x(s) * (v - s)? No, in discrete single bidder, 
    the standard payment rule for monotone x is:
    p(v) = v * x(v) - sum_{s < v} x(s)
    Wait, the continuous formula is p(v) = v x(v) - \int_0^v x(t) dt.
    In discrete, the IC condition u(v) - u(v') = \int_{v'}^v x(t) dt.
    If we set u(0) = 0, then u(v) = sum_{s=0}^{v-1} x(s).
    Then p(v) = v x(v) - u(v) = v x(v) - sum_{s=0}^{v-1} x(s).
    
    Let's check IC for discrete:
    u(v) + v x(v) >= u(v') + v x(v')
    sum_{s=0}^{v-1} x(s) + v x(v) >= sum_{s=0}^{v'-1} x(s) + v x(v')
    LHS - RHS = sum_{s=v'}^{v-1} x(s) + v x(v) - v x(v')  (assuming v > v')
    = sum_{s=v'}^{v-1} (x(s) - x(v')) + (v - v') x(v') + v x(v) - v x(v') ? No.
    
    Let's use the known result: For a single item, a mechanism (x, p) is IC if and only if x is monotone and p(v) = v x(v) - \int_0^v x(t) dt (continuous) or discrete equivalent.
    Discrete equivalent: p(v) = v x(v) - \sum_{s=0}^{v-1} x(s).
    
    Let's verify IR:
    u(v) = v x(v) - p(v) = \sum_{s=0}^{v-1} x(s) >= 0 since x >= 0.
    
    So the problem reduces to choosing x_g(v) for v in {0,1,2,3} such that x_g is non-decreasing.
    Objective for item g:
    Obj_g = sum_{v=0}^3 prob_g(v) * (v * x_g(v) - sum_{s=0}^{v-1} x_g(s))
    
    Let's rewrite the objective to see dependencies.
    Obj_g = sum_{v=0}^3 prob_g(v) * v * x_g(v) - sum_{v=0}^3 prob_g(v) * sum_{s=0}^{v-1} x_g(s)
    = sum_{v=0}^3 prob_g(v) * v * x_g(v) - sum_{s=0}^2 x_g(s) * sum_{v=s+1}^3 prob_g(v)
    
    Let C_g(s) = sum_{v=s+1}^3 prob_g(v).
    Then Obj_g = sum_{v=0}^3 prob_g(v) * v * x_g(v) - sum_{s=0}^2 C_g(s) * x_g(s)
    
    Note that for v=0, the second sum doesn't include x(0) because s goes up to v-1.
    Wait, the second term is sum_{v} prob(v) * sum_{s < v} x(s).
    Swap sums: sum_{s} x(s) * sum_{v > s} prob(v).
    Yes, C_g(s) = sum_{v=s+1}^3 prob_g(v).
    
    So the coefficient of x_g(v) in the objective is:
    coeff_g(v) = prob_g(v) * v - (C_g(v) if v < 3 else 0)
    
    Wait, the first term is sum_{v} prob(v) * v * x(v).
    The second term is sum_{s} C(s) * x(s).
    So coeff(v) = prob(v) * v - C(v).
    For v=3, C(3) = 0.
    
    We want to maximize sum_{v=0}^3 coeff_g(v) * x_g(v) subject to 0 <= x_g(0) <= x_g(1) <= x_g(2) <= x_g(3) <= 1.
    
    This is a linear program with simple constraints.
    Since the constraint matrix is totally unimodular (or just simple chain), the optimal solution will be at a vertex.
    The vertices of the polytope {0 <= x0 <= x1 <= x2 <= x3 <= 1} are sequences of 0s and 1s.
    i.e., there exists a threshold k such that x(v) = 0 for v < k and x(v) = 1 for v >= k.
    
    So we just need to try all 4 possible thresholds for each item (k=0,1,2,3) and pick the best one for each item independently.
    
    Let's calculate the revenue for each threshold.
    """
    t_start = time.time()
    
    # Parse the 16 types: (v1, v2) where v1, v2 in {0,1,2,3}
    # Type index: idx = v1 * 4 + v2
    types = [(v1, v2) for v1 in range(4) for v2 in range(4)]
    n_types = 16
    
    # Extract probabilities from distribution_spec
    if isinstance(distribution_spec, dict):
        if 'probs' in distribution_spec:
            probs = np.array(distribution_spec['probs'], dtype=float)
        elif 'probabilities' in distribution_spec:
            probs = np.array(distribution_spec['probabilities'], dtype=float)
        elif 'p' in distribution_spec:
            probs = np.array(distribution_spec['p'], dtype=float)
        else:
            probs = np.ones(n_types) / n_types
    elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
        probs = np.array(distribution_spec, dtype=float)
        if len(probs) != n_types:
            probs = np.ones(n_types) / n_types
    else:
        probs = np.ones(n_types) / n_types
    
    probs = np.array(probs, dtype=float).flatten()
    if len(probs) != n_types:
        probs = np.ones(n_types) / n_types
    probs = np.clip(probs, 0, None)
    prob_sum = probs.sum()
    if prob_sum > 0:
        probs = probs / prob_sum
    else:
        probs = np.ones(n_types) / n_types
    
    # Compute marginal distributions for item 1 and item 2
    # prob_g(v) = sum_{v_other} probs[idx]
    prob_1 = np.zeros(4)
    prob_2 = np.zeros(4)
    for v1 in range(4):
        for v2 in range(4):
            idx = v1 * 4 + v2
            prob_1[v1] += probs[idx]
            prob_2[v2] += probs[idx]
    
    # Function to find best threshold for an item given its marginal probability
    def best_threshold(prob_item):
        best_rev = -np.inf
        best_k = 0
        for k in range(4):
            # x(v) = 1 if v >= k else 0
            rev = 0.0
            for v in range(4):
                x_v = 1.0 if v >= k else 0.0
                # Payment: p(v) = v * x(v) - sum_{s < v} x(s)
                # Utility: u(v) = sum_{s < v} x(s)
                # Revenue contribution: prob(v) * (v * x(v) - u(v))
                u_v = sum(1.0 for s in range(v) if s >= k)
                rev += prob_item[v] * (v * x_v - u_v)
            if rev > best_rev:
                best_rev = rev
                best_k = k
        return best_k, best_rev
    
    k1, rev1 = best_threshold(prob_1)
    k2, rev2 = best_threshold(prob_2)
    
    # Construct allocations and payments
    allocations = np.zeros((n_types, 2))
    payments = np.zeros(n_types)
    
    for v1 in range(4):
        for v2 in range(4):
            idx = v1 * 4 + v2
            
            x1 = 1.0 if v1 >= k1 else 0.0
            x2 = 1.0 if v2 >= k2 else 0.0
            
            allocations[idx, 0] = x1
            allocations[idx, 1] = x2
            
            u1 = sum(1.0 for s in range(v1) if s >= k1)
            u2 = sum(1.0 for s in range(v2) if s >= k2)
            
            p1 = v1 * x1 - u1
            p2 = v2 * x2 - u2
            
            payments[idx] = p1 + p2
    
    # Ensure non-negative payments (should be by construction, but just in case)
    payments = np.maximum(payments, 0)
    
    return {'allocations': allocations, 'payments': payments}
# EVOLVE-BLOCK-END
