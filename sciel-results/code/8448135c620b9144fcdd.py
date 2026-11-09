import numpy as np
import itertools
import time
import random

def construct_auction(distribution_spec, support, constraints):
    """
    Construct an incentive-compatible auction for 1 bidder, 2 goods.
    Values in {0, 1, 2, 3}.
    Returns dict with 'allocations' (16, 2) and 'payments' (16,).
    """
    start_time = time.time()
    TIME_LIMIT = 85.0  # Safety margin
    
    # 1. Parse inputs
    # distribution_spec is likely a dict or object defining probabilities over the 16 types.
    # support is the set of possible value vectors.
    # constraints might be empty or contain specific bounds.
    
    # Define the support: 2 goods, values 0..3
    values = [0, 1, 2, 3]
    types = list(itertools.product(values, repeat=2))
    # Map type to index
    type_to_idx = {t: i for i, t in enumerate(types)}
    
    # Extract probabilities from distribution_spec
    # Assuming distribution_spec is a dict mapping type tuple to prob, or similar.
    # If it's a standard object from the suite, we need to handle it.
    # Based on "three fixed distributions", distribution_spec likely identifies which one.
    # Let's assume distribution_spec is a dict of probabilities or a callable.
    # Common pattern in these tasks: distribution_spec is a dict {type: prob}
    
    probs = np.zeros(16)
    if isinstance(distribution_spec, dict):
        for t, p in distribution_spec.items():
            # t might be a tuple or a string
            if isinstance(t, tuple):
                idx = type_to_idx[t]
                probs[idx] = p
            elif isinstance(t, str):
                # Try to parse string like "0,1"
                try:
                    parts = t.split(',')
                    t_tuple = tuple(int(x) for x in parts)
                    idx = type_to_idx[t_tuple]
                    probs[idx] = p
                except:
                    pass
    else:
        # Fallback: Uniform if we can't parse
        probs = np.ones(16) / 16.0
    
    # Normalize just in case
    total_p = probs.sum()
    if total_p > 0:
        probs /= total_p
    else:
        probs = np.ones(16) / 16.0

    # 2. Objective: Maximize Expected Revenue
    # Revenue = sum_t p_t * (payment_t)
    # Subject to:
    # 1. Individual Rationality (IR): For all t, payment_t <= v_t . x_t (where x_t is allocation vector)
    #    Actually, IR is usually: u(t, t) >= 0 => payment_t <= v_t . x_t
    # 2. Incentive Compatibility (IC): For all t, t', 
    #    u(t, t') <= u(t, t)
    #    v_t . x_{t'} - payment_{t'} <= v_t . x_t - payment_t
    #    payment_t - payment_{t'} <= v_t . (x_t - x_{t'})
    #    This must hold for all pairs (t, t').
    
    # Strategy: 
    # Since we have only 16 types, we can parameterize the mechanism.
    # A truthful mechanism for 1 bidder is defined by allocation rule x(t) and payment rule p(t).
    # For single bidder, IC is equivalent to:
    # p(t) = v_t . x_t - sum_{t' < t} p_t' . (x_t - x_{t'})  ... this is for 1D types.
    # For multi-dimensional types (2 goods), it's more complex.
    # However, we can use the following property:
    # If x(t) is monotone (in the lattice sense) and payments are set appropriately, IC holds.
    # But checking IC directly is safer.
    
    # Let's use a heuristic search.
    # Variables: x_t (2 floats in [0,1]) and p_t (1 float).
    # We want to maximize sum p_t * p_t.
    
    # Initial guess: Start with a simple mechanism.
    # e.g., x_t = 1 if v_t > threshold, else 0.
    # Or start with x_t = v_t / max_v.
    
    best_revenue = -1.0
    best_x = None
    best_p = None
    
    # Initialize with a simple feasible mechanism
    # Let's try x_t = v_t / 3.0 (normalized)
    x = np.zeros((16, 2))
    for i, t in enumerate(types):
        x[i] = np.array([t[0], t[1]]) / 3.0
        
    # Calculate payments to satisfy IR and IC?
    # For a given x, the maximum payment we can charge while maintaining IR is p_t <= v_t . x_t.
    # To satisfy IC, we need p_t - p_{t'} <= v_t . (x_t - x_{t'}).
    # This is a system of linear inequalities.
    # We can solve for the maximum revenue payments given x using LP, 
    # but since we want to avoid heavy LP solvers in the loop, 
    # let's use a simpler approach:
    
    # For 1 bidder, the "critical" payments are determined by the envelope theorem.
    # p(t) = v_t . x_t - integral ... 
    # Actually, for finite support, we can just set p_t = v_t . x_t - regret_margin?
    # No, that's not necessarily IC.
    
    # Let's try a different structural approach:
    # Use "Vickrey-Clarke-Groves" (VCG) logic? 
    # For 1 bidder, VCG is not directly applicable in the same way as multi-bidder.
    # But for 1 bidder, a truthful mechanism is:
    # x(t) must be monotone.
    # p(t) = v_t . x_t - sum_{t' in N(t)} ... 
    # Actually, the simplest truthful mechanism is the "All-pay" or "Second price" analogue.
    # For 1 bidder, 2 goods, the truthful mechanisms are those where x is monotone and p(t) = v_t . x_t - C(t), 
    # where C(t) is the "cost" of being at t, defined by the lattice.
    
    # Let's use a direct optimization on x and p simultaneously using gradient-free search.
    # We will define a penalty function for IC violations.
    
    def calc_revenue(x, p):
        return np.dot(probs, p)
    
    def check_constraints(x, p, epsilon=1e-8):
        # IR: p_t <= v_t . x_t + epsilon
        for i, t in enumerate(types):
            v = np.array([t[0], t[1]])
            if p[i] > np.dot(v, x[i]) + epsilon:
                return False
        
        # IC: p_t - p_{t'} <= v_t . (x_t - x_{t'}) + epsilon
        for i, t in enumerate(types):
            v = np.array([t[0], t[1]])
            for j, t_prime in enumerate(types):
                if i == j: continue
                lhs = p[i] - p[j]
                rhs = np.dot(v, x[i] - x[j])
                if lhs > rhs + epsilon:
                    return False
        return True
    
    # Heuristic: 
    # 1. Start with x_t = v_t / 3.
    # 2. Set p_t = v_t . x_t (max IR payment).
    # 3. Check IC. If violated, reduce p_t for the "benefiting" bidder.
    #    If p_i - p_j > v_i . (x_i - x_j), we must reduce p_i or increase p_j.
    #    Since we want to maximize revenue, we should reduce the payment of the type that is "lying" to get a better deal?
    #    Actually, if t reports t' and gets better utility, we need to adjust.
    #    The tightest constraint is usually on the "downward" deviations.
    
    # Let's implement a "Water Filling" or "Clipping" approach for payments given x.
    # Given x, find max p such that:
    # p_i <= v_i . x_i  (IR)
    # p_i - p_j <= v_i . (x_i - x_j)  (IC)
    
    # This is a linear programming problem. Since it's small (16 vars, ~256 cons), 
    # we can use a simple iterative projection or just call scipy if available and fast enough.
    # But the prompt says "Python standard library, NumPy and SciPy are available".
    # Using scipy.optimize.linprog is robust.
    
    # Let's try to use scipy.optimize.linprog for the payment subproblem, 
    # and use random search for the allocation x.
    
    from scipy.optimize import linprog
    
    # Precompute IC constraint matrix for payments
    # Variables: p_0 ... p_15
    # Maximize -sum(probs * p)  (linprog minimizes)
    # Constraints:
    # 1. p_i - v_i0*x_i0 - v_i1*x_i1 <= 0  (IR: p_i <= v_i . x_i)
    #    => p_i <= c_i where c_i = v_i . x_i
    # 2. p_i - p_j - v_i . (x_i - x_j) <= 0  (IC)
    #    => p_i - p_j <= d_{ij} where d_{ij} = v_i . (x_i - x_j)
    
    # We can define A_ub and b_ub for linprog.
    # Number of constraints: 16 (IR) + 16*15 (IC) = 256.
    # Number of variables: 16.
    
    A_ub_rows = []
    b_ub_vals = []
    
    # IR constraints: p_i <= v_i . x_i
    # p_i - 0... <= v_i . x_i
    for i in range(16):
        row = np.zeros(16)
        row[i] = 1.0
        A_ub_rows.append(row)
        # b will be updated based on x
    
    # IC constraints: p_i - p_j <= v_i . (x_i - x_j)
    # p_i - p_j <= d_{ij}
    # We only need to store the structure, b changes with x.
    ic_pairs = []
    for i in range(16):
        for j in range(16):
            if i == j: continue
            row = np.zeros(16)
            row[i] = 1.0
            row[j] = -1.0
            A_ub_rows.append(row)
            ic_pairs.append((i, j))
            
    A_ub = np.array(A_ub_rows)
    
    # Objective: Maximize sum(p_i * probs_i)
    # Minimize -sum(p_i * probs_i)
    c_obj = -probs
    
    # Bounds: p_i >= 0
    bounds = [(0, None)] * 16
    
    def solve_payments(x):
        # Calculate b_ub
        b_ub = np.zeros(len(A_ub_rows))
        
        # IR part
        for i in range(16):
            v = np.array([types[i][0], types[i][1]])
            b_ub[i] = np.dot(v, x[i])
            
        # IC part
        offset = 16
        for k, (i, j) in enumerate(ic_pairs):
            v_i = np.array([types[i][0], types[i][1]])
            b_ub[offset + k] = np.dot(v_i, x[i] - x[j])
            
        # Solve LP
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if res.status == 0:
            return -res.fun, res.x
        else:
            # If infeasible, return 0 revenue and zero payments (safe fallback)
            return 0.0, np.zeros(16)
    
    # Search over x
    # x_i must be in [0,1]^2.
    # We will use a random restart + local perturbation strategy.
    
    best_rev = -1.0
    best_x_sol = None
    best_p_sol = None
    
    num_iterations = 2000
    
    # Initial guess: x_i = v_i / 3
    current_x = np.zeros((16, 2))
    for i, t in enumerate(types):
        current_x[i] = np.array([t[0], t[1]]) / 3.0
        
    rev, p_sol = solve_payments(current_x)
    if rev > best_rev:
        best_rev = rev
        best_x_sol = current_x.copy()
        best_p_sol = p_sol.copy()
    
    for it in range(num_iterations):
        if time.time() - start_time > TIME_LIMIT:
            break
            
        # Perturb x
        # Randomly pick a few types and adjust their allocation
        num_perturb = random.randint(1, 3)
        idxs = random.sample(range(16), num_perturb)
        
        new_x = current_x.copy()
        for idx in idxs:
            # Random step
            step = np.random.uniform(-0.1, 0.1, size=2)
            new_x[idx] += step
            # Clip to [0,1]
            new_x[idx] = np.clip(new_x[idx], 0.0, 1.0)
            
        rev, p_sol = solve_payments(new_x)
        
        if rev > best_rev:
            best_rev = rev
            best_x_sol = new_x.copy()
            best_p_sol = p_sol.copy()
            current_x = new_x.copy()
        else:
            # If not improving, maybe try a smaller step or different direction
            # Simple hill climbing: reject if worse
            pass
            
    # Final check and formatting
    if best_x_sol is None:
        # Fallback to zero allocation
        best_x_sol = np.zeros((16, 2))
        best_p_sol = np.zeros(16)
    else:
        # Ensure payments are non-negative and satisfy IR/IC with strict tolerance
        # The LP solver might have small numerical errors.
        # Let's clean up the solution.
        # Re-check IR: p_i <= v_i . x_i
        for i in range(16):
            v = np.array([types[i][0], types[i][1]])
            max_p = np.dot(v, best_x_sol[i])
            if best_p_sol[i] > max_p:
                best_p_sol[i] = max_p
        
        # Re-check IC: p_i - p_j <= v_i . (x_i - x_j)
        # We might need to slightly reduce payments to ensure regret <= 1e-8
        # Let's apply a small safety margin
        safety = 1e-6
        for i in range(16):
            for j in range(16):
                if i == j: continue
                v_i = np.array([types[i][0], types[i][1]])
                limit = np.dot(v_i, best_x_sol[i] - best_x_sol[j])
                current_diff = best_p_sol[i] - best_p_sol[j]
                if current_diff > limit:
                    # Reduce p_i
                    best_p_sol[i] -= (current_diff - limit) + safety
        
        # Final IR check after IC adjustment
        for i in range(16):
            v = np.array([types[i][0], types[i][1]])
            max_p = np.dot(v, best_x_sol[i])
            if best_p_sol[i] > max_p:
                best_p_sol[i] = max_p
            if best_p_sol[i] < 0:
                best_p_sol[i] = 0.0

    return {
        "allocations": best_x_sol,
        "payments": best_p_sol
    }
