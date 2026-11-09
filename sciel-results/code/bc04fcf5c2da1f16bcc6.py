import numpy as np
import time
import scipy.optimize as opt
from itertools import product

def _solve_lp_dual(prizes, weights, capacities):
    """
    Solve the LP relaxation to get dual prices (shadow prices).
    Maximize sum(p_i * x_i) subject to W x <= C, 0 <= x <= 1.
    Dual: Minimize sum(C_j * y_j) + sum(x_i * z_i)
    s.t. W^T y + z >= p, y >= 0, z >= 0.
    
    We want reduced costs: r_i = p_i - sum_j w_ij y_j.
    Items with high r_i are 'efficient' in the LP sense.
    """
    n_items = prizes.shape[1]
    n_cons = prizes.shape[0]
    
    # We solve the dual directly or use scipy linprog on the primal to get marginals.
    # Using scipy.optimize.linprog on the primal is often more stable for getting marginals if we use method='highs'.
    # Primal: Max c^T x s.t. A x <= b, 0 <= x <= ub
    # c = prizes.sum(axis=0) ? No, the problem says "total prize". 
    # The task says "prizes ... of shape (5, 100)". 
    # "highest total prize found by any ant". 
    # Usually in MCKP, prize is a single vector. Here it is (5,100).
    # The prompt says: "prizes and weights, of shape (5, 100), are uniform in [0, 1]."
    # "Each ant adds items... until none fits."
    # "Scoring: mean over instances of the highest total prize found by any ant."
    # It is ambiguous if "total prize" is sum(prizes[:, item]) or just one row.
    # Given "prizes" is (5,100), it's likely each constraint has a "prize" component? 
    # Or is it 5 different instances? No, "3 fixed instances... 100 items and 5 constraints".
    # So one instance has 5 constraints and 100 items.
    # Why 5 prizes per item? 
    # Maybe the objective is sum of all prizes? Or max?
    # Let's assume the objective is the sum of prizes across the 5 dimensions, or perhaps the "prize" is just the first row?
    # Looking at the seed: `prizes / (max_weight + 1e-9)`. It treats `prizes` as a matrix.
    # If I sum prizes, I get a vector.
    # Let's assume the objective function for the ant is the sum of prizes of selected items.
    # i.e., value(item) = prizes[:, item].sum()
    
    c = prizes.sum(axis=0)
    A = weights # (5, 100)
    b = capacities # (5,)
    
    # We want to solve: Max c^T x s.t. A x <= b, 0 <= x <= 1
    # Linprog minimizes. So Min -c^T x.
    res = opt.linprog(-c, A_ub=A, b_ub=b, bounds=(0, 1), method='highs')
    
    if not res.success:
        # Fallback: simple ratio
        return None
        
    # res.ineqlin.marginals gives the dual variables for the constraints A_ub x <= b_ub
    # These are the shadow prices (lambda).
    # Note: linprog returns marginals for inequalities. 
    # For minimization of c^T x s.t. Ax <= b, the dual y >= 0 satisfies A^T y >= c.
    # The reduced cost for item i is c_i - A^T y.
    # Since we minimized -c^T x, the dual corresponds to max c^T x.
    # Let's verify signs. 
    # If we use method='highs', res.ineqlin.marginals are the dual values.
    # For a max problem converted to min, the marginals might be negative?
    # Let's just compute the reduced costs directly from the basis if possible, or use the marginals.
    
    # Actually, a safer way to get "efficiency" is to just use the LP solution itself.
    # But we need a heuristic for *all* items, including those with x=0.
    # The dual prices y_j represent the value of relaxing constraint j.
    # An item is attractive if its prize is high relative to the cost of resources it consumes.
    # Cost of item i = sum_j w_ij * y_j.
    # Net value (reduced cost) = prize_i - cost_i.
    
    # res.ineqlin.marginals for the primal constraints (A_ub x <= b_ub) in the minimization problem.
    # If the problem is Min -c^T x s.t. A x <= b, the dual is Max b^T y s.t. A^T y >= -c, y >= 0? No.
    # Standard form: Min c^T x s.t. Ax >= b, x >= 0. Dual: Max b^T y s.t. A^T y <= c.
    # Our form: Min c^T x s.t. Ax <= b, x >= 0.
    # Multiply by -1: Max -c^T x s.t. -Ax >= -b.
    # Dual: Max (-b)^T y s.t. (-A)^T y <= -c, y >= 0.
    # Let y' = -y. Max b^T (-y') s.t. A^T (-y') <= -c => -A^T y' <= -c => A^T y' >= c.
    # This is getting messy. 
    # Let's just use the fact that for the optimal LP solution, the reduced costs of non-basic variables are non-negative (for max) or non-positive (for min).
    # Actually, `res.ineqlin.marginals` in scipy highs for `A_ub x <= b_ub` in a minimization problem returns the dual variables associated with the constraints.
    # If we minimize -c^T x, the dual variables y are such that A^T y + s = -c (s >= 0 is slack in dual, which is reduced cost in primal).
    # So reduced cost (for max) = c - A^T y.
    # Let's extract y.
    
    y = res.ineqlin.marginals
    # y might be negative? In minimization with <= constraints, duals are <= 0?
    # Let's check. If we have 1 constraint, 1 item. Max x s.t. x <= 1.
    # Min -x s.t. x <= 1.
    # Optimal x=1. Constraint active.
    # Dual y. A^T y + s = -c => 1*y + s = -(-1) = 1.
    # If x is basic, s=0? No, reduced cost of basic var is 0.
    # So y = 1.
    # Reduced cost for max problem: c - A^T y = 1 - 1 = 0. Correct.
    # So y should be positive for active constraints in a max problem context?
    # Wait, if y is from minimization of -c, and constraint is <=, the dual variable is typically <= 0 in some conventions, or >= 0?
    # In scipy, for `A_ub x <= b_ub`, the marginals are the dual values.
    # Let's assume y is the shadow price.
    
    # Calculate "resource cost" for each item
    # cost_i = sum_j w_ij * y_j
    # We want items with high prize and low cost.
    # If y is negative, cost is negative, which is bad.
    # Let's take absolute values or ensure y is positive.
    # If y is negative, it means relaxing the constraint *decreases* the objective? That doesn't make sense for a resource constraint.
    # Shadow prices for resources in a max problem should be positive.
    
    # Let's just use the LP solution x* to identify "good" items and use that to build a heuristic?
    # No, we need a score for *all* items.
    
    # Alternative: Use the dual prices to define a "profitability" score.
    # score_i = c_i / (sum_j w_ij * max(y_j, epsilon) + epsilon)
    
    y_pos = np.maximum(y, 1e-6)
    resource_cost = weights @ y_pos # (100,)
    # Heuristic: prize / resource_cost
    heuristic_lp = c / (resource_cost + 1e-9)
    
    return heuristic_lp

def _greedy_heuristic(prizes, weights, capacities):
    """
    Run a greedy packing simulation to identify which items were selected.
    Items selected earlier are considered more attractive.
    """
    n_items = prizes.shape[1]
    total_prize = prizes.sum(axis=0)
    
    # Greedy by ratio
    ratios = total_prize / (weights.sum(axis=0) + 1e-9)
    order = np.argsort(-ratios)
    
    remaining = capacities.copy()
    selected = np.zeros(n_items)
    
    for idx in order:
        if np.all(weights[:, idx] <= remaining):
            selected[idx] = 1.0
            remaining -= weights[:, idx]
            
    # Heuristic: 1 for selected, small value for others, scaled by ratio
    heuristic_greedy = selected * 10.0 + ratios / (ratios.max() + 1e-9)
    return heuristic_greedy

def _simulate_aco(prizes, weights, capacities, heuristic, n_ants=20, n_iter=30, seed=42):
    """
    Fast vectorized ACO simulation.
    Returns the best total prize found.
    """
    n_items = prizes.shape[1]
    total_prize = prizes.sum(axis=0)
    
    # Pheromone initialization
    pheromone = np.ones(n_items)
    
    # Heuristic must be positive
    h = np.maximum(heuristic, 1e-9)
    
    best_total = 0.0
    
    for it in range(n_iter):
        iter_best = 0.0
        iter_best_mask = None
        
        for _ in range(n_ants):
            remaining = capacities.copy()
            selected_mask = np.zeros(n_items, dtype=bool)
            current_prize = 0.0
            
            # Build solution
            # To speed up, we can use a loop for selection but vectorize the probability calc?
            # With 100 items, a simple loop is fine. 20*30*100 = 60k steps. Very fast.
            
            for _ in range(n_items):
                # Check which items fit
                fits = np.all(weights[:, ~selected_mask] <= remaining, axis=0)
                # Map back to original indices
                candidate_indices = np.where(~selected_mask)[0]
                if len(candidate_indices) == 0:
                    break
                
                valid_candidates = candidate_indices[fits]
                if len(valid_candidates) == 0:
                    break
                
                # Calculate probabilities
                # prob_i = pheromone[i] * h[i]
                probs = pheromone[valid_candidates] * h[valid_candidates]
                
                # Normalize
                sum_probs = probs.sum()
                if sum_probs < 1e-12:
                    break
                probs /= sum_probs
                
                # Sample
                idx = np.random.choice(valid_candidates, p=probs)
                
                # Add item
                selected_mask[idx] = True
                remaining -= weights[:, idx]
                current_prize += total_prize[idx]
                
            if current_prize > iter_best:
                iter_best = current_prize
                iter_best_mask = selected_mask.copy()
                
        # Update pheromone
        pheromone *= 0.9
        if iter_best_mask is not None:
            pheromone[iter_best_mask] += iter_best / 10.0
            
        if iter_best > best_total:
            best_total = iter_best
            
    return best_total

def heuristics(prizes, weights):
    prizes = np.asarray(prizes, dtype=float)
    weights = np.asarray(weights, dtype=float)
    
    # Handle shape
    if prizes.shape != (5, 100) or weights.shape != (5, 100):
        # Fallback
        return np.ones(100)
        
    capacities = np.full(5, 25.0)
    
    start_time = time.time()
    deadline = start_time + 290.0 # Safety margin
    
    # Generate candidate heuristics
    candidates = []
    
    # 1. LP Dual based
    try:
        h_lp = _solve_lp_dual(prizes, weights, capacities)
        if h_lp is not None and np.all(np.isfinite(h_lp)):
            candidates.append(h_lp)
    except Exception:
        pass
        
    # 2. Greedy based
    try:
        h_greedy = _greedy_heuristic(prizes, weights, capacities)
        if np.all(np.isfinite(h_greedy)):
            candidates.append(h_greedy)
    except Exception:
        pass
        
    # 3. Simple Ratio (Current baseline)
    max_weight = weights.max(axis=0)
    h_ratio = (prizes.sum(axis=0) / (max_weight + 1e-9)) ** 2
    candidates.append(h_ratio)
    
    # 4. Min Weight Ratio
    min_weight = weights.min(axis=0)
    h_min_ratio = prizes.sum(axis=0) / (min_weight + 1e-9)
    candidates.append(h_min_ratio)
    
    # 5. Sum Weight Ratio
    sum_weight = weights.sum(axis=0)
    h_sum_ratio = prizes.sum(axis=0) / (sum_weight + 1e-9)
    candidates.append(h_sum_ratio)
    
    # Evaluate candidates
    best_score = -1.0
    best_heuristic = candidates[0] # Fallback
    
    np.random.seed(42) # Fixed seed for evaluation consistency
    
    for h in candidates:
        if time.time() > deadline:
            break
            
        # Normalize heuristic to be positive
        h_pos = np.maximum(h, 1e-9)
        
        # Run simulation
        score = _simulate_aco(prizes, weights, capacities, h_pos, n_ants=20, n_iter=30, seed=42)
        
        if score > best_score:
            best_score = score
            best_heuristic = h_pos
            
    # Ensure final output is valid
    best_heuristic = np.asarray(best_heuristic, dtype=float)
    if best_heuristic.shape != (100,):
        best_heuristic = np.ones(100)
        
    best_heuristic = np.where(np.isfinite(best_heuristic), best_heuristic, 1e-9)
    best_heuristic = np.maximum(best_heuristic, 1e-9)
    
    return best_heuristic
