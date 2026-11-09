# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s kill
    
    denominators = list(range(2, 49))  # [2, 3, ..., 48]
    n = len(denominators)
    
    # Precompute reciprocals as Fractions for exact arithmetic
    reciprocals = [Fraction(1, d) for d in denominators]
    
    # Greedy initial solution as fallback
    selected = []
    total = Fraction(0)
    for i in range(n):
        if total + reciprocals[i] < 1:
            selected.append(i)
            total += reciprocals[i]
    
    best_indices = selected[:]
    best_deficit = 1 - total
    
    # Now use MILP via scipy.optimize.milp
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        import numpy as np
        from math import lcm
        from functools import reduce
        
        # Compute LCM of all denominators
        L = reduce(lcm, denominators)
        
        # Weight for each item: LCM / denominator (integer)
        # We want to maximize sum of weights subject to sum < L
        # i.e., sum(weights_i * x_i) <= L - 1
        weights = np.array([L // d for d in denominators], dtype=np.int64)
        total_L = L
        
        # Objective: maximize sum(weights_i * x_i)
        # milp minimizes, so we negate: minimize -sum(weights_i * x_i)
        c = -weights.astype(np.float64)
        
        # Constraint: sum(weights_i * x_i) <= L - 1
        A = weights.astype(np.float64)
        A_ub = np.array([A])  # shape (1, n)
        b_ub = np.array([total_L - 1], dtype=np.float64)
        
        # Also: at least one item selected
        # sum(x_i) >= 1
        A_ge = np.ones((1, n), dtype=np.float64)
        b_ge = np.array([1], dtype=np.float64)
        
        # Combine constraints
        # We need: A_ub @ x <= b_ub AND A_ge @ x >= b_ge
        # scipy milp uses LinearConstraint(A, lb, ub) where lb <= A@x <= ub
        
        # For the sum constraint: 0 <= sum(x_i) <= n, and sum(x_i) >= 1
        # For the weight constraint: 0 <= sum(weights_i * x_i) <= L-1
        
        # Let's use a single LinearConstraint with 2 rows
        A_mat = np.vstack([A_ub, A_ge])  # 2 x n
        lb = np.array([0, 1], dtype=np.float64)
        ub = np.array([total_L - 1, n], dtype=np.float64)
        
        constraints = LinearConstraint(A_mat, lb, ub)
        
        # Bounds: each x_i in {0, 1}
        integrality = np.ones(n, dtype=np.int64)
        lb_bounds = np.zeros(n, dtype=np.float64)
        ub_bounds = np.ones(n, dtype=np.float64)
        bounds = Bounds(lb_bounds, ub_bounds)
        
        # Solve MILP
        result = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
        
        if result.success and result.x is not None:
            milp_indices = [i for i in range(n) if result.x[i] > 0.5]
            
            if len(milp_indices) > 0:
                milp_sum = sum(reciprocals[i] for i in milp_indices)
                if milp_sum < 1:
                    milp_deficit = 1 - milp_sum
                    if milp_deficit < best_deficit:
                        best_indices = milp_indices
                        best_deficit = milp_deficit
        
        # If MILP didn't find something better, try a few more approaches
        # with different constraint formulations or time limits
        
    except Exception:
        pass
    
    # Fallback: quick local search to try to improve
    if (time.time() - start_time) < deadline - 10:
        improved = True
        while improved and (time.time() - start_time) < deadline - 5:
            improved = False
            current_set = set(best_indices)
            current_sum = sum(reciprocals[i] for i in best_indices)
            
            # Try adding one element
            for add_idx in range(n):
                if add_idx in current_set:
                    continue
                new_sum = current_sum + reciprocals[add_idx]
                if new_sum < 1:
                    new_deficit = 1 - new_sum
                    if new_deficit < best_deficit:
                        best_indices.append(add_idx)
                        best_deficit = new_deficit
                        improved = True
                        break
            if improved:
                continue
            
            # Try 1-swap
            for remove_idx in list(current_set):
                for add_idx in range(n):
                    if add_idx in current_set:
                        continue
                    new_sum = current_sum - reciprocals[remove_idx] + reciprocals[add_idx]
                    if new_sum < 1:
                        new_deficit = 1 - new_sum
                        if new_deficit < best_deficit:
                            best_indices = [i for i in best_indices if i != remove_idx]
                            best_indices.append(add_idx)
                            best_deficit = new_deficit
                            improved = True
                            break
                if improved:
                    break
    
    # Final check
    if not best_indices:
        return [0]
    
    return best_indices
# EVOLVE-BLOCK-END
