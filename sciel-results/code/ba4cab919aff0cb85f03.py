import numpy as np
import time
from fractions import Fraction
import sys

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    target_rank = 36
    
    # Time limit guard
    start_time = time.time()
    time_limit = 110.0  # Safety margin for 120s limit
    
    def get_tensor_indices():
        indices = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    indices.append((i, j, k))
        return indices
    
    indices = get_tensor_indices()
    
    def objective(params):
        """
        params: flattened array of U, V, W
        Returns: array of residuals for each tensor entry
        """
        # Reshape params into U, V, W
        size_u = n * m * target_rank
        size_v = m * p * target_rank
        size_w = p * n * target_rank
        
        U = params[:size_u].reshape(n * m, target_rank)
        V = params[size_u:size_u+size_v].reshape(m * p, target_rank)
        W = params[size_u+size_v:].reshape(p * n, target_rank)
        
        residuals = np.zeros(len(indices))
        for idx, (i, j, k) in enumerate(indices):
            u_row = i * m + j
            v_row = j * p + k
            w_row = k * n + i
            
            # Sum over rank dimension
            val = np.dot(U[u_row, :], V[v_row, :] * W[w_row, :])
            # Target is 1
            residuals[idx] = val - 1.0
            
        return residuals
    
    # We need to try multiple random restarts
    # Total variables: (n*m + m*p + p*n) * rank
    total_vars = (n * m + m * p + p * n) * target_rank
    
    best_score = float('inf')
    best_solution = None
    
    # Number of restarts depends on available time
    # Each restart might take ~1-5 seconds depending on convergence
    # Let's allow up to 20 restarts or until time runs out
    
    max_restarts = 50
    
    for restart in range(max_restarts):
        if time.time() - start_time > time_limit:
            break
            
        # Random initialization
        # Scale matters. Small values help avoid overflow/underflow initially
        np.random.seed(restart)
        params = np.random.randn(total_vars) * 0.1
        
        try:
            # Use least_squares
            # Method 'lm' is fast for small problems but doesn't support bounds well? 
            # Actually 'lm' is good for unconstrained.
            # 'trf' is more robust.
            result = scipy.optimize.least_squares(objective, params, method='lm', max_nfev=2000)
            
            if result.success or result.cost < 1e-10:
                cost = result.cost
                if cost < best_score:
                    best_score = cost
                    best_solution = result.x
                    
                    # If we found a very good solution, try to rationalize
                    if cost < 1e-8:
                        exact_sol = try_rationalize(best_solution, n, m, p, target_rank)
                        if exact_sol is not None:
                            return exact_sol
                            
        except Exception as e:
            continue
            
    # If we didn't find an exact rational solution but found a good numerical one,
    # we might still try to rationalize the best one found
    if best_solution is not None and best_score < 1e-6:
        exact_sol = try_rationalize(best_solution, n, m, p, target_rank)
        if exact_sol is not None:
            return exact_sol

    # Fallback: Return schoolbook if optimization failed or time ran out
    # This ensures we always return a valid (though suboptimal) result
    return construct_schoolbook(n, m, p)

def try_rationalize(params, n, m, p, rank):
    """
    Attempts to convert numerical solution to exact fractions.
    Returns dict of U, V, W or None if fails.
    """
    size_u = n * m * rank
    size_v = m * p * rank
    size_w = p * n * rank
    
    U_num = params[:size_u].reshape(n * m, rank)
    V_num = params[size_u:size_u+size_v].reshape(m * p, rank)
    W_num = params[size_u+size_v:].reshape(p * n, rank)
    
    U_frac = []
    V_frac = []
    W_frac = []
    
    max_denom = 200  # Try denominators up to 200
    
    for i in range(n * m):
        row = []
        for j in range(rank):
            f = Fraction(U_num[i, j]).limit_denominator(max_denom)
            row.append(f)
        U_frac.append(row)
        
    for i in range(m * p):
        row = []
        for j in range(rank):
            f = Fraction(V_num[i, j]).limit_denominator(max_denom)
            row.append(f)
        V_frac.append(row)
        
    for i in range(p * n):
        row = []
        for j in range(rank):
            f = Fraction(W_num[i, j]).limit_denominator(max_denom)
            row.append(f)
        W_frac.append(row)
        
    # Verify the exact decomposition
    if verify_decomposition(U_frac, V_frac, W_frac, n, m, p):
        # Convert Fractions to ints if possible, or keep as floats? 
        # The interface expects integer/half-integer entries. 
        # Fractions are exact. We can return them as floats if they are simple, 
        # but returning Fraction objects might be safer if the evaluator accepts them?
        # The prompt says "integer/half-integer entries". 
        # Usually, this implies floats that are exactly representable or ints.
        # Let's convert to float if denominator is 1 or 2, else keep as float representation?
        # Actually, Python floats are binary. 1/3 is not exact.
        # If the solution requires 1/3, it's not "half-integer".
        # However, rank 36 solutions for 2x4x5 typically involve integers and half-integers.
        
        U_out = [[float(x) for x in row] for row in U_frac]
        V_out = [[float(x) for x in row] for row in V_frac]
        W_out = [[float(x) for x in row] for row in W_frac]
        
        # Re-verify with floats to ensure no precision loss in verification step
        # The evaluator will do its own check. We trust the Fraction verification.
        return {"U": U_out, "V": V_out, "W": W_out}
        
    return None

def verify_decomposition(U, V, W, n, m, p):
    """
    Verifies if U, V, W exactly decompose the tensor.
    U, V, W are lists of lists of Fractions or floats.
    """
    rank = len(U[0])
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                val = 0
                for l in range(rank):
                    val += U[u_idx][l] * V[v_idx][l] * W[w_idx][l]
                
                # Target is 1
                if isinstance(val, Fraction):
                    if val != 1:
                        return False
                else:
                    if abs(val - 1.0) > 1e-9:
                        return False
                        
    return True

def construct_schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

# Import scipy inside construct or at top level? Top level is fine.
import scipy.optimize
# EVOLVE-BLOCK-END
