# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog

def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.
    
    Returns:
        (domain, values): a tuple of two numpy arrays of equal length describing f
    """
    start_time = time.time()
    deadline = start_time + 580  # safety margin before 630s limit

    # Fallback to seed result
    def fallback():
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return domain, values

    # Try decreasing K values until feasible and time allows
    # We start with a moderate K to ensure the LP solves within time limits
    for K in [5000, 2000, 1000, 500, 200, 100]:
        if time.time() > deadline:
            return fallback()

        max_x = 10 * K
        
        # Dual LP:
        # Minimize sum_{x=1}^{10K} y(x)
        # Subject to sum_{x=k}^{10K} y(x) >= log(k)/k for k=2..K
        # y(x) >= 0
        
        # The constraint matrix A is K-1 by 10K.
        # A[i, x] = 1 if x >= k_i else 0, where k_i = i+2
        # This matrix is lower triangular in a cumulative sense.
        
        # To avoid constructing the full (K-1) x 10K matrix explicitly if it's too large,
        # we can exploit the structure. However, for K=5000, 5000*50000 = 250M entries.
        # This is ~2GB in float64. Might be too big or slow.
        
        # Let's try K=2000 first. 2000 * 20000 = 40M entries. ~320MB. Feasible.
        
        # Construct A
        # Rows correspond to k=2..K
        # Columns correspond to x=1..10K
        # A[i, j] = 1 if (j+1) >= (i+2) i.e., j >= i+1
        
        # We can build this efficiently or use scipy sparse? 
        # linprog with 'highs' supports sparse matrices.
        
        from scipy.sparse import csc_matrix
        import scipy.sparse as sp
        
        # Build A in CSR/CSC format
        # Row i (0-indexed) corresponds to k = i+2
        # It has 1s in columns j such that j+1 >= i+2 => j >= i+1
        # So for row i, columns i+1 to 10K-1 are 1.
        
        rows = []
        cols = []
        data = []
        
        # This loop might be slow in Python. Let's vectorize or use block construction.
        # Actually, A is a "staircase" matrix.
        # Row 0 (k=2): cols 1..10K-1 are 1
        # Row 1 (k=3): cols 2..10K-1 are 1
        # ...
        # Row K-2 (k=K): cols K-1..10K-1 are 1
        
        # We can construct the indices directly.
        # Number of non-zeros in row i is (10K - 1) - (i+1) + 1 = 10K - i - 1
        # Total non-zeros = sum_{i=0}^{K-2} (10K - i - 1)
        
        # For K=2000, 10K=20000.
        # Row 0: 19999 non-zeros
        # Row 1998: 20000 - 1998 - 1 = 18001 non-zeros
        
        # Let's just build the dense matrix if K is small enough, or sparse if not.
        # For K=2000, dense is 2000x20000 = 40M floats = 320MB. This is okay for most systems.
        # But let's use sparse to be safe and faster for the solver.
        
        # Efficient sparse construction:
        # For each row i, the start column is i+1, end is 10K-1.
        # We can use a loop to append rows, or pre-allocate.
        
        # Let's try a different approach: 
        # The dual of the dual is the primal. 
        # Primal: Max sum_{k=2}^K f(k) log(k)/k
        # s.t. sum_{k=2}^K f(k) floor(x/k) <= 1 for x=1..10K
        #      f(k) <= 10
        #      f(k) >= -10
        # Note: The constraint sum f(k)/k = 0 is handled by the evaluator via f(1).
        # In the primal formulation for the LP, we should include the constraint 
        # sum_{k=2}^K f(k)/k = 0 ? 
        # No, the evaluator adjusts f(1) to make sum f(k)/k = 0.
        # The objective a = - sum f(k) log(k)/k.
        # If we maximize a, we are minimizing sum f(k) log(k)/k.
        
        # Let's stick to the dual LP on y(x) as it avoids the f(1) complication in the LP itself.
        # The dual LP gives the optimal value of the primal (with f(1) free/unconstrained in a way that cancels out? 
        # Actually, the primal constraint M(x) <= 1 includes the term f(1)*x.
        # If we exclude k=1 from the LP, we are implicitly assuming f(1) is chosen to satisfy the constraints.
        # The dual LP formulation:
        # Min sum y(x)
        # s.t. sum_{x=k}^{10K} y(x) >= log(k)/k
        # This corresponds to the primal where f(k) are free variables?
        # If f(k) are free, the dual constraints are equalities. But here we have inequalities.
        # This implies f(k) >= 0 in the primal?
        # If f(k) can be negative, the dual constraint should be an equality?
        # Let's check the signs.
        # Primal: Max c^T f
        # s.t. A f <= b
        #      f <= 10
        #      f >= -10
        # Dual: Min b^T y + 10^T u - 10^T v
        # s.t. A^T y + u - v = c
        #      y, u, v >= 0
        
        # Here A is the matrix with A[x,k] = floor(x/k).
        # c[k] = -log(k)/k (since we want to maximize -sum f log/k, so c is negative? 
        # Wait, metric a = -sum f(k) log(k)/k. We want to MAXIMIZE a.
        # So Primal: Max sum (-log(k)/k) f(k)
        # c[k] = -log(k)/k.
        
        # If we drop the bounds on f(k) for a moment (assuming they are not active),
        # and ignore f(1), the dual is:
        # Min sum_{x=1}^{10K} y(x)
        # s.t. sum_{x=1}^{10K} floor(x/k) y(x) = -log(k)/k  (for k=2..K)
        # y(x) >= 0
        
        # But the current inspiration says:
        # "dual objective sum y(x) is minimized subject to sum_{x=k}^{10K} y(x) >= log(k)/k"
        # This implies the constraint matrix in the dual is different.
        # floor(x/k) = j for x in [jk, (j+1)k-1].
        # sum_{x=1}^{10K} floor(x/k) y(x) = sum_{j=1}^{floor(10K/k)} j * sum_{x=jk}^{(j+1)k-1} y(x)
        # This doesn't look like sum_{x=k}^{10K} y(x).
        
        # Let's re-read the inspiration carefully.
        # "sum_{x=k}^{10K} y(x) >= log(k)/k"
        # This constraint structure suggests that the primal variable f(k) is associated with a constraint that looks like a suffix sum.
        # This happens if A[x,k] is 1 for x>=k and 0 otherwise.
        # But A[x,k] = floor(x/k).
        
        # Maybe the inspiration is using a change of variables or a different primal formulation?
        # Or maybe I am misinterpreting the "dual" in the inspiration.
        
        # Let's try a different approach: Solve the Primal LP directly using scipy.optimize.linprog.
        # For K=500, variables f(2)..f(500) (499 vars).
        # Constraints:
        # 1. M(x) <= 1 for x=1..5000. (5000 constraints)
        # 2. f(k) <= 10 (499 constraints)
        # 3. f(k) >= -10 (499 constraints)
        # Total constraints ~ 6000.
        # Matrix A is 5000x499.
        # This is very small! We can go up to K=2000.
        # Constraints: 20000 + 2000 + 2000 = 24000.
        # Matrix A is 20000x1999.
        # This is ~40M entries. Sparse is good.
        
        # Let's implement the Primal LP.
        
        # Objective: Max a = - sum f(k) log(k)/k
        # linprog minimizes. So we minimize sum f(k) log(k)/k.
        # c = log(k)/k
        
        # Constraints:
        # M(x) = sum_{k=2}^K f(k) floor(x/k) <= 1 - f(1)*x
        # Wait, f(1) is determined by sum f(k)/k = 0 => f(1) = - sum_{k=2}^K f(k)/k.
        # So M(x) = sum_{k=2}^K f(k) floor(x/k) - (sum_{k=2}^K f(k)/k) * x <= 1
        
        # Let g(k) = f(k).
        # M(x) = sum_{k=2}^K g(k) ( floor(x/k) - x/k ) <= 1
        
        # Let B[x,k] = floor(x/k) - x/k.
        # Note that floor(x/k) - x/k is in [-1, 0).
        # So B is a matrix with negative or zero entries.
        
        # Constraint: B g <= 1
        
        # Also bounds: -10 <= g(k) <= 10.
        
        # We can solve this LP.
        
        # Let's try K=2000.
        
        # Build B matrix (sparse)
        # Rows x=1..10K
        # Cols k=2..K
        
        # B[x,k] = floor(x/k) - x/k
        
        # We can compute this row by row or vectorized.
        # For sparse matrix, we need rows, cols, data.
        
        # Let's try K=1000 first to be safe with memory/time.
        
        if K > 1000 and time.time() > start_time + 100:
            # If we are already using a lot of time, don't try larger K
            pass
            
        # Build B
        # x from 1 to 10K
        # k from 2 to K
        
        # We can use a loop over k to fill the column? 
        # Or loop over x?
        # Looping over x (10K iterations) and computing vector for all k (K size) might be fast.
        
        x_vals = np.arange(1, 10*K + 1)
        k_vals = np.arange(2, K + 1)
        
        # B is (10K) x (K-1)
        # B[i, j] = floor(x_vals[i] / k_vals[j]) - x_vals[i] / k_vals[j]
        
        # This is 10K * K operations. For K=1000, 10M ops. Fast in NumPy.
        
        # But we need sparse.
        # Many entries might be zero? 
        # floor(x/k) - x/k = 0 if x is divisible by k.
        # Otherwise it's fractional part negative.
        # So it's dense-ish.
        
        # Let's just use dense if K is small enough.
        # K=1000 -> 10000x999 ~ 10M floats = 80MB. OK.
        # K=2000 -> 20000x1999 ~ 40M floats = 320MB. OK.
        
        # Let's try dense for simplicity and speed of linprog setup.
        
        # However, linprog with 'highs' is fast.
        
        # Objective c
        c = np.log(k_vals.astype(np.float64)) / k_vals.astype(np.float64)
        
        # Constraint matrix A_ub
        # A_ub @ g <= b_ub
        # A_ub = B
        # b_ub = 1
        
        # Bounds
        bounds = [(-10, 10)] * len(k_vals)
        
        # Solve
        res = linprog(c, A_ub=B, b_ub=np.ones(10*K), bounds=bounds, method='highs')
        
        if res.status != 0:
            continue
            
        g = res.x
        
        # Check feasibility with M(x) > 1.0001
        # M(x) = sum g(k) * (floor(x/k) - x/k)
        # We used B in the LP, so B @ g <= 1 should hold.
        # Due to floating point, let's verify.
        
        M = B @ g
        if np.max(M) > 1.0001:
            # Clip or adjust? 
            # If slightly violated, we can scale g down.
            # But if significantly violated, infeasible.
            # Let's try scaling.
            scale = 1.0 / (np.max(M) + 1e-9)
            g = g * scale
            M = B @ g
            if np.max(M) > 1.0001:
                continue
                
        # Compute f(1)
        # sum_{k=2}^K f(k)/k + f(1) = 0
        # f(1) = - sum g(k)/k
        
        S = np.sum(g / k_vals)
        f1 = -S
        
        # Construct domain and values
        # Domain: 1, 2, ..., K
        # Values: f1, g(2), ..., g(K)
        
        # We can exclude k where g(k) is close to 0 to reduce m?
        # The problem says m >= 1. It doesn't require minimal m.
        # But smaller m is faster for evaluation? 
        # "cost about m * 10K". 
        # If m=K, cost is K * 10K = 10 K^2.
        # For K=1000, 10M ops. Fast.
        
        domain = np.arange(1, K + 1, dtype=np.int64)
        values = np.empty(K, dtype=np.float64)
        values[0] = f1
        values[1:] = g
        
        # Clip values to [-10, 10] just in case
        values = np.clip(values, -10, 10)
        
        # Recompute S and f1 after clipping? 
        # The evaluator clips values first, THEN adjusts f(1).
        # So we should return the unclipped g? 
        # No, if we return values that are clipped, the evaluator will clip them again (no change).
        # But the evaluator adjusts f(1) based on the CLIPPED values.
        # If we return f1 calculated from unclipped g, but evaluator uses clipped g, 
        # the sum won't be zero.
        
        # So we must calculate f1 based on the values we return.
        # If we clip g, we must recalculate f1.
        
        # Let's clip g first.
        g_clipped = np.clip(g, -10, 10)
        
        # Check if clipping changed M(x)
        # If g was within bounds, no change.
        # If g was clipped, M(x) might change.
        # Since g came from LP with bounds, g should be within [-10, 10].
        # So clipping should be a no-op.
        
        S_clipped = np.sum(g_clipped / k_vals)
        f1_clipped = -S_clipped
        
        values[0] = f1_clipped
        values[1:] = g_clipped
        
        # Verify M(x) with final values
        # M(x) = f1*x + sum g(k) floor(x/k)
        # = -S_clipped * x + sum g(k) floor(x/k)
        # = sum g(k) (floor(x/k) - x/k)
        # This is exactly B @ g_clipped.
        
        M_final = B @ g_clipped
        if np.max(M_final) > 1.0001:
            continue
            
        # Calculate score for logging (optional)
        a = -np.sum(values * np.log(domain.astype(np.float64)) / domain)
        
        return domain, values

    return fallback()

# EVOLVE-BLOCK-END
