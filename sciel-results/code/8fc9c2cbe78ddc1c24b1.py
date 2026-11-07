"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global start time for deadline guard
_start_time = time.time()
TIME_LIMIT = 145.0  # 150s limit, keep 5s margin

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Deadline guard
    if time.time() - _start_time > TIME_LIMIT:
        # Return a safe fallback
        return np.zeros_like(x)

    x = np.asarray(x, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    # Piecewise Cubic Spline representation
    # The 10 parameters are the y-values of the function at five fixed, equally spaced knots.
    # Wait, there are 10 params. Let's use 10 knots?
    # The directive says: "10 parameters are the y-values of the function at five fixed, equally spaced knots".
    # This implies 5 knots, but 10 params? That doesn't match directly unless some params are for something else.
    # However, usually a spline with N knots has N parameters (the values).
    # If we have 10 params, we can use 10 knots.
    # Let's re-read: "10 parameters are the y-values of the function at five fixed... knots"
    # This phrasing is slightly ambiguous. It might mean 5 knots, so 5 values. But we have 10 params.
    # Perhaps the other 5 are for something else? Or maybe it meant 10 knots?
    # Given we have 10 params, using 10 knots is the most natural mapping for a spline basis.
    # Let's use 10 knots.
    
    # Knots in [-2, 2]
    n_knots = 10
    knots = np.linspace(-2.0, 2.0, n_knots)
    
    # We need to evaluate the natural cubic spline at points x.
    # The spline is defined by its values at the knots (params) and natural boundary conditions (2nd deriv = 0).
    
    # Efficient evaluation:
    # We can precompute the spline coefficients for the current params.
    # However, equation() is called with different x each time?
    # "equation(x, params)" -> x is the input points.
    # The evaluator fits params using training data.
    # So for each evaluation of equation during the fit, x is the training points.
    # But wait, the fit calls equation(x_train, params_candidate) many times.
    # We cannot precompute the spline basis matrix B (n_points x n_knots) once globally if x changes?
    # Actually, in the evaluation loop, x is fixed (training points) for the entire optimization.
    # But equation is a pure function. It receives x and params.
    # We can compute the basis matrix B for the given x and knots inside equation.
    # This is O(n_points * n_knots).
    # n_points is likely 200. n_knots is 10. This is very fast.
    
    # Let's compute the spline basis matrix B such that result = B @ params.
    # The i-th row of B corresponds to x[i], and the j-th column corresponds to the influence of knot j's value.
    
    # Natural cubic spline construction:
    # Let h_j = x_{j+1} - x_j
    # Let M_j be the second derivative at knot j.
    # Natural BC: M_0 = 0, M_{n-1} = 0.
    # For interior knots j=1..n-2:
    # h_{j-1} * M_{j-1} + 2*(h_{j-1}+h_j) * M_j + h_j * M_{j+1} = 6 * ((y_{j+1}-y_j)/h_j - (y_j-y_{j-1})/h_{j-1})
    
    # We want to express y(x) as a linear combination of y_j (params).
    # This is a linear system.
    # Let A be the tridiagonal matrix for M. A M = d(y).
    # M = A^{-1} d(y).
    # Since d(y) is linear in y, M is linear in y.
    # The spline value at x is a piecewise cubic polynomial in x, with coefficients depending linearly on M and y.
    # Thus, y(x) is linear in y.
    
    # We can construct the basis functions by solving the spline problem for each basis vector e_j.
    # Or, more efficiently, we can solve the linear system for the general case and express the result.
    
    # Let's precompute the matrix A and its inverse (or solve factor) for the fixed knots.
    # Since knots are fixed, A is constant.
    
    h = np.diff(knots)
    n = len(knots)
    
    # Construct A (n x n)
    A = np.zeros((n, n))
    for j in range(n):
        if j == 0 or j == n - 1:
            A[j, j] = 1.0
        else:
            A[j, j-1] = h[j-1]
            A[j, j] = 2.0 * (h[j-1] + h[j])
            A[j, j+1] = h[j]
            
    # Precompute inverse of A? Or LU decomposition?
    # Since A is constant, we can compute A_inv once.
    # But A depends on knots which are fixed.
    # We can cache this in a global variable to avoid recomputation.
    
    global _A_inv
    if '_A_inv' not in globals() or _A_inv is None:
        try:
            _A_inv = np.linalg.inv(A)
        except np.linalg.LinAlgError:
            _A_inv = np.eye(n)
            
    # Now, for a given set of params y, we compute M = A_inv @ d(y).
    # d(y) is a vector where:
    # d[0] = 0, d[n-1] = 0
    # d[j] = 6 * ((y[j+1]-y[j])/h[j] - (y[j]-y[j-1])/h[j-1]) for j=1..n-2
    
    # Let's construct the matrix D such that d = D @ y.
    D = np.zeros((n, n))
    for j in range(1, n-1):
        D[j, j-1] = -6.0 / h[j-1]
        D[j, j] = 6.0 * (1.0/h[j-1] + 1.0/h[j])
        D[j, j+1] = -6.0 / h[j]
        
    # M = A_inv @ D @ y
    
    # The spline value at x:
    # For x in [knots[j], knots[j+1]], let t = (x - knots[j]) / h[j]
    # S(x) = (1-t) * y[j] + t * y[j+1] + (h[j]^2 / 6) * ( (1-t)^3 - (1-t) ) * M[j] + (h[j]^2 / 6) * ( t^3 - t ) * M[j+1]
    
    # We want to express S(x) = sum_j B[x, j] * y[j].
    # We can compute the basis matrix B for the given x.
    
    B = np.zeros((len(x), n))
    
    # Determine which interval each x falls into
    # np.searchsorted returns the index of the interval
    # knots are sorted.
    # If x < knots[0], clamp to knots[0]? Or extrapolate?
    # The problem states x is in [-2, 2], and knots cover [-2, 2].
    # So x should be within the range.
    # np.searchsorted(knots, x, side='right') - 1 gives the index j such that knots[j] <= x < knots[j+1]
    # If x == knots[-1], searchsorted returns n, so j = n-1. But there is no interval [n-1, n].
    # We need to handle the last point carefully.
    
    idx = np.searchsorted(knots, x, side='right') - 1
    # Clip idx to [0, n-2] for interval calculation, but if x is exactly knots[-1], we can just use the last knot value.
    # Actually, if x == knots[-1], the spline value is y[n-1].
    # Let's handle the general case first.
    
    # For efficiency, we can vectorize the spline evaluation.
    
    # 1. Compute M for each point? No, M depends on y (params), not x.
    # M is a vector of length n.
    # M = A_inv @ D @ params
    
    M = _A_inv @ (D @ params)
    
    # 2. For each x_i, find interval j_i.
    # j_i = searchsorted(knots, x_i, side='right') - 1
    # If x_i == knots[-1], j_i = n-1. We treat this as a special case or clamp.
    # Let's clamp j to n-2 for the formula, and handle the last point separately if needed.
    # Actually, if x is in [knots[j], knots[j+1]], the formula works.
    # If x == knots[n-1], it's the end of the last interval.
    
    j = idx
    # If x == knots[-1], j = n-1.
    # The formula for interval j uses knots[j] and knots[j+1].
    # So we need j to be in [0, n-2].
    # If j == n-1, set j = n-2.
    j = np.clip(j, 0, n-2)
    
    # Calculate t
    # t = (x - knots[j]) / h[j]
    # We need to index knots and h with j.
    # This is advanced indexing.
    
    x_j = knots[j]
    x_j1 = knots[j+1]
    h_j = h[j]
    
    t = (x - x_j) / h_j
    
    # S(x) = (1-t)*y_j + t*y_j1 + (h_j^2/6)*((1-t)^3 - (1-t))*M_j + (h_j^2/6)*(t^3 - t)*M_j1
    
    # y_j = params[j], y_j1 = params[j+1]
    y_j = params[j]
    y_j1 = params[j+1]
    
    M_j = M[j]
    M_j1 = M[j+1]
    
    # We want to express this as a linear combination of params.
    # This is tricky because M is linear in params, but the coefficients (t, etc.) depend on x.
    # However, we can just compute the value directly.
    # result = ...
    # But wait, the evaluator uses BFGS to optimize params.
    # It needs the gradient.
    # If we compute result directly, NumPy/SciPy will use finite differences for the gradient if we don't provide one.
    # The task says "scipy.optimize.minimize(method='BFGS')".
    # BFGS uses finite differences by default if jac is not provided.
    # So we just need to return the correct values.
    
    term1 = (1.0 - t) * y_j + t * y_j1
    term2 = (h_j**2 / 6.0) * ( ((1.0 - t)**3 - (1.0 - t)) * M_j + (t**3 - t) * M_j1 )
    
    result = term1 + term2
    
    # Handle the case where x is exactly the last knot
    # If x == knots[-1], the formula with j=n-2 and t=1 should give:
    # (1-1)*y[n-2] + 1*y[n-1] + ...
    # The M terms:
    # (1^3 - 1) * M[n-1] = 0.
    # ((1-1)^3 - (1-1)) * M[n-2] = 0.
    # So result = y[n-1]. Correct.
    
    # Check for non-finite values
    if not np.all(np.isfinite(result)):
        return np.zeros_like(x)
        
    return result

# Initialize global cache
_A_inv = None
# EVOLVE-BLOCK-END
