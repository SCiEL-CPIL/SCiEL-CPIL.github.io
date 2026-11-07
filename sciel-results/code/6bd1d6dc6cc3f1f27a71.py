"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    n = x.size
    if n == 0:
        return np.array([], dtype=np.float64)
    
    # Deadline guard: 150s limit, leave 10s safety margin
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Strategy: Overparameterized Linear Regression (NTK style)
    # We generate a large set of random nonlinear features.
    # Since the evaluator minimizes MSE over `params`, and `equation` returns `basis @ params`,
    # this is a linear least squares problem. We can solve it directly inside equation
    # IF we had the target. But we don't.
    # Wait, the evaluator calls equation(x, y, z, params) to get predictions.
    # The evaluator minimizes MSE(params) = mean((equation(x,y,z,params) - target)^2).
    # If equation is linear in params (basis @ params), the optimal params are given by the normal equations.
    # However, we cannot compute the optimal params inside equation because we don't have the target `y_target`.
    # We only have inputs x, y, z.
    # The evaluator will run BFGS to find params.
    # If we provide a very rich basis, BFGS might struggle if the basis is ill-conditioned or non-convex?
    # No, it's linear in params, so it's convex. BFGS should converge well if the basis is not too ill-conditioned.
    # The current program uses a 10-dim polynomial basis.
    # The new strategy: Use a much larger basis (e.g., 100-500 random ReLU features) to increase expressive power.
    # The evaluator's BFGS will fit the 10 params? 
    # WAIT. The params array has length 10. 
    # "The evaluator fits params, an array of length 10"
    # So I CANNOT increase the number of parameters. I am stuck with 10 parameters.
    # This changes everything. I cannot use a large basis matrix because I only have 10 weights to tune.
    # I must design 10 specific functions of x,y,z such that their linear combination approximates the target.
    # And the evaluator uses BFGS to fit these 10 weights.
    
    # Since the number of params is fixed at 10, I need to choose the 10 basis functions carefully.
    # The current program uses: 1, xs, ys, zs, xys, xzs, yzs, xs^2, ys^2, zs^2.
    # This is a complete quadratic polynomial.
    # Maybe the function is higher order or non-polynomial?
    # Let's try a mix of polynomial and trigonometric/exponential features to capture non-polynomial behavior.
    # Or, perhaps the function is separable?
    
    # Let's try a basis that includes:
    # 1. Constant
    # 2. x, y, z
    # 3. x^2, y^2, z^2
    # 4. x*y, x*z, y*z
    # This is the current one.
    
    # Alternative: Use radial basis functions (RBFs) centered at specific points?
    # With only 10 params, I can't do many RBFs.
    
    # Let's look at the inputs: [0.5, 2.0].
    # Maybe the function is something like sin(x)*cos(y)*exp(-z)?
    
    # Let's try a basis that is more flexible for non-polynomial shapes.
    # Features:
    # 1. 1
    # 2. x
    # 3. y
    # 4. z
    # 5. x^2
    # 6. y^2
    # 7. z^2
    # 8. sin(x)  # captures periodicity
    # 9. cos(y)
    # 10. exp(-z) # captures decay
    
    # Or maybe interactions are more important?
    # Let's try:
    # 1. 1
    # 2. x
    # 3. y
    # 4. z
    # 5. x*y
    # 6. x*z
    # 7. y*z
    # 8. x**2
    # 9. y**2
    # 10. z**2
    # This is the current one. It scored 7.39.
    
    # What if I scale the inputs differently?
    # The current scaling is (v-1.25)/0.75.
    # Maybe unscaled is better for BFGS? Or scaled to [0,1]?
    
    # Let's try a different set of 10 features that might be more robust.
    # How about Chebyshev polynomials? They are well-conditioned.
    
    def cheb(x):
        # Chebyshev of first kind
        # T0 = 1, T1 = x, T2 = 2x^2-1, T3 = 4x^3-3x
        pass
        
    # Let's stick to simple monomials but maybe add higher order terms if the quadratic isn't enough?
    # But we only have 10 slots.
    # If the function is a cubic, we need x^3, y^3, z^3, x^2y, etc. That's too many.
    
    # What if the function is simple, like x+y+z? The quadratic basis covers it.
    # What if it's x*y*z? Covered.
    # What if it's sin(x+y+z)? Not covered well by polynomials.
    
    # Let's try a basis that mixes polynomial and trigonometric.
    # 1. 1
    # 2. x
    # 3. y
    # 4. z
    # 5. x*y
    # 6. x*z
    # 7. y*z
    # 8. sin(x+y+z)
    # 9. cos(x-y)
    # 10. exp(-0.5*(x+y+z))
    
    # This is a guess. A safer bet for "unknown function" in these benchmarks is often a polynomial up to degree 2 or 3, or a simple rational function.
    
    # Let's try to improve the conditioning of the current polynomial basis.
    # The current basis is:
    # [1, xs, ys, zs, xys, xzs, yzs, xs^2, ys^2, zs^2]
    
    # Maybe the issue is that BFGS starts at all ones and gets stuck?
    # If the basis is linearly independent, the loss is quadratic in params, so BFGS should converge globally.
    # Unless the basis is ill-conditioned.
    
    # Let's try orthogonalizing the basis? No, we can't pre-compute because the inputs vary.
    # But we can construct an orthonormal basis on the fly?
    # QR decomposition of the basis matrix?
    # If we return Q @ params, and the evaluator minimizes ||Q @ params - y||^2,
    # this is equivalent to minimizing ||R @ params - Q^T y||^2 if we had y.
    # But the evaluator doesn't know Q. It just sees the function.
    # If we make the basis orthonormal, the Hessian is identity (or close), which is perfect for BFGS.
    
    # Let's implement an on-the-fly QR decomposition of the basis matrix.
    # Basis B (n x 10).
    # Compute Q, R = np.linalg.qr(B)
    # Return Q @ params.
    # Since Q is orthonormal, ||Q @ p - y||^2 = ||p - Q^T y||^2.
    # The optimal p is Q^T y.
    # BFGS will converge very fast because the problem is perfectly conditioned.
    # This might help if the previous failure was due to ill-conditioning causing BFGS to fail or take too long.
    
    # Let's try this.
    
    # 1. Construct basis
    x_s = (x - 1.25) / 0.75
    y_s = (y - 1.25) / 0.75
    z_s = (z - 1.25) / 0.75
    
    basis = np.column_stack([
        np.ones_like(x_s),
        x_s,
        y_s,
        z_s,
        x_s * y_s,
        x_s * z_s,
        y_s * z_s,
        x_s ** 2,
        y_s ** 2,
        z_s ** 2
    ])
    
    # 2. QR Decomposition
    # This makes the columns of Q orthonormal.
    # The evaluator minimizes ||Q @ p - y||^2.
    # This is equivalent to ||p - Q.T @ y||^2.
    # The gradient is 2 * Q.T @ (Q @ p - y) = 2 * (p - Q.T @ y).
    # The Hessian is 2 * I.
    # BFGS should converge in 1 step (or very few) if it uses the exact Hessian, but BFGS approximates.
    # However, with a diagonal Hessian, BFGS is very stable.
    
    try:
        Q, R = np.linalg.qr(basis)
        pred = Q @ params
    except np.linalg.LinAlgError:
        # Fallback if QR fails
        pred = basis @ params
        
    pred = np.clip(pred, -1e6, 1e6)
    
    # Check time
    if time.time() > deadline:
         return pred

    return pred

# EVOLVE-BLOCK-END
