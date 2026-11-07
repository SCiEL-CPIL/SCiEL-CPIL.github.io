import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Scale x to [0, 1]
    t = (x - 0.1) / 5.9
    
    # Clip to [0, 1] to handle numerical edge cases
    t = np.clip(t, 0.0, 1.0)
    
    # Define Cubic B-Spline basis with 10 coefficients
    # Knots: 0, 0, 0, 0, 1/6, 2/6, 3/6, 4/6, 5/6, 1, 1, 1, 1
    # This creates 10 basis functions.
    
    # We can compute B-splines using the Cox-de Boor recursion or scipy.interpolate
    # But scipy.interpolate.BSpline requires a specific knot vector format.
    # Let's use a simple implementation or pre-compute the matrix.
    
    # Since x is small (200 points), we can compute the basis matrix efficiently.
    
    # Knots
    knots = np.array([0, 0, 0, 0, 1/6, 2/6, 3/6, 4/6, 5/6, 1, 1, 1, 1])
    
    # We need to evaluate 10 cubic B-splines at each t.
    # Using scipy.interpolate.BSpline is easiest.
    # But we need the basis *matrix*, not just the function value.
    # We can create 10 BSpline objects, one for each basis function.
    
    # Alternatively, use the fact that B-splines are local.
    # But vectorized evaluation of 10 splines at 200 points is fast.
    
    from scipy.interpolate import BSpline
    
    # Create basis functions
    # Basis function i is defined by knots and order 4, with coefficient 1 at i and 0 elsewhere.
    # This is slow to create 10 objects inside the loop if called repeatedly, but equation is called once per evaluation?
    # No, equation is called many times during optimization.
    # Creating 10 BSpline objects every call is expensive.
    
    # Better: Pre-compute the basis matrix? No, x changes?
    # No, x is the input to equation. The evaluator calls equation(x, params) many times with the SAME x?
    # "The evaluator fits params ... minimizing the training MSE."
    # Yes, x is fixed during the fit.
    # But equation is a black box. We don't know if x is the same.
    # However, usually in these tasks, x is the training data.
    # To be safe, assume x can change.
    
    # Efficient B-Spline evaluation:
    # We can use the recurrence relation.
    # Or, since the knots are fixed, we can use a lookup table approach?
    # No, x is continuous.
    
    # Let's use a vectorized Cox-de Boor implementation or just scipy if we can cache?
    # We cannot cache easily in this stateless function.
    
    # Alternative: Use a simple RBF basis which is easier to vectorize.
    # RBFs with Gaussian kernels.
    # Centers: 8 centers in [0.1, 6].
    # Width: Fixed.
    
    # Let's go back to RBFs but with better parameters.
    # The previous RBF failed. Maybe the width was the issue.
    # Let's use a **Multi-scale** RBF? No, 10 params.
    
    # Let's try **Sigmoid** basis?
    # $1/(1+e^{-k(x-c)})$.
    # This is good for step functions.
    
    # Let's try **Fourier** again. It's the most robust "global" basis after polynomials.
    # If polynomials failed, Fourier is the best bet for oscillatory.
    # If the function is non-oscillatory, Fourier will still give a reasonable fit (best L2 approximation).
    # The score 1e9 for Chebyshev is suspicious.
    
    # Let's stick to the Fourier + Linear design. It's clean, fast, and well-conditioned.
    
    pi = np.pi
    
    # Basis
    p0 = np.ones_like(t)
    p1 = t
    
    # Harmonics
    # Use 4 harmonics to use remaining 8 params.
    # sin(2*pi*t), cos(2*pi*t)
    # sin(4*pi*t), cos(4*pi*t)
    # sin(6*pi*t), cos(6*pi*t)
    # sin(8*pi*t), cos(8*pi*t)
    
    ang1 = 2 * pi * t
    ang2 = 4 * pi * t
    ang3 = 6 * pi * t
    ang4 = 8 * pi * t
    
    p2 = np.sin(ang1)
    p3 = np.cos(ang1)
    p4 = np.sin(ang2)
    p5 = np.cos(ang2)
    p6 = np.sin(ang3)
    p7 = np.cos(ang3)
    p8 = np.sin(ang4)
    p9 = np.cos(ang4)
    
    basis = np.column_stack((p0, p1, p2, p3, p4, p5, p6, p7, p8, p9))
    
    return basis @ params
