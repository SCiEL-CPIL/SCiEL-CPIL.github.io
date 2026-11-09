"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Deadline guard
    t_start = time.time()
    
    n_points = len(x)
    if n_points == 0:
        return np.array([])
        
    # Map x from [-3, 3] to t in [-1, 1] for better numerical conditioning
    t = x / 3.0
    
    # We have 10 parameters.
    # Strategy: Hybrid basis combining well-conditioned Chebyshev polynomials
    # and low-frequency Fourier terms.
    #
    # Inspiration 1 (Chebyshev) scored poorly (5.36e6) likely due to ill-conditioning
    # or overfitting with 10 high-degree terms. Inspiration 3 (Linear) scored 1.0,
    # suggesting the target is relatively simple or the linear model is a strong baseline.
    # The current program (Hybrid Poly/Fourier) scored 782067, which is also very poor.
    #
    # Wait, the scores are "normalized MSE". Lower is better.
    # Score 1.0 is the seed program (linear).
    # Score 782067 is much worse than 1.0.
    # Score 5360288 is even worse.
    #
    # This implies that the current program and Inspiration 1 are performing *worse* than a simple linear model.
    # Why would a more complex model perform worse?
    # 1. BFGS starting from all ones might get stuck in a bad local minimum for complex non-linear bases.
    # 2. The basis functions might be ill-conditioned, causing numerical issues during optimization.
    # 3. The target function might be very simple (e.g., linear or quadratic) and the extra parameters
    #    allow the optimizer to fit noise or get stuck, leading to poor generalization.
    #
    # However, the problem states the target is noiseless. So overfitting noise isn't the issue.
    # It's likely that BFGS is failing to find the global minimum for these complex bases when starting from 1s.
    #
    # Let's try a different approach:
    # Use a basis that is more "linear-like" or easier for BFGS to optimize.
    # Or, perhaps the issue is that the basis functions are not orthogonal enough or scaled appropriately.
    #
    # Let's look at Inspiration 3 again: `return params[0] * x + params[1]`. Score 1.0.
    # This is the best score so far.
    #
    # Maybe we should stick closer to linear but add a few very safe, low-order terms that are
    # unlikely to cause instability.
    #
    # Let's try a pure polynomial basis of low degree, but scaled to be orthogonal-ish.
    # Or better yet, let's use the Chebyshev basis but only up to a low degree, and maybe
    # initialize the parameters differently? No, we can't change initialization.
    #
    # The key insight might be that BFGS from 1s struggles with highly non-linear or
    # oscillatory bases. The linear model works because it's convex.
    #
    # What if we use a basis that is a mixture of linear and very low-order polynomial terms,
    # but scaled such that they are well-behaved?
    #
    # Let's try:
    # p0: 1
    # p1: t
    # p2: t^2
    # p3: t^3
    # p4: t^4
    # p5: t^5
    # p6: t^6
    # p7: t^7
    # p8: t^8
    # p9: t^9
    # This is just monomials. Monomials are ill-conditioned for high degrees.
    #
    # Chebyshev is better conditioned. But Inspiration 1 used Chebyshev and got a terrible score.
    # Why? Maybe the target is NOT well-approximated by Chebyshev? Or maybe BFGS failed.
    #
    # Let's reconsider the scores.
    # Linear: 1.0
    # Chebyshev (9 deg): 5.36e6
    # Hybrid (Poly+Fourier): 7.8e5
    #
    # It seems that adding complexity makes it worse. This strongly suggests that BFGS is failing
    # to converge to the correct parameters for these complex models.
    #
    # How can we help BFGS?
    # 1. Use a basis with smaller dynamic range.
    # 2. Use a basis that is closer to linear.
    #
    # Let's try a basis that is just linear plus a few very small, smooth corrections.
    # For example:
    # p0: 1
    # p1: t
    # p2: t^2 (scaled by 0.1)
    # p3: t^3 (scaled by 0.1)
    # ...
    # But we don't control the scaling of params, only the basis.
    #
    # Actually, if we scale the basis functions down, the corresponding parameters will need to be larger.
    # If we scale them up, the parameters will be smaller.
    # BFGS works better when parameters are of similar magnitude.
    #
    # Let's try a basis of scaled monomials that are orthogonal on [-1,1] (Legendre polynomials).
    # Legendre polynomials are well-conditioned and orthogonal.
    #
    # Let's use Legendre polynomials P_0 to P_9.
    # P_0(x) = 1
    # P_1(x) = x
    # P_2(x) = (3x^2 - 1)/2
    # ...
    #
    # This might work better than Chebyshev if Chebyshev failed.
    #
    # Alternatively, maybe the target is actually linear, and any deviation hurts?
    # But the task is symbolic regression, so it's likely non-linear.
    #
    # Let's try a combination of Linear and a few very low-order terms, but ensure the basis
    # is well-scaled.
    #
    # Let's try:
    # basis[0] = 1
    # basis[1] = t
    # basis[2] = t^2
    # basis[3] = t^3
    # basis[4] = t^4
    # basis[5] = t^5
    # basis[6] = t^6
    # basis[7] = t^7
    # basis[8] = t^8
    # basis[9] = t^9
    #
    # This is monomials. Let's see if it's better than Chebyshev.
    #
    # Actually, let's look at the "Hybrid" score again: 782067.
    # And Chebyshev: 5360288.
    # Hybrid is better than Chebyshev.
    #
    # What if we just use Linear + Quadratic + Cubic?
    # params[0]: 1
    # params[1]: t
    # params[2]: t^2
    # params[3]: t^3
    # params[4..9]: 0 (or some other safe terms)
    #
    # If we set the higher order terms to 0 in the basis, they don't contribute.
    # But we have to use all 10 params.
    #
    # Let's try a basis that is mostly linear, with a few small corrections.
    #
    # basis[0] = 1
    # basis[1] = t
    # basis[2] = t^2
    # basis[3] = t^3
    # basis[4] = t^4
    # basis[5] = t^5
    # basis[6] = 0
    # basis[7] = 0
    # basis[8] = 0
    # basis[9] = 0
    #
    # This effectively reduces the model to a 5th-degree polynomial.
    #
    # Let's try Legendre polynomials instead of monomials, as they are orthogonal.
    #
    # P_0(t) = 1
    # P_1(t) = t
    # P_2(t) = 0.5 * (3*t^2 - 1)
    # P_3(t) = 0.5 * (5*t^3 - 3*t)
    # P_4(t) = (1/8) * (35*t^4 - 30*t^2 + 3)
    # P_5(t) = (1/8) * (63*t^5 - 70*t^3 + 15*t)
    # P_6(t) = (1/16) * (231*t^6 - 315*t^4 + 105*t^2 - 5)
    # P_7(t) = (1/16) * (429*t^7 - 693*t^5 + 315*t^3 - 35*t)
    # P_8(t) = (1/128) * (6435*t^8 - 12012*t^6 + 6930*t^4 - 1260*t^2 + 35)
    # P_9(t) = (1/128) * (12155*t^9 - 25740*t^7 + 18018*t^5 - 4620*t^3 + 315*t)
    #
    # These are well-behaved on [-1, 1].
    
    basis = np.zeros((10, n_points))
    basis[0, :] = 1.0
    basis[1, :] = t
    basis[2, :] = 0.5 * (3.0 * t * t - 1.0)
    basis[3, :] = 0.5 * (5.0 * t * t * t - 3.0 * t)
    basis[4, :] = (1.0/8.0) * (35.0 * t**4 - 30.0 * t**2 + 3.0)
    basis[5, :] = (1.0/8.0) * (63.0 * t**5 - 70.0 * t**3 + 15.0 * t)
    basis[6, :] = (1.0/16.0) * (231.0 * t**6 - 315.0 * t**4 + 105.0 * t**2 - 5.0)
    basis[7, :] = (1.0/16.0) * (429.0 * t**7 - 693.0 * t**5 + 315.0 * t**3 - 35.0 * t)
    basis[8, :] = (1.0/128.0) * (6435.0 * t**8 - 12012.0 * t**6 + 6930.0 * t**4 - 1260.0 * t**2 + 35.0)
    basis[9, :] = (1.0/128.0) * (12155.0 * t**9 - 25740.0 * t**7 + 18018.0 * t**5 - 4620.0 * t**3 + 315.0 * t)
    
    if time.time() - t_start > 100:
        return params[0] * t + params[1]
        
    return np.einsum('i,ij->j', params, basis)
# EVOLVE-BLOCK-END
