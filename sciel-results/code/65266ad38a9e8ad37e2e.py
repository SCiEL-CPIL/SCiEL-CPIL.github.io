import numpy as np
import numpy.polynomial.laguerre as npl
import time
from scipy.optimize import minimize_scalar
import math

def construct():
    start_time = time.time()
    deadline = 600.0
    
    def time_left():
        return deadline - (time.time() - start_time)
    
    try:
        # The task asks for "12 positive double-root locations".
        # In the context of Cohn-Goncalves and Fourier uncertainty, 
        # we are likely constructing a polynomial P(x) = \prod (x - r_i)^2 
        # or similar, where r_i are the roots.
        # The "score" is the upper bound on C4. 
        # For Fourier sign uncertainty, the constant C4 is related to the 
        # best constant C such that ||f||_2 <= C ||f'||_2 or similar.
        # Actually, the "refined uncertainty inequality" usually refers to:
        # ||f||_2^2 <= C * ||f'||_2^2 for bandlimited functions, or similar.
        
        # However, the specific "Cohn-Goncalves Laguerre-polynomial construction" 
        # suggests we are looking at the ratio of norms for Laguerre functions.
        # Let's hypothesize the score is related to the sum of roots or a specific 
        # functional of the roots that bounds the uncertainty.
        
        # A common metric in these problems is the "energy" or the value of the 
        # polynomial at specific points, or the ratio of coefficients.
        # Given the "double root" constraint, the polynomial is P(x) = \prod_{i=1}^{12} (x-r_i)^2.
        # This P(x) is a polynomial of degree 24.
        
        # Let's look at the "seed" program mentioned in inspirations:
        # Inspiration 3 says: "The current program just returns [4.0, 8.0, 12.0, ..., 48.0] as 12 positive double-root locations."
        # And it scores 1.0 (the baseline).
        # The current program (Laguerre roots) scores 0.4869.
        # Since lower is better for the raw value, and score = seed/value, 
        # if seed value is V_seed and current value is V_curr, 
        # Score_seed = 1.0 => V_seed is the reference.
        # Score_curr = V_seed / V_curr = 0.4869 => V_curr = V_seed / 0.4869 ≈ 2.05 * V_seed.
        # So the Laguerre roots produce a HIGHER raw value (worse bound) than the arithmetic progression [4, 8, ..., 48].
        
        # Wait, the prompt says "score is the upper bound on C4" and "lower is better".
        # But then it says "search score is seed/value, so 1.0 equals the seed program and above 1.0 is an improvement".
        # This implies:
        # Raw Value V (the upper bound C4). Lower V is better.
        # Score S = V_seed / V.
        # If V < V_seed, then S > 1.0.
        # The current program (Laguerre roots) has S = 0.4869.
        # This means V_laguerre > V_seed. The Laguerre roots are WORSE than the seed [4, 8, ..., 48].
        
        # My goal: Find roots r_i such that V(r_i) < V_seed.
        # Since [4, 8, ..., 48] gives V_seed, and Laguerre roots give a worse V, 
        # I should try to optimize the roots to minimize V.
        
        # What is V? 
        # In Fourier uncertainty, C4 often appears in:
        # ||f||_2 <= C4 * ||x f||_2 (Heisenberg) or similar.
        # But "sign uncertainty" is different.
        # Cohn-Goncalves construction for Fourier sign uncertainty:
        # The bound is often derived from the ratio of the L2 norm of the polynomial 
        # to the L2 norm of its derivative, or related to the roots' distribution.
        
        # Let's assume the "raw value" V is the value of the Cohn-Goncalves constant 
        # computed from the roots. A common proxy for the uncertainty constant 
        # associated with a polynomial P with roots r_i is:
        # V = \sum r_i^2 / \sum r_i ? No.
        
        # Let's look at the structure of the "C4" bound.
        # In many spectral concentration problems, the constant is related to the 
        # eigenvalues of a specific operator.
        
        # However, without the exact formula for V(r), I can use the fact that 
        # the seed [4, 8, ..., 48] is better than Laguerre roots.
        # The seed roots are arithmetic progression: r_i = 4i.
        # Laguerre roots of L_12 are clustered near 0 and extend to ~44.
        
        # Let's try to interpolate between the Laguerre roots and the Seed roots, 
        # or optimize a parametric family.
        # Since the seed is an arithmetic progression, maybe the optimal solution 
        # is close to an arithmetic progression but slightly adjusted.
        
        # Let's define a cost function that we believe correlates with V.
        # If V is the "upper bound on C4", and C4 is related to the "spread" or 
        # "concentration" of the polynomial, let's try to minimize the 
        # "variance" of the roots or the "energy".
        
        # Actually, let's look at the "double root" aspect.
        # P(x) = \prod (x-r_i)^2.
        # The L2 norm of P on [0, \infty) with weight e^{-x} is \int P(x)^2 e^{-x} dx.
        # The L2 norm of P' is \int P'(x)^2 e^{-x} dx.
        # The ratio ||P||^2 / ||P'||^2 is a standard uncertainty measure (Rayleigh quotient).
        # Minimizing this ratio minimizes the uncertainty constant.
        
        # Let's compute this ratio for the Seed and Laguerre roots to see which is smaller.
        
        def compute_ratio(roots):
            # roots: array of 12 positive numbers
            # Construct P(x) = prod (x - r_i)^2
            # This is a polynomial of degree 24.
            # We need ||P||_2^2 and ||P'||_2^2 with weight e^{-x} on [0, inf).
            # \int_0^\infty P(x)^2 e^{-x} dx.
            # This is the inner product <P, P> in the Laguerre space.
            # We can compute this by expanding P(x) in the Laguerre basis or using quadrature.
            # Using quadrature is safer for arbitrary roots.
            
            # Laguerre-Gauss quadrature of order N needs to be large enough to integrate 
            # a degree 24 polynomial squared (degree 48). 
            # A rule of thumb: N > 48/2 + 10 = 34. Let's use N=100 for safety.
            
            x_q, w_q = npl.laggauss(100)
            
            # Evaluate P(x_q)
            # P(x) = \prod_{i=1}^{12} (x - r_i)^2
            # To avoid overflow/underflow, we can sum logs or just multiply.
            # Roots are around 4-48. x_q goes up to ~100.
            # (100-4)^2 * ... * (100-48)^2. Large numbers.
            # Let's use log-sum-exp or just normalize.
            
            # Actually, for the ratio, scaling doesn't matter.
            # Let's compute log|P(x)| = \sum 2 \log|x - r_i|
            # But we need the sign? No, we square it.
            # P(x)^2 = \prod (x - r_i)^4.
            
            # Let's compute the integral using the quadrature points.
            # I_P = \sum w_q * P(x_q)^2
            # I_Pp = \sum w_q * P'(x_q)^2
            
            # P'(x) = P(x) * \sum_{i=1}^{12} \frac{2}{x - r_i}
            
            # To handle large numbers, let's normalize the roots or use a stable method.
            # Let's just compute the ratio directly.
            
            # Vectorized evaluation
            # x_q shape (100,), roots shape (12,)
            # diff = x_q[:, None] - roots[None, :]  # shape (100, 12)
            
            diff = x_q[:, np.newaxis] - roots[np.newaxis, :]
            
            # Avoid log(0) if x_q matches a root. Unlikely with Gauss nodes.
            # But if a root is very close, it's fine.
            
            # log_P_sq = 4 * sum(log(|diff|), axis=1)
            # This might be unstable if diff is negative? No, we square, so |diff|.
            
            # However, if x_q < r_i, diff is negative. log(abs(diff)) is fine.
            
            # Let's compute the integrals.
            # I_P = sum(w_q * exp(4 * sum(log(abs(diff)), axis=1)))
            # This can overflow.
            
            # Better: Use the property that the ratio is scale invariant?
            # No, the weight e^{-x} breaks scale invariance.
            
            # Let's use a simpler proxy if the exact integral is too slow or unstable.
            # Or, use the fact that for Laguerre polynomials, the roots are optimal 
            # for the weight e^{-x}. 
            # But the seed [4, 8, ...] is better?
            
            # Let's check the seed value vs Laguerre value using a simpler metric:
            # Sum of squares of roots?
            # Seed: sum((4i)^2) = 16 * sum(i^2) = 16 * 650 = 10400.
            # Laguerre L12 roots: approx [0.5, 2, 5, 9, 14, 20, 27, 34, 42, ...]
            # Sum of squares will be different.
            
            # Let's assume the "raw value" V is the Rayleigh Quotient 
            # R = ||P||_2^2 / ||P'||_2^2.
            # We want to MINIMIZE V (lower bound is better).
            # So we want to MINIMIZE R.
            
            # Let's compute R for the seed and Laguerre to calibrate.
            
            def rayleigh_quotient(roots):
                x_q, w_q = npl.laggauss(100)
                diff = x_q[:, np.newaxis] - roots[np.newaxis, :]
                
                # Compute log(P(x)^2) = 4 * sum(log|diff|)
                # Handle zeros in diff (should not happen for Gauss nodes vs arbitrary roots)
                # Add small epsilon for stability
                abs_diff = np.abs(diff) + 1e-10
                log_P_sq = 4.0 * np.sum(np.log(abs_diff), axis=1)
                
                # P'(x) = P(x) * S(x) where S(x) = sum(2/(x-r_i))
                S = np.sum(2.0 / diff, axis=1)
                
                # log(P'(x)^2) = log(P(x)^2) + 2*log|S(x)|
                # But S can be negative. We need S^2.
                # log(S^2) = 2 * log|S|
                
                # If S is close to 0, log is -inf.
                # S(x) = 0 when P'(x) = 0.
                # The roots of P' are interlaced with roots of P.
                # Gauss nodes are unlikely to hit them exactly, but might be close.
                
                # Let's compute P and P' directly using polynomial evaluation?
                # Degree 24 is small.
                
                # Construct polynomial coefficients
                # P(x) = prod (x^2 - 2 r_i x + r_i^2)
                # This is expensive to expand for each evaluation in optimization.
                
                # Let's stick to the log-sum-exp method but be careful.
                # To avoid overflow in exp(log_P_sq), we can normalize by the max.
                max_log_P_sq = np.max(log_P_sq)
                P_sq_vals = np.exp(log_P_sq - max_log_P_sq)
                
                # For P'_sq:
                abs_S = np.abs(S) + 1e-10
                log_Pp_sq = log_P_sq + 2.0 * np.log(abs_S)
                max_log_Pp_sq = np.max(log_P_Pp_sq)
                Pp_sq_vals = np.exp(log_Pp_sq - max_log_Pp_sq)
                
                I_P = np.sum(w_q * P_sq_vals)
                I_Pp = np.sum(w_q * Pp_sq_vals)
                
                # The ratio R = I_P / I_Pp
                # Note: The normalization constants max_log_P_sq and max_log_Pp_sq 
                # cancel out in the ratio?
                # I_P = exp(max_log_P_sq) * sum(w_q * exp(log_P_sq - max_log_P_sq))
                # I_Pp = exp(max_log_Pp_sq) * sum(w_q * exp(log_Pp_sq - max_log_Pp_sq))
                # R = exp(max_log_P_sq - max_log_Pp_sq) * (sum1 / sum2)
                
                R = np.exp(max_log_P_sq - max_log_Pp_sq) * (I_P / I_Pp)
                
                return R
            
            # Let's test this function on the seed and Laguerre roots.
            seed_roots = np.array([4.0 * (i + 1) for i in range(12)])
            lag_roots, _ = npl.laggauss(12)
            lag_roots = np.sort(lag_roots)
            
            R_seed = rayleigh_quotient(seed_roots)
            R_lag = rayleigh_quotient(lag_roots)
            
            # If R_seed < R_lag, then seed is better (lower V).
            # If R_seed > R_lag, then Laguerre is better.
            # Given Score_seed = 1.0 and Score_lag = 0.48, 
            # and Score = SeedVal / Val, 
            # Val_seed = V_seed. Val_lag = V_seed / 0.48.
            # So V_lag > V_seed.
            # If V is the Rayleigh Quotient R, then R_lag > R_seed.
            
            # So we want to MINIMIZE R.
            
            # Strategy: Optimize the roots to minimize R.
            # Parameterize the roots.
            # Since the seed is an arithmetic progression, let's start there 
            # and use gradient-free optimization (Nelder-Mead) or local search.
            
            # However, R is expensive to compute (quadrature).
            # We have 600 seconds.
            # One evaluation of R takes ~ms.
            # We can afford ~10,000 evaluations.
            
            # Let's use scipy.optimize.minimize with Nelder-Mead.
            # Variables: the 12 roots.
            # Constraint: roots must be positive and sorted? 
            # The product is symmetric, so order doesn't matter.
            # But for numerical stability, let's keep them sorted and positive.
            
            # To ensure positivity, optimize log(roots).
            # Let u_i = log(r_i). Then r_i = exp(u_i).
            
            def objective(u):
                if time_left() < 10:
                    return float('inf')
                r = np.exp(u)
                # Sort for stability in diff calculation? 
                # The formula for S = sum(2/(x-r_i)) is symmetric.
                # The formula for P is symmetric.
                # So order doesn't matter.
                return rayleigh_quotient(r)
            
            # Initial guess: log of seed roots
            u0 = np.log(seed_roots)
            
            # Run optimization
            # Use Nelder-Mead as it doesn't require gradients and is robust.
            # Max iterations to fit within time budget.
            # Each iteration is ~12 function evaluations.
            # Let's allow 500 iterations max.
            
            res = minimize_scalar(objective, bounds=None, method='Nelder-Mead', 
                                  options={'maxiter': 500, 'fatol': 1e-8, 'xatol': 1e-8})
            
            if res.success and time_left() > 5:
                best_u = res.x
                best_roots = np.exp(best_u)
                # Ensure positive
                best_roots = np.abs(best_roots)
                # Sort
                best_roots = np.sort(best_roots)
                
                # Check if it's better than seed
                R_opt = objective(best_u)
                R_seed_val = objective(u0)
                
                if R_opt < R_seed_val:
                    return best_roots.tolist()
                else:
                    return seed_roots.tolist()
            
            # Fallback to seed
            return seed_roots.tolist()

    except Exception as e:
        # Fallback to seed
        return [4.0 * (i + 1) for i in range(12)]

