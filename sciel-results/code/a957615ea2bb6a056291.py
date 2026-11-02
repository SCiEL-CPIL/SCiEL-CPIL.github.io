import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

# Deadline guard: 210s limit, use 190s for safety
_DEADLINE = time.time() + 190

def construct():
    """
    Returns three coefficients [c0, c4, c8] for the Hermite combination.
    The degree-12 coefficient c12 is derived to ensure g(0) = 0.
    The score is the upper bound on C4.
    """
    
    # Hermite values at 0 (Physicist's convention)
    # H0(0) = 1
    # H4(0) = 12
    # H8(0) = 1680
    # H12(0) = 665280
    H0_0 = 1.0
    H4_0 = 12.0
    H8_0 = 1680.0
    H12_0 = 665280.0
    
    # Weights for L2 norm squared: I0 = sum c_n^2 * 2^n * n!
    # n=0: 1
    # n=4: 16 * 24 = 384
    # n=8: 256 * 40320 = 10321920
    # n=12: 4096 * 479001600 = 1961988556800
    w0_0 = 1.0
    w0_4 = 384.0
    w0_8 = 10321920.0
    w0_12 = 1961988556800.0
    
    # Weights for I2 (likely related to ||x^2 g||^2 or similar moment)
    # The previous program used:
    # w2_4 = 1536 = 4 * 384
    # w2_8 = 82575360 = 8 * 10321920
    # w2_12 = 23543862681600 = 12 * 1961988556800
    # This implies I2 = sum c_n^2 * 2^n * n! * n
    # Let's verify if this is the correct objective.
    # If A(f) = ||f||_2, then A(f)*A(F(f)) = ||f||_2^2 by Plancherel.
    # Minimizing ||f||_2^2 subject to constraints is trivial (go to 0).
    # So A(f) must be a different norm, or the "upper bound" refers to a specific constant C4 in an inequality like:
    # ||f||_1 ||F(f)||_1 <= C4 ||f||_2^2 ?
    # Or maybe it's the uncertainty principle constant:
    # (var x)(var xi) >= C4 ||f||_2^4 ?
    
    # Let's look at the seed: [-1.13, 0.04, 0.000308642]
    # If the objective is simply I2/I0, let's calculate it for the seed.
    # c0 = -1.13, c4 = 0.04, c8 = 1/3240
    # c12 = -(c0 + 12c4 + 1680c8) / 665280
    # c12 = -(-1.13 + 0.48 + 1680/3240) / 665280
    # 1680/3240 = 0.518518...
    # c12 = -(-1.13 + 0.48 + 0.518518) / 665280 = -(-0.13148) / 665280 = 1.976e-7
    
    # I0 = 1.13^2 + 384*0.04^2 + 10321920*(1/3240)^2 + 1961988556800*(1.976e-7)^2
    # I0 = 1.2769 + 384*0.0016 + 10321920*9.645e-8 + 1961988556800*3.9e-14
    # I0 = 1.2769 + 0.6144 + 0.9955 + 0.00007 = 2.8868
    
    # I2 = 384*4*0.04^2 + 10321920*8*(1/3240)^2 + 1961988556800*12*(1.976e-7)^2
    # I2 = 1536*0.0016 + 82575360*9.645e-8 + 23543862681600*3.9e-14
    # I2 = 2.4576 + 7.964 + 0.918 = 11.34
    
    # Ratio = 11.34 / 2.8868 = 3.928
    
    # The seed score is 1.0. The current program (score 0.75) tried to minimize this ratio.
    # If minimizing the ratio gives a score < 1.0, it means the seed is NOT optimal for this ratio,
    # OR the scoring function is not simply this ratio.
    # However, the task says "score is the upper bound on C4".
    # If the seed is the "state of the art", it likely minimizes the actual bound.
    # The current program's low score suggests it found a local minimum of a WRONG objective,
    # or the objective is correct but the constraint was wrong (H12(0) error).
    
    # Let's assume the objective IS I2/I0 (or similar) but we need to be careful.
    # Actually, let's look at the structure of the uncertainty inequality.
    # For g(x) = P(x) e^{-x^2/2}, the L1 norm is hard to compute analytically.
    # But there is a known result for the "uncertainty inequality" constants.
    
    # Let's try a different approach: Numerical optimization of the actual bound.
    # What is the bound?
    # If it's ||g||_1 ||F(g)||_1, we can compute this numerically.
    
    def _compute_bound_numeric(c0, c4, c8):
        # Compute c12
        c12 = -(c0 + H4_0 * c4 + H8_0 * c8) / H12_0
        
        # Define g(x)
        # We need to evaluate |g(x)| and |F(g)(xi)|
        # Use a grid
        N = 4096
        x = np.linspace(-15, 15, N)
        dx = x[1] - x[0]
        
        # Hermite polynomials
        # H0 = 1
        # H4 = 16x^4 - 48x^2 + 12
        # H8 = 256x^8 - 4480x^6 + 26880x^4 - 67200x^2 + 40320
        # H12 = 4096x^12 - 147456x^10 + 2090112x^8 - 12902400x^6 + 34406400x^4 - 34406400x^2 + 10321920
        
        H0 = np.ones_like(x)
        H4 = 16*x**4 - 48*x**2 + 12
        H8 = 256*x**8 - 4480*x**6 + 26880*x**4 - 67200*x**2 + 40320
        H12 = (4096*x**12 - 147456*x**10 + 2090112*x**8 - 
               12902400*x**6 + 34406400*x**4 - 34406400*x**2 + 10321920)
        
        P = c0*H0 + c4*H4 + c8*H8 + c12*H12
        g = P * np.exp(-x**2 / 2.0)
        
        # L1 norm of g
        A_g = np.trapz(np.abs(g), x)
        
        # Fourier Transform of g
        # F(g)(xi) = 1/sqrt(2*pi) * integral g(x) e^{-ixxi} dx
        # Use FFT
        G = np.fft.fft(g) * dx / np.sqrt(2 * np.pi)
        xi = np.fft.fftfreq(N, d=dx) * 2 * np.pi
        G = np.fft.fftshift(G)
        xi = np.fft.fftshift(xi)
        
        # L1 norm of F(g)
        A_Fg = np.trapz(np.abs(G), xi)
        
        bound = A_g * A_Fg
        return bound

    # This numerical method is slow. Let's see if we can use it with a coarse search.
    # Or, stick to the analytical ratio if we can figure out the correct one.
    
    # Let's test the seed with the numerical method to see what value it produces.
    # If the seed produces value V_seed, and we find V_new < V_seed, we improve.
    
    # However, the numerical method might have discretization errors.
    # Let's try to optimize the ratio I2/I0 first, as it's fast, and see if it correlates.
    
    def objective_ratio(params):
        c0, c4, c8 = params
        c12 = -(c0 + H4_0 * c4 + H8_0 * c8) / H12_0
        
        I0 = w0_0 * c0**2 + w0_4 * c4**2 + w0_8 * c8**2 + w0_12 * c12**2
        I2 = w0_4 * 4 * c4**2 + w0_8 * 8 * c8**2 + w0_12 * 12 * c12**2
        
        if I0 == 0:
            return 1e10
        return I2 / I0

    # Let's try to optimize this ratio with a good global optimizer.
    # The previous program used Nelder-Mead which is local.
    # Let's use Differential Evolution for a global search.
    
    seed_c0, seed_c4, seed_c8 = -113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0
    
    # Bounds for coefficients
    # c0 is O(1), c4 is O(0.01), c8 is O(0.0001)
    bounds = [(-5.0, 5.0), (-1.0, 1.0), (-0.01, 0.01)]
    
    best_val = np.inf
    best_params = np.array([seed_c0, seed_c4, seed_c8])
    
    # 1. Try Differential Evolution on the ratio
    if time.time() < _DEADLINE - 30:
        try:
            de_res = differential_evolution(
                objective_ratio,
                bounds,
                seed=42,
                maxiter=100,
                tol=1e-10,
                polish=True,
                init='sobol'
            )
            if de_res.fun < best_val:
                best_val = de_res.fun
                best_params = de_res.x
        except Exception:
            pass
    
    # 2. Refine with Nelder-Mead from multiple starts
    starts = [
        best_params,
        np.array([seed_c0, seed_c4, seed_c8]),
        np.array([-1.0, 0.05, 0.0004]),
        np.array([-1.2, 0.03, 0.0002]),
        np.array([0.0, 0.0, 0.0]), # Will be projected
    ]
    
    for s in starts:
        if time.time() > _DEADLINE - 10:
            break
        try:
            res = minimize(objective_ratio, s, method='Nelder-Mead', options={'maxiter': 1000, 'xatol': 1e-15, 'fatol': 1e-15})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x
        except Exception:
            pass
    
    # Now, let's verify if this improvement in the ratio actually improves the score.
    # We can't easily verify without the evaluator, but we can assume the ratio is a proxy.
    # However, if the ratio minimization leads to a worse score (like 0.75), it means the proxy is wrong.
    
    # Let's try a different proxy: Minimize I0 (L2 norm) subject to I2 being fixed? No.
    
    # What if the objective is actually the L1 norm product?
    # Let's do a quick numerical check on the best_params found vs the seed.
    
    def _quick_numeric_score(c0, c4, c8):
        c12 = -(c0 + H4_0 * c4 + H8_0 * c8) / H12_0
        N = 2048
        x = np.linspace(-10, 10, N)
        dx = x[1] - x[0]
        H0 = np.ones_like(x)
        H4 = 16*x**4 - 48*x**2 + 12
        H8 = 256*x**8 - 4480*x**6 + 26880*x**4 - 67200*x**2 + 40320
        H12 = (4096*x**12 - 147456*x**10 + 2090112*x**8 - 
               12902400*x**6 + 34406400*x**4 - 34406400*x**2 + 10321920)
        P = c0*H0 + c4*H4 + c8*H8 + c12*H12
        g = P * np.exp(-x**2 / 2.0)
        A_g = np.trapz(np.abs(g), x)
        G = np.fft.fft(g) * dx / np.sqrt(2 * np.pi)
        xi = np.fft.fftfreq(N, d=dx) * 2 * np.pi
        G = np.fft.fftshift(G)
        xi = np.fft.fftshift(xi)
        A_Fg = np.trapz(np.abs(G), xi)
        return A_g * A_Fg

    seed_score = _quick_numeric_score(seed_c0, seed_c4, seed_c8)
    new_score = _quick_numeric_score(best_params[0], best_params[1], best_params[2])
    
    # If the new score is better (lower), keep it. Otherwise, return seed.
    if new_score < seed_score:
        return np.array(best_params)
    else:
        return np.array([seed_c0, seed_c4, seed_c8])
