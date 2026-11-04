import numpy as np
from scipy.optimize import minimize
import time

def construct():
    """Find coefficients c0, c4, c8 that minimize the upper bound C4.
    
    The function g(x) = (c0*H0 + c4*H4 + c8*H8 + c12*H12) * exp(-x²/2)
    with c12 determined by g(0) = 0.
    """
    start_time = time.time()
    deadline = start_time + 180.0  # 180s limit, use 170s to be safe
    
    # Hermite polynomial values at 0
    H0_0 = 1.0
    H4_0 = 12.0
    H8_0 = 1680.0
    H12_0 = 665280.0
    
    # c12 = -(c0 + 12*c4 + 1680*c8) / 665280
    def c12_from(c0, c4, c8):
        return -(c0 + H4_0 * c4 + H8_0 * c8) / H12_0
    
    def objective(coeffs):
        c0, c4, c8 = coeffs
        c12 = c12_from(c0, c4, c8)
        
        # Compute the L2 norm squared of g
        # ||g||^2 = sqrt(pi) * [c0^2 + 384*c4^2 + 10321920*c8^2 + 1962039475200*c12^2]
        norm_sq = np.sqrt(np.pi) * (c0**2 + 384.0 * c4**2 + 10321920.0 * c8**2 + 1962039475200.0 * c12**2)
        
        if norm_sq <= 0:
            return 1e30
        
        # For the uncertainty product, we need ||x*g|| and ||g'||
        # g(x) = P(x) * exp(-x²/2), P = c0*H0 + c4*H4 + c8*H8 + c12*H12
        # g'(x) = P'(x) * exp(-x²/2) - x * P(x) * exp(-x²/2) = (P'(x) - x*P(x)) * exp(-x²/2)
        
        # P'(x) = c4*H4'(x) + c8*H8'(x) + c12*H12'(x)  (H0'=0)
        # H_n'(x) = 2n * H_{n-1}(x)
        # H4' = 8*H3, H8' = 16*H7, H12' = 24*H11
        
        # ||g'||^2 = sqrt(pi) * [||P' - x*P||^2 in the exp(-x²) weight]
        # This is complex. Let me use the fact that for Hermite functions,
        # the Fourier transform preserves the L2 norm structure.
        
        # Actually, for the uncertainty product A(f)*A(F̂f), if A(f) = ||f||_2 * ||x f||_2
        # and A(F̂f) = ||F̂f||_2 * ||x F̂f||_2, then by Plancherel ||F̂f|| = ||f||,
        # and ||x F̂f|| = ||F̂f'|| (with appropriate normalization).
        
        # Let me compute this numerically using quadrature for accuracy.
        # But that might be slow. Let me try to compute analytically.
        
        # For g = P * exp(-x²/2):
        # ||x*g||^2 = ∫ x² P(x)² exp(-x²) dx
        # ||g'||^2 = ∫ (P'(x) - x*P(x))² exp(-x²) dx
        
        # These can be computed using the orthogonality of Hermite polynomials.
        # Let's expand P in the Hermite basis and use the recurrence relations.
        
        # P(x) = c0*H0 + c4*H4 + c8*H8 + c12*H12
        # x*H_n = (1/2)(H_{n+1} + n*H_{n-1})
        
        # x*P = c0*(1/2*H1) + c4*(1/2*H5 + 2*H3) + c8*(1/2*H9 + 4*H7) + c12*(1/2*H13 + 6*H11)
        
        # P'(x) = c4*8*H3 + c8*16*H7 + c12*24*H11
        
        # P' - x*P = c4*8*H3 + c8*16*H7 + c12*24*H11 
        #   - c0*(1/2*H1) - c4*(1/2*H5 + 2*H3) - c8*(1/2*H9 + 4*H7) - c12*(1/2*H13 + 6*H11)
        
        # = -c0/2 * H1 + c4*(8-2)*H3 - c4/2 * H5 + c8*(16-4)*H7 - c8/2 * H9 + c12*(24-6)*H11 - c12/2 * H13
        # = -c0/2 * H1 + 6*c4*H3 - c4/2 * H5 + 12*c8*H7 - c8/2 * H9 + 18*c12*H11 - c12/2 * H13
        
        # ||P' - x*P||² (with exp(-x²) weight) = sqrt(pi) * [(-c0/2)² * 1 + (6*c4)² * 2³*3! + (-c4/2)² * 2⁵*5! + (12*c8)² * 2⁷*7! + (-c8/2)² * 2⁹*9! + (18*c12)² * 2¹¹*11! + (-c12/2)² * 2¹³*13!]
        
        # Wait, ||H_n||² = ∫ H_n² exp(-x²) dx = sqrt(pi) * 2^n * n!
        
        norm_deriv_sq = np.sqrt(np.pi) * (
            (c0/2)**2 * 1 +
            (6*c4)**2 * 8 * 6 +  # 2^3 * 3! = 8*6=48
            (c4/2)**2 * 32 * 120 +  # 2^5 * 5! = 32*120=3840
            (12*c8)**2 * 128 * 5040 +  # 2^7 * 7! = 128*5040=645120
            (c8/2)**2 * 512 * 362880 +  # 2^9 * 9! = 512*362880=185794560
            (18*c12)**2 * 2048 * 39916800 +  # 2^11 * 11! = 2048*39916800=81749601280
            (c12/2)**2 * 8192 * 6227020800  # 2^13 * 13! = 8192*6227020800=51011754444800
        )
        
        # ||x*P||² = ∫ x² P² exp(-x²) dx
        # x*P = c0/2*H1 + c4*(2*H3 + H5/2) + c8*(4*H7 + H9/2) + c12*(6*H11 + H13/2)
        norm_xp_sq = np.sqrt(np.pi) * (
            (c0/2)**2 * 1 +
            (2*c4)**2 * 48 +  # 2^3 * 3!
            (c4/2)**2 * 3840 +  # 2^5 * 5!
            (4*c8)**2 * 645120 +  # 2^7 * 7!
            (c8/2)**2 * 185794560 +  # 2^9 * 9!
            (6*c12)**2 * 81749601280 +  # 2^11 * 11!
            (c12/2)**2 * 51011754444800  # 2^13 * 13!
        )
        
        # A(f) ~ ||g|| * ||x*g||, A(F̂f) ~ ||g|| * ||g'|| (by Plancherel and derivative property)
        # Upper bound C4 might be ||g||² * ||x*g|| * ||g'|| or similar
        
        # Let's compute the product
        A_f = np.sqrt(norm_sq * norm_xp_sq)
        A_Ff = np.sqrt(norm_sq * norm_deriv_sq)
        
        C4 = A_f * A_Ff
        return C4
    
    # Initial guess from current program
    x0 = np.array([-113/100, 1/25, 1/3240])
    
    best_x = x0.copy()
    best_val = objective(x0)
    
    # Try multiple starting points
    np.random.seed(42)
    starts = [x0.copy()]
    for _ in range(20):
        starts.append(np.random.randn(3) * 0.1 + x0)
    
    for x_init in starts:
        if time.time() > deadline - 30:
            break
        try:
            res = minimize(objective, x_init, method='L-BFGS-B', 
                          options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-12})
            if time.time() > deadline - 20:
                break
            if res.success and res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except:
            pass
    
    # Polish with tighter tolerance
    if time.time() < deadline - 10:
        try:
            res = minimize(objective, best_x, method='L-BFGS-B',
                          options={'maxiter': 5000, 'ftol': 1e-18, 'gtol': 1e-15})
            if res.success and res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except:
            pass
    
    return best_x
