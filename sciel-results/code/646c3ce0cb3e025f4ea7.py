import numpy as np
import scipy.optimize as opt
import time

def construct():
    """
    Optimizes the coefficients of a Hermite-Gaussian combination to minimize
    a proxy for the uncertainty product A(f)A(Fourier(f)).
    """
    start_time = time.time()
    time_limit = 180.0  # Safety margin before 210s kill

    # Hermite polynomials H_0, H_4, H_8, H_12 at x=0
    # H_0(x) = 1
    # H_4(x) = 16x^4 - 48x^2 + 12 -> H_4(0) = 12 ? 
    # Wait, Physicists' Hermite: H_0=1, H_1=2x, H_2=4x^2-2, H_3=8x^3-12x, H_4=16x^4-48x^2+12.
    # So H_4(0) = 12.
    # H_8(0): H_8(x) = 256x^8 - 3072x^6 + 11520x^4 - 13440x^2 + 1344. H_8(0) = 1344.
    # H_12(0): H_12(x) = ... + (-1)^6 * (12)! / (6!) * (x^0 term coeff).
    # General term for H_n(0) if n even: (-1)^{n/2} * n! / (n/2)!.
    # H_4(0) = (-1)^2 * 4! / 2! = 24/2 = 12. Correct.
    # H_8(0) = (-1)^4 * 8! / 4! = 40320 / 24 = 1680? 
    # Let's check H_8(0) from formula: 1344.
    # 8! / 4! = 40320 / 24 = 1680. 
    # Why discrepancy? 
    # H_2(0) = -2. Formula: (-1)^1 * 2!/1! = -2. Correct.
    # H_4(0) = 12. Formula: (-1)^2 * 24/2 = 12. Correct.
    # H_6(0) = -120? Formula: (-1)^3 * 720/6 = -120.
    # H_8(0) = 1344? Formula: (-1)^4 * 40320/24 = 1680.
    # Let's re-verify H_8.
    # H_6 = 64x^6 - 480x^4 + 720x^2 - 120. H_6(0) = -120.
    # H_8 = 256x^8 - 3072x^6 + 11520x^4 - 13440x^2 + 1344.
    # 1344 vs 1680.
    # Ah, the recurrence relation: H_{n+1} = 2x H_n - 2n H_{n-1}.
    # H_0 = 1
    # H_1 = 2x
    # H_2 = 2x(2x) - 2(1)(1) = 4x^2 - 2. H_2(0) = -2.
    # H_3 = 2x(4x^2-2) - 4(2x) = 8x^3 - 4x - 8x = 8x^3 - 12x. H_3(0)=0.
    # H_4 = 2x(8x^3-12x) - 8(4x^2-2) = 16x^4 - 24x^2 - 32x^2 + 16 = 16x^4 - 48x^2 + 16?
    # Wait. 2x H_3 = 16x^4 - 24x^2. 2(3) H_2 = 6(4x^2 - 2) = 24x^2 - 12.
    # H_4 = 16x^4 - 24x^2 - (24x^2 - 12) = 16x^4 - 48x^2 + 12.
    # So H_4(0) = 12.
    # My previous manual expansion had 12.
    # H_8(0) calculation:
    # H_5 = 2x H_4 - 10 H_3 = 2x(16x^4-48x^2+12) - 10(8x^3-12x) = 32x^5 - 96x^3 + 24x - 80x^3 + 120x = 32x^5 - 176x^3 + 144x.
    # H_6 = 2x H_5 - 20 H_4 = 2x(32x^5 - 176x^3 + 144x) - 20(16x^4 - 48x^2 + 12)
    # = 64x^6 - 352x^4 + 288x^2 - 320x^4 + 960x^2 - 240
    # = 64x^6 - 672x^4 + 1248x^2 - 240.
    # H_6(0) = -240.
    # Formula check: (-1)^3 * 6! / 3! = -720 / 6 = -120.
    # Discrepancy! -240 vs -120.
    # Recurrence is H_{n+1} = 2x H_n - 2n H_{n-1}.
    # H_6(0) = -20 H_4(0) = -20(12) = -240.
    # So the formula $(-1)^{n/2} n! / (n/2)!$ is for *Probabilists'* Hermite or different scaling?
    # Physicists' Hermite $H_n(x)$ has $H_n(0) = (-1)^{n/2} \frac{n!}{(n/2)!} 2^{n/2}$? No.
    # Let's stick to the recurrence or known values.
    # H_0(0) = 1
    # H_4(0) = 12
    # H_8(0):
    # H_7 = 2x H_6 - 26 H_5. H_7(0)=0.
    # H_8 = 2x H_7 - 30 H_6. H_8(0) = -30 H_6(0) = -30(-240) = 7200.
    # Let's re-verify H_8(0) via recurrence.
    # H_6(0) = -240.
    # H_8(0) = -30 * H_6(0) = 7200.
    
    # H_12(0):
    # H_10(0) = -40 H_8(0) = -40(7200) = -288000.
    # H_12(0) = -50 H_10(0) = -50(-288000) = 14400000.

    # Check values:
    # c12 = -(c0*1 + c4*12 + c8*7200) / 14400000
    
    # Objective Function
    def objective(coeffs):
        c0, c4, c8 = coeffs
        # Calculate c12
        c12 = -(c0 + 12*c4 + 7200*c8) / 14400000.0
        
        # Use Gauss-Hermite quadrature to integrate
        # Points and weights for e^{-x^2}
        # We want integrals of x^2 g^2, g^2, g'^2
        # g(x) = (c0 H0 + c4 H4 + c8 H8 + c12 H12) e^{-x^2/2}
        # g^2 = (Poly)^2 e^{-x^2}
        # Integrand for norm: (Poly)^2
        # Integrand for x^2 norm: x^2 (Poly)^2
        # Integrand for momentum: |g'|^2. 
        # g'(x) = Poly'(x) e^{-x^2/2} - x Poly(x) e^{-x^2/2} = (Poly'(x) - x Poly(x)) e^{-x^2/2}
        # So |g'|^2 = (Poly' - x Poly)^2 e^{-x^2}
        
        # Generate quadrature points
        # N=200 is sufficient for high accuracy
        x, w = np.polynomial.hermite.hermgauss(200)
        
        # Evaluate Hermite polynomials at x
        # We need H0, H4, H8, H12 and their derivatives
        # H_n'(x) = 2n H_{n-1}(x)
        
        H0 = np.ones_like(x)
        H4 = np.polynomial.hermite.hermval(x, [0, 0, 0, 0, 1]) # Coeffs for H4 in monomial basis? 
        # Easier: use scipy.special.eval_hermite
        from scipy.special import eval_hermite, eval_hermite_2d # eval_hermite is 1d
        
        H0_vals = eval_hermite(0, x)
        H4_vals = eval_hermite(4, x)
        H8_vals = eval_hermite(8, x)
        H12_vals = eval_hermite(12, x)
        
        # Derivatives: H_n' = 2n H_{n-1}
        H0_p = np.zeros_like(x)
        H4_p = 8 * eval_hermite(3, x)
        H8_p = 16 * eval_hermite(7, x)
        H12_p = 24 * eval_hermite(11, x)
        
        Poly = c0*H0_vals + c4*H4_vals + c8*H8_vals + c12*H12_vals
        Poly_p = c0*H0_p + c4*H4_p + c8*H8_p + c12*H12_p
        
        # Integrand for Norm: Poly^2
        integrand_norm = Poly**2
        # Integrand for X2 Norm: x^2 Poly^2
        integrand_x2 = x**2 * integrand_norm
        # Integrand for Momentum: (Poly_p - x Poly)^2
        momentum_term = Poly_p - x * Poly
        integrand_p = momentum_term**2
        
        # Integrate
        I0 = np.sum(w * integrand_norm)
        I2 = np.sum(w * integrand_x2)
        Ip = np.sum(w * integrand_p)
        
        if I0 < 1e-12:
            return 1e10
            
        Var_x = I2 / I0
        Var_p = Ip / I0
        
        return Var_x * Var_p

    # Initial guess from the "seed"
    x0 = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    
    # Optimization
    best_result = x0
    best_score = objective(x0)
    
    try:
        # Use BFGS for smooth optimization
        result = opt.minimize(objective, x0, method='BFGS', options={'disp': False, 'maxiter': 1000})
        
        if result.success and result.fun < best_score:
            best_result = result.x
            best_score = result.fun
            
        # Check time
        if time.time() - start_time > time_limit:
            pass # Just return best found so far
            
        # Try a second run with Nelder-Mead if time permits
        if time.time() - start_time < time_limit - 30:
            result2 = opt.minimize(objective, best_result, method='Nelder-Mead', options={'disp': False, 'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10})
            if result2.success and result2.fun < best_score:
                best_result = result2.x
                best_score = result2.fun
                
    except Exception:
        pass
        
    return best_result
