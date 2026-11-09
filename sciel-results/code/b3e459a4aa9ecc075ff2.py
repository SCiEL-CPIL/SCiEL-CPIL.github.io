import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize
from scipy.special import eval_hermite
import time

def construct():
    start_time = time.time()
    deadline = 210.0  # 210s limit, keep safety margin
    
    # Hermite polynomials H_0, H_4, H_8, H_12
    # Using physicists' Hermite polynomials
    # H_0(x) = 1
    # H_4(x) = 16x^4 - 48x^2 + 12
    # H_8(x) = 256x^8 - 3072x^6 + 11520x^4 - 13440x^2 + 1680
    # H_12(x) is large, we only need H_12(0) for the constraint and the polynomial for integration
    # H_n(0) for even n: (-1)^(n/2) * n! / (n/2)!
    # H_0(0) = 1
    # H_4(0) = 12
    # H_8(0) = 1680
    # H_12(0) = 665280
    
    H0_0 = 1.0
    H4_0 = 12.0
    H8_0 = 1680.0
    H12_0 = 665280.0

    def get_c12(c0, c4, c8):
        return -(c0 * H0_0 + c4 * H4_0 + c8 * H8_0) / H12_0

    def g(x, c0, c4, c8):
        c12 = get_c12(c0, c4, c8)
        # Evaluate H_k(x)
        # Using explicit formulas for speed
        h0 = 1.0
        h4 = 16.0*x**4 - 48.0*x**2 + 12.0
        h8 = 256.0*x**8 - 3072.0*x**6 + 11520.0*x**4 - 13440.0*x**2 + 1680.0
        # H12(x) coefficients:
        # H12(x) = 4096x^12 - 153600x^10 + 2419200x^8 - 16632000x^6 + 55574400x^4 - 79833600x^2 + 47900160
        # Let's verify H12(0) = 47900160? 
        # Formula: (-1)^6 * 12! / 6! = 479001600 / 720 = 665280.
        # My previous manual calc for H8(0) was 1680.
        # 8! / 4! = 40320 / 24 = 1680. Correct.
        # 12! / 6! = 479001600 / 720 = 665280.
        # So H12(0) = 665280.
        # The constant term in H12(x) is 665280.
        # Let's get the full polynomial for H12.
        # Recurrence: (n+1)H_{n+1} = 2x H_n - 2n H_{n-1}
        # H0=1
        # H1=2x
        # H2=4x^2-2
        # H3=8x^3-12x
        # H4=16x^4-48x^2+12
        # H5=32x^5-160x^3+120x
        # H6=64x^6-480x^4+720x^2-120
        # H7=128x^7-1344x^5+3360x^3-1680x
        # H8=256x^8-3072x^6+11520x^4-13440x^2+1680
        # H9=512x^9-7680x^7+34560x^5-60480x^3+30240x
        # H10=1024x^10-15360x^8+76800x^6-161280x^4+120960x^2-20160
        # H11=2048x^11-30720x^9+153600x^7-322560x^5+241920x^3-80640x
        # H12=4096x^12-49152x^10+229376x^8-458752x^6+359616x^4-118272x^2+16128? 
        # Let's re-calculate H12 constant term.
        # Constant term of H_n is (-1)^(n/2) n!/(n/2)! for even n.
        # H12(0) = 665280.
        # My recurrence gave 16128? 
        # H10 const: -20160. 10!/5! = 3628800/120 = 30240. (-1)^5 * 30240 = -30240.
        # My recurrence H10 const was -20160. Mismatch.
        # Let's use scipy.special.hermite to be sure or just numerical integration of the polynomial defined by roots?
        # Better: Use numpy.polynomial.hermite.herm2poly to get coefficients.
        
        # For speed in the objective, we can pre-compute the polynomial coefficients for H4, H8, H12.
        # But we are inside construct, so we can do it once.
        pass

    # Pre-compute Hermite polynomial coefficients
    # Using numpy.polynomial.hermite
    from numpy.polynomial.hermite import herm2poly, Hermite
    
    # Note: numpy.polynomial.hermite uses Physicists' Hermite polynomials?
    # "The Hermite polynomials are defined by the recurrence relation: (n+1)H_{n+1}(x) = 2xH_n(x) - 2nH_{n-1}(x)"
    # Yes, Physicists'.
    
    H4_poly = herm2poly(4)
    H8_poly = herm2poly(8)
    H12_poly = herm2poly(12)
    
    # Coefficients are in ascending order of power.
    # H4_poly: [12, 0, -48, 0, 16] -> 12 + 0x - 48x^2 + 0x^3 + 16x^4
    # H8_poly: [1680, 0, -13440, 0, 11520, 0, -3072, 0, 256]
    # H12_poly: [665280, 0, -1182720, ...]
    
    def eval_poly(coeffs, x):
        # coeffs ascending
        val = 0.0
        for i in range(len(coeffs)):
            val += coeffs[i] * (x ** i)
        return val

    def g_func(x, c0, c4, c8):
        c12 = get_c12(c0, c4, c8)
        h0 = 1.0
        h4 = eval_poly(H4_poly, x)
        h8 = eval_poly(H8_poly, x)
        h12 = eval_poly(H12_poly, x)
        p = c0*h0 + c4*h4 + c8*h8 + c12*h12
        return np.exp(-x**2/2.0) * p

    def norm_4_sq(c0, c4, c8):
        # Integral of |g(x)|^4 dx
        def integrand(x):
            val = g_func(x, c0, c4, c8)
            return val**4
        val, err = quad(integrand, -np.inf, np.inf, limit=200)
        return val

    def fourier_g(xi, c0, c4, c8):
        # Fourier transform of e^{-x^2/2} P(x)
        # F{e^{-x^2/2} x^n} = i^n sqrt(2pi) H_n(xi/2) e^{-xi^2/2} ?
        # Let's use the property: F{x^n e^{-x^2/2}} = (-i)^n \frac{d^n}{d\xi^n} F{e^{-x^2/2}}
        # F{e^{-x^2/2}} = \sqrt{2\pi} e^{-\xi^2/2}
        # So F{x^n e^{-x^2/2}} = (-i)^n \sqrt{2\pi} \frac{d^n}{d\xi^n} (e^{-\xi^2/2})
        # \frac{d^n}{d\xi^n} e^{-\xi^2/2} = (-1)^n H_n(\xi/\sqrt{2}) e^{-\xi^2/2} / 2^{n/2} ?
        # Actually, it's easier to just numerically integrate g(x) e^{-ixxi} if we can't derive it easily.
        # But g is a polynomial times Gaussian.
        # Let's use the known result:
        # If g(x) = e^{-x^2/2} P(x), then \hat{g}(\xi) = \frac{1}{\sqrt{2\pi}} \sqrt{2\pi} e^{-\xi^2/2} Q(\xi)
        # where Q is related to P.
        # Specifically, \hat{g}(\xi) = e^{-\xi^2/2} \tilde{P}(\xi)
        # The coefficients of \tilde{P} are related to c_k.
        # For H_k(x), the Fourier transform of e^{-x^2/2} H_k(x) is i^k e^{-\xi^2/2} H_k(\xi) ?
        # Check: F{e^{-x^2/2}} = \sqrt{2\pi} e^{-\xi^2/2}.
        # F{x e^{-x^2/2}} = i \sqrt{2\pi} \xi e^{-\xi^2/2}.
        # H_1(x) = 2x. F{e^{-x^2/2} 2x} = i \sqrt{2\pi} \xi e^{-\xi^2/2} = i \sqrt{2\pi} \frac{1}{2} H_1(\xi) e^{-\xi^2/2}.
        # So F{e^{-x^2/2} H_k(x)} = i^k \sqrt{2\pi} \frac{1}{2^{k/2}} H_k(\xi) e^{-\xi^2/2} ?
        # Let's check k=2. H_2(x) = 4x^2-2.
        # F{e^{-x^2/2}(4x^2-2)} = 4 F{x^2 e^{-x^2/2}} - 2 F{e^{-x^2/2}}
        # F{x^2 e^{-x^2/2}} = -\sqrt{2\pi} (\xi^2 - 1) e^{-\xi^2/2}
        # Result: 4(-\sqrt{2\pi})(\xi^2-1)e^{-\xi^2/2} - 2\sqrt{2\pi}e^{-\xi^2/2} = -\sqrt{2\pi} (4\xi^2 - 4 + 2) e^{-\xi^2/2} = -\sqrt{2\pi} (4\xi^2 - 2) e^{-\xi^2/2}
        # H_2(\xi) = 4\xi^2 - 2.
        # So F{...} = -\sqrt{2\pi} H_2(\xi) e^{-\xi^2/2}.
        # Formula: i^2 \sqrt{2\pi} \frac{1}{2} H_2(\xi) e^{-\xi^2/2} = -\sqrt{2\pi} \frac{1}{2} (4\xi^2-2) e^{-\xi^2/2} = -\sqrt{2\pi} (2\xi^2-1) e^{-\xi^2/2}.
        # Mismatch by factor of 2.
        # Correct relation: F{e^{-x^2/2} H_n(x)} = i^n \sqrt{2\pi} e^{-\xi^2/2} H_n(\xi) ?
        # For n=2: i^2 = -1. -\sqrt{2\pi} H_2(\xi) e^{-\xi^2/2}. Matches!
        # For n=1: i^1 \sqrt{2\pi} H_1(\xi) e^{-\xi^2/2} = i \sqrt{2\pi} 2\xi e^{-\xi^2/2}.
        # Earlier I got i \sqrt{2\pi} \xi e^{-\xi^2/2}.
        # Let's re-evaluate F{x e^{-x^2/2}}.
        # \int x e^{-x^2/2} e^{-ix\xi} dx = i \frac{d}{d\xi} \int e^{-x^2/2} e^{-ix\xi} dx = i \frac{d}{d\xi} (\sqrt{2\pi} e^{-\xi^2/2}) = i \sqrt{2\pi} (-\xi) e^{-\xi^2/2} = -i \sqrt{2\pi} \xi e^{-\xi^2/2}.
        # H_1(\xi) = 2\xi.
        # So F{x e^{-x^2/2}} = -i \sqrt{2\pi} \frac{1}{2} H_1(\xi) e^{-\xi^2/2}.
        # So F{e^{-x^2/2} H_1(x)} = 2 F{x e^{-x^2/2}} = -i \sqrt{2\pi} H_1(\xi) e^{-\xi^2/2}.
        # Formula $i^n \sqrt{2\pi} H_n(\xi) e^{-\xi^2/2}$ gives $i \sqrt{2\pi} H_1(\xi) e^{-\xi^2/2}$.
        # Sign mismatch.
        # It is $(-i)^n$?
        # For n=1: $(-i)^1 = -i$. Matches.
        # For n=2: $(-i)^2 = -1$. Matches.
        # So $\hat{g}(\xi) = \sqrt{2\pi} e^{-\xi^2/2} \sum c_k (-i)^k H_k(\xi)$.
        # Wait, the normalization of Fourier transform.
        # Unitary: $\hat{f}(\xi) = \frac{1}{\sqrt{2\pi}} \int f(x) e^{-ix\xi} dx$.
        # Then $\|f\|_2 = \|\hat{f}\|_2$.
        # $\hat{g}(\xi) = \frac{1}{\sqrt{2\pi}} \sum c_k (-i)^k \sqrt{2\pi} e^{-\xi^2/2} H_k(\xi) = e^{-\xi^2/2} \sum c_k (-i)^k H_k(\xi)$.
        
        # So if g(x) = e^{-x^2/2} P(x), then \hat{g}(\xi) = e^{-\xi^2/2} \hat{P}(\xi)
        # where \hat{P}(\xi) = \sum c_k (-i)^k H_k(\xi).
        # Since we only care about norms, and H_k are real, the magnitude is:
        # |\hat{g}(\xi)| = e^{-\xi^2/2} | \sum c_k (-i)^k H_k(\xi) |
        # Note that $(-i)^k$ is complex for odd k. But we only have even k (0,4,8,12).
        # For even k, $(-i)^k = (-1)^{k/2}$.
        # k=0: 1
        # k=4: 1
        # k=8: 1
        # k=12: 1
        # So all coefficients are real in the Fourier domain too!
        # \hat{g}(\xi) = e^{-\xi^2/2} (c_0 H_0(\xi) + c_4 H_4(\xi) + c_8 H_8(\xi) + c_{12} H_{12}(\xi))
        # This is the same polynomial structure!
        
        p_hat = c0*1.0 + c4*eval_poly(H4_poly, xi) + c8*eval_poly(H8_poly, xi) + c12*eval_poly(H12_poly, xi)
        return np.exp(-xi**2/2.0) * p_hat

    def norm_4_sq_hat(c0, c4, c8):
        def integrand(xi):
            val = fourier_g(xi, c0, c4, c8)
            return val**4
        val, err = quad(integrand, -np.inf, np.inf, limit=200)
        return val

    def objective(c):
        c0, c4, c8 = c
        n1 = norm_4_sq(c0, c4, c8)
        n2 = norm_4_sq_hat(c0, c4, c8)
        if n1 < 0 or n2 < 0:
            return 1e10
        return (n1 * n2) ** 0.25  # L4 norm product

    # Initial guess from seed
    x0 = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    
    # Check time
    if time.time() - start_time > deadline - 30:
        return x0

    # Optimize
    try:
        res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8})
        if time.time() - start_time > deadline - 10:
            # Return best so far if close to deadline
            return res.x if res.success else x0
        return res.x
    except Exception:
        return x0
