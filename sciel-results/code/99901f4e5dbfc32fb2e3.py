"""
Tuned Fourier-invariant Hermite-Gaussian test function for uncertainty inequality.

We seek coefficients (a0, a4, a8) for g(x) = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * exp(-x^2/2)
where a12 is derived from g(0)=0: a12 = -(a0 + a4 + a8)/H12(0).

The score is the upper bound on C4 = A(f)*A(Fourier(f)) for f = g * exp(-x^2/2) normalized.
Actually, the task says: "score is the upper bound on C4" and "raw value (lower is better)".

Let me think about this more carefully. The uncertainty inequality bound involves the
product of the L1 norm (or similar) of f and its Fourier transform. For a function
g(x) = sum a_k H_k(x) * exp(-x^2/2), we need to compute:

A(f) = integral |f(x)| dx  (or perhaps the L1 norm)
A(Fourier(f)) = integral |F(f)(xi)| dxi

And the bound C4 = A(f) * A(F(f)).

Actually, re-reading: "Tune a Fourier-invariant Hermite-polynomial times Gaussian test function to reduce the upper bound on A(f)*A(Fourier(f))."

So we're minimizing the product of the L1 norms of f and its Fourier transform.

For a function f(x) = g(x) where g is a combination of Hermite-Gaussian functions,
the Fourier transform of H_n(x) * exp(-x^2/2) is (-i)^n H_n(xi) * exp(-xi^2/2) (up to
normalization constants depending on the Fourier transform convention).

Let's use the convention: F(f)(xi) = integral f(x) exp(-i x xi) dx

For f(x) = H_n(x) exp(-x^2/2), F(f)(xi) = (-i)^n H_n(xi) exp(-xi^2/2) * sqrt(2*pi)

Wait, let me be more careful. The standard result is:
If f(x) = exp(-x^2/2), then F(f)(xi) = sqrt(2*pi) * exp(-xi^2/2)
If f(x) = H_n(x) exp(-x^2/2), then F(f)(xi) = (-i)^n sqrt(2*pi) H_n(xi) exp(-xi^2/2)

So for f(x) = sum_k a_k H_k(x) exp(-x^2/2):
F(f)(xi) = sqrt(2*pi) * sum_k a_k (-i)^k H_k(xi) exp(-xi^2/2)

The L1 norm: A(f) = integral |f(x)| dx
A(F(f)) = integral |F(f)(xi)| dxi = sqrt(2*pi) * integral |sum_k a_k (-i)^k H_k(xi)| exp(-xi^2/2) dxi

Note that since the coefficients are real, and (-i)^k cycles through 1, -i, -1, i, the
magnitude |sum a_k (-i)^k H_k(xi)| is not simply |sum a_k H_k(xi)|.

Actually, let's reconsider. The problem says "Fourier-invariant" which might mean that
the form is preserved under Fourier transform. For a combination of even-degree Hermite
polynomials (H0, H4, H8, H12 are all even), (-i)^k for k=0,4,8,12 gives:
k=0: (-i)^0 = 1
k=4: (-i)^4 = 1
k=8: (-i)^8 = 1
k=12: (-i)^12 = 1

So all the (-i)^k factors are 1! This means:
F(f)(xi) = sqrt(2*pi) * sum_k a_k H_k(xi) exp(-xi^2/2) = sqrt(2*pi) * f(xi)

So f is an eigenfunction of the Fourier transform with eigenvalue sqrt(2*pi).

Therefore:
A(f) = integral |f(x)| dx
A(F(f)) = sqrt(2*pi) * A(f)

And C4 = A(f) * A(F(f)) = sqrt(2*pi) * (A(f))^2

So we need to minimize (A(f))^2, which is equivalent to minimizing A(f) = integral |f(x)| dx.

Where f(x) = (a0*H0(x) + a4*H4(x) + a8*H8(x) + a12*H12(x)) * exp(-x^2/2)
and a12 = -(a0 + a4 + a8) / H12(0)

H12(0) for physicists' Hermite polynomials: H_n(0) for even n.
H0(0) = 1
H2(0) = -2
H4(0) = 12
H6(0) = -120
H8(0) = 1680
H10(0) = -30240
H12(0) = 665280

Wait, let me verify. The physicists' Hermite polynomials are:
H0(x) = 1
H1(x) = 2x
H2(x) = 4x^2 - 2
H3(x) = 8x^3 - 12x
H4(x) = 16x^4 - 48x^2 + 12
H5(x) = 32x^5 - 160x^3 + 120x
H6(x) = 64x^6 - 480x^4 + 720x^2 - 120
H7(x) = 128x^7 - 1344x^5 + 3360x^3 - 1680x
H8(x) = 256x^8 - 4480x^6 + 23520x^4 - 33600x^2 + 1680
H9(x) = 512x^9 - 12288x^7 + 107520x^5 - 362880x^3 + 302400x
H10(x) = 1024x^10 - 32768x^8 + 403200x^6 - 1848000x^4 + 3024000x^2 - 30240
H11(x) = 2048x^11 - 73728x^9 + 1128960x^7 - 7257600x^5 + 18662400x^3 - 15523200x
H12(x) = 4096x^12 - 184320x^10 + 3440640x^8 - 28828800x^6 + 118848000x^4 - 221625600x^2 + 665280

So H12(0) = 665280.

The constraint g(0) = 0 means:
a0*H0(0) + a4*H4(0) + a8*H8(0) + a12*H12(0) = 0
a0*1 + a4*12 + a8*1680 + a12*665280 = 0
a12 = -(a0 + 12*a4 + 1680*a8) / 665280

Now we need to minimize:
A(f) = integral_{-inf}^{inf} |(a0*H0(x) + a4*H4(x) + a8*H8(x) + a12*H12(x)) * exp(-x^2/2)| dx

This is a 3-parameter optimization problem. Let's use scipy.optimize to find the minimum.

The seed program returns [-113/100, 1/25, 1/3240] = [-1.13, 0.04, 0.000308642...].

Let me compute what the current score is. With these coefficients:
a12 = -(-1.13 + 12*0.04 + 1680*0.000308642) / 665280
     = -(-1.13 + 0.48 + 0.518518...) / 665280
     = -(-0.131482...) / 665280
     = 0.131482... / 665280
     ≈ 1.9764e-7

Let me write the code to numerically minimize this.
"""
# EVOLVE-BLOCK-START
import numpy as np
from scipy import optimize
from scipy.special import eval_hermite
import time

_DEADLINE = 180.0  # seconds

def _compute_A_f(a0, a4, a8):
    """Compute A(f) = integral |f(x)| dx for the given coefficients."""
    # a12 from g(0) = 0
    H0_0 = 1.0
    H4_0 = 12.0
    H8_0 = 1680.0
    H12_0 = 665280.0
    a12 = -(a0 * H0_0 + a4 * H4_0 + a8 * H8_0) / H12_0
    
    def f_abs(x):
        H0 = 1.0
        H4 = 16*x**4 - 48*x**2 + 12
        H8 = 256*x**8 - 4480*x**6 + 23520*x**4 - 33600*x**2 + 1680
        H12 = (4096*x**12 - 184320*x**10 + 3440640*x**8 - 28828800*x**6 
               + 118848000*x**4 - 221625600*x**2 + 665280)
        poly = a0*H0 + a4*H4 + a8*H8 + a12*H12
        return abs(poly) * np.exp(-x**2 / 2.0)
    
    # Integrate from -inf to inf
    result, err = optimize.quad(f_abs, -np.inf, np.inf, limit=200, epsabs=1e-12, epsrel=1e-10)
    return result

def _objective(params):
    """Objective function: A(f) (we minimize this, and score = sqrt(2*pi) * A(f)^2)."""
    a0, a4, a8 = params
    return _compute_A_f(a0, a4, a8)

def _score_from_Af(Af):
    """Convert A(f) to the score (upper bound on C4)."""
    return np.sqrt(2 * np.pi) * Af**2

def construct():
    """Return three coefficients (a0, a4, a8) minimizing the upper bound on C4."""
    start_time = time.time()
    
    # Seed value
    seed = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    best_params = seed.copy()
    best_Af = _compute_A_f(seed[0], seed[1], seed[2])
    best_score = _score_from_Af(best_Af)
    
    # Multi-start local optimization
    n_starts = 50
    rng = np.random.RandomState(42)
    
    for i in range(n_starts):
        elapsed = time.time() - start_time
        if elapsed > _DEADLINE - 30:  # Leave 30s safety margin
            break
        
        if i == 0:
            x0 = seed.copy()
        else:
            # Random perturbations around the seed and other starting points
            scale = 0.5 * (i % 5 + 1) / 10.0
            x0 = seed + scale * rng.randn(3)
        
        try:
            result = optimize.minimize(
                _objective, x0, method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            elapsed = time.time() - start_time
            if elapsed > _DEADLINE - 30:
                break
            
            if result.fun < best_Af:
                best_Af = result.fun
                best_params = result.x
                best_score = _score_from_Af(best_Af)
        except Exception:
            continue
    
    # Final polish with higher precision
    elapsed = time.time() - start_time
    if elapsed < _DEADLINE - 15:
        try:
            result = optimize.minimize(
                _objective, best_params, method='Nelder-Mead',
                options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            if result.fun < best_Af:
                best_Af = result.fun
                best_params = result.x
                best_score = _score_from_Af(best_Af)
        except Exception:
            pass
    
    return np.array(best_params)
# EVOLVE-BLOCK-END
