# EVOLVE-BLOCK-START
import time
import math
import numpy as np

_DEADLINE = 210.0  # wall-clock kill
_START = time.time()

def _remaining():
    return _DEADLINE - (time.time() - _START)

def _hermite_even(k, x):
    # Probabilists' He H_k(x) = (-1)^k e^{x^2/2} d^k/dx^k e^{-x^2/2}
    # H0=1, H2=x^2-1, H4=x^4-6x^2+3, H8=x^8-24x^6+72x^4-32x^2+15
    if k == 0:
        return np.ones_like(x)
    if k == 2:
        return x*x - 1.0
    if k == 4:
        return x**4 - 6.0*x*x + 3.0
    if k == 8:
        return x**8 - 24.0*x**6 + 72.0*x**4 - 32.0*x*x + 15.0
    if k == 12:
        # He_12 = x^12 - 60 x^10 + 540 x^8 - 2160 x^6 + 2880 x^4 - 720 x^2 + 105
        return (x**12 - 60.0*x**10 + 540.0*x**8 - 2160.0*x**6
                + 2880.0*x**4 - 720.0*x**2 + 105.0)
    raise ValueError("k must be even in {0,2,4,8,12}")

def _g_and_fourier(a0, a4, a8):
    """Return g(x) and its Fourier transform as functions of x (unitary convention)."""
    a12 = -(a0 + a4 + a8)  # impose g(0)=0
    def g(x):
        return (a0*_hermite_even(0,x) + a4*_hermite_even(4,x)
                + a8*_hermite_even(8,x) + a12*_hermite_even(12,x)) * np.exp(-x*x)
    def G(x):
        # Unitary FT: G(x) = (1/sqrt(2pi)) int g(y) e^{-ixy} dy
        # For g = (sum ak He_{2k}) e^{-x^2}, use the identity
        #   FT[He_{2k}(x) e^{-x^2}] = (1/sqrt(2pi)) He_{2k}(x) e^{-x^2} * (-1)^k
        # because He_{2k} e^{-x^2} is an eigenfunction of the (unitary) FT with eigenvalue (-1)^k.
        return (a0*_hermite_even(0,x) - a4*_hermite_even(4,x)
                + a8*_hermite_even(8,x) - a12*_hermite_even(12,x)) * np.exp(-x*x) / math.sqrt(2*math.pi)
    return g, G

def _bound_value(a0, a4, a8):
    """Upper bound on A(g)*A(G).  A(h) = sqrt( (int h^2)^2 + (int |h|^4) ).
    The Fourier-invariant Hermite bound: C4 = sup over the relevant quantity.
    We use the standard sharp bound: A(f)A(ĝ) <= C where
    C = sqrt( (1 + ||g||_4^4) (1 + ||G||_4^4) )  with ||h||_4^4 = int |h|^4 / (int |h|^2)^2.
    """
    g, G = _g_and_fourier(a0, a4, a8)
    # Integrate over a wide range; g decays like e^{-x^2}
    xs = np.linspace(-12.0, 12.0, 4001)
    dx = xs[1] - xs[0]
    gv = g(xs)
    Gv = G(xs)
    g2 = gv*gv
    G2 = Gv*Gv
    I_g2 = np.trapz(g2, xs)
    I_G2 = np.trapz(G2, xs)
    if I_g2 <= 0 or I_G2 <= 0:
        return float('inf')
    g4 = g2*g2
    G4 = G2*G2
    I_g4 = np.trapz(g4, xs)
    I_G4 = np.trapz(G4, xs)
    # normalized 4th moments
    n_g4 = I_g4 / (I_g2*I_g2)
    n_G4 = I_G4 / (I_G2*I_G2)
    # A(g) = sqrt(I_g2) * sqrt(1 + n_g4)  (since A(h)=sqrt((int h^2)^2 + int |h|^4))
    A_g = math.sqrt(I_g2) * math.sqrt(1.0 + n_g4)
    A_G = math.sqrt(I_G2) * math.sqrt(1.0 + n_G4)
    return A_g * A_G

def _optimize():
    from scipy.optimize import minimize
    best = _bound_value(-113/100, 1/25, 1/3240)
    best_x = np.array([-113/100, 1/25, 1/3240])

    def obj(v):
        try:
            return _bound_value(v[0], v[1], v[2])
        except Exception:
            return 1e9

    rng = np.random.default_rng(12345)
    # multi-start around prior
    starts = [np.array([-113/100, 1/25, 1/3240])]
    for _ in range(60):
        starts.append(rng.normal(0.0, 0.15, size=3))
    for _ in range(30):
        starts.append(rng.normal(0.0, 1.0, size=3))

    for s in starts:
        if _remaining() < 15.0:
            break
        try:
            res = minimize(obj, s, method='L-BFGS-B',
                           options={'maxiter': 400, 'ftol': 1e-12, 'gtol': 1e-10})
            if res.success and res.fun < best:
                best = res.fun
                best_x = res.x
        except Exception:
            pass
    return best_x, best

def construct():
    try:
        x, val = _optimize()
    except Exception:
        return np.array([-113/100, 1/25, 1/3240])
    # safety margin
    return np.asarray(x, dtype=float)
# EVOLVE-BLOCK-END
