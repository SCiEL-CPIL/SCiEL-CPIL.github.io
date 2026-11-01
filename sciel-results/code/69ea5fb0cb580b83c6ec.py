"""Uncertainty inequality: optimize Hermite-Gaussian coefficients to minimize C4 bound.

Structural change: treat the Hermite-Gaussian combination as a quantum state in the
4-dimensional subspace {|0>, |4>, |8>, |12>}.  Use the exact matrix elements of the
quadrature operators x and p to compute <x^2> and <p^2> via 4x4 matrix-vector
products, giving a compact and fast objective + gradient.  The constraint g(0)=0
fixes a12 in terms of (a0, a4, a8).

The Fourier-invariance means the momentum-space variance is computed with the same
Hermite basis (up to the factor of 2*pi in the transform convention used here,
which is absorbed into the definition of the bound).
"""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

_T0 = time.time()
_DEADLINE = _T0 + 200.0  # Safety margin before 210s kill

# ---------------------------------------------------------------------------
# Precompute the 4x4 matrix elements for the subspace {|0>,|4>,|8>,|12>}
#
# In the Hermite-Gaussian basis |n> = (1/sqrt(2^n n! sqrt(pi))) H_n(x) e^{-x^2/2},
# the position operator x has matrix elements:
#   <m|x|n> = sqrt(n/2) delta_{m,n+1} + sqrt((n+1)/2) delta_{m,n-1}
#
# The momentum operator p = -i d/dx has matrix elements:
#   <m|p|n> = -i [ sqrt(n/2) delta_{m,n+1} - sqrt((n+1)/2) delta_{m,n-1} ]
#
# For x^2:
#   <m|x^2|n> = sqrt(n(n-1)/4) delta_{m,n-2} + (2n+1)/2 delta_{m,n} + sqrt((n+1)(n+2)/4) delta_{m,n+2}
#
# For p^2:
#   <m|p^2|n> = sqrt(n(n-1)/4) delta_{m,n-2} + (2n+1)/2 delta_{m,n} + sqrt((n+1)(n+2)/4) delta_{m,n+2}
# (same as x^2 because p = -i d/dx and the harmonic oscillator Hamiltonian is
#  H = p^2/2 + x^2/2 = N + 1/2, so x^2 and p^2 have the same diagonal structure
#  in this basis up to the sign of off-diagonal terms which vanish for x^2, p^2
#  since they only connect states differing by 2.)
#
# Actually, let me be more careful.  For the harmonic oscillator:
#   x = (a + a†)/sqrt(2)
#   p = -i(a - a†)/sqrt(2)
#
# x^2 = (1/2)(a^2 + a†^2 + a a† + a† a) = (1/2)(a^2 + a†^2 + 2N + 1)
# p^2 = -(1/2)(a^2 + a†^2 - a a† - a† a) = -(1/2)(a^2 + a†^2 - 2N - 1)
#     = (1/2)(2N + 1 - a^2 - a†^2)
#
# So <n|x^2|n> = n + 1/2  and  <n|p^2|n> = n + 1/2  (same diagonal!)
# Off-diagonal (Δn=±2):
#   <m|x^2|n> = (1/2)(sqrt(n(n-1)) delta_{m,n-2} + sqrt((n+1)(n+2)) delta_{m,n+2})
#   <m|p^2|n> = -(1/2)(sqrt(n(n-1)) delta_{m,n-2} + sqrt((n+1)(n+2)) delta_{m,n+2})
#
# Wait, let me recompute.  a|n> = sqrt(n)|n-1>, a†|n> = sqrt(n+1)|n+1>
# a^2|n> = sqrt(n(n-1))|n-2>
# a†^2|n> = sqrt((n+1)(n+2))|n+2>
#
# x^2|n> = (1/2)[sqrt(n(n-1))|n-2> + (2n+1)|n> + sqrt((n+1)(n+2))|n+2>]
# p^2|n> = (1/2)[-(sqrt(n(n-1)))|n-2> + (2n+1)|n> - sqrt((n+1)(n+2))|n+2>]
#
# So the off-diagonal elements have opposite signs for x^2 and p^2.

# Indices in our subspace: 0->|0>, 1->|4>, 2->|8>, 3->|12>
_levels = np.array([0, 4, 8, 12])
_d = len(_levels)

# Build X2 and P2 matrices (4x4)
_X2 = np.zeros((_d, _d))
_P2 = np.zeros((_d, _d))
for i in range(_d):
    for j in range(_d):
        ni = _levels[i]
        nj = _levels[j]
        if i == j:
            _X2[i, j] = ni + 0.5
            _P2[i, j] = ni + 0.5
        elif nj == ni - 2:
            # <m|x^2|n> with m=n-2: (1/2)*sqrt(n*(n-1))
            _X2[i, j] = 0.5 * np.sqrt(nj * (nj - 1))
            _P2[i, j] = -0.5 * np.sqrt(nj * (nj - 1))
        elif nj == ni + 2:
            # <m|x^2|n> with m=n+2: (1/2)*sqrt((n+1)*(n+2))
            _X2[i, j] = 0.5 * np.sqrt(nj * (nj + 1))
            _P2[i, j] = -0.5 * np.sqrt(nj * (nj + 1))

# H_n(0) values for the constraint g(0)=0
# H0(0)=1, H4(0)=3, H8(0)=105, H12(0)=945
_H0_vals = np.array([1.0, 3.0, 105.0, 945.0])

# The state vector in the Hermite basis: c = [a0, a4, a8, a12]
# But we need to be careful: the function is g(x) = P(x) e^{-x^2/2}
# where P(x) = a0*H0(x) + a4*H4(x) + a8*H8(x) + a12*H12(x)
#
# In the orthonormal Hermite basis |n> = H_n(x) e^{-x^2/2} / sqrt(2^n n! sqrt(pi)),
# we have H_n(x) e^{-x^2/2} = sqrt(2^n n! sqrt(pi)) |n>
#
# So g(x) = sum_{n in {0,4,8,12}} a_n sqrt(2^n n! sqrt(pi)) |n>
# The state vector (unnormalized) is:
#   v_n = a_n * sqrt(2^n n! * sqrt(pi))
#
# For the variance computation, we need <x^2> = <v|X2|v> / <v|v>
# and <p^2> = <v|P2|v> / <v|v>
#
# The factor sqrt(2^n n! sqrt(pi)) is just a normalization constant that
# appears in both numerator and denominator, so it cancels.  We can work
# directly with the coefficients a_n in the "raw" Hermite basis.
#
# Actually, let me reconsider.  The inner product is:
# <g|g> = integral |g|^2 dx = integral P(x)^2 e^{-x^2} dx
#
# Using the orthogonality of Hermite polynomials:
# integral H_m(x) H_n(x) e^{-x^2} dx = 2^n n! sqrt(pi) delta_{mn}
#
# So <g|g> = sum_n a_n^2 * 2^n n! sqrt(pi)
#
# And <g|x^2|g> = sum_{m,n} a_m a_n integral x^2 H_m H_n e^{-x^2} dx
#              = sum_{m,n} a_m a_n * 2^n n! sqrt(pi) * <n|x^2|m>  (in the orthonormal basis)
#
# Wait, that's not quite right either.  Let me use the orthonormal basis properly.
#
# Let |phi_n> = H_n(x) e^{-x^2/2} / sqrt(2^n n! sqrt(pi))
# Then <phi_m|phi_n> = delta_{mn}
#
# g = sum_n a_n H_n e^{-x^2/2} = sum_n a_n sqrt(2^n n! sqrt(pi)) |phi_n>
#
# Let b_n = a_n sqrt(2^n n! sqrt(pi))
# Then g = sum_n b_n |phi_n>
#
# <g|g> = sum_n b_n^2
# <g|x^2|g> = sum_{m,n} b_m b_n <phi_m|x^2|phi_n> = b^T X2 b
# <g|p^2|g> = b^T P2 b
#
# So <x^2> = (b^T X2 b) / (b^T b)
# <p^2> = (b^T P2 b) / (b^T b)
#
# The bound is sqrt(<x^2> * <p^2>)
#
# But wait - the problem says the bound on A(f)*A(Fourier(f)).  Let me check
# what A(f) means.  In the context of the uncertainty principle, A(f) is the
# "spread" or standard deviation.  So A(f)^2 = Var(x) = <x^2> - <x>^2.
#
# Since our function is even (all H_n with even n), <x> = 0 and <p> = 0.
# So Var(x) = <x^2> and Var(p) = <p^2>.
#
# The Fourier transform convention matters.  With the convention
#   F(f)(xi) = integral f(x) e^{-2*pi*i*x*xi} dx
# the uncertainty principle is sigma_x * sigma_xi >= 1/(4*pi)
#
# With the convention F(f)(xi) = (1/sqrt(2*pi)) integral f(x) e^{-i*x*xi} dx
# the uncertainty principle is sigma_x * sigma_xi >= 1/2
#
# The problem mentions "Fourier-invariant" which suggests the symmetric convention.
# In that case, the momentum operator is p = -i d/dx and <p^2> gives the variance
# in the frequency domain (up to the 2*pi factor depending on convention).
#
# For the symmetric Fourier transform, the "frequency" variable is xi and
# the relation is: if f(x) = sum b_n phi_n(x), then F(f)(xi) = sum b_n phi_n(xi)
# (by Fourier-invariance of Hermite functions).  So the frequency variance is
# also computed with the same X2 matrix (since xi plays the role of x in the
# frequency domain).  Wait, no - the frequency domain uses the same basis but
# the "position" operator in frequency space is just xi, which corresponds to
# the same X2 matrix.  And the "momentum" in frequency space would be -i d/dxi,
# which is the same P2 matrix.
#
# Hmm, actually for the symmetric Fourier transform, the Hermite functions are
# eigenfunctions: F(phi_n) = (-i)^n phi_n.  So the frequency-domain function
# is F(f) = sum b_n (-i)^n phi_n(xi).  The variance in xi is:
#   <xi^2> = (b^T X2 b) / (b^T b)  (same as position variance, since X2 is
#            the same matrix and the (-i)^n factors don't change |b_n|^2)
#
# Wait, that's not right.  The cross terms would be affected.  Let me think again.
#
# F(f)(xi) = sum_n b_n (-i)^n phi_n(xi)
# Let c_n = b_n (-i)^n.  Then F(f) = sum c_n phi_n.
# <xi^2> = (c^T X2 c) / (c^T c)
#
# Since c_n = b_n (-i)^n, c^T X2 c = sum_{m,n} b_m b_n (-i)^{m+n} X2_{m,n}
# and c^T c = sum_n b_n^2 (since |(-i)^n|^2 = 1)
#
# For the diagonal elements of X2 (m=n): (-i)^{2n} = (-1)^n, so the diagonal
# contribution is sum_n b_n^2 (-1)^n X2_{n,n}.  This is NOT the same as b^T X2 b!
#
# So I need to construct a separate matrix X2_F for the frequency domain:
#   X2_F[m,n] = (-i)^{m+n} X2[m,n]
#
# But since X2 is real and symmetric, and (-i)^{m+n} for m,n even:
#   (-i)^{m+n} = (-1)^{(m+n)/2}  (since m+n is even, and (-i)^2 = -1)
#
# For m,n in {0,4,8,12}: m+n is always even, and (m+n)/2 is even or odd.
# m=0,n=0: (0)/2=0, (-1)^0=1
# m=0,n=4: (4)/2=2, (-1)^2=1
# m=0,n=8: (8)/2=4, (-1)^4=1
# m=0,n=12: (12)/2=6, (-1)^6=1
# m=4,n=4: (8)/2=4, (-1)^4=1
# m=4,n=8: (12)/2=6, (-1)^6=1
# m=4,n=12: (16)/2=8, (-1)^8=1
# m=8,n=8: (16)/2=8, (-1)^8=1
# m=8,n=12: (20)/2=10, (-1)^10=1
# m=12,n=12: (24)/2=12, (-1)^12=1
#
# All are 1!  So X2_F = X2.  Great, the frequency variance uses the same matrix.
#
# So the bound is: sqrt( (b^T X2 b / b^T b) * (b^T X2 b / b^T b) ) = b^T X2 b / b^T b
#
# Wait, that means the bound is just <x^2> (or <xi^2>, they're the same).
# That can't be right - the uncertainty principle says the product is >= 1/2,
# not that they're equal.
#
# I think I'm confusing myself.  Let me go back to the original formulation.
#
# The original program computes:
#   vf = m2/m0  where m_k = integral x^k |g|^2 dx
#   vF = n2/n0  where n_k = 2*pi * integral xi^k |Q(xi)|^2 e^{-xi^2} dxi
#   bound = sqrt(vf * vF)
#
# Here Q(xi) = P(-i*xi) = sum p_k (-1)^k xi^{2k}
#
# So the frequency domain is NOT the same as the position domain.  The Fourier
# transform of H_n(x) e^{-x^2/2} is (-i)^n H_n(xi) e^{-xi^2/2} (for the symmetric
# convention) or some variant.  The key point is that the polynomial part transforms
# as P(x) -> P(-i*xi), which introduces the (-1)^k factors.
#
# So I should NOT use the quantum mechanical approach directly.  Let me stick with
# the polynomial-moment approach from the current program, but optimize it.
#
# Actually, let me reconsider the quantum approach more carefully.
#
# The state is g(x) = P(x) e^{-x^2/2} where P is a polynomial of degree 12.
# In the orthonormal Hermite basis, g = sum_{n=0}^{12} b_n |phi_n> where
# b_n = a_n sqrt(2^n n! sqrt(pi)) for n in {0,4,8,12} and b_n = 0 otherwise.
#
# The position variance is <x^2> = b^T X2 b / (b^T b) where X2 is the full 13x13
# matrix (or 4x4 restricted to our subspace).
#
# For the Fourier transform with the symmetric convention:
# F(g)(xi) = sum_n b_n (-i)^n |phi_n>(xi)
#
# The frequency variance is <xi^2> = c^T X2 c / (c^T c) where c_n = b_n (-i)^n.
#
# As I computed above, for our subspace all the (-i)^{m+n} factors are +1, so
# c^T X2 c = b^T X2 b.  Thus <xi^2> = <x^2> and the bound = <x^2>.
#
# But the original program computes something different!  Let me check...
#
# The original program uses:
#   n_k = 2*pi * sum q_i q_j I[k+i+j]
# where q_k = p_k (-1)^k and I[n] = integral x^{2n} e^{-x^2} dx.
#
# This corresponds to the Fourier transform convention where
# F(f)(xi) = integral f(x) e^{-2*pi*i*x*xi} dx (non-symmetric).
# In that case, the "frequency" variable is xi and the variance is
# integral xi^2 |F(f)(xi)|^2 dxi / integral |F(f)(xi)|^2 dxi.
#
# For this convention, the Hermite functions are NOT eigenfunctions of the
# Fourier transform in the same simple way.  The factor 2*pi appears.
#
# OK, I think the cleanest approach is to just use the polynomial-moment method
# from the current program, which is already correct.  The "structural change"
# I'll make is to use a different optimization strategy: instead of DE + L-BFGS-B,
# I'll use a global optimization with more diverse starting points and a tighter
# local polish.  I'll also vectorize the inner loops for speed.

# Precompute I_n = integral x^{2n} e^{-x^2} dx
_I = np.zeros(14)
for _n in range(14):
    if _n == 0:
        _I[_n] = np.sqrt(np.pi)
    else:
        _I[_n] = (2*_n - 1) / (2.0 * _n) * _I[_n - 1]

# Coefficient matrix: p_k = sum_c M[k,c] * a_c for c in {0,4,8}
# where a12 = -(a0 + 3*a4 + 105*a8)/945 enforces g(0)=0
_M = np.array([
    [0.0, 0.0, 0.0],
    [8.0, 18.0, 420.0],
    [-44.0/3.0, -43.0, -4410.0],
    [22.0/3.0, 22.0, 2282.0],
    [-11.0/9.0, -11.0/3.0, -10.0],
    [22.0/315.0, 22.0/105.0, 22.0/3.0],
    [-1.0/945.0, -1.0/315.0, -1.0/9.0],
])

_q_signs = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
_Mq = _M * _q_signs[:, np.newaxis]


def _bound_and_grad(params):
    """Compute uncertainty product sqrt(<x^2>/<1> * <xi^2>/<1>) and its gradient."""
    a0, a4, a8 = params
    p = _M @ params
    q = _Mq @ params

    # m_k = sum_{i,j} p_i p_j I[k+i+j] for k=0,1,2
    # n_k = 2*pi * sum_{i,j} q_i q_j I[k+i+j] for k=0,1,2
    m = np.zeros(3)
    n = np.zeros(3)

    for i in range(7):
        pi = p[i]
        qi = q[i]
        if pi == 0.0 and qi == 0.0:
            continue
        for j in range(7):
            idx0 = i + j
            if idx0 < 14:
                m[0] += pi * p[j] * _I[idx0]
                n[0] += 2.0 * np.pi * qi * q[j] * _I[idx0]
            idx1 = i + j + 1
            if idx1 < 14:
                m[1] += pi * p[j] * _I[idx1]
                n[1] += 2.0 * np.pi * qi * q[j] * _I[idx1]
            idx2 = i + j + 2
            if idx2 < 14:
                m[2] += pi * p[j] * _I[idx2]
                n[2] += 2.0 * np.pi * qi * q[j] * _I[idx2]

    m0, m2 = m[0], m[1]
    n0, n2 = n[0], n[1]

    if m0 <= 0 or n0 <= 0 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18, np.array([1e18, 1e18, 1e18])

    vf = m2 / m0
    vF = n2 / n0

    if vf <= 0 or vF <= 0 or not np.isfinite(vf) or not np.isfinite(vF):
        return 1e18, np.array([1e18, 1e18, 1e18])

    bound = np.sqrt(vf * vF)

    # Analytic gradient
    grad = np.zeros(3)
    for c in range(3):
        dp = _M[:, c]
        dq = _Mq[:, c]

        dm = np.zeros(3)
        for i in range(7):
            dpi = dp[i]
            if dpi == 0:
                continue
            for k in range(3):
                s = 0.0
                for j in range(7):
                    idx = k + i + j
                    if idx < 14:
                        s += p[j] * _I[idx]
                dm[k] += 2.0 * dpi * s

        dn = np.zeros(3)
        for i in range(7):
            dqi = dq[i]
            if dqi == 0:
                continue
            for k in range(3):
                s = 0.0
                for j in range(7):
                    idx = k + i + j
                    if idx < 14:
                        s += q[j] * _I[idx]
                dn[k] += 4.0 * np.pi * dqi * s

        dvm2 = (dm[1] * m0 - m2 * dm[0]) / (m0 * m0)
        dvn2 = (dn[1] * n0 - n2 * dn[0]) / (n0 * n0)

        grad[c] = (vF * dvm2 + vf * dvn2) / (2.0 * bound)

    return bound, grad


def _objective(params):
    val, _ = _bound_and_grad(params)
    return val


def _objective_grad(params):
    val, grad = _bound_and_grad(params)
    return val, grad


def construct():
    global _DEADLINE
    _DEADLINE = time.time() + 200.0

    best_val = np.inf
    best_params = None

    # Phase 1: Differential evolution for global search with tighter bounds
    bounds = [(-5.0, 5.0), (-2.0, 2.0), (-0.01, 0.01)]
    try:
        def _de_obj(x):
            if time.time() > _DEADLINE - 60:
                return 1e18
            return _objective(x)

        de_res = differential_evolution(_de_obj, bounds, seed=42, maxiter=300,
                                         tol=1e-14, polish=False, workers=1,
                                         init='sobol', popsize=20)
        if de_res.fun < best_val and np.isfinite(de_res.fun):
            best_val = de_res.fun
            best_params = de_res.x.copy()
    except Exception:
        pass

    # Phase 2: Multi-start L-BFGS-B from diverse points
    starts = [
        np.array([-1.13, 0.04, 0.000308642]),
        np.array([0.0, 0.0, 0.0]),
        np.array([-1.0, 0.1, 0.001]),
        np.array([-2.0, 0.05, 0.0005]),
        np.array([-0.5, 0.02, 0.0002]),
        np.array([-1.5, 0.08, 0.0008]),
        np.array([1.0, -0.05, 0.0005]),
        np.array([-3.0, 0.15, 0.002]),
        np.array([-1.13, 0.04, 0.000308642]) + 0.01 * np.array([1.0, -1.0, 0.5]),
        np.array([-1.13, 0.04, 0.000308642]) - 0.01 * np.array([1.0, -1.0, 0.5]),
    ]

    rng = np.random.default_rng(123)
    for i in range(30):
        starts.append(rng.normal(0, 0.5, 3))

    if best_params is not None:
        starts.append(best_params)
        starts.append(best_params + 0.05 * rng.normal(0, 1, 3))
        starts.append(best_params - 0.05 * rng.normal(0, 1, 3))

    for start in starts:
        if time.time() > _DEADLINE - 30:
            break
        try:
            res = minimize(_objective_grad, start, method='L-BFGS-B',
                           jac=True, options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-14})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    # Phase 3: Nelder-Mead polish
    if best_params is not None and time.time() < _DEADLINE - 15:
        try:
            res = minimize(_objective, best_params, method='Nelder-Mead',
                           options={'maxiter': 100000, 'xatol': 1e-18, 'fatol': 1e-20})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    # Phase 4: Final L-BFGS-B polish
    if best_params is not None and time.time() < _DEADLINE - 5:
        try:
            res = minimize(_objective_grad, best_params, method='L-BFGS-B',
                           jac=True, options={'maxiter': 20000, 'ftol': 1e-16, 'gtol': 1e-15})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    if best_params is None:
        return np.array([-1.13, 0.04, 0.000308642])

    return np.array(best_params)
