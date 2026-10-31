"""Structural change: optimize the Hermite coefficients to minimize the
uncertainty product A(f)*A(F), modeled exactly in the Hermite basis, using
a global+local search with an explicit deadline guard."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np

_DEADLINE = 210.0
_SAFETY = 25.0


def _hermite_coeffs(n):
    """Coefficients of the probabilists' Hermite polynomial He_n(x) = n! sum H_k x^k/k!.
    Returns list of length n+1 (H_0..H_n)."""
    H = [0.0] * (n + 1)
    H[0] = 1.0
    if n >= 1:
        H[1] = 1.0
    for k in range(2, n + 1):
        H[k] = H[k - 1] - H[k - 2] / k
    return H


def _hermite_norm(n):
    """||He_n||^2 in L^2(R, dx) = 2^n n! sqrt(pi)."""
    return float((2.0 ** n) * math.factorial(n) * math.sqrt(math.pi))


def _hermite_cross(i, j):
    """<He_i, He_j> in L^2(R, dx) = 2^i i! sqrt(pi) if i==j else 0."""
    if i == j:
        return float((2.0 ** i) * math.factorial(i) * math.sqrt(math.pi))
    return 0.0


def _build_basis(max_n=12):
    """Return list of (coeffs, norm2) for He_0..He_max_n."""
    basis = []
    for n in range(max_n + 1):
        basis.append((_hermite_coeffs(n), _hermite_norm(n)))
    return basis


def _quad_form(coeffs_list, basis, weight):
    """Compute sum_k weight[k] * (sum_i c_i He_i(x_k))^2 (dx)  by expanding in the basis.
    weight: dict n -> scalar (the kernel weight for degree n)."""
    # Expand g = sum_i c_i He_i.  g^2 = sum_{i,j} c_i c_j He_i He_j.
    # <He_i He_j, w_n He_n> can be computed via the product expansion.
    # He_i He_j = sum_k C_{i,j,k} He_k where C_{i,j,k} = binom(i,j) ... use explicit product.
    # We precompute the product coefficients.
    # Represent each He_i as polynomial coeff array (deg i).
    # Product polynomial = conv(He_i, He_j). Then express in He basis.
    # <poly, He_n w_n> = w_n * <poly, He_n>.
    # <poly, He_n> = sum_m poly_m * <x^m, He_n>.
    # We precompute M[m,n] = <x^m, He_n> for needed m,n.
    raise RuntimeError("unused")


def _mom_matrix(max_deg, max_n):
    """M[m,n] = <x^m, He_n> for 0<=m<=max_deg, 0<=n<=max_n.
    He_n = n! sum_{k} H_k x^k/k!  (H_k from _hermite_coeffs).
    <x^m, He_n> = n! sum_k H_k/k! <x^m, x^k> = n! sum_k H_k/k! * 2^{(m+k)/2} Gamma((m+k+1)/2)/Gamma(1/2) ... 
    Use <x^m, x^k> = 0 if m+k odd, else 2^{(m+k)/2} Gamma((m+k+1)/2)/sqrt(pi)."""
    M = np.zeros((max_deg + 1, max_n + 1))
    for m in range(max_deg + 1):
        for n in range(max_n + 1):
            H = _hermite_coeffs(n)
            fact = math.factorial(n)
            val = 0.0
            for k in range(n + 1):
                s = m + k
                if s % 2 == 1:
                    continue
                # <x^m, x^k> = 2^{s/2} Gamma((s+1)/2)/sqrt(pi)
                integ = (2.0 ** (s / 2.0)) * math.gamma((s + 1) / 2.0) / math.sqrt(math.pi)
                val += H[k] / math.factorial(k) * integ
            M[m, n] = fact * val
    return M


def _kernel_weight(n, sigma):
    """w_n = <He_n, e^{-sigma^2 x^2}>.  Only even n contribute.
    e^{-sigma^2 x^2} = sum_{j} a_j He_{2j} (even).  <He_n, e^{-...}> = 0 unless n even."""
    # Expand e^{-sigma^2 x^2} in He basis: a_{2j} = <e^{-sigma^2 x^2}, He_{2j}>/||He_{2j}||^2.
    # But we need <He_n, e^{-sigma^2 x^2}> = a_n ||He_n||^2 for n even.
    # Compute <e^{-sigma^2 x^2}, He_n> = n! sum_k H_k/k! <e^{-sigma^2 x^2}, x^k>.
    # <e^{-sigma^2 x^2}, x^k> = 0 if k odd, else (k-1)!! sigma^{-k} (1/sqrt(1+sigma^2))^... 
    # Use: <e^{-sigma^2 x^2}, x^{2j}> = (2j-1)!! / (2 sigma^2)^j * (1/sqrt(1+sigma^2)).
    if n % 2 == 1:
        return 0.0
    H = _hermite_coeffs(n)
    fact = math.factorial(n)
    val = 0.0
    for k in range(n + 1):
        if k % 2 == 1:
            continue
        j = k // 2
        # (2j-1)!! with (-1)!! = 1
        dd = 1.0
        for t in range(1, j + 1):
            dd *= (2 * t - 1)
        integ = dd / ((2.0 * sigma * sigma) ** j) / math.sqrt(1.0 + sigma * sigma)
        val += H[k] / math.factorial(k) * integ
    return fact * val


def _build_quadratic(max_n=12, sigma=1.0):
    """Return matrix Q (size (max_n+1)^2) such that
       E[g^2 e^{-sigma^2 x^2}] = c^T Q c  where g = sum c_i He_i.
    Q_{i,j} = (1/2)(<He_i He_j, w> + <He_j He_i, w>) = <He_i He_j, w> (symmetric).
    w_n = kernel_weight(n, sigma)."""
    size = max_n + 1
    # Precompute M[m,n] for product degrees up to 2*max_n.
    M = _mom_matrix(2 * max_n, max_n)
    # Product He_i He_j as polynomial.
    Q = np.zeros((size, size))
    polys = [_hermite_coeffs(n) for n in range(size)]
    for i in range(size):
        for j in range(size):
            prod = np.convolve(polys[i], polys[j])
            # prod[m] = coeff of x^m
            s = 0.0
            for m in range(len(prod)):
                s += prod[m] * M[m, :] * np.array([_kernel_weight(n, sigma) for n in range(size)])
            Q[i, j] = s
    return Q


def _A_from_c(c, Q, sigma):
    """A(g) = (sum x^2 g^2 e^{-sigma^2 x^2}) / (sum g^2 e^{-sigma^2 x^2}).
    We need a second quadratic for the numerator (x^2 g^2)."""
    raise RuntimeError("need x2 quadratic")


def construct():
    t0 = time.time()
    best = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])

    max_n = 12
    sigma = 1.0
    size = max_n + 1

    # Build the two quadratic forms: denominator D = c^T Q0 c, numerator N = c^T Q1 c.
    # Q0: weight w_n = <He_n, e^{-sigma^2 x^2}>
    # Q1: weight v_n = <He_n, x^2 e^{-sigma^2 x^2}>
    M0 = _mom_matrix(2 * max_n, max_n)
    w = np.array([_kernel_weight(n, sigma) for n in range(size)])
    # v_n = <He_n, x^2 e^{-sigma^2 x^2}>.  x^2 e^{-sigma^2 x^2} expand:
    # <x^2 e^{-sigma^2 x^2}, He_n> = n! sum_k H_k/k! <x^2 e^{-sigma^2 x^2}, x^k>
    # = n! sum_k H_k/k! <e^{-sigma^2 x^2}, x^{k+2}>
    v = np.zeros(size)
    for n in range(size):
        if n % 2 == 1:
            v[n] = 0.0
            continue
        H = _hermite_coeffs(n)
        fact = math.factorial(n)
        val = 0.0
        for k in range(n + 1):
            kk = k + 2
            if kk % 2 == 1:
                continue
            j = kk // 2
            dd = 1.0
            for t in range(1, j + 1):
                dd *= (2 * t - 1)
            integ = dd / ((2.0 * sigma * sigma) ** j) / math.sqrt(1.0 + sigma * sigma)
            val += H[k] / math.factorial(k) * integ
        v[n] = fact * val

    polys = [_hermite_coeffs(n) for n in range(size)]
    Q0 = np.zeros((size, size))
    Q1 = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            prod = np.convolve(polys[i], polys[j])
            s0 = 0.0
            s1 = 0.0
            for m in range(len(prod)):
                s0 += prod[m] * (M0[m, :] * w)
                s1 += prod[m] * (M0[m, :] * v)
            Q0[i, j] = s0
            Q1[i, j] = s1

    # Constraint: g(0)=0 => sum_i c_i He_i(0) = 0.  He_i(0)=0 for i odd; He_i(0) = (-1)^{i/2} (i-1)!! for i even.
    # We fix c_12 so that sum_{i even} c_i He_i(0) = 0  =>  c_12 = - (sum_{i even, i<12} c_i He_i(0)) / He_12(0).
    def he0(n):
        if n % 2 == 1:
            return 0.0
        j = n // 2
        dd = 1.0
        for t in range(1, j + 1):
            dd *= (2 * t - 1)
        return ((-1.0) ** j) * dd

    # Free coefficients: c_0, c_4, c_8.  c_12 determined.  Others zero.
    def full_c(a, b, c):
        cc = np.zeros(size)
        cc[0] = a
        cc[4] = b
        cc[8] = c
        s = a * he0(0) + b * he0(4) + c * he0(8)
        cc[12] = -s / he0(12)
        return cc

    def obj(a, b, c):
        cc = full_c(a, b, c)
        D = cc @ Q0 @ cc
        N = cc @ Q1 @ cc
        if D <= 0:
            return np.inf
        return N / D

    # Global-ish random search + Nelder-Mead local polish.
    from scipy.optimize import minimize

    def obj3(v):
        return obj(v[0], v[1], v[2])

    rng = np.random.default_rng(0)
    best_val = obj3(best)

    # Random restarts.
    for _ in range(400):
        if time.time() > _DEADLINE - _SAFETY:
            break
        v = rng.normal(0, 0.05, 3)
        val = obj3(v)
        if val < best_val:
            res = minimize(obj3, v, method="Nelder-Mead",
                           options={"maxiter": 2000, "xatol": 1e-10, "fatol": 1e-12})
            if res.fun < best_val:
                best_val = res.fun
                best = np.array(res.x)
        else:
            # quick local polish from this point
            res = minimize(obj3, v, method="Nelder-Mead",
                           options={"maxiter": 500, "xatol": 1e-9, "fatol": 1e-11})
            if res.fun < best_val:
                best_val = res.fun
                best = np.array(res.x)

    if time.time() > _DEADLINE - _SAFETY:
        return best

    # A few more targeted starts around the found optimum.
    for _ in range(50):
        if time.time() > _DEADLINE - _SAFETY:
            break
        v = best + rng.normal(0, 1e-4, 3)
        res = minimize(obj3, v, method="Nelder-Mead",
                       options={"maxiter": 1000, "xatol": 1e-11, "fatol": 1e-13})
        if res.fun < best_val:
            best_val = res.fun
            best = np.array(res.x)

    return best
# EVOLVE-BLOCK-END
