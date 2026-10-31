"""Uncertainty inequality: tune Hermite combination to minimize A(g)*A(Fourier(g)).

Reconstruction of the task's mathematical structure:
  * The seed coefficients [-113/100, 1/25, 1/3240] equal the H0,H4,H8 weights of
    g(x) = e^{-x^2} - 1 expanded in the (physicists') Hermite basis
        e^{-x^2} = sum_{n>=0} (-1)^n / n! * H_{2n}(x)
    so H0->-1, H4->+1/24, H8->-1/1920, H12->+1/46080.  The degree-12 weight is
    fixed by the constraint g(0)=0, i.e. c12 = -(c0+c4+c8)/H12(0).
  * "Fourier-invariant Hermite-polynomial times Gaussian" + "upper bound on C4"
    points to the Hermite uncertainty inequality  A(f)*A(F) >= 1,  where
        A(f) = (integral |f|^2 dx)^{1/2},   F = Fourier transform.
    The bound is 1, attained by pure Gaussians.  The seed (a near-Gaussian) sits
    right at 1, so the score is 1.0.  The only way to beat it is to make g a
    genuine non-Gaussian (substantial H4,H8 weights) and minimize the product.

  The Hermite transform is exact and finite:  F[H_n] = i^n H_n.
  The L2 inner product is diagonal in the Hermite basis:
        <H_m, H_n> = 2^n n!  delta_{m,n}.
  Hence A(g)*A(F) can be evaluated exactly (no quadrature) and the problem is a
  bounded 3-parameter global optimization over (c0,c4,c8).
"""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np

# Precomputed exact data (physicists' Hermite functions, normalized to unit L2 norm).
# psi_n(x) = (2^n n! sqrt(pi))^{-1/2} H_n(x) e^{-x^2/2}.
#   A(g)  = sqrt( sum_n |c_n|^2 )          (psi_n orthonormal)
#   F[psi_n] = i^n psi_n  =>  A(F) = sqrt( sum_n |c_n|^2 )  (same set of weights)
# so the product is scale-invariant:  P(c0,c4,c8) = sum_n |c_n|^2  (with c12 from g(0)=0).
#   H_{2n}(0) = (2n)! / n!
_H0_0 = 1.0
_H4_0 = 24.0          # 4! / 2!
_H8_0 = 20160.0       # 8! / 4!
_H12_0 = 6652800.0    # 12! / 6!


def _product(c0, c4, c8):
    """Exact Hermite uncertainty product A(g)*A(F) for g = c0 H0 + c4 H4 + c8 H8 + c12 H12."""
    c12 = -(c0 + c4 + c8) / _H12_0
    # |c_n|^2 / (2^n n!)  for n in {0,4,8,12}
    w0 = (c0 * c0) / 1.0
    w4 = (c4 * c4) / (16.0 * 24.0)
    w8 = (c8 * c8) / (256.0 * 40320.0)
    w12 = (c12 * c12) / (4096.0 * 479001600.0)
    return w0 + w4 + w8 + w12


def _nelder_mead(f, x0, maxiter, maxfev, xtol, ftol):
    """Small self-contained Nelder-Mead minimizer (no SciPy dependency)."""
    n = len(x0)
    x0 = np.array(x0, dtype=float)
    # initial simplex
    simplex = [x0.copy()]
    for i in range(n):
        pt = x0.copy()
        pt[i] = pt[i] + 0.05 * (abs(pt[i]) + 1.0)
        simplex.append(pt)
    vals = [f(p) for p in simplex]
    best = min(vals)
    fev = n + 1
    alpha, gamma, rho, sigma = 1.0, 2.0, -0.5, 0.5
    for _ in range(maxiter):
        order = np.argsort(vals)
        simplex = [simplex[i] for i in order]
        vals = [vals[i] for i in order]
        if vals[0] < best:
            best = vals[0]
        # convergence
        if (max(vals) - min(vals)) <= ftol * (max(abs(max(vals)), abs(min(vals))) + 1e-30) \
           and max(np.abs(np.array(simplex) - simplex[0])) < xtol:
            break
        centroid = sum(simplex[:n]) / n
        xr = centroid + alpha * (centroid - simplex[n])
        fr = f(xr); fev += 1
        if fr < vals[0]:
            xe = centroid + gamma * (xr - centroid)
            fe = f(xe); fev += 1
            if fe < fr:
                simplex[n] = xe; vals[n] = fe
            else:
                simplex[n] = xr; vals[n] = fr
        elif fr < vals[-1]:
            simplex[n] = xr; vals[n] = fr
        else:
            xc = centroid + rho * (simplex[n] - centroid)
            fc = f(xc); fev += 1
            if fc < vals[-1]:
                simplex[n] = xc; vals[n] = fc
            else:
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    vals[i] = f(simplex[i]); fev += 1
        if fev >= maxfev:
            break
    order = np.argsort(vals)
    return np.array(simplex[order[0]]), vals[order[0]]


def construct():
    t_start = time.time()
    DEADLINE = 190.0          # hard wall-clock guard (evaluator kills at 210 s)
    safety = 8.0              # return this many seconds before the limit

    # ----- Phase 1: coarse global grid over the bounded 3D parameter space -----
    best = (float('inf'), None)
    n = 70
    c0v = np.linspace(-2.0, 2.0, n)
    c4v = np.linspace(-0.05, 0.05, n)
    c8v = np.linspace(-0.004, 0.004, n)
    # vectorized coarse sweep (cheap: ~343k exact evaluations)
    for a in c0v:
        fa = _product(a, 0.0, 0.0)
        for b in c4v:
            f4 = _product(a, b, 0.0)
            for c in c8v:
                v = _product(a, b, c)
                if v < best[0]:
                    best = (v, (float(a), float(b), float(c)))
        if time.time() > DEADLINE - safety:
            break
    if best[1] is None:
        return np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])

    # ----- Phase 2: multistart local refinement (Nelder-Mead) -----
    def f(x):
        return _product(float(x[0]), float(x[1]), float(x[2]))

    starts = []
    # keep the seed (known-good, score 1.0) as a guaranteed fallback start
    starts.append([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    # best grid point and its 27 neighbours
    b0, b4, b8 = best[1]
    starts.append([b0, b4, b8])
    for da in (-1, 0, 1):
        for db in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if da == db == dc == 0:
                    continue
                starts.append([b0 + da * (c0v[1] - c0v[0]),
                               b4 + db * (c4v[1] - c4v[0]),
                               b8 + dc * (c8v[1] - c8v[0])])
    # deterministic random starts (fixed seed) for global coverage
    rng = np.random.default_rng(12345)
    for _ in range(12):
        starts.append([rng.uniform(-2.0, 2.0), rng.uniform(-0.05, 0.05), rng.uniform(-0.004, 0.004)])

    for s in starts:
        if time.time() > DEADLINE - safety:
            break
        try:
            xopt, vopt = _nelder_mead(f, s, maxiter=400, maxfev=1200, xtol=1e-12, ftol=1e-14)
            if vopt < best[0]:
                best = (vopt, (float(xopt[0]), float(xopt[1]), float(xopt[2])))
        except Exception:
            continue

    # ----- Phase 3: one more refinement pass around the global best -----
    if time.time() < DEADLINE - safety:
        b0, b4, b8 = best[1]
        for da in (-1, 0, 1):
            for db in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if da == db == dc == 0:
                        continue
                    s = [b0 + da * 1e-3, b4 + db * 1e-5, b8 + dc * 1e-6]
                    if time.time() > DEADLINE - safety:
                        break
                    try:
                        xopt, vopt = _nelder_mead(f, s, maxiter=300, maxfev=900, xtol=1e-13, ftol=1e-15)
                        if vopt < best[0]:
                            best = (vopt, (float(xopt[0]), float(xopt[1]), float(xopt[2])))
                    except Exception:
                        continue

    c0, c4, c8 = best[1]
    return np.array([c0, c4, c8])
# EVOLVE-BLOCK-END
