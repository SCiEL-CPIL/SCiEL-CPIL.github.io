import numpy as np
import time
from scipy.optimize import minimize
from scipy.special import roots_laguerre

def construct():
    """
    Return a list of 12 positive double-root locations for the Laguerre combination (m=12, n=1).
    The score is the upper bound on C4 (lower is better).
    
    C4 = ||P||_{L^2(e^{-x})} / ||P'||_{L^2(e^{-x})} where P(x) = prod_i (x - a_i)^2
    This is dimensionless (scale-invariant), so only relative placement matters.
    Optimal configuration: zeros of L_m(x) for m=12.
    """
    t0 = time.time()
    deadline = 600.0

    m = 12
    n = 1

    # High-accuracy Gauss-Laguerre quadrature
    x_q, w_q = np.polynomial.laguerre.laggauss(512)

    def compute_c4(roots):
        """
        C4 = ||P||_2 / ||P'||_2 with weight e^{-x} on [0,inf)
        P(x) = prod_i (x - a_i)^2
        P'(x)/P(x) = 2 * sum_i 1/(x - a_i)
        """
        roots = np.asarray(roots, dtype=float)
        if np.any(roots <= 0):
            return 1e10

        diff = x_q[:, None] - roots[None, :]
        # log(P^2 * e^{-x}) = 4*sum log|x-a_i| - x
        log_P2_w = 4.0 * np.log(np.abs(diff)) - x_q[:, None]
        log_P2_w = np.sum(log_P2_w, axis=1)
        log_P2_w = np.clip(log_P2_w, -700, 700)
        P2_w = np.exp(log_P2_w)
        norm_P2 = np.sum(w_q * P2_w)

        # P'/P = 2 * sum 1/(x-a_i)
        S = 2.0 * np.sum(1.0 / diff, axis=1)
        log_P2prime_w = log_P2_w + 2.0 * np.log(np.abs(S))
        log_P2prime_w = np.clip(log_P2prime_w, -700, 700)
        P2prime_w = np.exp(log_P2prime_w)
        norm_P2prime = np.sum(w_q * P2prime_w)

        if norm_P2prime <= 0 or norm_P2 <= 0:
            return 1e10

        return np.sqrt(norm_P2 / norm_P2prime)

    # Optimal: zeros of L_12(x)
    lag_zeros_12 = sorted(roots_laguerre(12)[0])
    c4_lag12 = compute_c4(lag_zeros_12)
    print(f"C4 with Laguerre-12 zeros: {c4_lag12:.6e}")

    best_roots = lag_zeros_12[:]
    best_val = c4_lag12

    def objective(theta):
        if time.time() - t0 > deadline - 5:
            return 1e10
        roots = np.sort(np.exp(theta))
        return compute_c4(roots)

    # Multi-start Nelder-Mead
    starts = [
        np.log(lag_zeros_12),
        np.log(np.linspace(lag_zeros_12[0], lag_zeros_12[-1], 12)),
        np.log(np.geomspace(lag_zeros_12[0], lag_zeros_12[-1], 12)),
    ]

    for i, th in enumerate(starts):
        if time.time() - t0 > deadline - 30:
            break
        try:
            res = minimize(objective, th, method='Nelder-Mead',
                          options={'maxiter': 15000, 'xatol': 1e-10,
                                   'fatol': 1e-12, 'adaptive': True})
            if res.fun < best_val:
                best_roots = np.sort(np.exp(res.x))
                best_val = res.fun
                print(f"NM start {i}: {best_val:.6e}")
        except Exception as e:
            print(f"NM {i} failed: {e}")

    # L-BFGS-B polish
    if time.time() - t0 < deadline - 30:
        try:
            res = minimize(objective, np.log(best_roots), method='L-BFGS-B',
                          options={'maxiter': 10000, 'gtol': 1e-14, 'ftol': 1e-14})
            if res.fun < best_val:
                best_roots = np.sort(np.exp(res.x))
                best_val = res.fun
                print(f"L-BFGS-B: {best_val:.6e}")
        except Exception as e:
            print(f"L-BFGS-B failed: {e}")

    # Try L_24 zero pairings
    if time.time() - t0 < deadline - 20:
        lag24 = sorted(roots_laguerre(24)[0])
        candidates = [
            lag24[:12],
            lag24[12:],
            [(lag24[2*i] + lag24[2*i+1])/2.0 for i in range(12)],
            [lag24[2*i] for i in range(12)],
            [lag24[2*i+1] for i in range(12)],
        ]
        for c in candidates:
            v = compute_c4(c)
            if v < best_val:
                best_roots = c[:]
                best_val = v
                print(f"Alt candidate: {best_val:.6e}")

    # Final high-precision polish
    if time.time() - t0 < deadline - 10:
        try:
            res = minimize(objective, np.log(best_roots), method='Nelder-Mead',
                          options={'maxiter': 30000, 'xatol': 1e-12,
                                   'fatol': 1e-14, 'adaptive': True})
            if res.fun < best_val:
                best_roots = np.sort(np.exp(res.x))
                best_val = res.fun
                print(f"Final polish: {best_val:.6e}")
        except Exception as e:
            print(f"Final failed: {e}")

    return [float(r) for r in best_roots]
