"""Optimize Hermite-Gaussian coefficients to minimize the C4 upper bound.

The Fourier transform of H_n(x)*exp(-x^2/2) is (-i)^n H_n(x)*exp(-x^2/2),
so the L2 norm of g is invariant under the Fourier transform. Hence
A(f)*A(F(f)) = (int g^2 dx)^2, and minimizing the product is equivalent to
minimizing the quadratic form int g^2 dx over the three free coefficients
(a0, a4, a8), with the a12 coefficient fixed to enforce g(0)=0.
"""
import time
import numpy as np
from scipy.special import eval_hermite, roots_hermite
from scipy.optimize import minimize


def _gram_matrix(degrees, n_quad=600):
    """Return the Gram matrix G_ij = int H_i(x) H_j(x) exp(-x^2) dx."""
    # Gauss-Hermite quadrature for weight exp(-x^2).
    x, w = roots_hermite(n_quad)
    vals = np.array([eval_hermite(d, x) for d in degrees])  # (len(degrees), n_quad)
    # int f(x) g(x) exp(-x^2) dx = sum_k w_k f(x_k) g(x_k)
    # Equivalent to (vals * w) @ vals.T
    G = (vals * w) @ vals.T
    return G


def construct():
    deadline = time.time() + 170.0  # safety margin vs 180s limit
    best = np.array([-113.0 / 100, 1.0 / 25, 1.0 / 3240])  # seed fallback

    degrees = [0, 4, 8, 12]
    G = _gram_matrix(degrees, n_quad=600)

    # H12(0) and H4(0), H8(0) for the g(0)=0 constraint.
    H0_0 = eval_hermite(0, 0.0)
    H4_0 = eval_hermite(4, 0.0)
    H8_0 = eval_hermite(8, 0.0)
    H12_0 = eval_hermite(12, 0.0)

    # g(0) = (a0*H0_0 + a4*H4_0 + a8*H8_0 + a12*H12_0) * exp(0)
    # Set g(0)=0 => a12 = -(a0*H0_0 + a4*H4_0 + a8*H8_0)/H12_0
    #
    # Let v = [a0, a4, a8, a12]. Objective = v^T G v.
    # Substitute a12 = c - M·u where u=[a0,a4,a8], M = [H0_0,H4_0,H8_0]/H12_0, c=0.
    # v = [u; c - M u] = P u + q, with q = [0,0,0,c] = 0 (c=0).
    # So v = P u, P = [[I3],[ -M ]].
    H = np.array([H0_0, H4_0, H8_0])
    M = H / H12_0
    P = np.vstack([np.eye(3), -M.reshape(1, 3)])
    # Objective = u^T (P^T G P) u
    A = P.T @ G @ P

    def obj(u):
        return float(u @ A @ u)

    # Exact quadratic minimization: A is symmetric positive (semi)definite.
    # Solve A u = 0; if singular, use least-norm / pseudo-inverse path.
    try:
        eigvals, eigvecs = np.linalg.eigh(A)
        # Minimize over the null / lowest-eigenvalue direction.
        # Since objective is homogeneous (no linear term), the minimum over
        # nonzero u is 0 if A is singular; otherwise the infimum is 0 at u=0,
        # but u=0 gives g=0 (trivial). We need a nontrivial minimizer.
        # The true problem: minimize (int g^2) subject to a normalization.
        # The ratio A(f)*A(F) for a normalized f is scale-invariant, so we fix
        # a normalization, e.g. ||g||_2 = 1, and minimize is then just 1...
        # Instead, the C4 bound depends on the shape; we minimize the quadratic
        # form under a fixed norm constraint to get the optimal shape.
        # Minimize u^T A u subject to u^T u = 1 => smallest eigenvector.
        idx = int(np.argmin(eigvals))
        u_opt = eigvecs[:, idx]
        # Scale so that the resulting g has a sensible magnitude (norm 1).
        v_opt = P @ u_opt
        # normalize v so that int g^2 = 1
        norm2 = float(v_opt @ G @ v_opt)
        if norm2 > 0:
            scale = 1.0 / np.sqrt(norm2)
            v_opt = v_opt * scale
        cand = v_opt[:3].copy()
        # verify g(0) ~ 0
        a12 = float(v_opt[3])
        g0 = (cand[0] * H0_0 + cand[1] * H4_0 + cand[2] * H8_0 + a12 * H12_0)
        if abs(g0) < 1e-6 and np.all(np.isfinite(cand)):
            best = cand
    except Exception:
        pass

    # Refine with a smooth local polish on the three coefficients.
    def objective(u):
        if time.time() > deadline:
            return 1e300
        v = np.array([u[0], u[1], u[2], -(u[0] * H0_0 + u[1] * H4_0 + u[2] * H8_0) / H12_0])
        return float(v @ G @ v)

    try:
        x0 = best.copy()
        res = minimize(objective, x0, method="L-BFGS-B",
                       options={"maxiter": 200, "gtol": 1e-12, "ftol": 1e-14})
        if res.success and np.all(np.isfinite(res.x)):
            a12 = -(res.x[0] * H0_0 + res.x[1] * H4_0 + res.x[2] * H8_0) / H12_0
            g0 = (res.x[0] * H0_0 + res.x[1] * H4_0 + res.x[2] * H8_0 + a12 * H12_0)
            if abs(g0) < 1e-8:
                best = res.x.copy()
    except Exception:
        pass

    if time.time() > deadline:
        return np.array([-113.0 / 100, 1.0 / 25, 1.0 / 3240])
    return best
