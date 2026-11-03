"""Riemannian conjugate-gradient + L-BFGS polish for log-energy on S^2."""
import numpy as np
import time

N = 150
DEADLINE = 115.0  # wall-clock safety margin (evaluator kills at ~150s)


def _energy(P):
    # P: (N,3) unit vectors. Returns log(1/d) = -log(d) sum over i<j, chordal.
    D = 1.0 - P @ P.T
    np.fill_diagonal(D, 1.0)
    D = np.maximum(D, 1e-16)
    d = np.sqrt(D)
    iu = np.triu_indices(N, 1)
    return float(-np.log(d[iu]).sum())


def _grad(P):
    # Gradient of log(1/d) = -log|xi-xj| on R^3.
    # d/dxi [-log|xi-xj|] = (xi - xj)/|xi-xj|^2
    D = 1.0 - P @ P.T
    np.fill_diagonal(D, 1.0)
    D = np.maximum(D, 1e-16)
    d = np.sqrt(D)
    # G[i,j] = (xi - xj)/d^2  ; note xi - xj = P[i] - P[j]
    G = (P[:, None, :] - P[None, :, :]) / (D[:, :, None] * np.sqrt(D)[:, :, None])
    np.fill_diagonal(G, 0.0)
    return G.sum(axis=1)


def _project_tangent(P, V):
    # Project V onto tangent space at P: V - (V.P) P
    c = V * P
    c = c.sum(axis=1, keepdims=True)
    return V - c * P


def _exp_map(P, V, step):
    # Riemannian exponential map on S^2 with tangent vector step*V
    V = step * V
    n = np.linalg.norm(V, axis=1)
    # Handle n ~ 0
    out = np.empty_like(P)
    small = n < 1e-12
    if np.any(small):
        out[small] = P[small] + V[small]
        out[small] /= np.linalg.norm(out[small], axis=1, keepdims=True)
    if np.any(~small):
        Vs = V[~small]
        ns = n[~small]
        # Rodrigues rotation
        ax = Vs / ns[:, None]
        # cos(n), sin(n)
        cn = np.cos(ns)
        sn = np.sin(ns)
        cross = np.cross(P[~small], Vs)
        out[~small] = cn[:, None] * P[~small] + sn[:, None] * ax + (1.0 - cn)[:, None] * cross
    # re-normalize for numerical safety
    out /= np.linalg.norm(out, axis=1, keepdims=True)
    return out


def _line_search(P, f0, g, d, step0):
    # Armijo backtracking on sphere for direction d (tangent), with step.
    # f = energy, want decrease.
    c1 = 1e-4
    rho = 0.5
    step = step0
    Pn = _exp_map(P, d, step)
    fn = _energy(Pn)
    # gradient inner product for descent check: grad(f) . d
    # We need directional derivative = <grad, d> on sphere = <grad, d> (d tangent)
    dir_deriv = float((g * d).sum())
    k = 0
    while fn > f0 + c1 * step * dir_deriv and k < 30:
        step *= rho
        Pn = _exp_map(P, d, step)
        fn = _energy(Pn)
        k += 1
    return Pn, fn, step


def construct():
    t_start = time.time()

    # Initial: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    P = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    P = P / np.linalg.norm(P, axis=1, keepdims=True)

    f = _energy(P)
    best_P = P.copy()
    best_f = f

    def time_ok():
        return (time.time() - t_start) < DEADLINE

    # Riemannian conjugate gradient
    g = _grad(P)
    d = -_project_tangent(P, g)
    step = 0.05
    prev_g = g
    prev_d = d
    it = 0
    max_iter = 400

    while time_ok() and it < max_iter:
        Pn, fn, step_used = _line_search(P, f, g, d, step)
        step = max(step_used * 1.5, 1e-6)
        if not np.isfinite(fn):
            break
        gn = _grad(Pn)
        # Check improvement
        if fn < f:
            # Update best
            best_f = fn
            best_P = Pn.copy()
        # Polak-Ribiere beta
        num = float((gn * (gn - prev_g)).sum())
        den = float((prev_g * prev_g).sum())
        beta = 0.0 if den < 1e-16 else max(0.0, num / den)
        d = -_project_tangent(Pn, gn) + beta * _project_tangent(Pn, prev_d)
        P = Pn
        f = fn
        prev_g = gn
        prev_d = d
        it += 1
        # Early stop if gradient tiny
        if np.linalg.norm(gn) < 1e-10:
            break

    # Final polish with L-BFGS on the sphere via projection reparam
    # Use scipy L-BFGS-B on the 3N vector with a sphere-constraint via
    # re-normalization after each step is not directly supported; instead
    # do a projected gradient / L-BFGS with a penalty is messy.
    # Simpler: do a few more CG iterations with smaller steps for polish.
    if time_ok():
        g = _grad(best_P)
        d = -_project_tangent(best_P, g)
        step = 0.01
        prev_g = g
        prev_d = d
        f = best_f
        P = best_P
        for it2 in range(200):
            if not time_ok():
                break
            Pn, fn, step_used = _line_search(P, f, g, d, step)
            step = max(step_used * 1.5, 1e-7)
            if not np.isfinite(fn):
                break
            gn = _grad(Pn)
            if fn < f:
                f = fn
                best_f = fn
                best_P = Pn.copy()
            num = float((gn * (gn - prev_g)).sum())
            den = float((prev_g * prev_g).sum())
            beta = 0.0 if den < 1e-16 else max(0.0, num / den)
            d = -_project_tangent(Pn, gn) + beta * _project_tangent(Pn, prev_d)
            P = Pn
            prev_g = gn
            prev_d = d
            if np.linalg.norm(gn) < 1e-12:
                break

    # Final re-normalize and validity check
    out = best_P / np.linalg.norm(best_P, axis=1, keepdims=True)
    if not np.all(np.isfinite(out)):
        # fallback to initial
        out = P
    return out
