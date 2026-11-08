"""Bootstrap DIRECT solution: derive the Hermite-Gaussian coefficients from the
Fourier-invariant C4 uncertainty-product structure, with a hard wall-clock guard.
Primary method: analytic generalized-eigenvalue minimizer of the quadratic-form
ratio (variance / norm) under the g(0)=0 constraint; fallbacks: grid + Nelder-Mead;
final: high-precision polish."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from scipy.special import eval_hermite

# Wall-clock deadline guard (evaluator kills at 210 s; keep a safety margin).
T_START = time.time()
DEADLINE = 210.0 - 15.0  # 195 s hard stop


def _now():
    return time.time()


def _time_left():
    return DEADLINE - _now()


# Physically normalised probabilists' Hermite polynomials.
# The Fourier-invariant (self-dual up to sign) Gaussian-Hermite eigenfunctions use the
# physicists' form; here we work with a fixed basis so only the relative weights matter
# for the uncertainty product (a scale-invariant ratio). We use the standard physicists'
# H_n with the 2^{n/2} sqrt(n! pi) normalisation factor folded into the basis.
def _H(n, x):
    """Physicists' Hermite polynomial H_n(x)."""
    return eval_hermite(n, x)


def _make_basis():
    """Return basis vectors and precomputed moment matrices for fast uncertainty product evaluation.
    
    Precomputes:
    - N[i,j] = <b_i, b_j>
    - M[i,j] = <x * b_i, b_j>  
    - X2[i,j] = <x^2 * b_i, b_j>
    """
    xs = np.linspace(-40.0, 40.0, 40001)
    dx = xs[1] - xs[0]
    w = np.exp(-0.5 * xs * xs)
    degrees = (0, 4, 8, 12)
    basis = {}
    for n in degrees:
        h = _H(n, xs)
        f = h * w
        # normalise to unit L2 norm
        f = f / np.sqrt(np.sum(f * f) * dx + 1e-300)
        basis[n] = f
    
    # Precompute moment matrices
    N = np.zeros((4, 4))
    M = np.zeros((4, 4))
    X2 = np.zeros((4, 4))
    for i, ni in enumerate(degrees):
        bi = basis[ni]
        for j, nj in enumerate(degrees):
            bj = basis[nj]
            N[i, j] = np.sum(bi * bj) * dx
            M[i, j] = np.sum(xs * bi * bj) * dx
            X2[i, j] = np.sum(xs * xs * bi * bj) * dx
    
    return xs, dx, w, basis, N, M, X2, degrees


XS, DX, W, BASIS, N_MAT, M_MAT, X2_MAT, DEGREES = _make_basis()


def _coeff12(c0, c4, c8):
    """Derive the H12 weight so that the combination has P(0)=0 (g(0)=0).
    g(0) = c0*H0(0) + c4*H4(0) + c8*H8(0) + c12*H12(0) = 0
    => c12 = -(c0*H0(0) + c4*H4(0) + c8*H8(0)) / H12(0).
    We work in the unnormalised polynomial values at 0."""
    h0 = _H(0, 0.0)
    h4 = _H(4, 0.0)
    h8 = _H(8, 0.0)
    h12 = _H(12, 0.0)
    if abs(h12) < 1e-15:
        return 0.0
    return -(c0 * h0 + c4 * h4 + c8 * h8) / h12


def _uncertainty_product(c0, c4, c8):
    """Compute the C4 upper bound: A(f)*A(Fourier(f)) for the constructed f.
    
    Uses precomputed moment matrices for O(1) evaluation per coefficient set.
    """
    c12 = _coeff12(c0, c4, c8)
    # Coefficients in order [H0, H4, H8, H12]
    c = np.array([c0, c4, c8, c12])
    
    # norm_sq = c^T N c
    norm_sq = float(c @ N_MAT @ c)
    if norm_sq < 1e-30:
        return float('inf')
    
    # E[x] = <x*f, f> / norm_sq = c^T M c / norm_sq
    mean_val = float(c @ M_MAT @ c)
    mean_x = mean_val / norm_sq
    
    # E[x^2] = <x^2*f, f> / norm_sq = c^T X2 c / norm_sq
    ex2_val = float(c @ X2_MAT @ c)
    ex2 = ex2_val / norm_sq
    
    # var = E[x^2] - (E[x])^2
    var_x = ex2 - mean_x * mean_x
    var_x = max(var_x, 1e-15)
    return var_x


def _analytic_minimizer():
    """Analytic generalized-eigenvalue minimizer of the quadratic-form ratio
    (second-moment / zeroth-moment) under the g(0)=0 constraint.
    
    The C4 bound is scale-invariant and equals the variance of the constructed
    f (since the Fourier transform of a Gaussian-Hermite eigenfunction is itself
    up to a sign, the uncertainty product reduces to a function of the x-space
    variance). Minimizing var = (c^T X2 c - (c^T M c)^2 / (c^T N c)) / (c^T N c)
    over c = [c0, c4, c8, c12] with c12 = -(c0*h0 + c4*h4 + c8*h8)/h12.
    
    We reduce to 3 free variables and solve the resulting (nonlinear) minimization
    via a generalized eigenvalue on the constraint subspace, then polish.
    """
    h0 = float(_H(0, 0.0))
    h4 = float(_H(4, 0.0))
    h8 = float(_H(8, 0.0))
    h12 = float(_H(12, 0.0))
    if abs(h12) < 1e-15:
        return None, float('inf')
    
    # c12 = -(a0*c0 + a4*c4 + a8*c8) / h12  where a_i = h_i(0)
    # Define the linear map: c = B @ u where u = [c0, c4, c8]
    # c_full = [c0, c4, c8, c12]
    # Let's build matrices so that for any u in R^3, c_full = [u0, u1, u2, -(h0*u0+h4*u1+h8*u2)/h12]
    # Then norm_sq(u) = u^T A u, ex2(u) = u^T B u, mean(u) = u^T C u
    # where A = B_full^T N B_full, etc.
    # B_full is 4x3:
    B_full = np.zeros((4, 3))
    B_full[0, 0] = 1.0
    B_full[1, 1] = 1.0
    B_full[2, 2] = 1.0
    B_full[3, 0] = -h0 / h12
    B_full[3, 1] = -h4 / h12
    B_full[3, 2] = -h8 / h12
    
    A = B_full.T @ N_MAT @ B_full   # 3x3, norm quadratic form
    Bm = B_full.T @ X2_MAT @ B_full  # 3x3, x^2 quadratic form
    Cm = B_full.T @ M_MAT @ B_full   # 3x3, x quadratic form
    
    # var(u) = (u^T Bm u) / (u^T A u) - (u^T Cm u)^2 / (u^T A u)^2
    # This is a ratio of quadratics with a squared term in the mean.
    # For the minimum, we can use a generalized eigenvalue approach on an augmented space,
    # or simply do a high-precision Nelder-Mead on this 3D problem starting from
    # multiple initial points including the eigenvector of (Bm - lambda A).
    
    # First, get a good initial guess via generalized eigenvalue of (Bm, A):
    # This minimizes u^T Bm u / u^T A u, ignoring the mean-subtraction term.
    # It gives a direction close to the true minimum.
    try:
        # Solve Bm v = lambda A v
        # Use scipy.linalg.eigh for symmetric generalized eigenvalue
        from scipy.linalg import eigh
        # Ensure A is positive definite (it should be, as N is PSD and basis is independent)
        eigvals, eigvecs = eigh(Bm, A)
        # Take the eigenvector corresponding to the smallest eigenvalue
        idx = np.argmin(eigvals)
        v0 = eigvecs[:, idx]
        # Normalize to have c8 component = 1 for scale invariance (or just use as-is)
        # The objective is scale-invariant, so any scalar multiple works.
        # Use v0 directly.
        c0_init, c4_init, c8_init = v0[0], v0[1], v0[2]
    except Exception:
        c0_init, c4_init, c8_init = -1.13, 0.04, 0.000309
    
    # Polish with high-precision Nelder-Mead
    best, best_val = _local_refine(c0_init, c4_init, c8_init, iters=5000)
    return best, best_val


def _score(c0, c4, c8):
    """The upper bound C4 value (lower is better)."""
    try:
        return _uncertainty_product(c0, c4, c8)
    except Exception:
        return float('inf')


def _coarse_grid():
    """Direct structured sweep over the three free coefficients."""
    best = None
    best_val = float('inf')
    # Symmetry: coefficients can be scaled; the product is scale-invariant, so we
    # fix c8 = 1 (unless it would make the combination degenerate) and search c0/c8,
    # c4/c8 ratios. This reduces 3D search to 2D.
    # Use a finer grid since evaluation is now O(1)
    ratios_c0 = np.linspace(-50.0, 50.0, 201)
    ratios_c4 = np.linspace(-50.0, 50.0, 201)
    for r0 in ratios_c0:
        if _time_left() < 15.0:
            break
        for r4 in ratios_c4:
            v = _score(r0, r4, 1.0)
            if v < best_val:
                best_val = v
                best = (r0, r4, 1.0)
    return best, best_val


def _local_refine(c0, c4, c8, iters=2000, tol=1e-12):
    """Deterministic Nelder-Mead refinement around a start point.
    
    Uses vectorized evaluation for speed.
    """
    best = (c0, c4, c8)
    best_val = _score(c0, c4, c8)
    # Simple Nelder-Mead with many iterations (evaluation is now O(1))
    # Start with a reasonable simplex
    step = 1.0
    pts = [
        (c0, c4, c8),
        (c0 + step, c4, c8),
        (c0, c4 + step, c8),
        (c0, c4, c8 + step),
    ]
    vals = [_score(*p) for p in pts]
    for _ in range(iters):
        if _time_left() < 15.0:
            break
        order = np.argsort(vals)
        pts = [pts[i] for i in order]
        vals = [vals[i] for i in order]
        
        # Check for convergence
        if abs(vals[0] - vals[-1]) < tol:
            break
        
        centroid = tuple(np.mean(pts[:-1], axis=0))
        alpha = 1.0
        reflection = tuple(centroid[i] + alpha * (centroid[i] - pts[-1][i]) for i in range(3))
        vr = _score(*reflection)
        if vr < vals[0]:
            # expansion
            gamma = 2.0
            expansion = tuple(centroid[i] + gamma * (reflection[i] - pts[-1][i]) for i in range(3))
            ve = _score(*expansion)
            if ve < vr:
                pts[-1], vals[-1] = expansion, ve
            else:
                pts[-1], vals[-1] = reflection, vr
        elif vr < vals[-2]:
            pts[-1], vals[-1] = reflection, vr
        else:
            # contraction
            rho = 0.5
            contraction = tuple(centroid[i] + rho * (pts[-1][i] - centroid[i]) for i in range(3))
            vc = _score(*contraction)
            if vc < vals[-1]:
                pts[-1], vals[-1] = contraction, vc
            else:
                # shrink toward best
                for i in range(1, len(pts)):
                    pts[i] = tuple(pts[0][j] + 0.5 * (pts[i][j] - pts[0][j]) for j in range(3))
                    vals[i] = _score(*pts[i])
        cur_best = min(range(len(pts)), key=lambda i: vals[i])
        if vals[cur_best] < best_val:
            best_val = vals[cur_best]
            best = pts[cur_best]
    return best, best_val


def _final_polish(c0, c4, c8, iters=10000):
    """High-precision final polish with tighter tolerance and smaller step."""
    best, best_val = _local_refine(c0, c4, c8, iters=iters, tol=1e-14)
    # Additional very fine local search
    if _time_left() > 20.0:
        # Try a grid of small perturbations around the best
        eps = 1e-4
        for dc0 in [-eps, 0, eps]:
            for dc4 in [-eps, 0, eps]:
                for dc8 in [-eps, 0, eps]:
                    v = _score(c0 + dc0, c4 + dc4, c8 + dc8)
                    if v < best_val:
                        best_val = v
                        best = (c0 + dc0, c4 + dc4, c8 + dc8)
    return best, best_val


def construct():
    # Seed prior-art coefficients as a fallback / initial guess.
    seed = np.array([-113 / 100, 1 / 25, 1 / 3240])

    best = tuple(seed.tolist())
    best_val = _score(*best)

    # Phase 0: Analytic generalized-eigenvalue minimizer (primary method).
    if _time_left() > 60.0:
        eig_best, eig_val = _analytic_minimizer()
        if eig_best is not None and eig_val < best_val:
            best, best_val = eig_best, eig_val

    # Phase 1: coarse structured sweep (2D via scale-invariance, c8 fixed to 1).
    if _time_left() > 30.0:
        grid_best, grid_val = _coarse_grid()
        if grid_best is not None and grid_val < best_val:
            best, best_val = grid_best, grid_val
        # Also try the seed-scaled point and a few hand-derived structured points.
        for cand in [
            (-113.0, 1.0, 3240.0 / 100.0),  # seed ratios scaled to c8=1
            (-20.0, 4.0, 1.0),
            (-50.0, 10.0, 1.0),
            (0.0, 0.0, 1.0),
        ]:
            v = _score(*cand)
            if v < best_val:
                best, best_val = cand, v

    # Phase 2: local refinement from the best found (and from the seed).
    if _time_left() > 20.0:
        ref_best, ref_val = _local_refine(*best, iters=3000)
        if ref_val < best_val:
            best, best_val = ref_best, ref_val
    if _time_left() > 20.0:
        ref_best, ref_val = _local_refine(*seed.tolist(), iters=2000)
        if ref_val < best_val:
            best, best_val = ref_best, ref_val

    # Phase 3: High-precision final polish.
    if _time_left() > 20.0:
        pol_best, pol_val = _final_polish(*best)
        if pol_val < best_val:
            best, best_val = pol_best, pol_val

    # Return the three free coefficients (degrees 0, 4, 8).
    return np.array([best[0], best[1], best[2]])
# EVOLVE-BLOCK-END
