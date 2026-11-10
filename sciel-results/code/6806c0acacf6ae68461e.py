# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _union_length_1d_smooth(x, beta=50.0):
    """
    Compute a smooth approximation of the integral of union length of intervals
    [x_i + c*i/N, x_i + 1/N + c*i/N] for i=0..N-1, c in [0,1].
    
    Uses the fact that union length = max(end) - min(start) + overlaps correction,
    but for smooth optimization we use a different approach:
    
    Actually, the union of intervals on a line has length = (max right endpoint) - (min left endpoint)
    minus the gaps. But computing this smoothly is tricky.
    
    Better approach: The union length can be computed as the measure of the set
    {t : exists i, t in [x_i + c*i/N, x_i + 1/N + c*i/N]}
    
    For a fixed c, this is the length of the union of N intervals. We can compute
    this exactly by sorting endpoints, but for smoothness we use a soft version.
    
    Alternative: Use the fact that for intervals, union length = sum of lengths - sum of pairwise overlaps + ... 
    (inclusion-exclusion), but this is complex.
    
    Simplest smooth approach: approximate the indicator function and integrate.
    But that's expensive.
    
    Let me think differently. The union length at height c is:
    U(c) = length of union_{i=0}^{N-1} [x_i + c*i/N, x_i + 1/N + c*i/N]
    
    For N=8, we can compute this exactly for any c by sorting all endpoints.
    The function U(c) is piecewise linear in c (since all endpoints are linear in c).
    So the integral is a sum of integrals of linear pieces.
    
    But for gradient-based optimization, we need differentiability. The breakpoints
    in c occur when two endpoints cross, which is when x_i + c*i/N = x_j + c*j/N
    or similar. These are linear in c, so the breakpoints are fixed given x.
    
    Actually, let's just use a numerical quadrature with a smooth approximation.
    
    For the union of intervals, a smooth approximation:
    U(c) ≈ soft_union_length(x, c)
    
    One approach: The union length = max_i (x_i + c*i/N + 1/N) - min_i (x_i + c*i/N)
    minus the total gap. The gaps are hard to smooth.
    
    Alternative: Use the fact that for intervals sorted by start, the union length
    can be built up. But this is combinatorial.
    
    Let me try a different smooth surrogate: 
    U(c) = ∫ 1_{union}(t) dt ≈ ∫ [1 - ∏_i (1 - 1_{[x_i+c*i/N, x_i+1/N+c*i/N]}(t))] dt
    
    This is still hard to smooth.
    
    Practical approach: Use a fine grid in c, compute the exact union length at each
    grid point, and use a smooth interpolation. But this loses differentiability at
    the grid points.
    
    Better: Since U(c) is piecewise linear with breakpoints at values where two
    endpoints are equal, and these breakpoints depend linearly on the x_i, we can
    in principle compute the integral exactly as a function of x. But the piecewise
    structure changes as x changes.
    
    For optimization, let's use a moderate-beta smooth approximation of the
    interval indicator and integrate over c with Gauss-Legendre quadrature.
    """
    N = len(x)
    # Gauss-Legendre quadrature points in [0,1]
    n_quad = 20
    c_points, w_points = np.polynomial.legendre.leggauss(n_quad)
    c_points = 0.5 * (c_points + 1)  # map to [0,1]
    w_points = 0.5 * w_points
    
    total = 0.0
    for c, w in zip(c_points, w_points):
        # Compute union length of intervals [x_i + c*i/N, x_i + 1/N + c*i/N]
        starts = x + c * np.arange(N) / N
        ends = starts + 1.0 / N
        # Sort by start
        order = np.argsort(starts)
        s = starts[order]
        e = ends[order]
        # Compute union length exactly
        union_len = 0.0
        cur_start = s[0]
        cur_end = e[0]
        for k in range(1, N):
            if s[k] > cur_end:
                union_len += cur_end - cur_start
                cur_start = s[k]
                cur_end = e[k]
            else:
                cur_end = max(cur_end, e[k])
        union_len += cur_end - cur_start
        total += w * union_len
    
    return total

def _union_length_1d_smooth_grad(x, beta=50.0):
    """Compute union length integral and its gradient using finite differences."""
    N = len(x)
    eps = 1e-7
    f0 = _union_length_1d_smooth(x)
    grad = np.zeros(N)
    for i in range(N):
        xp = x.copy()
        xp[i] += eps
        xm = x.copy()
        xm[i] -= eps
        grad[i] = (_union_length_1d_smooth(xp) - _union_length_1d_smooth(xm)) / (2 * eps)
    return f0, grad

def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    deadline = time.time() + 600  # safety margin
    
    # Start from Keich construction
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
    
    # Optimize the 1D union length
    x0 = keich.copy()
    
    def objective(x):
        return _union_length_1d_smooth(x)
    
    def gradient(x):
        f, g = _union_length_1d_smooth_grad(x)
        return g
    
    # Try L-BFGS-B
    best_x = x0.copy()
    best_f = objective(x0)
    
    # Multiple restarts with different initializations
    rng = np.random.default_rng(42)
    n_restarts = 5
    
    for restart in range(n_restarts):
        if time.time() > deadline - 120:
            break
        
        if restart == 0:
            x_init = x0.copy()
        else:
            # Perturb the initial solution
            x_init = x0 + rng.normal(0, 0.1, N)
        
        try:
            res = minimize(
                objective,
                x_init,
                jac=gradient,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if res.fun < best_f:
                best_f = res.fun
                best_x = res.x.copy()
        except Exception:
            pass
    
    # Build the 2D positions from the optimized 1D positions
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    base_x = best_x[i_flat]
    base_y = best_x[j_flat]
    positions = np.stack([base_x, base_y], axis=1)
    
    return positions

# EVOLVE-BLOCK-END
