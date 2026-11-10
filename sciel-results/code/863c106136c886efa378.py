# EVOLVE-BLOCK-START
import time
import numpy as np

def kakeya_needle_2d_positions64() -> np.ndarray:
    """
    Two-dimensional Kakeya-needle area-minimization problem (n=64).
    Chooses base positions x_i to minimize the area of the union of 64 triangles.
    Uses L-BFGS-B gradient optimization from multiple diverse starts.
    """
    n = 64
    w = 1.0 / n
    a = (np.arange(n) + 1) / n  # a_i = (i+1)/n

    # Quadrature heights
    M = 20000
    y = (np.arange(M) + 0.5) / M  # shape (M,)

    # Precompute coefficients for L_i = x_i + a_i * y
    # L_i(y) = x_i + a_i * y, so at each height y_m, L_i = x_i + a_i * y_m
    # Cross-section width is w = (1-y)/64

    def union_area(x):
        """Compute the union area of the 64 triangles given base positions x."""
        # x shape (n,), y shape (M,)
        # L_i(y_m) = x_i + a_i * y_m  -> shape (M, n)
        L = x[None, :] + a[None, :] * y[:, None]  # (M, n)
        # Sort L along axis 1
        L_sorted = np.sort(L, axis=1)  # (M, n)
        # Width at each height: w_m = (1 - y_m) / 64
        w_m = (1.0 - y) / n  # (M,)
        # Union length = w_m + sum over consecutive pairs of min(L_{j+1} - L_j, w_m)
        diffs = np.diff(L_sorted, axis=1)  # (M, n-1)
        # min(diffs, w_m)
        mins = np.minimum(diffs, w_m[:, None])  # (M, n-1)
        union_len = w_m + mins.sum(axis=1)  # (M,)
        return union_len.mean()

    def union_area_grad(x):
        """Compute union area and its gradient w.r.t. x."""
        # x shape (n,)
        L = x[None, :] + a[None, :] * y[:, None]  # (M, n)
        L_sorted = np.sort(L, axis=1)
        w_m = (1.0 - y) / n  # (M,)
        diffs = np.diff(L_sorted, axis=1)  # (M, n-1)
        mins = np.minimum(diffs, w_m[:, None])
        union_len = w_m + mins.sum(axis=1)
        area = union_len.mean()

        # Gradient: d(area)/d(x_i)
        # area = (1/M) sum_m union_len_m
        # union_len_m = w_m + sum_j min(L_{j+1,m} - L_{j,m}, w_m)
        # d/dx_i of L_{k,m} = a_i if the i-th triangle is at sorted position k
        # We need to track which original index maps to which sorted position
        # Use argsort to get the permutation
        perm = np.argsort(L, axis=1)  # (M, n) perm[m, k] = original index of k-th smallest
        # For each m, for each consecutive pair (k, k+1) in sorted order:
        #   contribution = min(L_{k+1} - L_k, w_m)
        #   d/dx_i: if i == perm[m,k+1], +1 * (1 if diff < w_m else 0)
        #           if i == perm[m,k],   -1 * (1 if diff < w_m else 0)
        #   where diff = L_{k+1} - L_k
        #
        # Gradient w.r.t. x: grad[m, i] = sum over k where i is involved
        # Let's build it efficiently:
        # active = (diffs < w_m[:, None])  # (M, n-1) boolean
        active = (diffs < w_m[:, None])  # (M, n-1)
        # For each m, k: if active[m,k], then:
        #   grad[m, perm[m, k+1]] += 1
        #   grad[m, perm[m, k]]   -= 1
        # We can use np.add.at
        grad = np.zeros_like(x)
        # perm shape (M, n), diffs shape (M, n-1), active shape (M, n-1)
        # For the +1 contribution: indices (m, perm[m, k+1]) for k=0..n-2
        # perm[:, 1:] gives (M, n-1)
        # For the -1 contribution: indices (m, perm[m, k]) for k=0..n-2
        # perm[:, :-1] gives (M, n-1)
        # We need to add +1 to grad[m, idx] where active[m, k]
        # Use flat indexing:
        m_idx = np.arange(M)[:, None]  # (M, 1) -> broadcast to (M, n-1)
        m_idx = np.broadcast_to(m_idx, (M, n-1))
        # +1 contributions
        plus_idx = perm[:, 1:]  # (M, n-1)
        # -1 contributions
        minus_idx = perm[:, :-1]  # (M, n-1)
        # Only where active
        active_flat = active.ravel()
        m_flat = m_idx.ravel()
        plus_flat = plus_idx.ravel()
        minus_flat = minus_idx.ravel()
        # np.add.at(grad, (m_flat, plus_flat), 1.0) -- but grad is 1D, we need per-m
        # Actually grad[m, i] accumulates, but we return sum over m divided by M
        # Let's build a 2D temp or use a different approach
        # Simpler: build the gradient as (M, n) then average
        grad_2d = np.zeros((M, n))
        np.add.at(grad_2d, (m_flat[active_flat], plus_flat[active_flat]), 1.0)
        np.add.at(grad_2d, (m_flat[active_flat], minus_flat[active_flat]), -1.0)
        grad = grad_2d.mean(axis=0)  # (n,)
        return area, grad

    def keich_init():
        """Classical Keich construction."""
        k = int(np.log2(n))
        j = np.arange(1, k + 1)
        positions = []
        for i in range(n):
            eps = np.array(list(bin(i)[2:]), dtype=int)
            eps = np.pad(eps, (k - len(eps), 0), "constant")
            positions.append(np.sum((1 - j) * eps * (2.0 ** -j) / k))
        return np.array(positions, dtype=np.float64)

    # Time budget
    t0 = time.time()
    time_limit = 600.0
    safety_margin = 30.0  # seconds

    # Start from Keich
    x_best = keich_init()
    area_best = union_area(x_best)

    # Try L-BFGS-B from Keich
    from scipy.optimize import minimize

    def objective(x):
        return union_area(x)

    def objective_grad(x):
        area, grad = union_area_grad(x)
        return area, grad

    # Optimize from Keich
    try:
        res = minimize(objective, x_best, jac=objective_grad, method='L-BFGS-B',
                       options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50})
        if res.success and res.fun < area_best:
            x_best = res.x
            area_best = res.fun
    except Exception:
        pass

    # Try more diverse starts
    np.random.seed(42)
    for restart in range(10):
        if time.time() - t0 > time_limit - safety_margin:
            break
        # Generate diverse initial points
        if restart == 0:
            x0 = keich_init() + np.random.randn(n) * 0.01
        elif restart == 1:
            x0 = keich_init() + np.random.randn(n) * 0.05
        elif restart == 2:
            x0 = np.sort(np.random.uniform(-0.5, 1.5, n))
        elif restart == 3:
            x0 = np.sort(np.random.uniform(-0.2, 1.2, n))
        elif restart == 4:
            # Clustered around Keich with larger perturbation
            x0 = keich_init() + np.random.randn(n) * 0.1
        elif restart == 5:
            # Symmetric perturbation
            x0 = keich_init() + np.random.randn(n) * 0.02
            x0 = x0 - x0.mean() + keich_init().mean()
        elif restart == 6:
            # Random permutation of Keich
            x0 = keich_init()[np.random.permutation(n)]
        elif restart == 7:
            # Linear interpolation between Keich and uniform
            x_unif = np.linspace(-0.5, 1.5, n)
            x0 = 0.5 * keich_init() + 0.5 * x_unif
        elif restart == 8:
            # Small random noise
            x0 = keich_init() + np.random.randn(n) * 0.005
        else:
            x0 = keich_init() + np.random.randn(n) * 0.03

        try:
            res = minimize(objective, x0, jac=objective_grad, method='L-BFGS-B',
                           options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50})
            if res.success and res.fun < area_best:
                x_best = res.x
                area_best = res.fun
        except Exception:
            pass

    # Final polish with more iterations
    if time.time() - t0 < time_limit - safety_margin:
        try:
            res = minimize(objective, x_best, jac=objective_grad, method='L-BFGS-B',
                           options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50})
            if res.success and res.fun < area_best:
                x_best = res.x
                area_best = res.fun
        except Exception:
            pass

    # Ensure finite
    if not np.all(np.isfinite(x_best)):
        x_best = keich_init()

    return x_best.astype(np.float64)

# EVOLVE-BLOCK-END
