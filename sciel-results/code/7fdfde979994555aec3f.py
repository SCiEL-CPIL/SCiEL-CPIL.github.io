# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def kakeya_needle_3d8() -> np.ndarray:
    """
    Three-dimensional discrete Kakeya needle problem with grid parameter N = 8.

    We build N*N = 64 "generalized tubes" indexed by (i, j), i, j in {0, ..., N-1}.
    Tube (i, j) has a base position (x, y) and, at height c in [0, 1], occupies the
    axis-aligned square
        [x + c*i/N, x + 1/N + c*i/N] x [y + c*j/N, y + 1/N + c*j/N].

    Since the union is a product of 1D unions in x and y (when base_x depends only on i
    and base_y only on j), the union volume is:
        V = (integral_0^1 Lx(c) dc) * (integral_0^1 Ly(c) dc)
    where Lx(c) = length of union of intervals [x_i + c*i/N, x_i + 1/N + c*i/N] for i=0..N-1.

    We minimize V by minimizing f = Ix * Iy where Ix = integral_0^1 Lx(c) dc.
    By symmetry we set base_y = base_x, so we minimize f = Ix^2.

    We use scipy L-BFGS-B to optimize the N base positions x_0, ..., x_{N-1}.
    """
    N = 8
    w = 1.0 / N

    def union_length(xs, c):
        """Compute the length of the union of intervals [xs[i] + c*i/N, xs[i] + w + c*i/N]."""
        starts = xs + c * np.arange(N)
        ends = starts + w
        # Sort by start
        order = np.argsort(starts)
        s = starts[order]
        e = ends[order]
        # Merge intervals
        total = 0.0
        cur_start = s[0]
        cur_end = e[0]
        for k in range(1, N):
            if s[k] <= cur_end:
                cur_end = max(cur_end, e[k])
            else:
                total += cur_end - cur_start
                cur_start = s[k]
                cur_end = e[k]
        total += cur_end - cur_start
        return total

    def integral_x(xs):
        """Compute Ix = integral_0^1 Lx(c) dc using trapezoidal rule with fine grid."""
        # Use a fine grid for accurate integration
        c_vals = np.linspace(0, 1, 200)
        lengths = np.array([union_length(xs, c) for c in c_vals])
        return np.trapz(lengths, c_vals)

    def objective(xs):
        """Objective: Ix^2 (since base_y = base_x, V = Ix^2)."""
        ix = integral_x(xs)
        return ix * ix

    # Initial guess: evenly spaced positions
    x0 = np.linspace(0, 1, N)

    # Optimize using L-BFGS-B
    result = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-12})

    base_x = result.x

    # Construct the 64 positions
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()

    base_y = base_x  # By symmetry, same positions for y

    positions = np.stack([base_x[i_flat], base_y[j_flat]], axis=1)
    return positions


# EVOLVE-BLOCK-END
