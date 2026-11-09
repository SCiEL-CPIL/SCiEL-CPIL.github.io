# EVOLVE-BLOCK-START
import numpy as np
import time


def kakeya_needle_3d8() -> np.ndarray:
    """
    Three-dimensional discrete Kakeya needle problem with grid parameter N = 8.

    We build N*N = 64 "generalized tubes" indexed by (i, j), i, j in {0, ..., N-1}.
    Tube (i, j) has a base position (x, y) and, at height c in [0, 1], occupies the
    axis-aligned square
        [x + c*i/N, x + 1/N + c*i/N] x [y + c*j/N, y + 1/N + c*j/N].

    The union of all tubes is a product set: the x-projection is the union over i of
    intervals [x_i + c*i/N, x_i + 1/N + c*i/N] for c in [0,1], and similarly for y.
    Thus the union volume is V = Vx * Vy * 1, where Vx and Vy are the measures of the
    1-D unions in x and y respectively.

    We minimize Vx and Vy independently using a spectral approach: we represent the
    displacement function d(t) = c*i/N as a function of c in [0,1], and the 1-D union
    measure is the measure of the union of intervals [x_i + d_i(c), x_i + 1/N + d_i(c)]
    for c in [0,1].

    For the 1-D problem with N intervals of width 1/N, each shifted by d_i(c) = c*i/N,
    the union measure over c in [0,1] can be minimized by choosing base positions x_i
    to maximize overlap. The optimal strategy is to make all intervals overlap as much
    as possible.

    Key insight: if we set x_i = -d_i(0) = 0 for all i, then at c=0 all intervals are
    [0, 1/N], and at c=1 they are [i/N, (i+1)/N]. The union at each c is the set of
    points covered by at least one interval. The union measure over c is the integral
    over c of the 1-D measure of the union of intervals at height c.

    For a given c, the intervals are [c*i/N, c*i/N + 1/N] for i=0,...,N-1. These are
    N intervals of width 1/N, spaced by 1/N apart (the i-th starts at c*i/N). So they
    tile [0, c*(N-1)/N + 1/N] with gaps if c > 0. Actually, interval i is [c*i/N,
    c*i/N + 1/N]. The gap between interval i and i+1 is (c*(i+1)/N) - (c*i/N + 1/N)
    = c/N - 1/N = (c-1)/N. For c < 1, this is negative, meaning they overlap. So for
    c in [0,1), the intervals overlap and form a single interval from 0 to c*(N-1)/N
    + 1/N. The measure is c*(N-1)/N + 1/N. Integrating over c from 0 to 1:
    integral_0^1 (c*(N-1)/N + 1/N) dc = (N-1)/(2N) + 1/N = (N-1+2)/(2N) = (N+1)/(2N).

    For N=8, this gives 9/16 = 0.5625. So Vx = Vy = 9/16, and V = (9/16)^2 = 81/256
    ≈ 0.3164.

    But we can do better by choosing x_i more carefully. The idea is to make the
    intervals overlap more at each c. If we shift x_i by a constant offset, the
    measure doesn't change. But if we use a non-uniform placement, we might reduce
    the union.

    Actually, the Keich construction for 2-D Kakeya gives a better result. Let me
    implement the Keich binary construction properly and then try to optimize the
    base positions using a local search.

    For the 1-D problem: minimize the integral over c in [0,1] of the measure of the
    union of intervals [x_i + c*i/N, x_i + 1/N + c*i/N] for i=0,...,N-1.

    The Keich construction uses x_i = sum_{k=1}^{log2(N)} eps_{i,k} * (1 - 2^k) *
    2^{-k} / log2(N) where eps_{i,k} are the binary digits of i.

    Let me compute this and then do a local optimization.
    """
    deadline = time.time() + 600.0
    N = 8
    width = 1.0 / N

    # Compute Keich base positions for the 1-D problem
    pos = int(np.log2(N))
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        # Get binary representation of idx, padded to pos bits
        bits = np.zeros(pos, dtype=int)
        for k in range(pos):
            bits[k] = (idx >> k) & 1
        # Keich: x_i = sum_{k=0}^{pos-1} eps_{i,k} * (1 - 2^{k+1}) * 2^{-(k+1)} / pos
        # where eps are the binary digits
        terms = np.arange(1, pos + 1)
        keich[idx] = np.sum(bits * (1 - 2.0 ** terms) * (2.0 ** -terms) / pos)

    # Build the 64 base positions using Keich in x and y
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()

    base_x = keich[i_flat].copy()
    base_y = keich[j_flat].copy()
    positions = np.stack([base_x, base_y], axis=1)

    # Now try to optimize using local search
    # The 1-D union measure for a set of base positions x_i is:
    # integral_0^1 measure(union_i [x_i + c*i/N, x_i + width + c*i/N]) dc
    #
    # We can estimate this numerically by sampling c values.

    def compute_1d_union_measure(x_base, n_samples=200):
        """Compute the 1-D union measure for given base positions."""
        c_samples = np.linspace(0, 1, n_samples)
        measures = np.zeros(n_samples)
        for ci, c in enumerate(c_samples):
            # For each i, the interval is [x_base[i] + c*i/N, x_base[i] + width + c*i/N]
            starts = x_base + c * np.arange(N) / N
            ends = starts + width
            # Compute the measure of the union of these intervals
            # Sort by start
            order = np.argsort(starts)
            s_sorted = starts[order]
            e_sorted = ends[order]
            # Sweep to find union measure
            union = 0.0
            cur_start = s_sorted[0]
            cur_end = e_sorted[0]
            for k in range(1, N):
                if s_sorted[k] <= cur_end:
                    cur_end = max(cur_end, e_sorted[k])
                else:
                    union += cur_end - cur_start
                    cur_start = s_sorted[k]
                    cur_end = e_sorted[k]
            union += cur_end - cur_start
            measures[ci] = union
        # Trapezoidal rule
        return np.trapz(measures, c_samples)

    def compute_volume_1d(x_base, n_samples=200):
        """Compute the 1-D union measure."""
        return compute_1d_union_measure(x_base, n_samples)

    # Initial volume estimate
    vol_x = compute_1d_union_measure(base_x)
    vol_y = compute_1d_union_measure(base_y)
    best_vol = vol_x * vol_y
    best_x = base_x.copy()
    best_y = base_y.copy()

    # Local optimization using coordinate descent
    # Try perturbing each base position
    rng = np.random.default_rng(42)
    n_iter = 0
    max_iter = 50000

    step_size = 0.05
    while time.time() < deadline - 30 and n_iter < max_iter:
        # Try a random perturbation
        dim = N  # We optimize the 1-D x positions
        idx = rng.integers(0, dim)
        delta = rng.uniform(-step_size, step_size)
        new_x = best_x.copy()
        new_x[idx] += delta
        new_vol_x = compute_1d_union_measure(new_x, n_samples=100)
        new_vol = new_vol_x * vol_y

        if new_vol < best_vol:
            best_x = new_x
            best_vol = new_vol
            step_size = max(step_size * 0.95, 0.001)
        else:
            step_size = min(step_size * 1.05, 0.2)
        n_iter += 1

        # Also try optimizing y
        idx_y = rng.integers(0, dim)
        delta_y = rng.uniform(-step_size, step_size)
        new_y = best_y.copy()
        new_y[idx_y] += delta_y
        new_vol_y = compute_1d_union_measure(new_y, n_samples=100)
        new_vol_y_total = vol_x * new_vol_y

        if new_vol_y_total < best_vol:
            best_y = new_y
            vol_y = new_vol_y
            best_vol = new_vol_y_total
            step_size = max(step_size * 0.95, 0.001)
        else:
            step_size = min(step_size * 1.05, 0.2)
        n_iter += 1

    # Rebuild positions from optimized 1-D bases
    base_x_opt = np.zeros(N * N)
    base_y_opt = np.zeros(N * N)
    for i in range(N):
        for j in range(N):
            k = i * N + j
            base_x_opt[k] = best_x[i]
            base_y_opt[k] = best_y[j]

    positions = np.stack([base_x_opt, base_y_opt], axis=1)

    # Validate
    if not np.all(np.isfinite(positions)):
        # Fall back to Keich
        base_x = keich[i_flat]
        base_y = keich[j_flat]
        positions = np.stack([base_x, base_y], axis=1)

    return positions


# EVOLVE-BLOCK-END
