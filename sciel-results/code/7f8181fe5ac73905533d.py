# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_area(x, a, y, w):
    """
    Fast computation of union area.
    x: (64,) base positions
    a: (64,) apex offsets
    y: (20000,) quadrature heights
    w: (20000,) widths at each height
    Returns: mean union length (area)
    """
    # L_i(y) = x_i + a_i * y, shape (20000, 64)
    L = x[None, :] + a[None, :] * y[:, None]
    L_sorted = np.sort(L, axis=1)
    diffs = L_sorted[:, 1:] - L_sorted[:, :-1]
    contribs = np.minimum(diffs, w[:, None])
    union_lengths = w + np.sum(contribs, axis=1)
    return np.mean(union_lengths)


def _keich_positions(n):
    """Generate classical Keich construction positions."""
    k = int(np.log2(n))
    j = np.arange(1, k + 1)
    positions = []
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        positions.append(np.sum((1 - j) * eps * (2.0 ** -j) / k))
    return np.array(positions, dtype=np.float64)


def kakeya_needle_2d_positions64() -> np.ndarray:
    """
    Two-dimensional Kakeya-needle area-minimization problem for n=64.
    Starts from the Keich construction and refines using coordinate descent
    and small random perturbations to escape local minima.
    """
    n = 64
    # Precompute fixed values
    a = np.arange(1, n + 1) / n  # a_i = (i+1)/64
    m = np.arange(20000)
    y = (m + 0.5) / 20000.0
    w = (1.0 - y) / n

    # Start from Keich construction
    x = _keich_positions(n)
    best_x = x.copy()
    best_area = _compute_area(best_x, a, y, w)

    start_time = time.time()
    deadline = start_time + 600  # Safety margin for 630s limit

    # Coordinate descent refinement
    def try_coordinate_update(x, idx, delta):
        x_new = x.copy()
        x_new[idx] += delta
        return x_new

    # Phase 1: Coordinate descent with adaptive step size
    step_sizes = [0.01, 0.005, 0.001, 0.0005, 0.0001]
    for step in step_sizes:
        if time.time() > deadline - 60:
            break
        for _ in range(3):  # Multiple passes
            if time.time() > deadline - 60:
                break
            improved = False
            for idx in range(n):
                if time.time() > deadline - 60:
                    break
                # Try positive and negative steps
                for sign in [1, -1]:
                    x_new = try_coordinate_update(best_x, idx, sign * step)
                    area_new = _compute_area(x_new, a, y, w)
                    if area_new < best_area:
                        best_x = x_new
                        best_area = area_new
                        improved = True
            if not improved:
                break

    # Phase 2: Random perturbations to escape local minima
    np.random.seed(42)
    perturb_scales = [0.05, 0.02, 0.01, 0.005, 0.001]
    for scale in perturb_scales:
        if time.time() > deadline - 30:
            break
        for _ in range(10):
            if time.time() > deadline - 30:
                break
            # Perturb a random subset of coordinates
            num_perturb = np.random.randint(3, 10)
            indices = np.random.choice(n, num_perturb, replace=False)
            x_new = best_x.copy()
            x_new[indices] += np.random.normal(0, scale, num_perturb)
            area_new = _compute_area(x_new, a, y, w)
            if area_new < best_area:
                best_x = x_new
                best_area = area_new
                # After a good perturbation, do a quick coordinate descent
                for step in [0.005, 0.001]:
                    if time.time() > deadline - 30:
                        break
                    for _ in range(2):
                        if time.time() > deadline - 30:
                            break
                        for idx in range(n):
                            if time.time() > deadline - 30:
                                break
                            for sign in [1, -1]:
                                x_try = best_x.copy()
                                x_try[idx] += sign * step
                                area_try = _compute_area(x_try, a, y, w)
                                if area_try < best_area:
                                    best_x = x_try
                                    best_area = area_try

    # Final safety check
    if time.time() > deadline - 10:
        return best_x

    # One more round of fine-tuning
    for step in [0.0001, 0.00005]:
        if time.time() > deadline - 10:
            break
        for _ in range(2):
            if time.time() > deadline - 10:
                break
            for idx in range(n):
                if time.time() > deadline - 10:
                    break
                for sign in [1, -1]:
                    x_new = best_x.copy()
                    x_new[idx] += sign * step
                    area_new = _compute_area(x_new, a, y, w)
                    if area_new < best_area:
                        best_x = x_new
                        best_area = area_new

    return best_x


# EVOLVE-BLOCK-END
