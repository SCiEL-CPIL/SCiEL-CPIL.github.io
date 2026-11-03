import numpy as np
import time

N = 100
TARGET_SUM = N / 2.0  # 50.0


def _max_overlap(h):
    """Calculate the maximum overlap of h with shifted 1-h (circular)."""
    max_val = 0.0
    for k in range(N):
        shifted_h = np.roll(h, k)
        overlap = np.sum(h * (1.0 - shifted_h))
        if overlap > max_val:
            max_val = overlap
    return max_val


def _is_valid(h):
    return (np.all(h >= -1e-9) and np.all(h <= 1.0 + 1e-9)
            and abs(np.sum(h) - TARGET_SUM) < 1e-6)


def _project(h):
    """Project h onto {x in [0,1]^N : sum x = TARGET_SUM} via bisection on a
    water-filling threshold. This is exact and fast."""
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        # x_i = clip(h_i - mid, 0, 1); we want sum = TARGET_SUM
        x = np.clip(h - mid, 0.0, 1.0)
        s = np.sum(x)
        if s > TARGET_SUM:
            lo = mid
        else:
            hi = mid
    x = np.clip(h - 0.5 * (lo + hi), 0.0, 1.0)
    # tiny residual correction spread over free (interior) coords
    diff = TARGET_SUM - np.sum(x)
    if abs(diff) > 1e-12:
        free = (x > 1e-9) & (x < 1.0 - 1e-9)
        if np.any(free):
            x[free] += diff / np.sum(free)
            x = np.clip(x, 0.0, 1.0)
    return x


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin under the 150 s kill

    # The uniform solution h_i = 0.5 is provably optimal:
    #   C_k = sum_i h_i(1 - h_{i+k}) = S - A_k,  S = sum h_i = 50
    #   sum_k A_k = S^2 = 2500  =>  mean_k A_k = 25  =>  min_k A_k <= 25
    #   =>  max_k C_k >= 50 - 25 = 25, with equality at uniform.
    # So 25 is the floor; no feasible h can beat it.
    h_uniform = np.full(N, 0.5)
    best_h = h_uniform.copy()
    best_score = _max_overlap(best_h)  # exactly 25.0

    # Confirm optimality with a short, cheap local search. It cannot improve
    # below 25 (the proven floor), but it guards against any arithmetic slip
    # and is bounded well within the time budget.
    np.random.seed(42)
    h = h_uniform.copy()
    steps = 0
    for _ in range(400):
        if time.time() > deadline:
            break
        steps += 1
        # small random perturbation of a few coordinates
        mask = np.random.choice(N, 12, replace=False)
        h[mask] += np.random.uniform(-0.04, 0.04, 12)
        h = _project(h)
        if not _is_valid(h):
            h = h_uniform.copy()
            continue
        score = _max_overlap(h)
        if score < best_score - 1e-12:
            best_score = score
            best_h = h.copy()

    # Final validity guard: if anything is off, fall back to the exact uniform.
    if not _is_valid(best_h):
        best_h = h_uniform.copy()

    return best_h
