# EVOLVE-BLOCK-START
import numpy as np
import itertools


def _union_length_1d_fast(base, N, n_c=50):
    """Vectorized computation of integral of union length over c."""
    c_samples = np.linspace(0, 1, n_c)
    i_vals = np.arange(N) / N
    starts = base[:, None] + c_samples[None, :] * i_vals[:, None]
    ends = starts + 1.0 / N
    
    lengths = np.zeros(n_c)
    for k in range(n_c):
        s = starts[:, k]
        e = ends[:, k]
        order = np.argsort(s)
        s_sorted = s[order]
        e_sorted = e[order]
        merged_start = s_sorted[0]
        merged_end = e_sorted[0]
        total = 0.0
        for m in range(1, N):
            if s_sorted[m] <= merged_end:
                if e_sorted[m] > merged_end:
                    merged_end = e_sorted[m]
            else:
                total += merged_end - merged_start
                merged_start = s_sorted[m]
                merged_end = e_sorted[m]
        total += merged_end - merged_start
        lengths[k] = total
    return np.trapz(lengths, c_samples)


def kakeya_needle_3d8() -> np.ndarray:
    """
    Three-dimensional discrete Kakeya needle problem with grid parameter N = 8.
    
    Uses spectral grid-search over Fourier coefficients of the tube displacement
    functions to minimize the union volume.
    """
    N = 8
    
    def base_from_fourier(coeffs):
        i_vals = np.arange(N)
        theta = 2.0 * np.pi * np.outer(i_vals, np.arange(1, 4)) / N
        base = np.zeros(N)
        base += coeffs[0] * np.cos(theta[:, 0]) + coeffs[1] * np.sin(theta[:, 0])
        base += coeffs[2] * np.cos(theta[:, 1]) + coeffs[3] * np.sin(theta[:, 1])
        base += coeffs[4] * np.cos(theta[:, 2]) + coeffs[5] * np.sin(theta[:, 2])
        base += coeffs[6] * np.cos(2.0 * np.pi * i_vals * 4 / N)
        return base
    
    def evaluate(coeffs):
        base = base_from_fourier(coeffs)
        return _union_length_1d_fast(base, N, n_c=50)
    
    # Try linear shear family first
    def evaluate_linear(alpha):
        base = alpha * np.arange(N) / N
        return _union_length_1d_fast(base, N, n_c=100)
    
    alphas = np.linspace(-1, 1, 500)
    vals = [evaluate_linear(a) for a in alphas]
    best_alpha = alphas[np.argmin(vals)]
    best_val_linear = min(vals)
    base_linear = best_alpha * np.arange(N) / N
    
    # Fourier grid search
    levels = np.array([-0.5, 0.0, 0.5])
    best_coeffs = None
    best_score = np.inf
    
    for combo in itertools.product(levels, repeat=7):
        coeffs = np.array(combo)
        score = evaluate(coeffs)
        if score < best_score:
            best_score = score
            best_coeffs = coeffs
    
    # Fine grid
    if best_coeffs is not None:
        step = 0.25
        for combo in itertools.product(np.linspace(-step, step, 5), repeat=7):
            coeffs = best_coeffs + combo
            score = evaluate(coeffs)
            if score < best_score:
                best_score = score
                best_coeffs = coeffs
    
    if best_score < best_val_linear:
        base_1d = base_from_fourier(best_coeffs)
    else:
        base_1d = base_linear
    
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    base_x = base_1d[i_flat]
    base_y = base_1d[j_flat]
    positions = np.stack([base_x, base_y], axis=1)
    return positions


# EVOLVE-BLOCK-END
