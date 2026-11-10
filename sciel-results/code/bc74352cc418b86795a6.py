"""
Flattest plus-minus-one polynomials with 24 coefficients (minimum sup norm).
Combines fast precomputed-evaluation local search (single + pair flips) with
golden-section refinement for the final score.
"""
import time
import numpy as np
import random
from scipy.optimize import minimize_scalar

# EVOLVE-BLOCK-START

# --- Precomputation ---
_N_SAMPLES = 2048
_Z = np.exp(2j * np.pi * np.arange(_N_SAMPLES) / _N_SAMPLES)

# Precompute z^k for k=0..23 at all sample points
# z_powers[k][j] = z_j^k
_z_powers = np.ones((24, _N_SAMPLES), dtype=complex)
for _k in range(1, 24):
    _z_powers[_k] = _z_powers[_k - 1] * _Z


def _eval_poly(coeffs, z):
    """Evaluate polynomial P(z) = sum c_k z^k at complex z using Horner's method."""
    result = 0.0
    for c in reversed(coeffs):
        result = result * z + c
    return result


def _eval_all(coeffs, num_samples=_N_SAMPLES):
    """Evaluate P at all sample points. Returns complex array."""
    vals = np.zeros(num_samples, dtype=complex)
    for c in reversed(coeffs):
        vals = vals * _Z[:num_samples] + c
    return vals


def _sup_norm_refined(coeffs, num_samples=4096):
    """
    Compute sup norm with golden-section refinement of top peaks.
    """
    n = len(coeffs)
    angles = 2 * np.pi * np.arange(num_samples) / num_samples
    zs = np.exp(1j * angles)

    vals = np.zeros(num_samples, dtype=complex)
    for c in reversed(coeffs):
        vals = vals * zs + c

    mag = np.abs(vals)

    top_k = 8
    if num_samples > top_k:
        peak_indices = np.argsort(mag)[-top_k:][::-1]
    else:
        peak_indices = np.arange(num_samples)

    best_val = float(np.max(mag))
    window = 2 * np.pi / num_samples

    for idx in peak_indices:
        center = angles[idx]
        center = center % (2 * np.pi)

        def obj(theta):
            z = np.exp(1j * theta)
            return -abs(_eval_poly(coeffs, z))

        lower = center - window
        upper = center + window

        try:
            res = minimize_scalar(obj, bounds=(lower, upper),
                                  method='bounded', options={'xatol': 1e-11})
            if res.success:
                peak_mag = -res.fun
                if peak_mag > best_val:
                    best_val = peak_mag
        except Exception:
            pass

    return best_val


def _score_fast(coeffs, num_samples=_N_SAMPLES):
    """Fast score: max |P(z)| / sqrt(n) using precomputed roots."""
    n = len(coeffs)
    vals = np.zeros(num_samples, dtype=complex)
    for c in reversed(coeffs):
        vals = vals * _Z[:num_samples] + c
    return float(np.max(np.abs(vals))) / np.sqrt(n)


def _local_search(coeffs, time_limit=3.0, num_samples=_N_SAMPLES):
    """
    Local search with single and pair flips using fast precomputed evaluation.
    Returns (coeffs, score).
    """
    start = time.time()
    current = list(coeffs)
    n = len(current)
    zp = _z_powers[:n]

    # Precompute current values at all sample points
    vals = np.zeros(num_samples, dtype=complex)
    for c in reversed(current):
        vals = vals * _Z[:num_samples] + c
    current_score = float(np.max(np.abs(vals))) / np.sqrt(n)

    improved = True
    while improved:
        improved = False
        if time.time() - start > time_limit:
            break

        # Single flips
        for i in range(n):
            if time.time() - start > time_limit:
                break
            # delta_j = -4*c_i*Re(vals[j] * conj(z_j^i)) + 4
            delta = -4.0 * current[i] * np.real(vals * np.conj(zp[i])) + 4.0
            new_mag_sq = np.abs(vals) ** 2 + delta
            new_max = float(np.max(new_mag_sq))
            new_score = np.sqrt(new_max) / np.sqrt(n)
            if new_score < current_score - 1e-12:
                current[i] = -current[i]
                vals = vals - 2.0 * current[i] * zp[i]
                current_score = new_score
                improved = True

        # Pair flips
        if time.time() - start > time_limit:
            break
        for i in range(n):
            if time.time() - start > time_limit:
                break
            for j in range(i + 1, n):
                if time.time() - start > time_limit:
                    break
                new_vals = vals - 2.0 * current[i] * zp[i] - 2.0 * current[j] * zp[j]
                new_max = float(np.max(np.abs(new_vals) ** 2))
                new_score = np.sqrt(new_max) / np.sqrt(n)
                if new_score < current_score - 1e-12:
                    current[i] = -current[i]
                    current[j] = -current[j]
                    vals = new_vals
                    current_score = new_score
                    improved = True

    return current, current_score


def _grs_sequence(length=24):
    """Generate Golay-Rudin-Shapiro sequence truncated to given length."""
    p, q = [1], [1]
    while len(p) < length:
        p, q = p + q, p + [-c for c in q]
    return p[:length]


def _all_reversals(coeffs):
    """Generate all reversals of the coefficient sequence."""
    n = len(coeffs)
    results = [list(coeffs)]
    for k in range(1, n):
        rev = coeffs[:k] + list(reversed(coeffs[k:]))
        results.append(list(rev))
    return results


def construct():
    deadline = time.time() + 115  # Safety margin before 150s kill

    # Initialize with GRS and its reversals
    grs = _grs_sequence(24)
    candidates = _all_reversals(grs)

    best_coeffs = None
    best_score = float('inf')

    # Evaluate all GRS reversals with local search
    for cand in candidates:
        if time.time() > deadline - 30:
            break
        coeffs, score = _local_search(cand, time_limit=2.0)
        if score < best_score:
            best_score = score
            best_coeffs = list(coeffs)

    # Random restarts
    while time.time() < deadline - 25:
        coeffs = [random.choice([-1, 1]) for _ in range(24)]
        coeffs, score = _local_search(coeffs, time_limit=3.0)
        if score < best_score:
            best_score = score
            best_coeffs = list(coeffs)

    # Final refinement of the best candidate
    final_score = _sup_norm_refined(best_coeffs, num_samples=4096) / np.sqrt(24)

    return best_coeffs

# EVOLVE-BLOCK-END
