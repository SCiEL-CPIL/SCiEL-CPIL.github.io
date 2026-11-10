"""Optimal +/-1 polynomial with 24 coefficients minimizing sup norm on |z|=1."""
import time
import numpy as np
from scipy.optimize import minimize_scalar

# EVOLVE-BLOCK-START
import random
import itertools

# Precomputed DFT matrices for the decomposition P(z) = A(z) + z^12 * B(z)
# where A and B each have 12 coefficients.
# |P(z)|^2 = |A|^2 + |B|^2 + 2*Re(A * conj(B) * z^12)
# At sample point j: z^12 = omega^j where omega = exp(2*pi*i/12)

_N_SAMPLES = 2048
_M = 12  # size of each half
_OMEGA = np.exp(2j * np.pi / _M)  # 12th root of unity

# Precompute: for each half, DFT matrix W[j, k] = exp(2*pi*i*j*k/N)
_W_A = np.exp(2j * np.pi * np.outer(np.arange(_N_SAMPLES), np.arange(_M)) / _N_SAMPLES)
_W_B = np.exp(2j * np.pi * np.outer(np.arange(_N_SAMPLES), np.arange(_M)) / _N_SAMPLES)
# z^12 at each sample point
_Z12 = np.exp(2j * np.pi * np.arange(_N_SAMPLES) * _M / _N_SAMPLES)
# = np.ones(_N_SAMPLES) since z^12 for z = exp(2*pi*i*j/N) gives exp(2*pi*i*j*12/N)
# Actually: z_j = exp(2*pi*i*j/N), z_j^12 = exp(2*pi*i*j*12/N)
# Since N=2048 and M=12: z_j^12 = exp(2*pi*i*j*12/2048)
# This is NOT all ones. Let me recompute properly.

# Actually let me think more carefully.
# P(z) = sum_{k=0}^{23} c_k z^k = sum_{k=0}^{11} c_k z^k + z^12 sum_{k=0}^{11} c_{k+12} z^k
# = A(z) + z^12 * B(z)
# where A(z) = sum_{k=0}^{11} c_k z^k, B(z) = sum_{k=0}^{11} c_{k+12} z^k
#
# |P(z_j)|^2 = |A(z_j)|^2 + |B(z_j)|^2 + 2*Re(A(z_j) * conj(B(z_j)) * z_j^12)
#
# A(z_j) = W_A[j,:] @ a  where a = [c_0,...,c_11]
# B(z_j) = W_B[j,:] @ b  where b = [c_12,...,c_23]
#
# We can precompute |A|^2 and |B|^2 contributions using the fact that
# |sum_k a_k W[j,k]|^2 = sum_k sum_l a_k a_l W[j,k] conj(W[j,l])
# = sum_k a_k^2 * |W[j,k]|^2 + 2*Re(sum_{k<l} a_k a_l W[j,k] conj(W[j,l]))
# Since |W[j,k]| = 1, this = sum_k a_k^2 + 2*Re(sum_{k<l} a_k a_l W[j,k] conj(W[j,l]))
# = 12 + 2*Re(sum_{k<l} a_k a_l W[j,k] conj(W[j,l]))  (since a_k in {+1,-1}, a_k^2=1)
#
# Even better: we can precompute the full contribution.
# Let's use a different approach: precompute for each pair (i,j) in first half,
# the contribution to |A|^2 at each sample point.

# Actually, the fastest approach: precompute the full |P|^2 as a function of the 24 bits.
# |P(z_j)|^2 = sum_k sum_l c_k c_l z_j^{k-l}
# This is a quadratic form in the c's.
# |P(z_j)|^2 = c^T Q_j c where Q_j[k,l] = z_j^{k-l}
#
# For a flip of bit i: delta = |P_new|^2 - |P_old|^2
# = 2*c_i * sum_{l != i} c_l z_j^{i-l} + 2*(1 - 1) ... 
# Actually: if we flip c_i -> -c_i, then
# |P_new|^2 = |P_old - 2*c_i*z^i|^2 = |P_old|^2 - 4*c_i*Re(P_old * conj(z^i)) + 4
# So delta_j = -4*c_i*Re(P_old(z_j) * conj(z_j^i)) + 4
#
# This means if we precompute P_old(z_j) for all j, a single flip costs O(N).
# For pair flips it's more complex but still manageable.

# Let me use the simplest fast approach: precompute z_j powers.
_Z = np.exp(2j * np.pi * np.arange(_N_SAMPLES) / _N_SAMPLES)

def _eval_all(coeffs):
    """Evaluate P at all sample points. Returns complex array of length _N_SAMPLES."""
    n = len(coeffs)
    vals = np.zeros(_N_SAMPLES, dtype=complex)
    for c in reversed(coeffs):
        vals = vals * _Z + c
    return vals

def _score_fast(coeffs, num_samples=_N_SAMPLES):
    """Fast score: max |P(z)| / sqrt(n) using precomputed roots."""
    n = len(coeffs)
    vals = np.zeros(num_samples, dtype=complex)
    for c in reversed(coeffs):
        vals = vals * _Z[:num_samples] + c
    return float(np.max(np.abs(vals))) / np.sqrt(n)

def _eval_poly(coeffs, z):
    """Evaluate polynomial P(z) = sum c_k z^k at complex z using Horner's method."""
    result = 0.0
    for c in reversed(coeffs):
        result = result * z + c
    return result

def _sup_norm(coeffs, num_samples=4096, refine=True):
    """Compute sup norm of P(z) over |z|=1 via dense sampling + golden-section refinement."""
    n = len(coeffs)
    angles = 2 * np.pi * np.arange(num_samples) / num_samples
    zs = np.exp(1j * angles)
    
    vals = np.zeros(num_samples, dtype=complex)
    for c in reversed(coeffs):
        vals = vals * zs + c
    
    mag = np.abs(vals)
    
    if not refine:
        return float(np.max(mag))
    
    top_k = 8
    peak_indices = np.argsort(mag)[-top_k:][::-1]
    
    best_val = float(np.max(mag))
    window = 2 * np.pi / num_samples
    
    for idx in peak_indices:
        center = angles[idx]
        def obj(theta):
            z = np.exp(1j * theta)
            return -abs(_eval_poly(coeffs, z))
        res = minimize_scalar(obj, bounds=(center - window, center + window),
                              method='bounded', options={'xatol': 1e-11})
        if res.success:
            peak_mag = -res.fun
            if peak_mag > best_val:
                best_val = peak_mag
    
    return best_val

def _score(coeffs, num_samples=4096):
    """Score: max |P(z)| / sqrt(n)"""
    return _sup_norm(coeffs, num_samples) / np.sqrt(len(coeffs))

def _local_search(coeffs, time_limit=2.0, fast=True, num_samples=_N_SAMPLES):
    """Local search with single and pair flips using fast precomputed evaluation."""
    start = time.time()
    current = list(coeffs)
    n = len(current)
    
    # Precompute current values at all sample points
    vals = np.zeros(num_samples, dtype=complex)
    for c in reversed(current):
        vals = vals * _Z[:num_samples] + c
    current_score = float(np.max(np.abs(vals))) / np.sqrt(n)
    
    # Precompute z^k for k=0..23 at all sample points for flip delta computation
    # z_powers[k][j] = z_j^k
    z_powers = np.ones((n, num_samples), dtype=complex)
    for k in range(1, n):
        z_powers[k] = z_powers[k-1] * _Z[:num_samples]
    
    improved = True
    while improved:
        improved = False
        if time.time() - start > time_limit:
            break
        # Single flips: delta_j = -4*c_i*Re(vals[j] * conj(z_j^i)) + 4
        for i in range(n):
            if time.time() - start > time_limit:
                break
            # Compute new |vals|^2 for all j
            # |vals_new|^2 = |vals - 2*c_i*z^i|^2 = |vals|^2 - 4*c_i*Re(vals*conj(z^i)) + 4
            delta = -4.0 * current[i] * np.real(vals * np.conj(z_powers[i])) + 4.0
            new_mag_sq = np.abs(vals)**2 + delta
            new_max = float(np.max(new_mag_sq))
            new_score = np.sqrt(new_max) / np.sqrt(n)
            if new_score < current_score - 1e-12:
                current[i] = -current[i]
                vals = vals - 2.0 * current[i] * z_powers[i]
                current_score = new_score
                improved = True
        # Pair flips
        if time.time() - start > time_limit:
            break
        for i in range(n):
            if time.time() - start > time_limit:
                break
            for j in range(i+1, n):
                if time.time() - start > time_limit:
                    break
                # Flip both i and j
                # new_vals = vals - 2*c_i*z^i - 2*c_j*z^j
                new_vals = vals - 2.0 * current[i] * z_powers[i] - 2.0 * current[j] * z_powers[j]
                new_max = float(np.max(np.abs(new_vals)**2))
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
    """Generate all reversal variants."""
    n = len(coeffs)
    variants = []
    for shift in range(n):
        rotated = coeffs[shift:] + coeffs[:shift]
        variants.append(rotated)
        variants.append(list(reversed(rotated)))
    return variants

def construct():
    deadline = time.time() + 140  # Safety margin before 150s kill
    random.seed(42)
    
    # Generate initial candidates from known constructions
    candidates = []
    
    # Rudin-Shapiro and variants
    grs = _grs_sequence(24)
    candidates.append(grs)
    candidates.append(list(reversed(grs)))
    candidates.append([-c for c in grs])
    candidates.append([-c for c in reversed(grs)])
    
    # Try all shifts and reversals of GRS
    for variant in _all_reversals(grs):
        candidates.append(variant)
        candidates.append([-c for c in variant])
    
    # Random candidates
    for _ in range(100):
        candidates.append([random.choice([-1, 1]) for _ in range(24)])
    
    # Additional structured candidates
    for period in range(2, 13):
        base = [random.choice([-1, 1]) for _ in range(period)]
        seq = [base[i % period] for i in range(24)]
        candidates.append(seq)
    
    # Deduplicate
    seen = set()
    unique_candidates = []
    for c in candidates:
        key = tuple(c)
        if key not in seen:
            seen.add(key)
            unique_candidates.append(c)
    
    # Phase 1: Quick screening
    best_coeffs = None
    best_score = float('inf')
    
    for c in unique_candidates:
        if time.time() > deadline - 80:
            break
        s = _score_fast(c)
        if s < best_score:
            best_score = s
            best_coeffs = list(c)
    
    # Phase 2: Local search from top candidates
    scored = [(c, _score_fast(c)) for c in unique_candidates]
    scored.sort(key=lambda x: x[1])
    
    for c, s in scored[:30]:
        if time.time() > deadline - 60:
            break
        coeffs, score = _local_search(c, time_limit=3.0, fast=True)
        if score < best_score:
            best_score = score
            best_coeffs = coeffs
    
    # Phase 3: Many random restarts with local search (key improvement)
    # The fast local search enables many more restarts
    num_restarts = 500
    for r in range(num_restarts):
        if time.time() > deadline - 30:
            break
        coeffs = [random.choice([-1, 1]) for _ in range(24)]
        coeffs, score = _local_search(coeffs, time_limit=2.0, fast=True)
        if score < best_score:
            best_score = score
            best_coeffs = coeffs
    
    # Phase 4: Perturbation + local search from best
    if time.time() < deadline - 20:
        for _ in range(50):
            if time.time() > deadline - 15:
                break
            perturbed = list(best_coeffs)
            for _ in range(random.randint(1, 5)):
                idx = random.randint(0, 23)
                perturbed[idx] = -perturbed[idx]
            coeffs, score = _local_search(perturbed, time_limit=1.5, fast=True)
            if score < best_score:
                best_score = score
                best_coeffs = coeffs
    
    # Phase 5: Final refinement with full precision scoring
    if time.time() < deadline - 10:
        coeffs, score = _local_search(best_coeffs, time_limit=min(8.0, deadline - time.time() - 5), fast=False)
        if score < best_score:
            best_score = score
            best_coeffs = coeffs
    
    # Final verification
    final_score = _score(best_coeffs)
    print(f"Final score: {final_score:.9f}")
    
    return best_coeffs
# EVOLVE-BLOCK-END
