import time
import numpy as np
from scipy.optimize import minimize_scalar

# Global deadline to ensure we finish well before the 150s limit
# We aim to finish by 110s to be safe.
_GLOBAL_DEADLINE = time.time() + 110.0

# Precomputation for fast evaluation
_N_SAMPLES = 2048
_Z = np.exp(2j * np.pi * np.arange(_N_SAMPLES) / _N_SAMPLES)
_N = 24
_Z_POWERS = np.ones((_N, _N_SAMPLES), dtype=complex)
for k in range(1, _N):
    _Z_POWERS[k] = _Z_POWERS[k-1] * _Z

def _eval_all(coeffs):
    """Evaluate P at all sample points. Returns complex array of length _N_SAMPLES."""
    vals = np.zeros(_N_SAMPLES, dtype=complex)
    for c in reversed(coeffs):
        vals = vals * _Z + c
    return vals

def _fast_score(coeffs):
    """Fast score: max |P(z)| / sqrt(n) using precomputed roots."""
    vals = _eval_all(coeffs)
    return float(np.max(np.abs(vals))) / np.sqrt(_N)

def _eval_poly(coeffs, z):
    """Evaluate polynomial P(z) = sum c_k z^k at complex z using Horner's method."""
    result = 0.0 + 0.0j
    for c in reversed(coeffs):
        result = result * z + c
    return result

def _sup_norm_high_precision(coeffs, num_samples=20000, refine=True):
    """
    Compute the sup norm of the polynomial on the unit circle with high precision.
    1. Dense sampling to find candidate peaks.
    2. Golden-section refinement (via scipy) around the top peaks.
    """
    n = len(coeffs)
    if n == 0:
        return 0.0
        
    # 1. Dense sampling
    angles = 2 * np.pi * np.arange(num_samples) / num_samples
    zs = np.exp(1j * angles)
    
    vals = np.zeros(num_samples, dtype=complex)
    for c in reversed(coeffs):
        vals = vals * zs + c
        
    mag = np.abs(vals)
    
    if not refine:
        return float(np.max(mag))
        
    # 2. Refine top peaks
    top_k = 20
    peak_indices = np.argsort(mag)[-top_k:][::-1]
    
    best_val = float(np.max(mag))
    window = 2 * np.pi / num_samples
    
    for idx in peak_indices:
        center = angles[idx]
        
        def obj(theta):
            z = np.exp(1j * theta)
            return -abs(_eval_poly(coeffs, z))
            
        try:
            res = minimize_scalar(obj, bounds=(center - window, center + window),
                                  method='bounded', options={'xatol': 1e-12})
            if res.success:
                peak_mag = -res.fun
                if peak_mag > best_val:
                    best_val = peak_mag
        except Exception:
            pass
            
    return best_val

def _local_search_vectorized(coeffs, time_limit=5.0):
    """
    Local search with vectorized single flips and sampled pair flips.
    Uses precomputed Z powers and incremental updates for speed.
    """
    start_time = time.time()
    current = np.array(coeffs, dtype=np.int8)
    n = len(current)
    
    vals = _eval_all(current)
    current_score = float(np.max(np.abs(vals))) / np.sqrt(n)
    
    improved = True
    while improved:
        improved = False
        if time.time() > start_time + time_limit or time.time() > _GLOBAL_DEADLINE:
            break
        
        # Single flips
        for i in range(n):
            if time.time() > _GLOBAL_DEADLINE:
                return current.tolist(), current_score
            
            # delta_j = -4*c_i*Re(vals[j] * conj(z_j^i)) + 4
            # new_mag_sq = |vals|^2 + delta
            delta = -4.0 * current[i] * np.real(vals * np.conj(_Z_POWERS[i])) + 4.0
            new_mag_sq = np.abs(vals)**2 + delta
            
            # Check for any negative values due to numerical error (shouldn't happen much)
            # If negative, it means the approximation is bad, but let's proceed carefully.
            # Actually |vals_new|^2 = |vals - 2*c_i*z^i|^2.
            # Let's just compute max directly to be safe, but the delta formula is exact.
            new_max = float(np.max(new_mag_sq))
            new_score = np.sqrt(new_max) / np.sqrt(n)
            
            if new_score < current_score - 1e-12:
                # Accept flip
                current[i] = -current[i]
                # Update vals: vals_new = vals - 2*c_old_i*z^i. Since c_new = -c_old, 
                # vals_new = vals - 2*c_old*z^i.
                # Note: c_old was current[i] before flip.
                vals = vals - 2.0 * (-current[i]) * _Z_POWERS[i] # Wait.
                # Let's be precise.
                # old_c = current[i] (before flip)
                # new_c = -old_c
                # P_new = P_old - 2*old_c*z^i
                # So we need old_c.
                # We already flipped current[i]. So old_c = -current[i].
                vals = vals - 2.0 * (-current[i]) * _Z_POWERS[i]
                current_score = new_score
                improved = True
                
        # Pair flips (sampled to save time, or full if fast enough)
        # With N=24, there are 276 pairs. 
        # Each pair flip requires O(N_SAMPLES) ops.
        # 276 * 2048 ~ 560k ops. This is very fast. We can do full pair flips.
        if time.time() < start_time + time_limit * 0.5:
            for i in range(n):
                if time.time() > _GLOBAL_DEADLINE:
                    return current.tolist(), current_score
                for j in range(i + 1, n):
                    if time.time() > _GLOBAL_DEADLINE:
                        return current.tolist(), current_score
                    
                    # Flip both i and j
                    # P_new = P_old - 2*c_i*z^i - 2*c_j*z^j
                    # new_vals = vals - 2*c_i*z^i - 2*c_j*z^j
                    new_vals = vals - 2.0 * current[i] * _Z_POWERS[i] - 2.0 * current[j] * _Z_POWERS[j]
                    new_max = float(np.max(np.abs(new_vals)**2))
                    new_score = np.sqrt(new_max) / np.sqrt(n)
                    
                    if new_score < current_score - 1e-12:
                        current[i] = -current[i]
                        current[j] = -current[j]
                        vals = new_vals
                        current_score = new_score
                        improved = True
                        
                    if time.time() > start_time + time_limit:
                        break
                if time.time() > start_time + time_limit:
                    break

    return current.tolist(), current_score

def _grs_sequence(length=24):
    """
    Generate Rudin-Shapiro sequence of given length.
    """
    p, q = [1], [1]
    while len(p) < length:
        p, q = p + q, p + [-c for c in q]
    return p[:length]

def construct():
    # 1. Generate diverse initial candidates
    grs = _grs_sequence(24)
    
    candidates = []
    # Rudin-Shapiro and its symmetries
    candidates.append(grs)
    candidates.append([-c for c in grs])
    candidates.append(list(reversed(grs)))
    candidates.append([-c for c in reversed(grs)])
    
    # Add a few random walks to explore other basins
    np.random.seed(42)
    for _ in range(10):
        random_seq = np.random.choice([1, -1], size=24).tolist()
        candidates.append(random_seq)

    best_seq = None
    best_score = float('inf')
    
    # 2. Run local search on each candidate
    # We allocate time dynamically based on remaining time
    num_candidates = len(candidates)
    
    for i, cand in enumerate(candidates):
        if time.time() > _GLOBAL_DEADLINE - 20: # Stop if we only have 20s left for final refinement
            break
            
        # Run local search
        # Allocate more time if we have more candidates, less if fewer.
        # But 5s per candidate is usually enough for convergence.
        time_per_candidate = 5.0
        searched_seq, local_score = _local_search_vectorized(cand, time_limit=time_per_candidate)
        
        if local_score < best_score:
            best_score = local_score
            best_seq = searched_seq
            
        # If we found a very good score early, we can stop searching other candidates
        # The theoretical minimum is ~1.258. If we get below 1.26, it's likely optimal.
        if best_score < 1.26:
            break

    if best_seq is None:
        # Fallback to the first candidate if no search completed
        best_seq = candidates[0]
        best_score = _fast_score(best_seq)

    # 3. Final high-precision refinement and verification
    # We use the best sequence found.
    final_score = _sup_norm_high_precision(best_seq)
    
    # Optional: If time permits, try to improve the best sequence with a very tight local search pass
    # using the high-precision metric? No, too slow.
    # Instead, ensure the sequence is "locally optimal" with respect to the fast metric.
    # The local search above already did this.
    
    return best_seq

# Test block (optional, usually removed in final submission but good for sanity)
if __name__ == "__main__":
    start = time.time()
    coeffs = construct()
    end = time.time()
    print(f"Time taken: {end - start:.2f}s")
    print(f"Coeffs: {coeffs}")
    score = _sup_norm_high_precision(coeffs)
    print(f"Score: {score}")
    print(f"Target: 1.2583758")
    print(f"Ratio: {score / 1.2583758}")
