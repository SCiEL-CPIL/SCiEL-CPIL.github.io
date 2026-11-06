import numpy as np
import time

def construct():
    N = 24
    half_N = 12
    num_half = 2 ** half_N  # 4096
    
    start_time = time.time()
    time_limit = 110.0
    
    # Stage 1: Coarse Search with MITM
    M_search = 2048
    theta = np.linspace(0, 2 * np.pi, M_search, endpoint=False)
    
    indices = np.arange(num_half)
    
    # Generate coefficient matrices C_A and C_B (4096 x 12)
    C_A = np.ones((num_half, half_N), dtype=np.float64)
    C_B = np.ones((num_half, half_N), dtype=np.float64)
    for k in range(half_N):
        bits = (indices >> k) & 1
        C_A[:, k] = 1 - 2 * bits
        C_B[:, k] = 1 - 2 * bits
        
    # Basis matrices (12 x M)
    k_range = np.arange(half_N)
    B_A = np.exp(1j * np.outer(k_range, theta))
    B_B = np.exp(1j * np.outer(k_range, theta))
    
    # Compute values (4096 x M)
    V_A = C_A @ B_A
    V_B = C_B @ B_B
    
    phase_shift = np.exp(1j * 12 * theta)
    U_B = V_B * phase_shift[None, :]
    
    # Store top candidates
    # We want the top 10 pairs (i, k) with lowest max norm
    # We can't store all 4096x4096 scores, so we iterate and keep a heap or list.
    # A simple list of top 10 is sufficient.
    top_candidates = []
    
    for i in range(num_half):
        if time.time() - start_time > time_limit - 20: # Leave 20s for refinement
            break
            
        S = np.abs(V_A[i, None, :] + U_B)
        scores = np.max(S, axis=1)
        
        # Get indices of top 10 lowest scores for this i
        # We only need the global top 10, so we can just check if any of these are in the global top 10
        # Simpler: just get the min for this i, and if it's better than current worst in top_candidates, add it.
        # But we might miss a candidate that is 2nd best for this i but 1st globally?
        # No, if it's 2nd best for this i, it's worse than the 1st best for this i.
        # So the global best for a fixed i is the 1st best.
        # So we just need the min score for each i.
        
        min_score = np.min(scores)
        k_idx = np.argmin(scores)
        
        # Add to top candidates list
        # We maintain a list of (score, i, k)
        top_candidates.append((min_score, i, k_idx))
        
        # Keep only top 10
        if len(top_candidates) > 10:
            top_candidates.sort(key=lambda x: x[0])
            top_candidates = top_candidates[:10]
            
    if not top_candidates:
        return [1] * 24

    # Sort top candidates
    top_candidates.sort(key=lambda x: x[0])
    
    # Stage 2: Refine top 10 candidates with a much finer grid
    M_refine = 100000
    theta_ref = np.linspace(0, 2 * np.pi, M_refine, endpoint=False)
    
    # Precompute basis for refinement?
    # For each candidate, we just compute P(z) on the fine grid.
    # P(z) = sum c_k z^k.
    # We can use np.polyval or direct sum.
    # Direct sum with vectorized exp is fast enough for 10 candidates.
    
    best_score_refined = np.inf
    best_coeffs = None
    
    for score, i, k in top_candidates[:5]: # Only refine top 5 to save time
        if time.time() - start_time > time_limit - 5:
            break
            
        # Reconstruct coeffs
        coeffs = np.empty(24, dtype=np.float64)
        for kk in range(half_N):
            bit_i = (i >> kk) & 1
            coeffs[kk] = 1 if bit_i == 0 else -1
            bit_k = (k >> kk) & 1
            coeffs[12 + kk] = 1 if bit_k == 0 else -1
            
        # Evaluate on fine grid
        # z = exp(i theta)
        # P(z) = sum coeffs[k] * z^k
        # We can compute powers of z
        # z_powers[j, k] = exp(i * k * theta_ref[j])
        # This is a (100000, 24) matrix.
        # 100000 * 24 * 16 bytes = 38 MB. OK.
        
        k_range_full = np.arange(24)
        # Compute powers
        # To save memory, we can compute in chunks or just use the fact that z^k = (z)^k
        # Let's compute z_ref = exp(1j * theta_ref)
        z_ref = np.exp(1j * theta_ref)
        
        # P_vals[j] = sum_k coeffs[k] * z_ref[j]^k
        # We can use Horner's method or just powers.
        # Powers:
        # z_powers = z_ref[:, None] ** k_range_full[None, :]
        # This might be slow due to power operation.
        # Better: iterative multiplication.
        # Or just use the basis matrix approach again but for this specific coeff vector.
        
        # Let's use the basis matrix approach for consistency and speed.
        # Basis_ref[k, j] = exp(i * k * theta_ref[j])
        # We can compute this as:
        # base = exp(i * theta_ref)
        # Basis_ref[0] = 1
        # Basis_ref[k] = Basis_ref[k-1] * base
        
        Basis_ref = np.empty((24, M_refine), dtype=np.complex128)
        Basis_ref[0, :] = 1.0
        base = np.exp(1j * theta_ref)
        for kk in range(1, 24):
            Basis_ref[kk, :] = Basis_ref[kk-1, :] * base
            
        P_vals = coeffs @ Basis_ref
        max_norm = np.max(np.abs(P_vals))
        
        if max_norm < best_score_refined:
            best_score_refined = max_norm
            best_coeffs = coeffs.astype(np.int8)
            
    if best_coeffs is None:
        # Fallback to the best from coarse search
        score, i, k = top_candidates[0]
        best_coeffs = np.empty(24, dtype=np.int8)
        for kk in range(half_N):
            bit_i = (i >> kk) & 1
            best_coeffs[kk] = 1 if bit_i == 0 else -1
            bit_k = (k >> kk) & 1
            best_coeffs[12 + kk] = 1 if bit_k == 0 else -1
    else:
        best_coeffs = best_coeffs.astype(np.int8)
        
    return best_coeffs.tolist()
