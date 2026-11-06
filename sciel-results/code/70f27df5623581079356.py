import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    N = 24  # number of coefficients
    
    # =====================================================================
    # Phase 1: Exhaustive search with batched evaluation on a coarse grid
    # Fix c[0] = 1 (global sign symmetry), enumerate 2^23 candidates
    # =====================================================================
    M_screen = 512  # Coarse grid for fast screening
    angles = np.linspace(0, 2 * np.pi, M_screen, endpoint=False)
    z = np.exp(1j * angles)
    
    # Precompute Z[k, m] = z[m]^k
    Z = np.zeros((N, M_screen), dtype=complex)
    Z[0, :] = 1.0
    for k in range(1, N):
        Z[k, :] = Z[k-1, :] * z
    
    # base = sum of all z^k (all coefficients +1)
    base = np.sum(Z, axis=0)  # shape (M_screen,)
    Z_flip = Z[1:N]  # shape (23, M_screen)
    
    total_candidates = 1 << 23  # 8,388,608
    B = 4096  # Batch size
    
    best_score = np.inf
    best_idx = 0
    best_coeffs = np.ones(N)
    
    # Track top few candidates for refinement
    top_scores = []  # list of (score, idx)
    
    for b in range((total_candidates + B - 1) // B):
        if time.time() > deadline - 20.0:
            break
        
        idx_start = b * B
        idx_end = min(idx_start + B, total_candidates)
        B_actual = idx_end - idx_start
        
        indices = np.arange(idx_start, idx_end, dtype=np.int64)
        
        # Build flip matrix: (B_actual, 23)
        flips = np.empty((B_actual, 23), dtype=np.float64)
        for i in range(23):
            flips[:, i] = (indices >> i) & 1
        
        # batch_P = base[None,:] - 2 * (flips @ Z_flip)
        # (B_actual, 23) @ (23, M_screen) = (B_actual, M_screen)
        batch_P = base[None, :] - 2.0 * (flips @ Z_flip)
        
        scores = np.max(np.abs(batch_P), axis=1)
        
        # Find best in batch
        local_best_idx = np.argmin(scores)
        local_best_score = scores[local_best_idx]
        
        if local_best_score < best_score:
            best_score = local_best_score
            best_idx = idx_start + local_best_idx
        
        # Keep top candidates (we'll refine later)
        # For memory, just track the global best and a few runners-up
        if len(top_scores) < 50:
            top_scores.append((local_best_score, idx_start + local_best_idx))
            top_scores.sort()
            if len(top_scores) > 50:
                top_scores.pop()
        else:
            if local_best_score < top_scores[-1][0]:
                top_scores[-1] = (local_best_score, idx_start + local_best_idx)
                top_scores.sort()
    
    # Reconstruct best coefficients from best_idx
    best_coeffs = np.ones(N)
    for i in range(23):
        if (best_idx >> i) & 1:
            best_coeffs[i + 1] = -1.0
    
    # =====================================================================
    # Phase 2: Refine with a finer grid
    # =====================================================================
    M_fine = 4096
    angles_fine = np.linspace(0, 2 * np.pi, M_fine, endpoint=False)
    z_fine = np.exp(1j * angles_fine)
    Z_fine = np.zeros((N, M_fine), dtype=complex)
    Z_fine[0, :] = 1.0
    for k in range(1, N):
        Z_fine[k, :] = Z_fine[k-1, :] * z_fine
    
    def get_peak_fine(c):
        vals = Z_fine.T @ c
        return np.max(np.abs(vals))
    
    # Re-evaluate top candidates on fine grid
    refined_candidates = []
    for score_coarse, idx_cand in top_scores:
        coeffs_cand = np.ones(N)
        for i in range(23):
            if (idx_cand >> i) & 1:
                coeffs_cand[i + 1] = -1.0
        score_fine = get_peak_fine(coeffs_cand)
        refined_candidates.append((score_fine, coeffs_cand))
    
    refined_candidates.sort()
    best_coeffs = refined_candidates[0][1].copy()
    best_fine_score = refined_candidates[0][0]
    
    # =====================================================================
    # Phase 3: Local search (single-bit flips) from best candidates
    # =====================================================================
    def local_search(coeffs, current_peak):
        improved = True
        while improved:
            if time.time() > deadline - 10.0:
                break
            improved = False
            for i in range(N):
                coeffs[i] *= -1
                new_peak = get_peak_fine(coeffs)
                if new_peak < current_peak:
                    current_peak = new_peak
                    improved = True
                else:
                    coeffs[i] *= -1  # flip back
        return current_peak
    
    # Run local search from top refined candidates
    for score_c, coeffs_c in refined_candidates[:5]:
        if time.time() > deadline - 15.0:
            break
        coeffs_work = coeffs_c.copy()
        score_ls = local_search(coeffs_work, score_c)
        if score_ls < best_fine_score:
            best_fine_score = score_ls
            best_coeffs = coeffs_work.copy()
    
    # =====================================================================
    # Phase 4: Try 2-bit flip moves to escape local minima
    # =====================================================================
    if time.time() < deadline - 15.0:
        coeffs_work = best_coeffs.copy()
        current_peak = best_fine_score
        improved = True
        while improved:
            if time.time() > deadline - 10.0:
                break
            improved = False
            for i in range(N):
                for j in range(i + 1, N):
                    if time.time() > deadline - 10.0:
                        break
                    coeffs_work[i] *= -1
                    coeffs_work[j] *= -1
                    new_peak = get_peak_fine(coeffs_work)
                    if new_peak < current_peak:
                        current_peak = new_peak
                        improved = True
                    else:
                        coeffs_work[i] *= -1
                        coeffs_work[j] *= -1
                if time.time() > deadline - 10.0:
                    break
            if time.time() > deadline - 10.0:
                break
        
        if current_peak < best_fine_score:
            best_fine_score = current_peak
            best_coeffs = coeffs_work.copy()
    
    # =====================================================================
    # Phase 5: Golden-section refinement of the peak for accurate scoring
    # (Not strictly needed for returning coefficients, but good practice)
    # =====================================================================
    # The evaluator will compute the score; we just need to return coefficients.
    # But let's do a final check that we haven't regressed.
    
    # Convert to integers
    return [1 if c > 0 else -1 for c in best_coeffs]
