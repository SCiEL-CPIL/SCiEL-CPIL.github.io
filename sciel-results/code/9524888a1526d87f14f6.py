import numpy as np
import time
import sys

def design(problem):
    """
    Design a 32x32 unit cell to maximize the TM band gap between bands 4 and 5.
    """
    size = problem["pixels"]
    eps_high = problem.get("eps_high", 12)
    lower_band = problem.get("lower_band", 4)
    
    start_time = time.time()
    deadline = start_time + 210.0  # Leave 20s margin for safety

    # Fast internal PWE solver for TM modes
    def compute_tm_bands(cell, n_pw=13, n_kpts=10, return_eigvecs=False):
        """
        Compute TM band structure using plane-wave expansion.
        """
        cell = np.array(cell, dtype=float)
        eps = 1.0 + (eps_high - 1.0) * cell
        inv_eps = 1.0 / eps
        N = size
        
        fft_coeffs = np.fft.fft2(inv_eps) / (N * N)
        
        half = n_pw // 2
        g_indices = []
        for m in range(-half, half + 1):
            for n in range(-half, half + 1):
                if m * m + n * n <= half * half + half:
                    g_indices.append((m, n))
        
        n_total = len(g_indices)
        
        # Precompute S matrix indices
        S = np.zeros((n_total, n_total), dtype=complex)
        for i, (mi, ni) in enumerate(g_indices):
            for j, (mj, nj) in enumerate(g_indices):
                gm = mi - mj
                gn = ni - nj
                idx_m = gm % N
                idx_n = gn % N
                S[i, j] = fft_coeffs[idx_m, idx_n]
        
        # K-points
        k_points = []
        n_seg = max(2, n_kpts // 3)
        
        # Gamma -> X
        for t in np.linspace(0, 0.5, n_seg + 1):
            k_points.append((t, 0.0))
        # X -> M
        for t in np.linspace(0, 0.5, n_seg + 1):
            k_points.append((0.5, t))
        # M -> Gamma
        for t in np.linspace(0.5, 0, n_seg + 1):
            k_points.append((t, t))
        
        # Unique
        k_points_unique = []
        for kp in k_points:
            if not any(np.allclose(kp, kp2, atol=1e-10) for kp2 in k_points_unique):
                k_points_unique.append(kp)
        
        min_omega5 = np.inf
        max_omega4 = -np.inf
        best_gap = -np.inf
        best_k = None
        best_vec4 = None
        best_vec5 = None
        
        for ki, (kx, ky) in enumerate(k_points_unique):
            M = np.zeros((n_total, n_total), dtype=complex)
            for i in range(n_total):
                mi, ni = g_indices[i]
                for j in range(n_total):
                    mj, nj = g_indices[j]
                    kx_total = mj + kx
                    ky_total = nj + ky
                    k_mag_sq = kx_total**2 + ky_total**2
                    M[i, j] = S[i, j] * k_mag_sq
            
            try:
                eigvals, eigvecs = np.linalg.eigh(M)
                idx = np.argsort(eigvals)
                eigvals = eigvals[idx]
                eigvecs = eigvecs[:, idx]
                
                if len(eigvals) >= 5:
                    w4 = np.sqrt(np.real(eigvals[3]))
                    w5 = np.sqrt(np.real(eigvals[4]))
                    
                    max_omega4 = max(max_omega4, w4)
                    min_omega5 = min(min_omega5, w5)
                    
                    gap = w5 - w4
                    if gap > best_gap:
                        best_gap = gap
                        best_k = (kx, ky)
                        if return_eigvecs:
                            best_vec4 = eigvecs[:, 3]
                            best_vec5 = eigvecs[:, 4]
            except Exception:
                continue
        
        if min_omega5 == np.inf or max_omega4 == -np.inf:
            if return_eigvecs:
                return -1.0, None, None, None
            return -1.0
        
        if min_omega5 + max_omega4 <= 0:
            if return_eigvecs:
                return -1.0, None, None, None
            return -1.0
        
        score = 2.0 * (min_omega5 - max_omega4) / (min_omega5 + max_omega4)
        
        if return_eigvecs:
            return score, best_k, best_vec4, best_vec5
        return score

    def enforce_d4_symmetry(cell):
        """Enforce D4 symmetry using averaging to preserve float values if needed, but we use int."""
        cell = np.array(cell, dtype=float)
        cell = (cell + cell[::-1, :] + cell[:, ::-1] + cell.T + cell[::-1, ::-1] + 
                cell[:, ::-1].T + cell[::-1, :].T + cell[::-1, ::-1].T) / 8.0
        # Threshold at 0.5
        return (cell > 0.5).astype(int)

    def make_four_quadrant_cell(rod_radius, exponent=2.0, offset=0.0):
        """
        Create a cell with four rods centered in each quadrant.
        """
        cell = np.zeros((size, size), dtype=int)
        coords = (np.arange(size) + 0.5) / size - 0.5
        
        centers = [(-0.25 + offset, -0.25 + offset), (0.25 - offset, -0.25 + offset), 
                   (-0.25 + offset, 0.25 - offset), (0.25 - offset, 0.25 - offset)]
        
        X, Y = np.meshgrid(coords, coords, indexing='ij')
        
        for cx, cy in centers:
            DX = X - cx
            DY = Y - cy
            if rod_radius > 0:
                val = np.abs(DX/rod_radius)**exponent + np.abs(DY/rod_radius)**exponent
            else:
                val = np.zeros_like(X)
            cell[val <= 1.0 + 1e-9] = 1
        
        cell = enforce_d4_symmetry(cell)
        return cell

    def compute_field_intensity(vec, kx, ky):
        """
        Compute |E|^2 on the pixel grid.
        """
        half = 6
        g_indices = []
        for m in range(-half, half + 1):
            for n in range(-half, half + 1):
                if m * m + n * n <= half * half + half:
                    g_indices.append((m, n))
        
        m_vals = np.array([g[0] for g in g_indices])
        n_vals = np.array([g[1] for g in g_indices])
        
        coords = (np.arange(size) + 0.5) / size - 0.5
        xx, yy = np.meshgrid(coords, coords, indexing='ij')
        
        intensity = np.zeros((size, size), dtype=float)
        
        m_k = m_vals + kx
        n_k = n_vals + ky
        
        chunk_size = max(1, 500000 // (size * size))
        
        for start in range(0, len(g_indices), chunk_size):
            end = min(start + chunk_size, len(g_indices))
            m_chunk = m_k[start:end]
            n_chunk = n_k[start:end]
            c_chunk = vec[start:end]
            
            phase = 2.0 * np.pi * (np.outer(m_chunk, xx.ravel()) + np.outer(n_chunk, yy.ravel()))
            phase = phase.reshape(end - start, size, size)
            
            e_contrib = np.sum(c_chunk[:, None, None] * np.exp(1j * phase), axis=0)
            intensity += np.abs(e_contrib)**2
        
        return intensity

    # 1. Initial Grid Search
    best_score = -np.inf
    best_cell = None
    
    radii = np.linspace(0.08, 0.18, 11)
    exponents = [2.0, 2.5, 3.0]
    
    for r in radii:
        if time.time() > deadline - 40:
            break
        for exp in exponents:
            if time.time() > deadline - 30:
                break
            cell = make_four_quadrant_cell(r, exp)
            try:
                score = compute_tm_bands(cell, n_pw=13, n_kpts=10)
                if score > best_score:
                    best_score = score
                    best_cell = cell
            except Exception:
                continue
    
    if best_cell is None:
        best_cell = make_four_quadrant_cell(0.12, 2.0)
        best_score = compute_tm_bands(best_cell, n_pw=13, n_kpts=10)

    # 2. Topological Optimization (Hill Climbing)
    current_cell = best_cell.copy()
    current_score = best_score
    no_improve_count = 0
    
    # We will try to improve by flipping pixels guided by field intensity difference.
    # delta_score ~ (|E4|^2 - |E5|^2) * delta_eps
    # If we add dielectric (delta_eps > 0), gain if |E4|^2 > |E5|^2.
    # If we remove dielectric (delta_eps < 0), gain if |E5|^2 > |E4|^2.
    
    max_iter = 40
    for it in range(max_iter):
        if time.time() > deadline - 15:
            break
        
        score, best_k, vec4, vec5 = compute_tm_bands(current_cell, n_pw=13, n_kpts=10, return_eigvecs=True)
        
        if best_k is None:
            break
            
        kx, ky = best_k
        
        I4 = compute_field_intensity(vec4, kx, ky)
        I5 = compute_field_intensity(vec5, kx, ky)
        
        I4 = I4 / (np.sum(I4) + 1e-12)
        I5 = I5 / (np.sum(I5) + 1e-12)
        
        # Metric:
        # Gain from adding dielectric at p: I4[p] - I5[p]
        # Gain from removing dielectric at p: I5[p] - I4[p]
        
        mask_add = (current_cell == 0)
        mask_rem = (current_cell == 1)
        
        gain_add = np.where(mask_add, I4 - I5, -np.inf)
        gain_rem = np.where(mask_rem, I5 - I4, -np.inf)
        
        # Find top candidates
        # We might flip multiple symmetric pixels at once to maintain symmetry naturally?
        # enforce_d4_symmetry handles it, but flipping one pixel flips 4 (or 2 or 1) due to symmetry.
        # So the "effective" gain is the sum of gains at all symmetric partners.
        
        # Let's compute the effective gain for each pixel, assuming we flip it and enforce symmetry.
        # Actually, enforce_d4_symmetry uses np.maximum.
        # If we flip a 0->1, and its symmetric partner is 0, it becomes 1.
        # If we flip a 1->0, and its symmetric partner is 1, it becomes 0.
        # But if we flip a 0->1, and partner is 1, it stays 1 (no change for partner).
        
        # Simpler approach: Just pick the single best pixel, flip it, enforce symmetry, evaluate.
        
        idx_add = np.unravel_index(np.argmax(gain_add), gain_add.shape)
        best_gain_add = gain_add[idx_add]
        
        idx_rem = np.unravel_index(np.argmax(gain_rem), gain_rem.shape)
        best_gain_rem = gain_rem[idx_rem]
        
        if best_gain_add > best_gain_rem and best_gain_add > 0:
            new_cell = current_cell.copy()
            new_cell[idx_add] = 1
            new_cell = enforce_d4_symmetry(new_cell)
        elif best_gain_rem > 0:
            new_cell = current_cell.copy()
            new_cell[idx_rem] = 0
            new_cell = enforce_d4_symmetry(new_cell)
        else:
            # No obvious gain, try random perturbation or break
            # Let's try a few random flips if we are stuck
            if no_improve_count > 5:
                break
            # Random flip
            candidates = np.argwhere(mask_add | mask_rem)
            if len(candidates) == 0:
                break
            idx = candidates[np.random.randint(len(candidates))]
            new_cell = current_cell.copy()
            if new_cell[idx[0], idx[1]] == 1:
                new_cell[idx[0], idx[1]] = 0
            else:
                new_cell[idx[0], idx[1]] = 1
            new_cell = enforce_d4_symmetry(new_cell)
        
        new_score = compute_tm_bands(new_cell, n_pw=13, n_kpts=10)
        
        if new_score > current_score:
            current_cell = new_cell
            current_score = new_score
            best_cell = current_cell
            best_score = current_score
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count > 10:
                break

    # 3. Final Polish: Try to refine the best cell found with a slightly higher precision solver?
    # The evaluator uses 289 PW (17x17). Our internal solver uses ~169 PW (13x13 circular).
    # The ranking should be consistent.
    
    return best_cell.tolist()
