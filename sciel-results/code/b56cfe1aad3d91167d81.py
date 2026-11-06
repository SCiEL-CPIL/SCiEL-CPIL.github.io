import numpy as np
import time
import sys

# EVOLVE-BLOCK-START
def design(problem):
    """
    Design a 32x32 unit cell to maximize the TM band gap between bands 4 and 5.
    
    Strategy:
    1. Use a fast internal PWE solver with vectorized S-matrix construction.
    2. Generate diverse candidate cells: four-quadrant rods, central ring, staggered rods.
    3. Enforce exact D4 symmetry by constructing from a single quadrant and mirroring.
    4. Perform multi-round local search with smooth surrogate objective.
    5. Final high-precision polish pass with more k-points.
    6. Maintain strict time budget with early termination.
    """
    size = problem["pixels"]
    eps_high = problem.get("eps_high", 12)
    lower_band = problem.get("lower_band", 4)
    
    # Time budget: 230s limit, keep 15s margin for overhead
    start_time = time.time()
    deadline = start_time + 215.0
    
    def time_left():
        return deadline - time.time()
    
    # Fast internal PWE solver for TM modes with vectorized S-matrix
    def compute_tm_bands(cell, n_pw=13, n_kpts=8):
        """
        Compute TM band structure using plane-wave expansion.
        Uses vectorized S-matrix construction for speed.
        """
        cell = np.array(cell, dtype=float)
        eps = 1.0 + (eps_high - 1.0) * cell
        inv_eps = 1.0 / eps
        
        # Compute Fourier coefficients of 1/epsilon using FFT
        fft_coeffs = np.fft.fft2(inv_eps) / (size * size)
        
        # Define the set of G vectors to include in the basis
        M = n_pw // 2
        g_indices = []
        for m in range(-M, M + 1):
            for n in range(-M, M + 1):
                if m*m + n*n <= (M + 1)*(M + 1):
                    g_indices.append((m, n))
        
        n_total = len(g_indices)
        g_array = np.array(g_indices, dtype=int)  # shape (n_total, 2)
        
        # Vectorized S-matrix construction
        # S[i,j] = fft_coeffs[(g_i - g_j) mod size]
        # Compute all differences at once
        diff_m = g_array[:, 0:1] - g_array[1:, 0:1].T  # This won't work directly
        # Instead: for each pair (i,j), compute (mi-mj, ni-nj)
        diff_m = g_array[:, 0].reshape(-1, 1) - g_array[:, 0].reshape(1, -1)
        diff_n = g_array[:, 1].reshape(-1, 1) - g_array[:, 1].reshape(1, -1)
        
        # Index into fft_coeffs with wrapping
        idx_m = diff_m % size
        idx_n = diff_n % size
        S = fft_coeffs[idx_m, idx_n]
        
        # Define k-points in the irreducible Brillouin zone
        k_points = []
        n_seg = max(3, n_kpts // 3)
        for t in np.linspace(0, 0.5, n_seg):
            k_points.append((t, 0))
        for t in np.linspace(0, 0.5, n_seg):
            k_points.append((0.5, t))
        for t in np.linspace(0.5, 0, n_seg):
            k_points.append((t, t))
        
        # Remove duplicates
        k_points_unique = []
        for kp in k_points:
            if not any(np.allclose(kp, kp2, atol=1e-10) for kp2 in k_points_unique):
                k_points_unique.append(kp)
        
        n_kpts_actual = len(k_points_unique)
        omega = np.zeros((n_kpts_actual, n_total))
        
        # Pre-compute G vectors
        G_vecs = np.array([(2.0 * np.pi * m, 2.0 * np.pi * n) for m, n in g_indices])
        
        for ki, (kx, ky) in enumerate(k_points_unique):
            Gk_x = G_vecs[:, 0] + 2.0 * np.pi * kx
            Gk_y = G_vecs[:, 1] + 2.0 * np.pi * ky
            k_mag_sq = Gk_x**2 + Gk_y**2
            
            M_mat = S * k_mag_sq[np.newaxis, :]
            
            try:
                eigvals = np.linalg.eigvalsh(M_mat)
                eigvals = np.sort(np.real(eigvals))
                eigvals = np.maximum(eigvals, 0)
                omega[ki, :] = np.sqrt(eigvals)
            except Exception:
                omega[ki, :] = np.inf
        
        return omega

    # Enforce exact D4 symmetry by constructing from top-left quadrant
    def enforce_d4_symmetry(cell):
        """
        Enforce D4 symmetry: cell == cell[::-1] row-wise and column-wise, and cell == its transpose.
        We take the top-left quadrant and mirror it.
        """
        cell = np.array(cell, dtype=int)
        half = size // 2
        tl = cell[:half, :half].copy()
        cell_new = np.zeros((size, size), dtype=int)
        cell_new[:half, :half] = tl
        cell_new[:half, half:] = tl[:, ::-1]
        cell_new[half:, :half] = tl[::-1, :]
        cell_new[half:, half:] = tl[::-1, ::-1]
        return cell_new
    
    def quadrant_to_full(tl):
        """Convert top-left quadrant to full D4-symmetric cell."""
        half = size // 2
        cell = np.zeros((size, size), dtype=int)
        cell[:half, :half] = tl
        cell[:half, half:] = tl[:, ::-1]
        cell[half:, :half] = tl[::-1, :]
        cell[half:, half:] = tl[::-1, ::-1]
        return cell

    def make_four_quadrant_cell(rod_radius, exponent=2.0):
        """
        Create a cell with four rods centered in each quadrant.
        Enforces D4 symmetry.
        """
        cell = np.zeros((size, size), dtype=int)
        coords = (np.arange(size) + 0.5) / size - 0.5
        
        # Rod centers in normalized coordinates [-0.5, 0.5]
        centers = [(-0.25, -0.25), (0.25, -0.25), (-0.25, 0.25), (0.25, 0.25)]
        
        X, Y = np.meshgrid(coords, coords, indexing='ij')
        
        for cx, cy in centers:
            DX = X - cx
            DY = Y - cy
            if rod_radius > 0:
                val = np.abs(DX/rod_radius)**exponent + np.abs(DY/rod_radius)**exponent
            else:
                val = np.zeros_like(X)
            cell[val <= 1.0 + 1e-9] = 1
        
        return enforce_d4_symmetry(cell).tolist()

    def make_central_ring_cell(inner_radius, outer_radius, exponent=2.0):
        """
        Create a cell with a central ring (annulus).
        Enforces D4 symmetry.
        """
        cell = np.zeros((size, size), dtype=int)
        coords = (np.arange(size) + 0.5) / size - 0.5
        X, Y = np.meshgrid(coords, coords, indexing='ij')
        
        R = np.sqrt(X**2 + Y**2)
        
        # Ring: inner_radius <= R <= outer_radius
        if outer_radius > 0 and inner_radius >= 0:
            val = np.abs(R/outer_radius)**exponent - np.abs(inner_radius/outer_radius)**exponent
            cell[val >= 0] = 1
        
        return enforce_d4_symmetry(cell).tolist()

    def make_staggered_rods_cell(rod_radius, exponent=2.0, offset=0.15):
        """
        Create a cell with staggered rods.
        Enforces D4 symmetry.
        """
        cell = np.zeros((size, size), dtype=int)
        coords = (np.arange(size) + 0.5) / size - 0.5
        X, Y = np.meshgrid(coords, coords, indexing='ij')
        
        # Rods at positions that maintain D4 symmetry
        # Centers: (±offset, ±offset) and (±(0.5-offset), ±(0.5-offset))
        centers = []
        for sx in [-1, 1]:
            for sy in [-1, 1]:
                centers.append((sx*offset, sy*offset))
                centers.append((sx*(0.5-offset), sy*(0.5-offset)))
        
        for cx, cy in centers:
            DX = X - cx
            DY = Y - cy
            if rod_radius > 0:
                val = np.abs(DX/rod_radius)**exponent + np.abs(DY/rod_radius)**exponent
            else:
                val = np.zeros_like(X)
            cell[val <= 1.0 + 1e-9] = 1
        
        return enforce_d4_symmetry(cell).tolist()

    def score_cell(cell, n_pw=13, n_kpts=8):
        """
        Score a cell by computing the gap-midgap ratio between bands 4 and 5.
        Uses exact min/max for the final metric.
        """
        try:
            bands = compute_tm_bands(cell, n_pw=n_pw, n_kpts=n_kpts)
            omega_4 = bands[:, 3]
            omega_5 = bands[:, 4]
            
            max_omega_4 = np.max(omega_4)
            min_omega_5 = np.min(omega_5)
            
            if min_omega_5 > max_omega_4:
                gap = min_omega_5 - max_omega_4
                ratio = 2.0 * gap / (min_omega_5 + max_omega_4)
            else:
                overlap = max_omega_4 - min_omega_5
                midgap = 0.5 * (min_omega_5 + max_omega_4)
                ratio = -2.0 * overlap / midgap if midgap > 0 else -1.0
            return ratio
        except Exception:
            return -1.0
    
    def smooth_score_cell(cell, n_pw=13, n_kpts=8, beta=50.0):
        """
        Smooth surrogate for the gap-midgap ratio using soft-min/soft-max.
        beta controls smoothness: larger beta -> closer to exact min/max.
        """
        try:
            bands = compute_tm_bands(cell, n_pw=n_pw, n_kpts=n_kpts)
            omega_4 = bands[:, 3]
            omega_5 = bands[:, 4]
            
            # Soft max of omega_4: log-sum-exp
            max_omega_4 = np.max(omega_4)
            soft_max_4 = max_omega_4 + np.log(np.sum(np.exp(omega_4 - max_omega_4) * beta)) / beta
            
            # Soft min of omega_5: -soft_max(-omega_5)
            min_omega_5 = np.min(omega_5)
            soft_min_5 = min_omega_5 - np.log(np.sum(np.exp(-(omega_5 - min_omega_5)) * beta)) / beta
            
            gap = soft_min_5 - soft_max_4
            midgap = 0.5 * (soft_min_5 + soft_max_4)
            
            if midgap > 1e-10:
                ratio = 2.0 * gap / midgap
            else:
                ratio = -1.0
            return ratio
        except Exception:
            return -1.0

    # Generate and evaluate candidates
    best_score = -np.inf
    best_cell = None
    
    # Phase 1: Grid search over four-quadrant rods - expanded range
    radii = np.linspace(0.06, 0.20, 15)
    exponents = [1.5, 2.0, 2.5, 3.0, 4.0, 6.0]
    
    for r in radii:
        if time_left() < 15:
            break
        for exp in exponents:
            if time_left() < 5:
                break
            cell = make_four_quadrant_cell(r, exp)
            score = score_cell(cell)
            if score > best_score:
                best_score = score
                best_cell = cell
    
    # Phase 2: Try central rings - expanded range
    if time_left() > 15:
        inner_radii = np.linspace(0.03, 0.18, 9)
        outer_radii = np.linspace(0.12, 0.28, 9)
        for ir in inner_radii:
            if time_left() < 10:
                break
            for orad in outer_radii:
                if time_left() < 5:
                    break
                if orad <= ir:
                    continue
                cell = make_central_ring_cell(ir, orad)
                score = score_cell(cell)
                if score > best_score:
                    best_score = score
                    best_cell = cell
    
    # Phase 3: Try staggered rods - expanded range
    if time_left() > 15:
        radii_s = np.linspace(0.04, 0.16, 11)
        offsets = np.linspace(0.08, 0.22, 9)
        for r in radii_s:
            if time_left() < 10:
                break
            for off in offsets:
                if time_left() < 5:
                    break
                cell = make_staggered_rods_cell(r, 2.0, off)
                score = score_cell(cell)
                if score > best_score:
                    best_score = score
                    best_cell = cell
    
    # Phase 4: Local refinement using smooth surrogate for faster convergence
    if best_cell is not None and time_left() > 25:
        best_arr = np.array(best_cell, dtype=int)
        half = size // 2
        tl_best = best_arr[:half, :half].copy()
        
        # Build candidate list: boundary pixels first, then all
        boundary_candidates = []
        all_pixels = []
        for i in range(half):
            for j in range(half):
                all_pixels.append((i, j))
                neighbors = []
                for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
                    ni, nj = i+di, j+dj
                    if 0 <= ni < half and 0 <= nj < half:
                        neighbors.append(tl_best[ni, nj])
                if neighbors and tl_best[i, j] != np.mean(neighbors):
                    boundary_candidates.append((i, j))
        
        # Use smooth surrogate for local search (faster to evaluate, smoother landscape)
        np.random.seed(42)
        
        # Round 1: Single pixel flips on boundary + random
        search_list = boundary_candidates + all_pixels
        np.random.shuffle(search_list)
        
        n_evals = 0
        for (i, j) in search_list:
            if time_left() < 10:
                break
            if n_evals >= 300:
                break
            tl_new = tl_best.copy()
            tl_new[i, j] = 1 - tl_new[i, j]
            cell_new = quadrant_to_full(tl_new)
            
            score = smooth_score_cell(cell_new.tolist(), beta=50.0)
            n_evals += 1
            if score > best_score:
                # Verify with exact score before accepting
                exact_score = score_cell(cell_new.tolist())
                if exact_score > best_score:
                    best_score = exact_score
                    best_cell = cell_new.tolist()
                    tl_best = cell_new[:half, :half].copy()
        
        # Round 2: Multi-pixel flips (2-3 pixels at a time)
        if time_left() > 15:
            np.random.seed(123)
            for _ in range(200):
                if time_left() < 8:
                    break
                # Pick 2-3 random pixels to flip
                n_flip = np.random.choice([2, 3])
                flips = np.random.choice(len(all_pixels), size=n_flip, replace=False)
                tl_new = tl_best.copy()
                for f in flips:
                    i, j = all_pixels[f]
                    tl_new[i, j] = 1 - tl_new[i, j]
                cell_new = quadrant_to_full(tl_new)
                
                score = smooth_score_cell(cell_new.tolist(), beta=50.0)
                if score > best_score:
                    exact_score = score_cell(cell_new.tolist())
                    if exact_score > best_score:
                        best_score = exact_score
                        best_cell = cell_new.tolist()
                        tl_best = cell_new[:half, :half].copy()
        
        # Round 3: Focused local search with tighter acceptance
        if time_left() > 10:
            # Rebuild boundary candidates from current best
            boundary2 = []
            for i in range(half):
                for j in range(half):
                    neighbors = []
                    for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
                        ni, nj = i+di, j+dj
                        if 0 <= ni < half and 0 <= nj < half:
                            neighbors.append(tl_best[ni, nj])
                    if neighbors and tl_best[i, j] != np.mean(neighbors):
                        boundary2.append((i, j))
            
            np.random.seed(456)
            np.random.shuffle(boundary2)
            
            for (i, j) in boundary2:
                if time_left() < 5:
                    break
                tl_new = tl_best.copy()
                tl_new[i, j] = 1 - tl_new[i, j]
                cell_new = quadrant_to_full(tl_new)
                
                score = smooth_score_cell(cell_new.tolist(), beta=100.0)
                if score > best_score - 0.01:  # tight acceptance
                    exact_score = score_cell(cell_new.tolist())
                    if exact_score > best_score:
                        best_score = exact_score
                        best_cell = cell_new.tolist()
                        tl_best = cell_new[:half, :half].copy()
    
    # If no cell was found, return a default
    if best_cell is None:
        best_cell = make_four_quadrant_cell(0.12, 2.0)
    
    # Final high-precision polish: re-evaluate with more k-points to find true optimum
    if time_left() > 20:
        # Try a few variations around the best with higher precision
        best_arr = np.array(best_cell, dtype=int)
        half = size // 2
        tl_best = best_arr[:half, :half].copy()
        
        # Fine-tune: try flipping each boundary pixel with high-precision scoring
        boundary_final = []
        for i in range(half):
            for j in range(half):
                neighbors = []
                for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
                    ni, nj = i+di, j+dj
                    if 0 <= ni < half and 0 <= nj < half:
                        neighbors.append(tl_best[ni, nj])
                if neighbors and tl_best[i, j] != np.mean(neighbors):
                    boundary_final.append((i, j))
        
        for (i, j) in boundary_final:
            if time_left() < 8:
                break
            tl_new = tl_best.copy()
            tl_new[i, j] = 1 - tl_new[i, j]
            cell_new = quadrant_to_full(tl_new)
            
            # Use more k-points for precision
            score = score_cell(cell_new.tolist(), n_pw=13, n_kpts=12)
            if score > best_score:
                best_score = score
                best_cell = cell_new.tolist()
                tl_best = cell_new[:half, :half].copy()
    
    # Final validation: ensure D4 symmetry
    best_arr = np.array(best_cell, dtype=int)
    best_arr = enforce_d4_symmetry(best_arr)
    
    return best_arr.tolist()

# EVOLVE-BLOCK-END
