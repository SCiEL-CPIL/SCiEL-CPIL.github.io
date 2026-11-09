import numpy as np
import time

def design(problem):
    start_time = time.time()
    deadline = start_time + 210.0  # Safety margin before 230s kill

    size = problem["pixels"]
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)

    # --- Fast 289-PW evaluator (matches task: 17x17 PW grid) ---
    N_half = 8  # -8..8 => 17 values, 17*17=289 PWs
    G_vals = np.arange(-N_half, N_half + 1, dtype=np.int32)
    Gx, Gy = np.meshgrid(G_vals, G_vals, indexing='ij')
    Gx_flat = Gx.ravel()  # shape (289,)
    Gy_flat = Gy.ravel()

    # k-points in irreducible BZ: 30 points as per task
    # Boundary: Gamma-M (kx,0), M-X (0.5,ky), Gamma-X (kx,kx)
    # We'll use a dense set of 30 points
    k_points = []
    # Along Gamma-M: (t, 0) for t in [0, 0.5]
    for t in np.linspace(0, 0.5, 10):
        k_points.append((t, 0.0))
    # Along M-X: (0.5, t) for t in [0, 0.5]
    for t in np.linspace(0, 0.5, 10):
        k_points.append((0.5, t))
    # Along Gamma-X: (t, t) for t in [0, 0.5]
    for t in np.linspace(0, 0.5, 10):
        k_points.append((t, t))
    # Remove duplicates
    seen = set()
    unique_k = []
    for k in k_points:
        key = (round(k[0], 12), round(k[1], 12))
        if key not in seen:
            seen.add(key)
            unique_k.append(k)
    k_points = np.array(unique_k)  # shape (30, 2)

    # Precompute (k+G)^2 for all 30 k-points and 289 G-vectors
    # k_points: (30, 2), Gx_flat: (289,), Gy_flat: (289,)
    # omega_sq[k_idx, g_idx] = (kx+Gx)^2 + (ky+Gy)^2
    kx_all = k_points[:, 0:1]  # (30, 1)
    ky_all = k_points[:, 1:2]  # (30, 1)
    Gx_2d = Gx_flat[None, :]   # (1, 289)
    Gy_2d = Gy_flat[None, :]   # (1, 289)
    omega_sq = (kx_all + Gx_2d)**2 + (ky_all + Gy_2d)**2  # (30, 289)

    def build_coeff_matrix(inv_eps_grid):
        """Compute the 289x289 coefficient matrix from 1/epsilon grid."""
        # FFT of 1/eps, normalized
        fft_inv = np.fft.fft2(inv_eps_grid) / (size * size)
        
        # For each pair (i,j), coeff[i,j] = fft_inv[(Gx_i - Gx_j) % size, (Gy_i - Gy_j) % size]
        # Gx_flat and Gy_flat are the 289 G-vectors
        # dx = Gx_flat[i] - Gx_flat[j], dy = Gy_flat[i] - Gy_flat[j]
        # We need to compute this for all 289x289 pairs efficiently.
        
        # dx[i,j] = Gx_flat[i] - Gx_flat[j]
        dx = Gx_flat[:, None] - Gx_flat[None, :]  # (289, 289)
        dy = Gy_flat[:, None] - Gy_flat[None, :]  # (289, 289)
        
        # Wrap to [0, size)
        ix = dx % size
        iy = dy % size
        
        # Index into fft_inv
        coeff_matrix = fft_inv[ix, iy]  # (289, 289)
        return coeff_matrix

    def compute_gap(eps_grid):
        """Compute the gap-midgap ratio between bands 4 and 5."""
        inv_eps = 1.0 / eps_grid
        coeff_matrix = build_coeff_matrix(inv_eps)
        
        max_omega_4 = 0.0
        min_omega_5 = np.inf
        
        for k_idx in range(len(k_points)):
            # M = coeff_matrix * omega_sq[k_idx, :] (broadcast over columns)
            M = coeff_matrix * omega_sq[k_idx, :]
            
            # Solve eigenvalue problem: M v = omega^2 v
            # M is complex symmetric (not Hermitian), so use eigvals
            try:
                eigvals = np.linalg.eigvals(M)
            except np.linalg.LinAlgError:
                continue
            
            real_eigvals = np.sort(np.real(eigvals))
            
            if len(real_eigvals) >= 5:
                # Bands are indexed 1-based, so band 4 = index 3, band 5 = index 4
                w4_sq = real_eigvals[3]
                w5_sq = real_eigvals[4]
                
                if w4_sq > 0 and w5_sq > 0:
                    w4 = np.sqrt(w4_sq)
                    w5 = np.sqrt(w5_sq)
                    max_omega_4 = max(max_omega_4, w4)
                    min_omega_5 = min(min_omega_5, w5)
        
        if min_omega_5 == np.inf or max_omega_4 <= 0:
            return -1.0
        
        gap = min_omega_5 - max_omega_4
        midgap = 0.5 * (min_omega_5 + max_omega_4)
        if midgap <= 0:
            return -1.0
        ratio = 2.0 * gap / midgap
        return ratio

    # --- Quadrant-based binary search ---
    # The cell is 32x32, symmetric under C4v.
    # Fundamental domain: top-left 16x16 quadrant (i in [0,15], j in [0,15])
    # Actually, for 32x32 with symmetry about center, the fundamental is 16x16.
    # But we need to be careful: the symmetry is about the cell center.
    # cell[i][j] == cell[31-i][31-j] (180 rot)
    # cell[i][j] == cell[j][i] (transpose)
    # cell[i][j] == cell[31-j][i] (reflection)
    
    # Let's use the top-left 16x16 as the free parameters.
    quad_size = size // 2  # 16
    quad = np.zeros((quad_size, quad_size), dtype=np.int32)
    
    def expand_quad(q):
        """Expand 16x16 quadrant to full 32x32 symmetric cell."""
        full = np.zeros((size, size), dtype=np.int32)
        for i in range(quad_size):
            for j in range(quad_size):
                val = q[i, j]
                # All 8 symmetric positions
                full[i, j] = val
                full[size-1-i, j] = val
                full[i, size-1-j] = val
                full[size-1-i, size-1-j] = val
                full[j, i] = val
                full[size-1-j, i] = val
                full[j, size-1-i] = val
                full[size-1-j, size-1-i] = val
        return full

    def quad_to_eps(q):
        """Convert quadrant to full epsilon grid."""
        full_bin = expand_quad(q)
        return eps_low + (eps_high - eps_low) * full_bin

    # Initial guess: dielectric rod in center
    # In the 16x16 quadrant, the center of the full cell is at (15.5, 15.5)
    # So the center of the quadrant is at (15, 15)
    # A rod of radius ~4-5 pixels centered at (15,15) in the quadrant
    for i in range(quad_size):
        for j in range(quad_size):
            dx = i - (quad_size - 1)
            dy = j - (quad_size - 1)
            if dx*dx + dy*dy <= 16:  # radius 4
                quad[i, j] = 1

    # Evaluate initial
    eps_init = quad_to_eps(quad)
    best_score = compute_gap(eps_init)
    best_quad = quad.copy()

    # Hill climbing: flip single pixels in the quadrant
    num_quad_pixels = quad_size * quad_size
    
    # Precompute pixel indices for the quadrant
    pixel_indices = [(i, j) for i in range(quad_size) for j in range(quad_size)]
    
    # Keep track of recent evaluations to avoid redundant work
    # Use a simple approach: evaluate all 256 pixels each iteration
    
    iterations = 0
    while time.time() < deadline - 5:
        improved = False
        # Evaluate all possible single-pixel flips
        best_flip_score = best_score
        best_flip_idx = -1
        
        for idx in range(num_quad_pixels):
            i, j = pixel_indices[idx]
            # Flip this pixel
            quad[i, j] = 1 - quad[i, j]
            eps_grid = quad_to_eps(quad)
            score = compute_gap(eps_grid)
            quad[i, j] = 1 - quad[i, j]  # Undo
            
            if score > best_flip_score:
                best_flip_score = score
                best_flip_idx = idx
        
        if best_flip_idx >= 0:
            # Apply the best flip
            i, j = pixel_indices[best_flip_idx]
            best_quad[i, j] = 1 - best_quad[i, j]
            quad = best_quad.copy()
            best_score = best_flip_score
            improved = True
            iterations += 1
        else:
            # No improvement from single-pixel flips; try a random restart
            # Generate a new random quadrant with bias towards dielectric regions
            if time.time() < deadline - 10:
                # Random restart with some structure
                for _ in range(5):
                    random_quad = np.zeros((quad_size, quad_size), dtype=np.int32)
                    # Place a few dielectric blobs
                    for _ in range(np.random.randint(2, 5)):
                        cx = np.random.randint(8, 16)
                        cy = np.random.randint(8, 16)
                        r = np.random.randint(2, 5)
                        for di in range(-r, r+1):
                            for dj in range(-r, r+1):
                                ni, nj = cx+di, cy+dj
                                if 0 <= ni < quad_size and 0 <= nj < quad_size:
                                    if di*di + dj*dj <= r*r:
                                        random_quad[ni, nj] = 1
                    eps_grid = quad_to_eps(random_quad)
                    score = compute_gap(eps_grid)
                    if score > best_score:
                        best_score = score
                        best_quad = random_quad.copy()
                        quad = random_quad.copy()
                        improved = True
                        break
            if not improved:
                break  # Stuck, exit

    # Final: expand best quadrant to full cell
    full_bin = expand_quad(best_quad)
    
    # Convert to nested list
    result = []
    for i in range(size):
        row = []
        for j in range(size):
            row.append(int(full_bin[i, j]))
        result.append(row)
    
    return result
