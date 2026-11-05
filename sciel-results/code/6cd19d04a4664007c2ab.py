import time
import math
import numpy as np

def design(problem):
    start_time = time.time()
    deadline = start_time + 220  # 10s margin before the 230s kill
    
    size = problem["pixels"]  # 32
    eps_low = problem["eps_low"]  # 1
    eps_high = problem["eps_high"]  # 12
    
    # -----------------------------------------------------------------------
    # Fast TM PWE solver
    # -----------------------------------------------------------------------
    N_pw = 13  # 13x13 = 169 plane waves, good balance of speed and accuracy
    half_N = N_pw // 2
    
    # Generate plane wave index pairs
    pw_indices = []
    for nx in range(-half_N, half_N + 1):
        for ny in range(-half_N, half_N + 1):
            pw_indices.append((nx, ny))
    pw_indices = np.array(pw_indices)
    N_actual = len(pw_indices)
    
    # Sample k-points in irreducible BZ: (0,0) -> (0.5,0) -> (0.5,0.5)
    # Add some interior points for better gap estimation
    k_points = [
        (0.0, 0.0),
        (0.1, 0.0),
        (0.2, 0.0),
        (0.3, 0.0),
        (0.4, 0.0),
        (0.5, 0.0),
        (0.5, 0.1),
        (0.5, 0.2),
        (0.5, 0.3),
        (0.5, 0.4),
        (0.5, 0.5),
        (0.25, 0.25),
        (0.35, 0.15),
        (0.45, 0.25),
    ]
    
    def compute_fourier_inv_eps(cell):
        """Compute Fourier coefficients of 1/eps using FFT."""
        eps = np.where(cell == 1, eps_high, eps_low)
        inv_eps = 1.0 / eps
        fft_inv_eps = np.fft.fft2(inv_eps)
        fft_inv_eps /= (size * size)
        fft_inv_eps = np.fft.fftshift(fft_inv_eps)
        return fft_inv_eps
    
    def solve_bands_fast(cell, kx, ky, num_bands=5):
        """Solve for the first num_bands TM bands at a given k-point."""
        fft_inv_eps = compute_fourier_inv_eps(cell)
        
        # k-vectors in units of 2pi/a
        k_vecs = pw_indices + np.array([kx, ky])
        
        # Build the PWE matrix H
        # For TM: H_ij = (1/eps)_{k_i - k_j} * |k_i - k_j|^2
        # Actually, let me reconsider the TM formulation.
        # Standard TM: the eigenvalue equation is
        # sum_j (1/eps)_{k_i-k_j} |k_j|^2 c_j = omega^2 c_i  -- No
        # Let me use the standard form from Joannopoulos:
        # For TM polarization, the equation is:
        # sum_q (1/eps)_q (k_i - q)^2 c_{k_i-q} = omega^2 c_{k_i}
        # In matrix form with c indexed by plane waves:
        # H_ij = (1/eps)_{k_i - k_j} * |k_i - k_j|^2
        # Wait, that's not quite right either. Let me be careful.
        
        # The correct TM PWE matrix element is:
        # H_ij = (1/eps)_{k_i - k_j} * |k_i - k_j|^2
        # This makes H Hermitian since (1/eps)_{-q} = conj((1/eps)_q) for real eps.
        
        # Precompute k_i - k_j differences
        # k_i - k_j = (nx_i - nx_j, ny_i - ny_j) since kx, ky cancel
        k_diffs = np.zeros((N_actual, N_actual, 2))
        for i in range(N_actual):
            for j in range(N_actual):
                k_diffs[i, j, 0] = k_vecs[i, 0] - k_vecs[j, 0]
                k_diffs[i, j, 1] = k_vecs[i, 1] - k_vecs[j, 1]
        
        k_diff_sq = np.sum(k_diffs**2, axis=2)
        
        # Look up (1/eps)_q for each q = k_i - k_j
        # q is an integer vector (nx_i - nx_j, ny_i - ny_j)
        # fft_inv_eps is fftshifted, so index (0,0) is at (half_N, half_N)
        # But we need to handle the range properly.
        # Our FFT grid covers -15..15 (for size=32, but we only use -13..13 for N_pw=13)
        # Actually the FFT gives us all 32x32 frequencies.
        # The integer q_x ranges from -31 to 31.
        # After fftshift, index 0 corresponds to q_x = -16, index 15 to q_x = -1, etc.
        # Wait, for size=32, fftshift moves the zero frequency to index 16.
        # So index i corresponds to q = i - 16 for i=0..31.
        
        # Let me redo this more carefully.
        # np.fft.fft2 gives DFT: F[m,n] = sum_{p,q} f[p,q] exp(-2pi i (mp/N + nq/N))
        # After fftshift, F_shifted[m,n] = F[(m+16)%32, (n+16)%32]
        # So F_shifted[m,n] corresponds to frequency (m-16, n-16) for m,n in 0..31.
        
        # We need (1/eps)_q where q = (qx, qy) is integer.
        # The index in the shifted array is (qx + 16, qy + 16) mod 32.
        
        H = np.zeros((N_actual, N_actual), dtype=np.float64)
        for i in range(N_actual):
            for j in range(N_actual):
                qx = int(round(k_diffs[i, j, 0]))
                qy = int(round(k_diffs[i, j, 1]))
                # Wrap to FFT range
                idx_x = (qx + 16) % 32
                idx_y = (qy + 16) % 32
                coeff = fft_inv_eps[idx_y, idx_x].real  # Should be real for symmetric eps
                H[i, j] = coeff * k_diff_sq[i, j]
        
        # Solve eigenvalue problem
        vals, vecs = np.linalg.eigh(H)
        
        # Filter out very small eigenvalues (numerical artifacts near zero)
        mask = vals > 1e-10
        vals = vals[mask]
        vecs = vecs[:, mask]
        
        if len(vals) < num_bands:
            return None, None
        
        return vals[:num_bands], vecs[:, :num_bands]
    
    def evaluate_gap(cell):
        """Evaluate the gap-midgap ratio between bands 4 and 5."""
        max_omega_4 = 0.0
        min_omega_5 = float('inf')
        
        for kx, ky in k_points:
            vals, _ = solve_bands_fast(cell, kx, ky, num_bands=5)
            if vals is None:
                continue
            omegas = np.sqrt(vals)
            if omegas[3] > max_omega_4:
                max_omega_4 = omegas[3]
            if omegas[4] < min_omega_5:
                min_omega_5 = omegas[4]
        
        if min_omega_5 == float('inf') or max_omega_4 == 0.0:
            return -1.0
        
        gap = min_omega_5 - max_omega_4
        midgap = (min_omega_5 + max_omega_4) / 2.0
        if midgap < 1e-10:
            return 0.0
        return gap / midgap
    
    # -----------------------------------------------------------------------
    # Symmetric cell construction from 8x8 fundamental domain
    # -----------------------------------------------------------------------
    def expand_symmetric(fund):
        """Expand an 8x8 fundamental domain to a full 32x32 C4v-symmetric cell."""
        # fund is 8x8 representing the top-left quadrant
        # The full 32x32 cell has 16x16 quadrant (since 32/2=16)
        # Wait, the cell is 32x32. The symmetry is about the center.
        # C4v symmetry: invariant under 90-degree rotations and reflections.
        # The fundamental domain is 8x8 (top-left 1/4 of the 16x16 half-quadrant? No.)
        # For a 32x32 cell with C4v symmetry about the center:
        # The fundamental domain is 8x8 pixels (top-left corner of the quadrant).
        # Actually, for a 32x32 grid, the center is between pixels.
        # The four quadrants are 16x16 each.
        # Under C4v, the fundamental domain is 8x8 (the 1/4 of a 16x16 quadrant? No.)
        # Let me think again.
        # 32x32 cell. Symmetry about the center point (16, 16) in continuous coordinates.
        # The pixel centers are at (i+0.5, j+0.5) for i,j = 0..31.
        # The center of the cell is at (16, 16) in pixel-center coordinates.
        # Reflection about x=16: pixel i maps to pixel 31-i.
        # Reflection about y=16: pixel j maps to pixel 31-j.
        # 90-degree rotation: (i,j) -> (j, 31-i) or similar.
        
        # The fundamental domain for C4v on a 32x32 grid:
        # We can define it as the top-left 8x8 block, then mirror.
        # Actually, the full symmetry group has 8 elements.
        # The orbit of a pixel (i,j) under C4v has up to 8 pixels.
        # For a 32x32 grid, the fundamental domain is 8x8 = 64 pixels.
        # Wait, 32*32/8 = 128. So the fundamental domain should be 128 pixels.
        # Hmm, let me reconsider.
        
        # For C4v symmetry on an NxN grid where N is divisible by 4:
        # The fundamental domain is (N/4) x (N/4) if we also account for the diagonal.
        # Actually, for N=32, N/4 = 8. The fundamental domain is 8x8 = 64 pixels.
        # But 64*8 = 512, not 1024. So some pixels are on symmetry axes.
        
        # Let me just use a different approach: define the top-left 16x16 quadrant,
        # then reflect to get the full 32x32 cell.
        # The 16x16 quadrant itself needs to be symmetric under the diagonal? No.
        # For C4v, the 16x16 quadrant (top-left) is the fundamental domain if we
        # also require that it's symmetric under the main diagonal (i<->j).
        # So the fundamental domain is the upper triangle of the 16x16 quadrant, which is 136 pixels.
        
        # This is getting complicated. Let me use a simpler approach:
        # Define the top-left 8x8 block, then construct the full cell by reflections.
        
        # For a 32x32 cell with C4v symmetry:
        # Let's define the top-left 8x8 as the fundamental domain.
        # Then:
        # - Top-left 16x16 quadrant: mirror 8x8 horizontally to get 8x16, then vertically to get 16x16.
        #   Wait, that's not quite right.
        
        # Let me use explicit pixel mapping.
        # The C4v group acting on pixel (i, j) in a 32x32 grid:
        # 1. Identity: (i, j)
        # 2. Reflection about x=16 (vertical mirror): (31-i, j)
        # 3. Reflection about y=16 (horizontal mirror): (i, 31-j)
        # 4. 180-degree rotation: (31-i, 31-j)
        # 5. 90-degree CCW rotation: (j, 31-i)
        # 6. 90-degree CW rotation: (31-j, i)
        # 7. Reflection about main diagonal: (j, i)
        # 8. Reflection about anti-diagonal: (31-j, 31-i)
        
        # Wait, I need to be careful. Let me define the operations properly.
        # For a pixel at position (row, col) in a 32x32 array:
        # - Identity: (r, c)
        # - H-flip (about horizontal axis through center): (31-r, c)
        # - V-flip (about vertical axis through center): (r, 31-c)
        # - 180 rot: (31-r, 31-c)
        # - 90 CW rot: (c, 31-r)
        # - 90 CCW rot: (31-c, r)
        # - Main diag transpose: (c, r)
        # - Anti-diag transpose: (31-c, 31-r) -- same as 180 rot
        
        # Hmm, some of these are the same. Let me list distinct operations:
        # 1. (r, c)
        # 2. (31-r, c) -- H-flip
        # 3. (r, 31-c) -- V-flip
        # 4. (31-r, 31-c) -- 180 rot = H-flip + V-flip
        # 5. (c, r) -- main diagonal transpose
        # 6. (31-c, r) -- 90 CCW
        # 7. (c, 31-r) -- 90 CW
        # 8. (31-c, 31-r) -- anti-diagonal = same as 180 rot
        
        # So we have 8 distinct operations, but #4 and #8 are the same.
        # Actually, for C4v, we have: E, C4, C2, C4^3, sigma_v(x), sigma_v(y), sigma_d(1), sigma_d(2)
        # That's 8 elements.
        
        # Let me just use the approach: define top-left 8x8, then for each pixel in the
        # full 32x32, determine which fundamental pixel it maps to.
        
        cell = np.zeros((size, size), dtype=int)
        
        # For each pixel (r, c), find the corresponding fundamental pixel.
        # The fundamental domain is the top-left 8x8: rows 0-7, cols 0-7.
        # A pixel (r, c) maps to fundamental pixel (r_f, c_f) where:
        # r_f = min(r, 31-r, c, 31-c) -- no, this isn't right either.
        
        # Let me think of it differently. The C4v fundamental domain for a 32x32 grid
        # is actually 8x8 if we consider that:
        # - The grid can be divided into four 16x16 quadrants.
        # - Each 16x16 quadrant can be divided into four 8x8 sub-quadrants.
        # - Under C4v, all 16 sub-quadrants are equivalent.
        # So the fundamental domain is 8x8.
        
        # To map (r, c) to the fundamental 8x8:
        # First, determine which 16x16 quadrant: 
        #   qr = r // 16 (0 or 1), qc = c // 16 (0 or 1)
        # Then within that quadrant, the local coordinates are:
        #   lr = r % 16, lc = c % 16
        # Now, within the 16x16 quadrant, we need to map to 8x8:
        #   The 16x16 quadrant has C4v symmetry? No, only the full cell does.
        #   Within a 16x16 quadrant, we have mirror symmetry about its center.
        #   The center of the 16x16 quadrant is at (7.5, 7.5) in local coords.
        #   So we can mirror: lr_f = min(lr, 15-lr), lc_f = min(lc, 15-lc)
        #   This gives an 8x8 fundamental domain.
        
        # But wait, this only gives dihedral symmetry within each quadrant, not the full C4v.
        # The full C4v also relates the four quadrants.
        # Since all four quadrants are equivalent under C4v, we can just use the
        # top-left 16x16 quadrant and mirror it to the other three.
        
        # So: fundamental domain is 8x8 (top-left 8x8 of the top-left 16x16 quadrant).
        # To get the full 32x32:
        # 1. Expand 8x8 to 16x16 by mirroring within the quadrant.
        # 2. Expand 16x16 to 32x32 by mirroring the quadrant to the other three.
        
        # Step 1: Expand 8x8 to 16x16
        quad = np.zeros((16, 16), dtype=int)
        for i in range(8):
            for j in range(8):
                quad[i, j] = fund[i, j]
                quad[15-i, j] = fund[i, j]  # H-mirror within quadrant
                quad[i, 15-j] = fund[i, j]  # V-mirror within quadrant
                quad[15-i, 15-j] = fund[i, j]  # 180 rot within quadrant
        
        # Step 2: Expand 16x16 to 32x32
        cell[:16, :16] = quad
        cell[16:, :16] = quad[::-1, :]  # H-mirror the quadrant
        cell[:16, 16:] = quad[:, ::-1]  # V-mirror the quadrant
        cell[16:, 16:] = quad[::-1, ::-1]  # 180 rot of the quadrant
        
        return cell
    
    # -----------------------------------------------------------------------
    # Initial design: 4 circular rods
    # -----------------------------------------------------------------------
    y, x = np.mgrid[0:size, 0:size]
    y = y + 0.5
    x = x + 0.5
    
    centers = [
        (size/4, size/4),
        (size/4, 3*size/4),
        (3*size/4, size/4),
        (3*size/4, 3*size/4)
    ]
    
    r = 5.5
    cell = np.zeros((size, size), dtype=int)
    for cx, cy in centers:
        dist = (x - cx)**2 + (y - cy)**2
        cell[dist < r**2] = 1
    
    # Extract fundamental domain from initial cell
    fund = cell[:8, :8].copy()
    
    # Verify symmetry
    cell = expand_symmetric(fund)
    
    # Initial evaluation
    current_score = evaluate_gap(cell)
    best_cell = cell.copy()
    best_score = current_score
    
    # -----------------------------------------------------------------------
    # Simulated annealing optimization on the 8x8 fundamental domain
    # -----------------------------------------------------------------------
    rng = np.random.default_rng(42)
    T_start = 1.0
    T_end = 0.01
    alpha = 0.995
    
    T = T_start
    steps = 0
    max_steps = 20000
    
    # Precompute k-point list for faster evaluation
    while time.time() < deadline - 5 and steps < max_steps:
        steps += 1
        
        # Mutate: Flip a random pixel in the fundamental domain
        i = rng.integers(0, 8)
        j = rng.integers(0, 8)
        
        old_val = fund[i, j]
        fund[i, j] = 1 - old_val
        
        # Expand to full cell
        new_cell = expand_symmetric(fund)
        
        # Evaluate
        new_score = evaluate_gap(new_cell)
        
        # Accept/Reject
        if new_score > current_score:
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_cell = new_cell.copy()
        else:
            delta = new_score - current_score
            if delta < 0:
                prob = np.exp(delta * 10.0 / T)
                if rng.random() < prob:
                    current_score = new_score
                else:
                    fund[i, j] = old_val
        
        # Cool down
        T *= alpha
        if T < T_end:
            T = T_end
    
    # Return best found
    return best_cell.tolist()
