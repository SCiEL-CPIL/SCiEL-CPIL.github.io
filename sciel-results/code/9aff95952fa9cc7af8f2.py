"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import math

# EVOLVE-BLOCK-START

def _symmetrize_quadrant(q):
    """Take a 16x16 top-left quadrant and produce a full 32x32 symmetric cell.
    The cell must satisfy:
      cell == cell[::-1] row-wise  (vertical flip)
      cell == cell[::-1] column-wise (horizontal flip)
      cell == cell.T (transpose)
    So the cell is symmetric under D4 (square symmetry group).
    We define the quadrant as top-left 16x16.
    Full cell[i][j] = q[i][j] for i,j in 0..15
    By horizontal symmetry: cell[i][31-j] = cell[i][j]
    By vertical symmetry: cell[31-i][j] = cell[i][j]
    By transpose: cell[i][j] = cell[j][i]
    
    So for any (i,j):
      cell[i][j] = q[min(i,31-i)][min(j,31-j)] if we use the quadrant... 
      Actually, let's think more carefully.
      
    The center is between pixels 15 and 16 (for 32 pixels, indices 0..31).
    Symmetry about cell centre means:
      cell[i][j] = cell[31-i][31-j] (180 deg rotation)
      cell[i][j] = cell[i][31-j] (horizontal mirror)
      cell[i][j] = cell[31-i][j] (vertical mirror)
      cell[i][j] = cell[j][i] (transpose)
    
    So the fundamental domain is the top-left quadrant where i <= 15 and j <= 15,
    but with the constraint that the quadrant itself must be symmetric under transpose
    for the parts that map within the quadrant? No, actually:
    
    For i in 0..15, j in 0..15:
      cell[i][j] = q[i][j]
      cell[i][31-j] = q[i][j]  (horizontal)
      cell[31-i][j] = q[i][j]  (vertical)
      cell[31-i][31-j] = q[i][j] (both)
      cell[j][i] = q[i][j]  (transpose)
      
    But wait, if i <= 15 and j <= 15, then cell[j][i] is also in the top-left quadrant,
    so q[j][i] must equal q[i][j]. So the quadrant must be symmetric!
    
    Actually no. The transpose symmetry says cell[i][j] = cell[j][i]. 
    If both i,j are in 0..15, then cell[i][j] = q[i][j] and cell[j][i] = q[j][i],
    so we need q[i][j] = q[j][i].
    
    So the 16x16 quadrant must be symmetric under transpose.
    """
    size = 32
    half = size // 2
    cell = np.zeros((size, size), dtype=np.int8)
    # Fill from quadrant
    for i in range(half):
        for j in range(half):
            val = q[i, j]
            cell[i, j] = val
            cell[i, size-1-j] = val
            cell[size-1-i, j] = val
            cell[size-1-i, size-1-j] = val
    return cell

def _cell_to_quadrant(cell):
    """Extract the top-left 16x16 quadrant from a 32x32 symmetric cell."""
    return cell[:16, :16].copy()

def _pw_evaluate(cell, eps_low=1.0, eps_high=12.0):
    """
    Evaluate the TM band structure using plane-wave expansion.
    Returns the gap-midgap ratio between bands 4 and 5.
    
    For TM polarization:
      The eigenvalue equation is:
        sum_G' [k+G]^2 * (1/eps(G-G')) * c(G') = omega^2 * c(G)
    
    Where 1/eps is the inverse dielectric function.
    """
    size = 32
    Npw = 17  # Use 17x17 = 289 plane waves
    half_N = Npw // 2
    
    # Compute Fourier coefficients of 1/eps
    # eps(x,y) = eps_low + (eps_high - eps_low) * cell(x,y)
    # 1/eps(x,y) = 1 / (eps_low + (eps_high - eps_low) * cell(x,y))
    
    # For each pixel, compute 1/eps
    inv_eps = np.zeros((size, size), dtype=np.float64)
    for i in range(size):
        for j in range(size):
            eps_val = eps_low + (eps_high - eps_low) * cell[i, j]
            inv_eps[i, j] = 1.0 / eps_val
    
    # Compute Fourier coefficients of inv_eps
    # inv_eps(G) = (1/A) * sum_{x,y} inv_eps(x,y) * exp(-i G . r)
    # where G = (Gx, Gy) * 2*pi (in units of 1/a, a=1)
    
    # G vectors: Gx = -half_N to half_N, Gy = -half_N to half_N
    Gx_vals = np.arange(-half_N, half_N + 1, dtype=np.float64)
    Gy_vals = np.arange(-half_N, half_N + 1, dtype=np.float64)
    
    # Precompute exp(-i * 2*pi * G * x) for each Gx and x
    # x = (i + 0.5)/size
    xs = (np.arange(size) + 0.5) / size
    ys = (np.arange(size) + 0.5) / size
    
    # Fourier coefficient: f(Gx, Gy) = (1/N^2) * sum_{i,j} inv_eps[i,j] * exp(-i*2*pi*(Gx*x_i + Gy*y_j))
    # We can separate: f(Gx, Gy) = (1/N^2) * sum_i [sum_j inv_eps[i,j] * exp(-i*2*pi*Gy*y_j)] * exp(-i*2*pi*Gx*x_i)
    
    # Compute 2D FFT approach
    # Actually, let's just compute directly using matrix operations
    # exp(-i*2*pi*Gx*x_i) for each Gx and i
    exp_x = np.exp(-1j * 2.0 * np.pi * np.outer(Gx_vals, xs))  # (Npw, size)
    exp_y = np.exp(-1j * 2.0 * np.pi * np.outer(Gy_vals, ys))  # (Npw, size)
    
    # inv_eps_fft[Gx_idx, Gy_idx] = (1/N^2) * sum_i sum_j inv_eps[i,j] * exp_x[Gx_idx, i] * exp_y[Gy_idx, j]
    # = (1/N^2) * exp_x @ inv_eps @ exp_y.T
    inv_eps_fft = (exp_x @ inv_eps @ exp_y.T) / (size * size)
    
    # Now build the eigenvalue matrix for each k-point
    # For TM: A(G, G') = (k+G)^2 * inv_eps_fft[G-G']
    # where (k+G)^2 = (kx+Gx)^2 + (ky+Gy)^2
    
    # K-points: 30 points covering the irreducible BZ
    # Irreducible BZ of square lattice: triangle from Gamma(0,0) to X(pi,0) to M(pi,pi) to X(pi,0)... 
    # Actually: Gamma(0,0), X(pi,0), M(pi,pi)
    # The irreducible BZ is the triangle with vertices (0,0), (pi,0), (pi,pi)
    
    # Generate 30 k-points
    k_points = []
    # Along Gamma-X: (t*pi, 0) for t in [0,1]
    for t in np.linspace(0, 1, 10):
        k_points.append((t * np.pi, 0.0))
    # Along X-M: (pi, t*pi) for t in [0,1]
    for t in np.linspace(0, 1, 10):
        k_points.append((np.pi, t * np.pi))
    # Along M-Gamma: (t*pi, t*pi) for t in [0,1]
    for t in np.linspace(0, 1, 10):
        k_points.append((t * np.pi, t * np.pi))
    
    # Remove duplicates
    k_points = list(set(k_points))
    # Ensure we have a good set
    if len(k_points) < 30:
        # Add more points
        for t in np.linspace(0, 1, 15):
            k_points.append((t * np.pi, 0.0))
        for t in np.linspace(0, 1, 15):
            k_points.append((np.pi, t * np.pi))
        for t in np.linspace(0, 1, 15):
            k_points.append((t * np.pi, t * np.pi))
        k_points = list(set(k_points))
    
    k_points = k_points[:30]
    
    # For each k-point, compute the 5th smallest eigenvalue and 4th largest... 
    # Actually we need min omega_5 and max omega_4 over all k-points
    # omega_4 = 4th band (index 3), omega_5 = 5th band (index 4)
    
    max_omega_4 = 0.0
    min_omega_5 = float('inf')
    
    Gx_list = Gx_vals  # -half_N to half_N
    Gy_list = Gy_vals
    
    # Precompute G vector indices
    G_indices = []
    for ix, gx in enumerate(Gx_list):
        for iy, gy in enumerate(Gy_list):
            G_indices.append((gx, gy, ix, iy))
    
    for kx, ky in k_points:
        # Build the Npw x Npw matrix
        N = Npw
        A = np.zeros((N, N), dtype=np.float64)
        
        for i, (gx1, gy1, ix1, iy1) in enumerate(G_indices):
            kx_total = kx + gx1
            ky_total = ky + gy1
            k_sq = kx_total * kx_total + ky_total * ky_total
            
            for j, (gx2, gy2, ix2, iy2) in enumerate(G_indices):
                dgx = ix1 - ix2
                dgy = iy1 - iy2
                # inv_eps_fft[dgx, dgy] but we need to handle the indexing
                # G1 - G2 = (gx1-gx2, gy1-gy2)
                # In our FFT array, the index for G=(Gx, Gy) is (Gx+half_N, Gy+half_N)
                # So inv_eps_fft at (gx1-gx2+half_N, gy1-gy2+half_N)
                # But gx1-gx2 ranges from -2*half_N to 2*half_N
                # We only have coefficients for G in [-half_N, half_N]
                # For a 289-plane-wave calculation with 17x17, the convolution 
                # requires coefficients up to 2*half_N = 16, but we only computed up to 8.
                # This is a problem. We need to compute more Fourier coefficients.
                
                # Actually, for the Galerkin method with 289 plane waves, we need
                # 1/eps(G) for G in the range that appears in the convolution.
                # The matrix element is (k+G1)^2 * (1/eps)(G1-G2)
                # G1-G2 can range from -2*half_N to 2*half_N in each component.
                # So we need Fourier coefficients up to 2*half_N = 16.
                
                # Let me recompute with a larger range
                pass
        
        # This approach is too slow. Let me use a more efficient method.
        # Actually, let's precompute the full Fourier spectrum needed.
        break
    
    # Let me redo this more carefully
    # We need 1/eps(G) for G in [-2*half_N, 2*half_N] x [-2*half_N, 2*half_N]
    # That's a 33x33 = 1089 coefficient computation
    
    max_G = 2 * half_N  # 16
    Gx_full = np.arange(-max_G, max_G + 1, dtype=np.float64)
    Gy_full = np.arange(-max_G, max_G + 1, dtype=np.float64)
    
    exp_x_full = np.exp(-1j * 2.0 * np.pi * np.outer(Gx_full, xs))  # (33, 32)
    exp_y_full = np.exp(-1j * 2.0 * np.pi * np.outer(Gy_full, ys))  # (33, 32)
    
    inv_eps_full = (exp_x_full @ inv_eps @ exp_y_full.T) / (size * size)  # (33, 33)
    
    # Now for each k-point, build the 17x17 = 289 matrix
    for kx, ky in k_points:
        N = Npw
        A = np.zeros((N, N), dtype=np.float64)
        
        for i in range(N):
            gx1 = Gx_vals[i // Npw]
            gy1 = Gx_vals[i % Npw]  # Wait, this is wrong. Let me fix the indexing.
            # Actually G_indices[i] = (gx, gy, ix, iy) where ix, iy are indices in Gx_vals, Gy_vals
            # Let me just use G_indices
            pass
        
        # Let me use a cleaner approach
        for i, (gx1, gy1, ix1, iy1) in enumerate(G_indices):
            kx_total = kx + gx1
            ky_total = ky + gy1
            k_sq = kx_total * kx_total + ky_total * ky_total
            
            for j, (gx2, gy2, ix2, iy2) in enumerate(G_indices):
                dgx = int(gx1 - gx2)
                dgy = int(gy1 - gy2)
                # Index in inv_eps_full: dgx + max_G, dgy + max_G
                idx_gx = dgx + max_G
                idx_gy = dgy + max_G
                if 0 <= idx_gx < len(Gx_full) and 0 <= idx_gy < len(Gy_full):
                    coeff = inv_eps_full[idx_gx, idx_gy].real  # Should be real for symmetric cell
                else:
                    coeff = 0.0
                A[i, j] = k_sq * coeff
        
        # Solve eigenvalue problem
        try:
            eigvals = np.linalg.eigvalsh(A)
            # Take the 5 smallest positive eigenvalues
            # Filter out very small or negative values
            eigvals = eigvals[eigvals > 1e-10]
            if len(eigvals) >= 5:
                omega_4 = eigvals[3]  # 4th band (0-indexed: 3)
                omega_5 = eigvals[4]  # 5th band (0-indexed: 4)
                max_omega_4 = max(max_omega_4, omega_4)
                min_omega_5 = min(min_omega_5, omega_5)
        except Exception:
            continue
    
    if min_omega_5 == float('inf'):
        return -1.0
    
    gap = min_omega_5 - max_omega_4
    midgap = (min_omega_5 + max_omega_4) / 2.0
    if midgap <= 0:
        return -1.0
    
    return 2.0 * gap / (min_omega_5 + max_omega_4)

def _local_search(initial_cell, max_time=180.0):
    """Local search by flipping individual pixels while maintaining symmetry."""
    start_time = time.time()
    best_cell = initial_cell.copy()
    best_score = _pw_evaluate(best_cell)
    
    size = 32
    half = 16
    
    # Get the quadrant
    quadrant = _cell_to_quadrant(best_cell)
    
    # Precompute which quadrant pixels are independent
    # Due to transpose symmetry, q[i][j] = q[j][i]
    # So we only need to consider i <= j
    independent_pixels = []
    for i in range(half):
        for j in range(i, half):
            independent_pixels.append((i, j))
    
    improved = True
    while improved and (time.time() - start_time) < max_time:
        improved = False
        for i, j in independent_pixels:
            if (time.time() - start_time) > max_time - 5.0:
                break
            
            # Try flipping
            old_val = quadrant[i, j]
            new_val = 1 - old_val
            quadrant[i, j] = new_val
            quadrant[j, i] = new_val  # Maintain transpose symmetry
            
            cell = _symmetrize_quadrant(quadrant)
            score = _pw_evaluate(cell)
            
            if score > best_score:
                best_score = score
                best_cell = cell.copy()
                improved = True
            else:
                # Revert
                quadrant[i, j] = old_val
                quadrant[j, i] = old_val
    
    return best_cell, best_score

def _gaussian_mixture_cell(peaks, size=32, threshold=0.5):
    """
    Generate a symmetric cell from a set of C4v-symmetric Gaussian peaks.
    peaks: list of (x, y, amplitude, width) where (x,y) is in the top-left quadrant
    The full field is constructed by reflecting across all symmetry axes.
    """
    half = size // 2
    # Create the top-left quadrant field
    field = np.zeros((half, half), dtype=np.float64)
    
    xs = (np.arange(half) + 0.5) / size
    ys = (np.arange(half) + 0.5) / size
    
    for (px, py, amp, width) in peaks:
        # px, py are in [0, 0.5] (fractional coordinates in the unit cell)
        for i in range(half):
            for j in range(half):
                dx = xs[i] - px
                dy = ys[j] - py
                field[i, j] += amp * np.exp(-(dx*dx + dy*dy) / (2.0 * width*width))
    
    # Apply threshold
    quadrant = (field >= threshold).astype(np.int8)
    
    # Ensure transpose symmetry in the quadrant
    for i in range(half):
        for j in range(i, half):
            val = max(quadrant[i, j], quadrant[j, i])
            quadrant[i, j] = val
            quadrant[j, i] = val
    
    return _symmetrize_quadrant(quadrant)

def design(problem):
    """
    Design a symmetric 32x32 unit cell to maximize the TM band gap between bands 4 and 5.
    Uses local search starting from a good initial guess.
    """
    start_time = time.time()
    max_time = 210.0  # Safety margin
    
    size = problem.get("pixels", 32)
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)
    
    # Start with a circular rod
    cell = np.zeros((size, size), dtype=np.int8)
    center = (size - 1) / 2.0
    radius = 0.2 * size
    for i in range(size):
        for j in range(size):
            dx = i - center
            dy = j - center
            if dx*dx + dy*dy < radius*radius:
                cell[i, j] = 1
    
    # Ensure symmetry
    quadrant = _cell_to_quadrant(cell)
    for i in range(16):
        for j in range(i, 16):
            val = max(quadrant[i, j], quadrant[j, i])
            quadrant[i, j] = val
            quadrant[j, i] = val
    cell = _symmetrize_quadrant(quadrant)
    
    # Local search
    cell, score = _local_search(cell, max_time=max_time - (time.time() - start_time))
    
    # Convert to nested list
    result = [[int(cell[i, j]) for j in range(size)] for i in range(size)]
    
    # Verify symmetry
    for i in range(size):
        for j in range(size):
            if result[i][j] != result[j][i]:
                # Fix transpose
                result[i][j] = result[j][i] = 1 if (result[i][j] or result[j][i]) else 0
            if result[i][j] != result[size-1-i][size-1-j]:
                result[i][j] = result[size-1-i][size-1-j] = 1 if (result[i][j] or result[size-1-i][size-1-j]) else 0
            if result[i][j] != result[i][size-1-j]:
                result[i][j] = result[i][size-1-j] = 1 if (result[i][j] or result[i][size-1-j]) else 0
            if result[i][j] != result[size-1-i][j]:
                result[i][j] = result[size-1-i][j] = 1 if (result[i][j] or result[size-1-i][j]) else 0
    
    return result

# EVOLVE-BLOCK-END
