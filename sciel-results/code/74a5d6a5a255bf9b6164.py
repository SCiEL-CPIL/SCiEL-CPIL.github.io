import numpy as np
import time
import math
import random

def design(problem):
    size = problem["pixels"]      # 32
    eps_low = problem["eps_low"]  # 1
    eps_high = problem["eps_high"]# 12

    t0 = time.time()
    DEADLINE = 215.0

    def time_left():
        return DEADLINE - (time.time() - t0)

    # ---------- Symmetry ----------
    # D4 symmetry for 32x32: cell[i,j] == cell[31-i,j] == cell[i,31-j] == cell[j,i]
    # Free region: i in 0..15, j in i..15 (upper-left triangle including diagonal)
    # We'll represent the cell by its free triangle and expand.

    def expand_symmetric(tri):
        """tri: 1D array of length 136 (the free triangle). Returns 32x32 int array."""
        cell = np.zeros((size, size), dtype=int)
        idx = 0
        for i in range(16):
            for j in range(i, 16):
                v = tri[idx]
                idx += 1
                cell[i, j] = v
                cell[31 - i, j] = v
                cell[i, 31 - j] = v
                cell[31 - i, 31 - j] = v
                cell[j, i] = v
                cell[31 - j, i] = v
                cell[j, 31 - i] = v
                cell[31 - j, 31 - i] = v
        return cell

    def cell_to_triangle(cell):
        """Extract the free triangle from a 32x32 symmetric cell."""
        tri = np.zeros(136, dtype=int)
        idx = 0
        for i in range(16):
            for j in range(i, 16):
                tri[idx] = cell[i, j]
                idx += 1
        return tri

    # ---------- Fast surrogate band solver ----------
    # Use a 9x9 plane-wave set (Gx, Gy in -4..4) and FFT to get Fourier coeffs.
    # This is much faster than the current 7x7 with nested pixel loops.

    # Precompute G-vector list and indices
    G_half = 4
    G_list = [(gx, gy) for gx in range(-G_half, G_half + 1) for gy in range(-G_half, G_half + 1)]
    nG = len(G_list)  # 81
    G_idx = {g: i for i, g in enumerate(G_list)}

    # K-points in irreducible BZ (in units of 2*pi/a, a=1)
    # Irreducible BZ: 0 <= ky <= kx, kx in [0, 0.5]
    k_points = [
        (0.0, 0.0),
        (0.1, 0.0), (0.2, 0.0), (0.3, 0.0), (0.4, 0.0), (0.5, 0.0),
        (0.5, 0.1), (0.5, 0.2), (0.5, 0.3), (0.5, 0.4), (0.5, 0.5),
        (0.3, 0.1), (0.4, 0.2), (0.3, 0.2), (0.2, 0.1), (0.4, 0.3),
        (0.35, 0.05), (0.45, 0.15), (0.25, 0.15), (0.35, 0.25),
    ]

    def compute_inv_eps_fourier(cell):
        """Compute Fourier coefficients of 1/epsilon using FFT."""
        eps = eps_low + (eps_high - eps_low) * cell.astype(float)
        inv_eps = 1.0 / eps
        # 2D FFT
        F = np.fft.fft2(inv_eps)
        F = np.fft.fftshift(F)
        # Now F[gx+16, gy+16] = sum_{i,j} inv_eps[i,j] * exp(2*pi*i*(gx*i+gy*j)/32)
        # We need (1/N^2) * sum exp(-2*pi*i*(gx*i+gy*j)/N)
        # np.fft.fft2 gives sum exp(-2*pi*i*(k*i)/N) for positive k
        # Let's use ifft2 convention: ifft2 gives (1/N^2) sum exp(+2*pi*i*(k*i)/N)
        # So we need fft2 conjugate... 
        # Actually: fft2(f)[k] = sum_n f[n] exp(-2*pi*i*k*n/N)
        # We want c_G = (1/N^2) sum_n f[n] exp(-2*pi*i*G*n/N) = fft2(f)[G] / N^2
        # But G can be negative. fft2 handles this via periodicity.
        # For G in -4..4, index in fft2 output is G mod 32.
        # So c_G = fft2(inv_eps)[G mod 32] / 32^2
        result = {}
        N = size
        for gx, gy in G_list:
            ix = gx % N
            iy = gy % N
            result[(gx, gy)] = F[iy, ix] / (N * N)
        return result

    def solve_bands_at_k(inv_eps_F, kx, ky):
        """Solve the TM eigenvalue problem at a given k-point."""
        # M[G, G'] = (G-k)·(G'-k) * inv_eps_F[G-G']
        # Build matrix
        M = np.zeros((nG, nG), dtype=float)
        for ig, (gx, gy) in enumerate(G_list):
            Gkx = gx - kx
            Gky = gy - ky
            for igp, (gxp, gyp) in enumerate(G_list):
                Gpkx = gxp - kx
                Gpky = gyp - ky
                dGx = gx - gxp
                dGy = gy - gyp
                if (dGx, dGy) in inv_eps_F:
                    coeff = inv_eps_F[(dGx, dGy)]
                    # coeff is complex, but for real symmetric epsilon, it should be real
                    M[ig, igp] = (Gkx * Gpkx + Gky * Gpky) * coeff.real
        # Ensure symmetry
        M = (M + M.T) / 2.0
        eigvals = np.linalg.eigvalsh(M)
        return eigvals

    def compute_gap_ratio(cell):
        """Compute the band 4-5 gap ratio using the surrogate solver."""
        inv_eps_F = compute_inv_eps_fourier(cell)
        
        max_omega_4 = 0.0
        min_omega_5 = float('inf')
        
        for kx, ky in k_points:
            eigvals = solve_bands_at_k(inv_eps_F, kx, ky)
            if len(eigvals) >= 5:
                omega_4 = np.sqrt(max(0.0, eigvals[3]))
                omega_5 = np.sqrt(max(0.0, eigvals[4]))
                max_omega_4 = max(max_omega_4, omega_4)
                min_omega_5 = min(min_omega_5, omega_5)
        
        if min_omega_5 == float('inf') or max_omega_4 == 0.0:
            return -1.0
        
        gap_ratio = 2.0 * (min_omega_5 - max_omega_4) / (min_omega_5 + max_omega_4)
        return gap_ratio

    # ---------- Initial designs ----------
    def make_circular_rod(radius_frac):
        """Create a centered circular rod. radius_frac is fraction of cell size."""
        cell = np.zeros((size, size), dtype=int)
        cx = cy = (size - 1) / 2.0
        r = radius_frac * size
        for i in range(size):
            for j in range(size):
                dx = i - cx
                dy = j - cy
                if dx * dx + dy * dy <= r * r:
                    cell[i, j] = 1
        return cell

    def make_square_rod(half_width):
        """Create a centered square rod."""
        cell = np.zeros((size, size), dtype=int)
        center = size // 2
        for i in range(size):
            for j in range(size):
                if abs(i - center + 0.5) <= half_width and abs(j - center + 0.5) <= half_width:
                    cell[i, j] = 1
        return cell

    def make_diamond_rod(half_width):
        """Create a centered diamond (rotated square) rod."""
        cell = np.zeros((size, size), dtype=int)
        center = (size - 1) / 2.0
        for i in range(size):
            for j in range(size):
                if abs(i - center) + abs(j - center) <= half_width:
                    cell[i, j] = 1
        return cell

    # Generate initial candidates
    initial_cells = []
    for r in [0.1, 0.12, 0.15, 0.18, 0.2, 0.22, 0.25, 0.28, 0.3]:
        initial_cells.append(make_circular_rod(r))
    for hw in [4, 5, 6, 7, 8]:
        initial_cells.append(make_square_rod(hw))
    for hw in [5, 6, 7, 8, 9]:
        initial_cells.append(make_diamond_rod(hw))

    # Evaluate all initial candidates
    best_cell = None
    best_score = -float('inf')
    
    for cell in initial_cells:
        if time_left() < 5.0:
            break
        score = compute_gap_ratio(cell)
        if score > best_score:
            best_score = score
            best_cell = cell.copy()

    # ---------- Local search: random symmetric mutations ----------
    def random_symmetric_cell():
        """Generate a random D4-symmetric cell."""
        tri = np.random.randint(0, 2, size=136)
        return expand_symmetric(tri)

    def mutate_cell(cell, num_flips=3):
        """Randomly flip some pixels in the free triangle, maintaining symmetry."""
        tri = cell_to_triangle(cell)
        tri = tri.copy()
        for _ in range(num_flips):
            idx = random.randint(0, 135)
            tri[idx] = 1 - tri[idx]
        return expand_symmetric(tri)

    # Run local search
    search_time = 120.0  # max time for local search
    search_start = time.time()
    
    while (time.time() - search_start) < search_time and time_left() > 10.0:
        # Try a few mutations
        for _ in range(10):
            if time_left() < 5.0:
                break
            new_cell = mutate_cell(best_cell, num_flips=random.choice([1, 2, 3, 5]))
            new_score = compute_gap_ratio(new_cell)
            if new_score > best_score:
                best_score = new_score
                best_cell = new_cell.copy()
        
        # Occasionally try a completely random restart
        if random.random() < 0.05:
            rand_cell = random_symmetric_cell()
            rand_score = compute_gap_ratio(rand_cell)
            if rand_score > best_score:
                best_score = rand_score
                best_cell = rand_cell.copy()

    # ---------- Final: try a few more targeted perturbations ----------
    if time_left() > 15.0:
        # Try flipping single pixels in the free triangle
        tri = cell_to_triangle(best_cell)
        for idx in range(136):
            if time_left() < 5.0:
                break
            tri_flipped = tri.copy()
            tri_flipped[idx] = 1 - tri_flipped[idx]
            new_cell = expand_symmetric(tri_flipped)
            new_score = compute_gap_ratio(new_cell)
            if new_score > best_score:
                best_score = new_score
                best_cell = new_cell.copy()
                tri = cell_to_triangle(best_cell)

    return best_cell.tolist()
