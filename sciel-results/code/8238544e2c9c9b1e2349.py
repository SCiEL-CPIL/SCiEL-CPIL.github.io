"""Photonic crystal unit cell design for large TM band gap between bands 4 and 5."""
import time
import numpy as np

def design(problem):
    """Design a 32x32 symmetric unit cell maximizing TM band gap between bands 4 and 5."""
    size = problem["pixels"]
    eps_high = problem.get("eps_high", 12)
    
    start_time = time.time()
    deadline = 220.0  # seconds, with safety margin
    margin = 15.0  # seconds before hard kill
    
    def elapsed():
        return time.time() - start_time
    
    def time_ok():
        return elapsed() < deadline - margin
    
    # --- Band structure computation via plane wave method ---
    
    # Use 289 plane waves (17x17 grid)
    pw_cutoff = 8  # G_max such that (Gx^2 + Gy^2) <= cutoff^2
    # For a square lattice, plane waves are indexed by (nx, ny) where G = 2pi*(nx, ny)
    # We need |G| <= cutoff * (2pi/a). In pixel units, a = 1.
    # So we need nx^2 + ny^2 <= cutoff^2
    # For cutoff=8: sum over nx,ny in [-8,8] of (nx^2+ny^2 <= 64)
    # Let's count: this gives approximately pi*64 ≈ 201, but we need exactly 289 = 17^2
    # Actually, let's use a rectangular grid of 17x17 = 289 plane waves
    # This means nx, ny in [-8, 8]
    
    nx_range = list(range(-8, 9))  # -8 to 8, 17 values
    plane_waves = []
    for nx in nx_range:
        for ny in nx_range:
            plane_waves.append((nx, ny))
    
    N_pw = len(plane_waves)  # should be 289
    
    # Precompute G vectors
    G_vectors = np.array(plane_waves, dtype=float)  # (N_pw, 2)
    
    # Precompute G^2 for each plane wave
    G_sq = np.sum(G_vectors ** 2, axis=1)
    
    # --- Fourier coefficients of epsilon ---
    
    def compute_fourier_coeffs(cell):
        """Compute Fourier coefficients of epsilon for the given cell."""
        # epsilon(x,y) = eps_low + (eps_high - eps_low) * f(x,y)
        # where f(x,y) is 1 for dielectric, 0 for air
        # Fourier coefficient: eps(G) = (1/a^2) integral epsilon(x,y) exp(-i G·x) dxdy
        # For a pixel-based cell, we approximate the integral as a sum over pixels
        # Each pixel has area (1/size)^2 and is centered at ((i+0.5)/size - 0.5, (j+0.5)/size - 0.5)
        
        eps_low = 1.0
        delta_eps = eps_high - eps_low  # 11
        
        # For pixel (i,j) with center (xi, yj):
        # Contribution to eps(G) from this pixel:
        # delta_eps * f(i,j) * (1/size^2) * exp(-i 2pi (nx*xi + ny*yj))
        # where f(i,j) = 1 if dielectric, 0 if air
        
        coeffs = np.zeros(N_pw, dtype=complex)
        
        for nx, ny in plane_waves:
            total = 0.0
            for i in range(size):
                xi = (i + 0.5) / size - 0.5
                phase_x = np.exp(-1j * 2 * np.pi * nx * xi)
                for j in range(size):
                    yj = (j + 0.5) / size - 0.5
                    phase = phase_x * np.exp(-1j * 2 * np.pi * ny * yj)
                    total += cell[i][j] * phase
            coeffs[plane_waves.index((nx, ny))] = delta_eps * total / (size * size)
        
        # eps(G) = eps_low * delta(G,0) + delta_eps * f(G)
        # where f(G) is the Fourier coefficient of the binary pattern
        # For G=0: eps(0) = eps_low + delta_eps * (fill fraction)
        # For G!=0: eps(G) = delta_eps * f(G)
        
        result = np.zeros(N_pw, dtype=complex)
        for idx, (nx, ny) in enumerate(plane_waves):
            if nx == 0 and ny == 0:
                fill = np.sum(cell) / (size * size)
                result[idx] = eps_low + delta_eps * fill
            else:
                result[idx] = coeffs[idx]
        
        return result
    
    # --- Band structure at a given k-point ---
    
    def compute_bands_at_k(kx, ky, eps_coeffs):
        """Compute TM bands at k-point (kx, ky) using 289 plane waves."""
        # For TM polarization, the eigenvalue problem is:
        # sum_G' [(G+K)^2 / eps(G'-G)] E(G') = omega^2 E(G)
        # where K is the wavevector in reciprocal space units
        # In our units, k is in units of 2pi/a, so G+K has components (nx+kx, ny+ky)
        
        # Build the matrix
        # M(G1, G2) = (G1+K)^2 / eps(G1-G2)
        # where G1-G2 corresponds to the difference in Fourier indices
        
        # First, build a map from (nx, ny) to index
        pw_index = {}
        for idx, (nx, ny) in enumerate(plane_waves):
            pw_index[(nx, ny)] = idx
        
        # Precompute eps for all differences
        # eps(G1-G2) where G1-G2 = (nx1-nx2, ny1-ny2)
        # We need to find which plane wave index corresponds to each difference
        
        # For the 17x17 grid, differences can range from -16 to 16 in each component
        # Not all differences are in our plane wave set, so we need to handle this
        # Actually, for the Galerkin method with a finite plane wave set, we only
        # include terms where G1-G2 is in the plane wave set
        
        M = np.zeros((N_pw, N_pw), dtype=complex)
        
        for i, (nx1, ny1) in enumerate(plane_waves):
            for j, (nx2, ny2) in enumerate(plane_waves):
                # (G1+K)^2
                gx = nx1 + kx
                gy = ny1 + ky
                g_sq = gx * gx + gy * gy
                
                # G1 - G2
                diff_nx = nx1 - nx2
                diff_ny = ny1 - ny2
                
                # Check if this difference is in our plane wave set
                # If not, we can't compute eps for it directly
                # In a proper implementation, we'd need to include more plane waves
                # or use a different approach
                # For now, let's only include terms where the difference is in our set
                
                key = (diff_nx, diff_ny)
                if key in pw_index:
                    eps_val = eps_coeffs[pw_index[key]]
                    if abs(eps_val) > 1e-10:
                        M[i, j] = g_sq / eps_val
        
        # Solve eigenvalue problem
        # Use eigvalsh for Hermitian matrix
        # The matrix should be Hermitian if eps is real and symmetric
        # Let's symmetrize to be safe
        M = (M + M.conj().T) / 2
        
        try:
            eigenvalues = np.linalg.eigvalsh(M)
            # eigenvalues are omega^2, so omega = sqrt(eigenvalues)
            # Sort by omega
            valid = eigenvalues > 0
            omega = np.sqrt(eigenvalues[valid])
            omega = np.sort(omega)
            return omega
        except:
            return None
    
    # --- Evaluate gap ratio ---
    
    # Irreducible Brillouin zone for square lattice: triangle with vertices (0,0), (0.5,0), (0.5,0.5)
    # Sample 30 k-points along the boundary and interior
    
    def get_k_points(n_per_edge=10):
        """Generate k-points in the irreducible BZ."""
        k_points = []
        
        # Along M-G (0,0) to (0.5, 0)
        for i in range(n_per_edge + 1):
            t = i / n_per_edge
            k_points.append((t * 0.5, 0.0))
        
        # Along M-X (0.5, 0) to (0.5, 0.5)
        for i in range(1, n_per_edge + 1):
            t = i / n_per_edge
            k_points.append((0.5, t * 0.5))
        
        # Along X-G (0.5, 0.5) to (0, 0)
        for i in range(1, n_per_edge + 1):
            t = i / n_per_edge
            k_points.append((0.5 * (1 - t), 0.5 * (1 - t)))
        
        return k_points
    
    k_points = get_k_points(10)  # 31 points, close to 30
    
    def compute_gap_ratio(cell):
        """Compute the gap-midgap ratio between bands 4 and 5."""
        eps_coeffs = compute_fourier_coeffs(cell)
        
        max_omega4 = 0.0
        min_omega5 = float('inf')
        
        for kx, ky in k_points:
            omega = compute_bands_at_k(kx, ky, eps_coeffs)
            if omega is None or len(omega) < 5:
                continue
            max_omega4 = max(max_omega4, omega[3])  # 4th band (index 3)
            min_omega5 = min(min_omega5, omega[4])  # 5th band (index 4)
        
        if min_omega5 == float('inf') or max_omega4 == 0.0:
            return -1.0
        
        gap = min_omega5 - max_omega4
        midgap = (min_omega5 + max_omega4) / 2.0
        if midgap < 1e-10:
            return -1.0
        
        return gap / midgap
    
    # --- Symmetry enforcement ---
    
    def enforce_symmetry(cell):
        """Enforce square symmetry by averaging over the d4 group."""
        size = len(cell)
        sym_cell = [[0] * size for _ in range(size)]
        for i in range(size):
            for j in range(size):
                vals = [
                    cell[i][j],
                    cell[size - 1 - i][size - 1 - j],  # 180° rotation
                    cell[j][i],  # transpose
                    cell[size - 1 - j][size - 1 - i]  # 180° + transpose
                ]
                sym_cell[i][j] = 1 if sum(vals) >= 2 else 0
        return sym_cell
    
    # --- Initial candidate patterns ---
    
    def make_cell_from_pattern(pattern_func):
        """Create a cell from a pattern function, ensuring symmetry."""
        cell = [[0] * size for _ in range(size)]
        for i in range(size):
            for j in range(size):
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                cell[i][j] = 1 if pattern_func(x, y) else 0
        return enforce_symmetry(cell)
    
    def rod_pattern(radius):
        def f(x, y):
            return x * x + y * y < radius * radius
        return f
    
    def four_rod_pattern(radius, offset):
        def f(x, y):
            for sx in [1, -1]:
                for sy in [1, -1]:
                    dx = x - sx * offset
                    dy = y - sy * offset
                    if dx * dx + dy * dy < radius * radius:
                        return True
            return False
        return f
    
    def ring_pattern(inner_r, outer_r):
        def f(x, y):
            r2 = x * x + y * y
            return inner_r * inner_r < r2 < outer_r * outer_r
        return f
    
    def cross_pattern(arm_width, arm_length):
        def f(x, y):
            if abs(y) < arm_width / 2 and abs(x) < arm_length / 2:
                return True
            if abs(x) < arm_width / 2 and abs(y) < arm_length / 2:
                return True
            return False
        return f
    
    # --- Evaluate initial candidates ---
    
    candidates = []
    
    # Single rod
    for r in [0.15, 0.18, 0.20, 0.22, 0.25]:
        cell = make_cell_from_pattern(rod_pattern(r))
        candidates.append(("rod_r%.2f" % r, cell))
    
    # Four rods
    for r in [0.08, 0.10, 0.12, 0.14]:
        for off in [0.25, 0.28, 0.30]:
            cell = make_cell_from_pattern(four_rod_pattern(r, off))
            candidates.append(("4rod_r%.2f_o%.2f" % (r, off), cell))
    
    # Rings
    for ir, orr in [(0.10, 0.20), (0.12, 0.22), (0.15, 0.25)]:
        cell = make_cell_from_pattern(ring_pattern(ir, orr))
        candidates.append(("ring_i%.2f_o%.2f" % (ir, orr), cell))
    
    # Crosses
    for aw, al in [(0.15, 0.40), (0.20, 0.45), (0.25, 0.50)]:
        cell = make_cell_from_pattern(cross_pattern(aw, al))
        candidates.append(("cross_w%.2f_l%.2f" % (aw, al), cell))
    
    best_cell = None
    best_ratio = -1.0
    
    for name, cell in candidates:
        if not time_ok():
            break
        
        ratio = compute_gap_ratio(cell)
        if ratio > best_ratio:
            best_ratio = ratio
            best_cell = cell
    
    if best_cell is None:
        best_cell = make_cell_from_pattern(rod_pattern(0.20))
    
    # --- Local search: flip pixels and evaluate ---
    
    current_cell = [row[:] for row in best_cell]
    current_ratio = best_ratio
    
    # Get independent pixels (upper triangle including diagonal for transpose symmetry,
    # and only half for 180° symmetry)
    # For d4 symmetry, the fundamental domain is 1/8 of the cell
    # Let's just shuffle all pixels and try flipping
    
    np.random.seed(42)
    indices = [(i, j) for i in range(size) for j in range(size)]
    
    no_improve = 0
    max_no_improve = 10
    
    while time_ok() and no_improve < max_no_improve:
        improved = False
        np.random.shuffle(indices)
        
        for (i, j) in indices:
            if not time_ok():
                break
            
            # Flip the pixel
            old_val = current_cell[i][j]
            current_cell[i][j] = 1 - old_val
            new_cell = enforce_symmetry(current_cell)
            
            # Check if the cell actually changed
            changed = any(new_cell[pi][pj] != current_cell[pi][pj] for pi in range(size) for pj in range(size))
            
            if not changed:
                current_cell[i][j] = old_val
                continue
            
            # Evaluate
            new_ratio = compute_gap_ratio(new_cell)
            
            if new_ratio > current_ratio + 1e-6:
                current_cell = new_cell
                current_ratio = new_ratio
                improved = True
                no_improve = 0
            else:
                # Revert
                current_cell[i][j] = old_val
                current_cell = enforce_symmetry(current_cell)
                no_improve += 1
            
            if not time_ok():
                break
        
        if not improved:
            no_improve += 1
    
    return current_cell

# EVOLVE-BLOCK-END
