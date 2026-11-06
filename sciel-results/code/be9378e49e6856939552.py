"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import random
import numpy as np

# EVOLVE-BLOCK-START

def _enforce_symmetry(cell):
    """Enforce 4-fold symmetry (D4) about the cell centre."""
    size = len(cell)
    half = size // 2
    # Work on a copy
    c = [row[:] for row in cell]
    # For a 32x32 cell, symmetry about center means:
    # c[i][j] == c[size-1-i][j] (vertical flip)
    # c[i][j] == c[i][size-1-j] (horizontal flip)
    # c[i][j] == c[j][i] (transpose)
    # So we only need to define the top-left 16x16 quadrant and mirror it.
    for i in range(half):
        for j in range(half):
            val = c[i][j]
            # Top-left is already set
            # Top-right: mirror horizontally
            c[i][size - 1 - j] = val
            # Bottom-left: mirror vertically
            c[size - 1 - i][j] = val
            # Bottom-right: mirror both
            c[size - 1 - i][size - 1 - j] = val
    return c

def _compute_fourier_coefficients(cell, size, max_g=8):
    """
    Compute Fourier coefficients of the permittivity function.
    epsilon(r) = eps_avg + sum_{G!=0} eps_G exp(iG.r)
    where G = (2π/nx, 2π/ny) for integer (nx, ny)
    
    For a square lattice, the Fourier coefficients are:
    eps_G = (1/A) ∫∫ epsilon(r) exp(-iG.r) dA
    
    For pixel-based: eps_G = (1/N^2) sum_{i,j} eps_{i,j} exp(-iG.x_i) exp(-iG.y_j)
    """
    eps_low = 1.0
    eps_high = 12.0
    
    # Average permittivity
    total = 0.0
    for row in cell:
        for val in row:
            total += eps_high if val == 1 else eps_low
    eps_avg = total / (size * size)
    
    # Compute Fourier coefficients for G vectors up to max_g in each direction
    coeffs = {}
    # Precompute exp(-i * 2π * k * (j+0.5)/size) for efficiency
    # For G = (2π*nx, 2π*ny), the coefficient is:
    # eps_{nx,ny} = (1/N^2) sum_{i,j} eps_{i,j} exp(-i 2π nx (i+0.5)/N) exp(-i 2π ny (j+0.5)/N)
    
    # For a symmetric cell, many coefficients will be real
    for nx in range(-max_g, max_g + 1):
        for ny in range(-max_g, max_g + 1):
            if nx == 0 and ny == 0:
                continue
            # Use the pixel sum
            real_sum = 0.0
            imag_sum = 0.0
            for i in range(size):
                phase_i = -2.0 * math.pi * nx * (i + 0.5) / size
                ci = math.cos(phase_i)
                si = math.sin(phase_i)
                for j in range(size):
                    phase_j = -2.0 * math.pi * ny * (j + 0.5) / size
                    cj = math.cos(phase_j)
                    sj = math.sin(phase_j)
                    eps_val = eps_high if cell[i][j] == 1 else eps_low
                    # exp(i(phase_i + phase_j)) = (ci + i*si)(cj + i*sj)
                    # But we need exp(-i(G.r)) which is what we have
                    # Actually: exp(-i 2π nx (i+0.5)/N) * exp(-i 2π ny (j+0.5)/N)
                    # = (ci - i*si)(cj - i*sj) where ci=cosh, si=sinh... no
                    # Let me be careful:
                    # exp(-iθ) = cos(θ) - i sin(θ)
                    # So exp(-i(phase_i + phase_j)) = (ci - i*si)(cj - i*sj)
                    # = ci*cj - ci*sj*i - si*cj*i + si*sj*i^2
                    # = ci*cj - si*sj - i(ci*sj + si*cj)
                    real_part = ci * cj - si * sj
                    imag_part = -(ci * sj + si * cj)
                    real_sum += eps_val * real_part
                    imag_sum += eps_val * imag_part
            coeffs[(nx, ny)] = (real_sum - eps_avg * size * size * 0) / (size * size)
            # Wait, the average is already subtracted in the definition
            # eps_G for G!=0 is just (1/N^2) sum eps_{ij} exp(-iG.r_ij)
            # since the G=0 term is the average
            coeffs[(nx, ny)] = (real_sum, imag_sum) / (size * size)
    
    return coeffs, eps_avg

def _compute_band_structure(cell, size, max_g=8, k_points=15):
    """
    Compute TM band structure using plane-wave expansion.
    For TM polarization, the eigenvalue problem is:
    sum_{G'} (K+G)^2 * eps^{-1}_{G-G'} * C_{G'} = omega^2 * C_G
    
    where K is the wave vector, G are reciprocal lattice vectors,
    and eps^{-1} is the Fourier transform of 1/epsilon.
    
    We need to compute the Fourier coefficients of 1/epsilon.
    """
    eps_low = 1.0
    eps_high = 12.0
    
    # Compute Fourier coefficients of epsilon
    eps_coeffs = {}
    for nx in range(-max_g, max_g + 1):
        for ny in range(-max_g, max_g + 1):
            real_sum = 0.0
            imag_sum = 0.0
            for i in range(size):
                phase_i = -2.0 * math.pi * nx * (i + 0.5) / size
                ci = math.cos(phase_i)
                si = math.sin(phase_i)
                for j in range(size):
                    phase_j = -2.0 * math.pi * ny * (j + 0.5) / size
                    cj = math.cos(phase_j)
                    sj = math.sin(phase_j)
                    eps_val = eps_high if cell[i][j] == 1 else eps_low
                    real_part = ci * cj - si * sj
                    imag_part = -(ci * sj + si * cj)
                    real_sum += eps_val * real_part
                    imag_sum += eps_val * imag_part
            eps_coeffs[(nx, ny)] = (real_sum, imag_sum) / (size * size)
    
    # For TM, we need Fourier coefficients of 1/epsilon
    # 1/eps = 1/eps_low for air, 1/eps_high for dielectric
    inv_eps_coeffs = {}
    for nx in range(-max_g, max_g + 1):
        for ny in range(-max_g, max_g + 1):
            real_sum = 0.0
            imag_sum = 0.0
            for i in range(size):
                phase_i = -2.0 * math.pi * nx * (i + 0.5) / size
                ci = math.cos(phase_i)
                si = math.sin(phase_i)
                for j in range(size):
                    phase_j = -2.0 * math.pi * ny * (j + 0.5) / size
                    cj = math.cos(phase_j)
                    sj = math.sin(phase_j)
                    inv_eps_val = 1.0 / (eps_high if cell[i][j] == 1 else eps_low)
                    real_part = ci * cj - si * sj
                    imag_part = -(ci * sj + si * cj)
                    real_sum += inv_eps_val * real_part
                    imag_sum += inv_eps_val * imag_part
            inv_eps_coeffs[(nx, ny)] = (real_sum, imag_sum) / (size * size)
    
    # Irreducible Brillouin zone points (fractional coordinates)
    # For square lattice, IBZ is: 0 <= kx <= ky, kx^2 + ky^2 <= 1 (in units of pi/a)
    # Actually, the BZ is [-π/a, π/a] x [-π/a, π/a], IBZ is 0 <= kx <= ky, kx^2+ky^2 <= (π/a)^2
    # Let's use a set of representative points
    k_points_list = []
    # High symmetry points
    k_points_list.append((0.0, 0.0))       # Gamma
    k_points_list.append((0.5, 0.0))       # X (in units of π/a, so this is (π/2a, 0))
    k_points_list.append((0.5, 0.5))       # M
    # Path points
    for t in np.linspace(0, 1, 10):
        k_points_list.append((t * 0.5, 0.0))  # Gamma to X
        k_points_list.append((0.5, t * 0.5))  # X to M
        k_points_list.append((t * 0.5, t * 0.5))  # Gamma to M
    
    # Remove duplicates
    unique_k = []
    for k in k_points_list:
        if not any(abs(k[0] - u[0]) < 1e-6 and abs(k[1] - u[1]) < 1e-6 for u in unique_k):
            unique_k.append(k)
    k_points_list = unique_k
    
    # Plane wave list: G vectors with |G| <= 2π*max_g/a
    # In units where a=1, G = (2π*nx, 2π*ny)
    g_vectors = []
    for nx in range(-max_g, max_g + 1):
        for ny in range(-max_g, max_g + 1):
            g_vectors.append((2.0 * math.pi * nx, 2.0 * math.pi * ny))
    n_pw = len(g_vectors)
    
    # Compute bands
    band_energies = []  # List of (kx, ky, [omega_1, ..., omega_N])
    
    for kx_frac, ky_frac in k_points_list:
        # Convert to actual wave vector (in units of 1/a, with a=1)
        # The BZ boundary is at π/a, so kx = kx_frac * π
        kx = kx_frac * math.pi
        ky = ky_frac * math.pi
        
        # Build the matrix H where H_{G,G'} = (K+G)^2 * inv_eps_{G-G'}
        # But we need to be careful: the eigenvalue equation is
        # sum_{G'} (K+G)^2 * inv_eps_{G-G'} * C_{G'} = omega^2 * C_G
        # Wait, that's not quite right. Let me recall the plane-wave method for TM.
        #
        # For TM (Ez nonzero), Maxwell's equations give:
        # ∇^2 Ez + (omega/c)^2 epsilon(r) Ez = 0
        # In plane-wave expansion: Ez = sum_G C_G exp(i(K+G).r)
        # This leads to: sum_{G'} inv_eps_{G-G'} (K+G')^2 C_{G'} = (omega/c)^2 C_G
        # Wait, I need to double-check. The standard formulation is:
        # For TE (H_parallel): sum_{G'} eps_{G-G'} (K+G')^2 C_{G'} = (omega/c)^2 C_G
        # For TM (E_parallel): sum_{G'} inv_eps_{G-G'} (K+G')^2 C_{G'} = (omega/c)^2 C_G
        #
        # So for TM, the matrix element is: H_{G,G'} = (K+G')^2 * inv_eps_{G-G'}
        # And we solve H C = (omega/c)^2 C
        
        # Build matrix
        H = np.zeros((n_pw, n_pw), dtype=complex)
        for idx_g, (gx, gy) in enumerate(g_vectors):
            for idx_gp, (gpx, gpy) in enumerate(g_vectors):
                # (K+G')^2
                kx_gp = kx + gpx
                ky_gp = ky + gpy
                kp_sq = kx_gp * kx_gp + ky_gp * ky_gp
                
                # G - G'
                dx = gx - gpx
                dy = gy - gpy
                # In units of 2π/a, so the Fourier index is (dx/(2π), dy/(2π))
                nx_diff = round(dx / (2.0 * math.pi))
                ny_diff = round(dy / (2.0 * math.pi))
                
                if -max_g <= nx_diff <= max_g and -max_g <= ny_diff <= max_g:
                    inv_eps_val = inv_eps_coeffs.get((nx_diff, ny_diff), (0, 0))
                    H[idx_g, idx_gp] = kp_sq * (inv_eps_val[0] + 1j * inv_eps_val[1])
        
        # Solve eigenvalue problem
        try:
            eigvals = np.linalg.eigvalsh(H)
            # Take positive eigenvalues
            eigvals = np.sort(np.real(eigvals))
            # omega^2 = eigvals * c^2, but we work in units where c=1 and a=1
            # So omega = sqrt(eigval)
            omegas = np.sqrt(np.maximum(eigvals, 0))
            band_energies.append((kx_frac, ky_frac, omegas))
        except:
            continue
    
    # Compute the 4-5 band gap
    # For each k point, get omega_4 and omega_5
    max_omega4 = 0.0
    min_omega5 = float('inf')
    
    for kx_frac, ky_frac, omegas in band_energies:
        if len(omegas) >= 5:
            omega4 = omegas[3]  # 4th band (0-indexed: 3)
            omega5 = omegas[4]  # 5th band (0-indexed: 4)
            max_omega4 = max(max_omega4, omega4)
            min_omega5 = min(min_omega5, omega5)
    
    if max_omega4 == 0.0 or min_omega5 == float('inf'):
        return -1.0  # Invalid
    
    gap = min_omega5 - max_omega4
    midgap = 0.5 * (min_omega5 + max_omega4)
    
    if midgap <= 0:
        return -1.0
    
    ratio = 2.0 * gap / midgap
    return ratio

def design(problem):
    """Design a photonic crystal unit cell maximizing the 4-5 TM band gap."""
    size = problem["pixels"]
    start_time = time.time()
    deadline = start_time + 180  # Safety margin before 230s limit
    
    # Start with a reasonable initial design
    # Use a checkerboard-like pattern as initial guess
    cell = []
    for i in range(size):
        row = []
        for j in range(size):
            # Checkerboard pattern with some structure
            if (i // 4) % 2 == (j // 4) % 2:
                row.append(1)
            else:
                row.append(0)
        cell.append(row)
    cell = _enforce_symmetry(cell)
    
    best_score = _compute_band_structure(cell, size, max_g=6, k_points=10)
    best_cell = [row[:] for row in cell]
    
    # Local search: flip random pixels (respecting symmetry)
    random.seed(42)
    iteration = 0
    while time.time() < deadline:
        iteration += 1
        # Create a perturbation
        new_cell = [row[:] for row in best_cell]
        
        # Flip a few pixels in the fundamental domain (top-left 16x16)
        n_flips = random.randint(1, 3)
        for _ in range(n_flips):
            i = random.randint(0, size // 2 - 1)
            j = random.randint(0, size // 2 - 1)
            new_cell[i][j] = 1 - new_cell[i][j]
        
        new_cell = _enforce_symmetry(new_cell)
        
        # Evaluate
        score = _compute_band_structure(new_cell, size, max_g=6, k_points=10)
        
        if score > best_score:
            best_score = score
            best_cell = [row[:] for row in new_cell]
        
        # Occasionally try larger jumps
        if iteration % 10 == 0 and time.time() < deadline - 60:
            # Try a completely different pattern
            pattern_type = random.randint(0, 3)
            alt_cell = []
            if pattern_type == 0:
                # Circular holes
                for i in range(size):
                    row = []
                    for j in range(size):
                        x = (i + 0.5) / size - 0.5
                        y = (j + 0.5) / size - 0.5
                        row.append(0 if x*x + y*y < 0.15 else 1)
                    alt_cell.append(row)
            elif pattern_type == 1:
                # Square holes
                for i in range(size):
                    row = []
                    for j in range(size):
                        x = (i + 0.5) / size - 0.5
                        y = (j + 0.5) / size - 0.5
                        row.append(0 if abs(x) < 0.15 and abs(y) < 0.15 else 1)
                    alt_cell.append(row)
            elif pattern_type == 2:
                # Cross pattern
                for i in range(size):
                    row = []
                    for j in range(size):
                        x = (i + 0.5) / size - 0.5
                        y = (j + 0.5) / size - 0.5
                        row.append(1 if abs(x) < 0.1 or abs(y) < 0.1 else 0)
                    alt_cell.append(row)
            else:
                # Random symmetric
                for i in range(size):
                    row = []
                    for j in range(size):
                        row.append(random.randint(0, 1))
                    alt_cell.append(row)
            alt_cell = _enforce_symmetry(alt_cell)
            
            alt_score = _compute_band_structure(alt_cell, size, max_g=6, k_points=10)
            if alt_score > best_score:
                best_score = alt_score
                best_cell = [row[:] for row in alt_cell]
    
    return best_cell

# EVOLVE-BLOCK-END
