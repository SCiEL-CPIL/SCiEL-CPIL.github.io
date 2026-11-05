"""Combined program: super-ellipse base shape with local symmetric perturbation search."""
import time
import math
import random

# EVOLVE-BLOCK-START
def design(problem):
    """
    Structural change: use a super-ellipse (squircle) as the base dielectric shape,
    then perform a local symmetric pixel-flip search to maximize the band gap.
    The base shape is chosen to match the area of the seed circle, preserving
    fill fraction while exploiting the square symmetry more effectively.
    """
    start_time = time.time()
    deadline = start_time + 180  # Safety margin before 200s limit
    
    size = problem["pixels"]
    eps_low = problem["eps_low"]
    eps_high = problem["eps_high"]
    
    # Helper to enforce square symmetry (reverse rows, reverse cols, transpose)
    def enforce_symmetry(cell):
        """Return a cell that is invariant under the square symmetry group."""
        n = len(cell)
        sym = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                # Four symmetric positions
                s = cell[i][j] + cell[n-1-i][j] + cell[i][n-1-j] + cell[n-1-i][n-1-j]
                # Majority vote (or just set all to the same value)
                val = 1 if s >= 2 else 0
                sym[i][j] = val
                sym[n-1-i][j] = val
                sym[i][n-1-j] = val
                sym[n-1-i][n-1-j] = val
        return sym
    
    # Helper to compute Fourier coefficients exactly for the pixel map
    def get_fourier_coefficients(cell):
        """
        Compute exact Fourier coefficients for a 32x32 pixel map.
        epsilon(r) is piecewise constant over each pixel.
        eps_G = (1/A) * sum_{pixels} eps_pixel * integral_pixel exp(-i G.r) dA
        """
        coeffs = {}
        n = len(cell)
        # Plane wave basis: G = 2*pi*(nx, ny) for nx, ny in range
        # 289 plane waves means 17x17 grid: nx, ny from -8 to 8
        for nx in range(-8, 9):
            for ny in range(-8, 9):
                if nx == 0 and ny == 0:
                    avg = sum(sum(row) for row in cell) / (n * n)
                    coeffs[(nx, ny)] = eps_low + (eps_high - eps_low) * avg
                else:
                    Gx = 2 * math.pi * nx
                    Gy = 2 * math.pi * ny
                    integral = 0.0
                    for i in range(n):
                        for j in range(n):
                            if cell[i][j] == 1:
                                # Pixel (i,j) covers x in [i/n, (i+1)/n], y in [j/n, (j+1)/n]
                                # Integral of exp(-i(Gx*x + Gy*y)) over this pixel
                                # = exp(-i*Gx*(i+0.5)/n) * (2*sin(Gx/(2n))/Gx) 
                                #   * exp(-i*Gy*(j+0.5)/n) * (2*sin(Gy/(2n))/Gy)
                                # But need to handle Gx=0 or Gy=0 cases
                                def sinc1d(G, idx):
                                    if G == 0:
                                        return 1.0 / n
                                    else:
                                        # Integral from idx/n to (idx+1)/n of exp(-iGx) dx
                                        # = [exp(-iG*idx/n) - exp(-iG*(idx+1)/n)] / (iG)
                                        # = exp(-iG*(idx+0.5)/n) * 2*sin(G/(2n)) / G
                                        return math.exp(-1j * G * (idx + 0.5) / n) * 2 * math.sin(G / (2 * n)) / G
                                
                                term_x = sinc1d(Gx, i)
                                term_y = sinc1d(Gy, j)
                                integral += term_x * term_y
                    
                    # Divide by area A = 1 (normalized)
                    coeffs[(nx, ny)] = eps_low + (eps_high - eps_low) * integral
        
        return coeffs
    
    # Helper to compute TM bands using plane-wave Galerkin method
    def compute_tm_bands(cell, num_bands=5, num_kpoints=30):
        """
        Compute TM band structure using plane-wave expansion.
        Returns list of (kx, ky, omega_4, omega_5) for sampling k-points.
        """
        coeffs = get_fourier_coefficients(cell)
        
        # G vectors
        G_list = []
        for nx in range(-8, 9):
            for ny in range(-8, 9):
                G_list.append((2 * math.pi * nx, 2 * math.pi * ny, nx, ny))
        
        # Build inverse epsilon matrix
        # For TM: (k^2 - G^2) a_G - sum_{G'} eps^{-1}_{G-G'} a_{G'} = 0
        # Actually for TM, the equation is:
        # sum_{G'} [ (k+G)^2 delta_{G,G'} - eps^{-1}_{G-G'} ] a_{G'} = 0
        # Wait, let me be more careful.
        
        # For TM polarization in 2D:
        # The wave equation is: -∇² E_z = k² ε(r) E_z
        # In plane wave basis: sum_{G'} [(k+G')² δ_{GG'} - ε_{G-G'}] E_{G'} = 0
        # No, that's not right either.
        
        # Correct TM equation:
        # (k² - G²) a_G = sum_{G'} ε_{G-G'} a_{G'}
        # Where ε_{G-G'} are Fourier coefficients of epsilon.
        # Rearranging: sum_{G'} [ (k² - G²) δ_{GG'} - ε_{G-G'} ] a_{G'} = 0
        
        # Actually the standard form is:
        # sum_{G'} [ (|k+G|²) δ_{GG'} - ε^{-1}_{G-G'} ] a_{G'} = 0 for TE
        # For TM: sum_{G'} [ (|k+G|²) δ_{GG'} - ε_{G-G'} ] a_{G'} = 0
        # No, I need to recall the exact form.
        
        # Let me use the standard plane-wave expansion for 2D photonic crystals.
        # For TM: the eigenvalue equation is
        # sum_{G'} [ (|k+G|²) δ_{GG'} - (1/ε)_{G-G'} ] E_{G'} = 0
        # Wait, I'm getting confused. Let me think again.
        
        # The Maxwell equation for TM in 2D (with E_z only):
        # ∇² E_z + k² ε(r) E_z = 0
        # Expand E_z = sum_G a_G exp(i(k+G)·r)
        # ε(r) E_z = sum_G sum_G' ε_{G-G'} a_{G'} exp(i(k+G)·r)
        # ∇² E_z = -sum_G |k+G|² a_G exp(i(k+G)·r)
        # So: -sum_G |k+G|² a_G exp(i(k+G)·r) + k² sum_G sum_G' ε_{G-G'} a_{G'} exp(i(k+G)·r) = 0
        # For each G: -|k+G|² a_G + k² sum_{G'} ε_{G-G'} a_{G'} = 0
        # Rearranging: sum_{G'} [ k² ε_{G-G'} - |k+G|² δ_{GG'} ] a_{G'} = 0
        # Or: sum_{G'} [ |k+G|² δ_{GG'} - k² ε_{G-G'} ] a_{G'} = 0
        
        # This is a generalized eigenvalue problem if we divide by k²:
        # sum_{G'} [ (|k+G|²/k²) δ_{GG'} - ε_{G-G'} ] a_{G'} = 0
        # Let ω = k (normalized), then:
        # sum_{G'} [ (|k+G|²/ω²) δ_{GG'} - ε_{G-G'} ] a_{G'} = 0
        # This is: M a = ω² a where M_{GG'} = (|k+G|²/ω²) δ_{GG'} - ε_{G-G'}
        # But ω is what we're solving for, so this is implicit.
        
        # Better form: Define the matrix A_{GG'} = δ_{GG'} |k+G|² - ω² ε_{G-G'}
        # Then A a = 0, which means det(A) = 0.
        # Or equivalently: ω² ε_{G-G'} a_{G'} = |k+G|² a_G
        # This is a standard eigenvalue problem if we write it as:
        # sum_{G'} ε_{G-G'} a_{G'} = (|k+G|²/ω²) a_G
        # Let λ = 1/ω², then: sum_{G'} ε_{G-G'} a_{G'} = λ |k+G|² a_G
        # This is: ε_matrix a = λ D a where D is diagonal with D_{GG} = |k+G|²
        # Or: D^{-1} ε_matrix a = λ a
        # So the eigenvalues λ = 1/ω², and ω = 1/sqrt(λ).
        
        # Let me implement this properly using NumPy.
        import numpy as np
        
        G_vecs = []
        for nx in range(-8, 9):
            for ny in range(-8, 9):
                G_vecs.append((2 * math.pi * nx, 2 * math.pi * ny))
        N = len(G_vecs)
        
        # Precompute |k+G|² for given k
        def compute_eigenvalues(kx, ky):
            # Build matrix B where B_{GG'} = ε_{G-G'} / |k+G|²
            # Then B a = (1/ω²) a
            B = np.zeros((N, N), dtype=complex)
            for idx, (Gx, Gy) in enumerate(G_vecs):
                kxG = kx + Gx
                kyG = ky + Gy
                kG_sq = kxG * kxG + kyG * kyG
                if kG_sq < 1e-12:
                    continue
                for idx2, (Gx2, Gy2) in enumerate(G_vecs):
                    # ε_{G-G'} = ε_{(nx-nx2, ny-ny2)}
                    nx1 = int(round(Gx / (2 * math.pi)))
                    ny1 = int(round(Gy / (2 * math.pi)))
                    nx2 = int(round(Gx2 / (2 * math.pi)))
                    ny2 = int(round(Gy2 / (2 * math.pi)))
                    dx = nx1 - nx2
                    dy = ny1 - ny2
                    if -8 <= dx <= 8 and -8 <= dy <= 8:
                        eps_val = coeffs.get((dx, dy), 0)
                        B[idx, idx2] = eps_val / kG_sq
                    else:
                        B[idx, idx2] = 0
            
            # Solve eigenvalue problem
            eigvals = np.linalg.eigvalsh(B)
            eigvals = np.sort(np.real(eigvals))
            # ω = 1/sqrt(λ), so bands are ordered by decreasing λ (increasing ω)
            # Actually ω = 1/sqrt(λ), so larger λ means smaller ω.
            # We want the 4th and 5th bands, which correspond to the 4th and 5th smallest ω.
            # ω_i = 1/sqrt(λ_i), so we need λ in decreasing order for ω in increasing order.
            eigvals_sorted = np.sort(eigvals)[::-1]  # descending λ
            # ω = 1/sqrt(λ)
            omegas = [1.0 / math.sqrt(v) for v in eigvals_sorted if v > 1e-12]
            # Take the first 5 bands
            if len(omegas) >= 5:
                return omegas[3], omegas[4]  # 4th and 5th bands (0-indexed: 3 and 4)
            return None, None
        
        # Sample k-points in the irreducible Brillouin zone
        # For square lattice: kx, ky in [0, π] with kx >= ky (or some other convention)
        # The irreducible BZ for square lattice is the triangle with vertices (0,0), (π,0), (π/2,π/2)
        # Actually it's the region: 0 <= ky <= kx <= π, or similar.
        
        # Let me use a grid of k-points
        kpoints = []
        # Along the boundary
        for i in range(15):
            t = i / 14.0
            kpoints.append((math.pi * t, 0))  # X direction
            kpoints.append((math.pi, math.pi * t))  # X to M
            kpoints.append((math.pi * (1 - t), math.pi * t))  # M to Γ
        # Add interior points
        kpoints.append((0, 0))
        kpoints.append((math.pi / 2, math.pi / 2))
        
        # Compute gap at each k-point
        min_omega_5 = float('inf')
        max_omega_4 = float('-inf')
        for kx, ky in kpoints:
            o4, o5 = compute_eigenvalues(kx, ky)
            if o4 is not None and o5 is not None:
                max_omega_4 = max(max_omega_4, o4)
                min_omega_5 = min(min_omega_5, o5)
        
        if min_omega_5 == float('inf') or max_omega_4 == float('-inf'):
            return 0.0
        
        gap = min_omega_5 - max_omega_4
        midgap = (min_omega_5 + max_omega_4) / 2.0
        if midgap == 0:
            return 0.0
        return gap / midgap
    
    # Generate initial super-ellipse cell
    def generate_superellipse(beta, r_beta):
        cell = [[0] * size for _ in range(size)]
        for i in range(size):
            for j in range(size):
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                if abs(x / r_beta) ** beta + abs(y / r_beta) ** beta < 1.0:
                    cell[i][j] = 1
        return cell
    
    # Initial parameters: super-ellipse with beta=4, matching area of circle radius 0.2
    beta = 4.0
    r_circle = 0.2
    area_circle = math.pi * r_circle ** 2
    # Area of super-ellipse: 4 * Gamma(1+1/beta)^2 / Gamma(1+2/beta) * R^2
    # For beta=4: Gamma(1.25) ≈ 0.9064, Gamma(1.5) ≈ 0.8862
    area_coeff = 4 * (0.9064 ** 2 / 0.8862)
    r_beta = math.sqrt(area_circle / area_coeff)
    
    best_cell = generate_superellipse(beta, r_beta)
    best_cell = enforce_symmetry(best_cell)
    best_gap = compute_tm_bands(best_cell)
    
    # Local search: flip symmetric groups of pixels
    random.seed(42)
    
    # Get all symmetric groups (orbits under square symmetry)
    def get_symmetric_groups(n):
        visited = set()
        groups = []
        for i in range(n):
            for j in range(n):
                if (i, j) not in visited:
                    group = set()
                    # Generate orbit under square symmetry
                    positions = [(i, j), (n-1-i, j), (i, n-1-j), (n-1-i, n-1-j)]
                    for pos in positions:
                        group.add(pos)
                        visited.add(pos)
                    groups.append(group)
        return groups
    
    groups = get_symmetric_groups(size)
    
    # Local search
    for iteration in range(200):
        if time.time() > deadline:
            break
        
        # Pick a random group
        idx = random.randint(0, len(groups) - 1)
        group = groups[idx]
        
        # Try flipping all pixels in the group
        new_cell = [row[:] for row in best_cell]
        current_val = new_cell[list(group)[0][0]][list(group)[0][1]]
        new_val = 1 - current_val
        for (i, j) in group:
            new_cell[i][j] = new_val
        
        new_cell = enforce_symmetry(new_cell)
        new_gap = compute_tm_bands(new_cell)
        
        if new_gap > best_gap:
            best_cell = new_cell
            best_gap = new_gap
    
    return best_cell
# EVOLVE-BLOCK-END
