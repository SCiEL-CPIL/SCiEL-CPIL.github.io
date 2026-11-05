import numpy as np
import time

def design(problem):
    start_time = time.time()
    deadline = start_time + 180  # 180s deadline
    
    size = problem["pixels"]
    eps_low = problem["eps_low"]
    eps_high = problem["eps_high"]
    
    # Helper to generate symmetric cell
    def make_cell(shape_type, param1, param2=None):
        cell = [[0]*size for _ in range(size)]
        center = (size - 1) / 2.0
        for i in range(size):
            for j in range(size):
                # Use symmetry to only compute one quadrant? 
                # For simplicity, compute all, then enforce symmetry by averaging or just checking.
                # Since the shapes are symmetric, just computing is fine if float precision is ok.
                # To be safe, compute i, j and mirror.
                pass
        
        # Better: generate based on distance from center
        for i in range(size):
            for j in range(size):
                x = (i - center) / (size/2.0)
                y = (j - center) / (size/2.0)
                # x, y in [-1, 1]
                is_dielectric = False
                if shape_type == 'circle':
                    r = param1
                    if x*x + y*y < r*r:
                        is_dielectric = True
                elif shape_type == 'square':
                    side = param1
                    if abs(x) < side/2 and abs(y) < side/2:
                        is_dielectric = True
                elif shape_type == 'super_ellipse':
                    r = param1
                    n = param2
                    if abs(x/r)**n + abs(y/r)**n < 1.0:
                        is_dielectric = True
                        
                if is_dielectric:
                    cell[i][j] = 1
                else:
                    cell[i][j] = 0
                    
        # Enforce symmetry explicitly to handle floating point errors
        # The cell must be invariant under square symmetry group
        # 1. Reverse rows
        # 2. Reverse cols
        # 3. Transpose
        # Since we generated it symmetrically, it should be fine.
        # But to be absolutely sure, we can take the majority vote of symmetric pixels.
        # For 32x32, center is 15.5.
        # Pixel (i,j) is symmetric to (31-i, j), (i, 31-j), (31-i, 31-j).
        for i in range(size):
            for j in range(size):
                # Average the symmetric pixels
                s = cell[i][j] + cell[31-i][j] + cell[i][31-j] + cell[31-i][31-j]
                if s >= 2:
                    # Set all to 1 or 0?
                    # If they are not consistent, it's a problem.
                    # But our generation is deterministic.
                    pass
        return cell

    def compute_gap(cell):
        # Compute Fourier coefficients
        # epsilon(r) = eps_low + (eps_high - eps_low) * cell(r)
        # Discrete Fourier Transform of the 32x32 cell
        # But we need continuous Fourier coefficients for the plane wave method?
        # The evaluator uses "exact Fourier coefficients of the pixel map".
        # This implies we treat the pixel map as a square wave function.
        # eps(x,y) is piecewise constant.
        # epsilon_G = (1/A) * integral eps(x,y) exp(-i G.r) dx dy
        # For a pixel map, this is an analytical integration of rectangles.
        
        # However, implementing exact analytical integration for 289 G vectors might be complex.
        # Approximation: Use DFT of the pixel grid?
        # The prompt says "exact-coefficient plane-wave method".
        # If I use DFT, it might differ from the evaluator.
        # But for a first pass, DFT is a good approximation.
        # Or I can compute the integral analytically.
        
        # Let's stick to DFT for speed and simplicity, assuming it's close enough.
        # Actually, the "pixel map" usually implies the discrete values are used to define the dielectric function.
        # If the evaluator uses exact coefficients, it integrates the square pixels.
        # I will implement the exact integration for the rectangular pixels.
        
        def get_fourier_coefficients(cell):
            coeffs = {}
            # G vectors
            # Range of G: -8 to 8 (for 17x17)
            # a = 1
            # G = 2*pi*n
            for nx in range(-8, 9):
                for ny in range(-8, 9):
                    Gx = 2 * np.pi * nx
                    Gy = 2 * np.pi * ny
                    if nx == 0 and ny == 0:
                        # Average
                        avg = sum(sum(row) for row in cell) / (size*size)
                        eps_G = eps_low + (eps_high - eps_low) * avg
                        coeffs[(nx, ny)] = eps_G
                    else:
                        # Integral over the cell
                        # Sum over all pixels (i,j) where cell[i][j] == 1
                        # Pixel (i,j) covers x in [i/size, (i+1)/size], y in [j/size, (j+1)/size]
                        # Integral of exp(-i(Gx x + Gy y)) over the pixel
                        # = exp(-i Gx (i+0.5)/size) * (2 sin(Gx/2size) / Gx) 
                        #   * exp(-i Gy (j+0.5)/size) * (2 sin(Gy/2size) / Gy)
                        # If Gx == 0, term is 1/size.
                        
                        def sinc_term(G, n):
                            if G == 0:
                                return 1.0 / size
                            else:
                                # Pixel center at (n+0.5)/size
                                # Integral from n/size to (n+1)/size
                                # = [exp(-iGx) - exp(-iG(n+1)/size)] / (-iG) * exp(iG/2size)?
                                # Simpler: 
                                # int_n^{n+1} exp(-i G x/size) dx = size * exp(-i G (n+0.5)/size) * sinc(G/2)
                                # where sinc(u) = sin(pi u) / (pi u)
                                # Here argument is G/(2*size)? No.
                                # Let u = G/size.
                                # int = size * exp(-i u (n+0.5)) * sin(u/2) / (u/2)
                                # = 2 size / u * exp(-i u (n+0.5)) * sin(u/2)
                                # = 2 / (G/size) * exp(-i G (n+0.5)/size) * sin(G/(2size))
                                # = 2 size / G * exp(-i G (n+0.5)/size) * sin(G/(2size))
                                return (2.0 / G) * np.exp(-1j * G * (n + 0.5) / size) * np.sin(G / (2.0 * size))
                        
                        sum_val = 0.0
                        for i in range(size):
                            for j in range(size):
                                if cell[i][j] == 1:
                                    term_x = sinc_term(Gx, i)
                                    term_y = sinc_term(Gy, j)
                                    sum_val += term_x * term_y
                        # The integral is over the unit cell area 1x1.
                        # The formula above gives the integral.
                        # epsilon_G = (eps_high - eps_low) * sum_val
                        # Note: The DFT normalization might vary.
                        # Standard: eps_G = (1/A) int eps(r) e^{-iG.r} dr
                        # A=1.
                        eps_G = (eps_high - eps_low) * sum_val
                        coeffs[(nx, ny)] = eps_G
            return coeffs

        # This analytical integration is O(N^2 * N_G^2) which is 1024 * 289 = 300k ops. Fast.
        
        coeffs = get_fourier_coefficients(cell)
        
        # Define G vectors
        G_vecs = []
        for nx in range(-8, 9):
            for ny in range(-8, 9):
                G_vecs.append((2*np.pi*nx, 2*np.pi*ny, nx, ny))
                
        # K points in IBZ
        # IBZ: 0 <= kx <= ky <= pi
        # Sample 10x10 grid in the triangle
        k_points = []
        for i in range(10):
            for j in range(10):
                kx = i / 9.0 * np.pi
                ky = j / 9.0 * np.pi
                if kx <= ky:
                    k_points.append((kx, ky))
                    
        # Also include high symmetry points
        k_points.append((0, 0)) # Gamma
        k_points.append((np.pi, 0)) # X
        k_points.append((np.pi, np.pi)) # M
        k_points.append((np.pi/2, np.pi/2)) # Center of M-X?
        
        # For each k point, solve eigenvalue problem
        # E c = mu D c
        # E is Toeplitz from coeffs
        # D is diagonal |k+G|^2
        
        all_omegas = []
        
        for kx, ky in k_points:
            # Build E matrix (289x289)
            E = np.zeros((289, 289), dtype=complex)
            D = np.zeros(289, dtype=float)
            
            for idx_g, (Gx, Gy, nx, ny) in enumerate(G_vecs):
                # D[idx_g] = |k+G|^2
                kx_g = kx + Gx
                ky_g = ky + Gy
                D[idx_g] = kx_g*kx_g + ky_g*ky_g
                
                for idx_gp, (Gxp, Gyp, nxp, nyp) in enumerate(G_vecs):
                    # E[idx_g, idx_gp] = eps_{G - G'}
                    # G - G' = (nx - nxp, ny - nyp) * 2pi
                    dx = nx - nxp
                    dy = ny - nyp
                    if (dx, dy) in coeffs:
                        E[idx_g, idx_gp] = coeffs[(dx, dy)]
                    else:
                        # If G-G' is outside the calculated range, it should be 0?
                        # With 17x17, max diff is 16.
                        # Range -8..8 covers diff -16..16.
                        # So it should be in coeffs.
                        E[idx_g, idx_gp] = 0.0
            
            # Solve E c = mu D c  =>  D^{-1} E c = mu c
            # D is diagonal, so D_inv_E = E / D[:, None]
            D_inv_E = E / D[:, None]
            
            # Eigenvalues
            # D_inv_E is not Hermitian?
            # E is Hermitian (eps is real). D is real diagonal.
            # D^{-1} E is similar to a Hermitian matrix?
            # Let c' = D^{1/2} c.
            # D^{-1} E D^{1/2} c' = mu c'
            # This matrix is Hermitian if E is Hermitian.
            # (D^{-1} E D^{1/2})^dagger = D^{1/2} E^dagger D^{-1} = D^{1/2} E D^{-1}.
            # Not exactly the same.
            # But the eigenvalues are real.
            # We can just use np.linalg.eigvals.
            
            try:
                mu_vals = np.linalg.eigvals(D_inv_E)
                # mu = c^2 / omega^2. Let c=1.
                # omega = 1 / sqrt(mu)
                # Sort mu
                mu_real = np.real(mu_vals)
                # Filter out negative or zero (numerical errors)
                mu_valid = mu_real[mu_real > 1e-10]
                if len(mu_valid) < 5:
                    continue
                mu_sorted = np.sort(mu_valid)
                # We want the 4th and 5th bands.
                # Smallest mu -> Largest omega.
                # Largest mu -> Smallest omega.
                # Wait.
                # omega = 1/sqrt(mu).
                # Band 1 (lowest omega) corresponds to largest mu.
                # Band 5 (highest omega in the gap region) corresponds to smaller mu.
                # Let's list omegas.
                omegas = 1.0 / np.sqrt(mu_sorted)
                omegas = np.sort(omegas) # Ascending order
                # Band 1 is omegas[0], Band 4 is omegas[3], Band 5 is omegas[4]
                if len(omegas) >= 5:
                    all_omegas.append(omegas)
            except Exception:
                continue
        
        if not all_omegas:
            return 0.0
            
        # Find max of band 4 and min of band 5
        # Band 4 is index 3
        # Band 5 is index 4
        max_band4 = max(o[3] for o in all_omegas)
        min_band5 = min(o[4] for o in all_omegas)
        
        gap = min_band5 - max_band4
        midgap = (min_band5 + max_band4) / 2.0
        
        if midgap == 0:
            return 0.0
            
        ratio = gap / midgap
        return ratio

    # Search
    candidates = [
        ('circle', 0.2, None),
        ('circle', 0.25, None),
        ('circle', 0.3, None),
        ('square', 0.4, None),
        ('square', 0.5, None),
        ('super_ellipse', 0.25, 4.0),
        ('super_ellipse', 0.3, 4.0),
    ]
    
    best_score = -1.0
    best_cell = None
    
    for shape, p1, p2 in candidates:
        if time.time() > deadline:
            break
            
        cell = make_cell(shape, p1, p2)
        score = compute_gap(cell)
        
        if score > best_score:
            best_score = score
            best_cell = cell
            
    if best_cell is None:
        # Fallback
        best_cell = make_cell('circle', 0.2, None)
        
    return best_cell
