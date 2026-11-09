import time
import math
import numpy as np

def design(problem):
    """Design a 32x32 photonic crystal unit cell maximizing the 4-5 TM band gap."""
    start_time = time.time()
    deadline = start_time + 100  # 100s safety margin within 120s budget
    
    size = problem["pixels"]  # 32
    eps_low = problem.get("eps_low", 1)
    eps_high = problem.get("eps_high", 12)
    
    # Plane-wave setup: 289 = 17x17
    N = 17  # odd, so -8..8
    nmax = (N - 1) // 2  # 8
    
    # Generate G-vectors
    G = []
    for mx in range(-nmax, nmax + 1):
        for my in range(-nmax, nmax + 1):
            G.append((mx, my))
    G = np.array(G, dtype=float)
    NG = len(G)  # 289
    
    # G^2 magnitudes
    G2 = np.sum(G * G, axis=1)
    
    def compute_fourier(cell):
        """Compute Fourier coefficients of 1/eps for the cell."""
        # cell is 32x32, 1 = dielectric (eps_high), 0 = air (eps_low)
        eps = np.where(np.array(cell) == 1, eps_high, eps_low)
        inv_eps = 1.0 / eps
        
        # Compute Fourier coefficients via FFT
        # The pixel grid: pixel (i,j) covers [i/32, (i+1)/32] x [j/32, (j+1)/32]
        # We need integral of inv_eps(x,y) * exp(-i G . r) over the unit cell
        # Using FFT: F[G] = sum_{i,j} inv_eps[i,j] * integral over pixel
        # For a pixel at (i,j), the integral of exp(-i*(mx*2pi*x + my*2pi*y)) dx dy
        # = [1 - exp(-i*mx*2pi/32)] / (-i*mx*2pi) * [1 - exp(-i*my*2pi/32)] / (-i*my*2pi)
        # for mx,my != 0; = 1/32^2 for mx=my=0
        
        # Actually, let's use the standard approach:
        # F[mx,my] = (1/32^2) * sum_{i,j} inv_eps[i,j] * exp(-i*2*pi*(mx*(i+0.5)/32 + my*(j+0.5)/32))
        # This is the DFT of the pixel values (with the pixel-center phase)
        
        # Compute DFT
        F = np.zeros((N, N), dtype=complex)
        for mx in range(N):
            gmx = mx - nmax
            for my in range(N):
                gmy = my - nmax
                if gmx == 0 and gmy == 0:
                    F[mx, my] = np.mean(inv_eps)
                else:
                    # Compute sum over pixels
                    # Phase: exp(-i*2*pi*(gmx*(i+0.5)/32 + gmy*(j+0.5)/32))
                    i_idx = np.arange(size)
                    j_idx = np.arange(size)
                    phase_i = np.exp(-1j * 2 * np.pi * gmx * (i_idx + 0.5) / size)
                    phase_j = np.exp(-1j * 2 * np.pi * gmy * (j_idx + 0.5) / size)
                    # F = (1/size^2) * sum_i sum_j inv_eps[i,j] * phase_i[i] * phase_j[j]
                    # = (1/size^2) * (phase_i^T @ inv_eps @ phase_j)
                    F[mx, my] = (phase_i @ inv_eps @ phase_j) / (size * size)
        
        return F
    
    def solve_bands(F):
        """Solve the eigenvalue problem for TM bands at given k-points."""
        # For TM polarization: (G - k)^2 * F[G - G'] * c[G'] = omega^2 * c[G]
        # where F is the Fourier transform of 1/eps
        
        # k-points in irreducible BZ: 30 points
        # The irreducible BZ of square lattice is the triangle from (0,0) to (pi,0) to (pi,pi)
        # Let's use a set of k-points
        
        k_points = []
        # Along Gamma-X: (0,0) to (pi,0)
        for t in np.linspace(0, 1, 8):
            k_points.append((t * np.pi, 0))
        # Along X-M: (pi,0) to (pi,pi)
        for t in np.linspace(0, 1, 8):
            k_points.append((np.pi, t * np.pi))
        # Along M-Gamma: (pi,pi) to (0,0)
        for t in np.linspace(0, 1, 8):
            k_points.append((t * np.pi, t * np.pi))
        # Interior points
        k_points.append((np.pi/2, np.pi/4))
        k_points.append((3*np.pi/4, np.pi/4))
        k_points.append((np.pi/2, np.pi/2))
        
        max_omega4 = 0
        min_omega5 = np.inf
        
        for kx, ky in k_points:
            # Build the matrix
            # A[G, G'] = (G - k)^2 * F[G - G']
            # where F is indexed by G - G'
            
            # For each G_i, G_j, we need F[(G_i - G_j)]
            # G_i - G_j has components in range [-16, 16], but we only have [-8, 8]
            # For a periodic dielectric, F is periodic in reciprocal space, so
            # F[mx,my] = F[mx + 32, my] etc. But since we're using a finite number of PWs,
            # we need to handle this carefully.
            
            # Actually, for a unit cell with period 1, the Fourier series is:
            # 1/eps(r) = sum_{G} F[G] exp(i G . r)
            # where G = (2pi*mx, 2pi*my)
            
            # The eigenvalue equation is:
            # sum_{G'} (G - k)^2 F[G - G'] c[G'] = omega^2 c[G]
            
            # Let's build this matrix
            A = np.zeros((NG, NG), dtype=complex)
            for i in range(NG):
                Gi = G[i]
                for j in range(NG):
                    Gj = G[j]
                    dGx = int(Gi[0] - Gj[0])
                    dGy = int(Gi[1] - Gj[1])
                    # Map to F index
                    fi = dGx + nmax
                    fj = dGy + nmax
                    if 0 <= fi < N and 0 <= fj < N:
                        Fval = F[fi, fj]
                    else:
                        Fval = 0  # Truncation: ignore higher harmonics
                    A[i, j] = (Gi[0] - kx)**2 + (Gi[1] - ky)**2 if j == i else 0
                    # Wait, that's wrong. Let me redo.
                    # A[i,j] = (G_i - k)^2 * F[G_i - G_j]
                    # But (G_i - k)^2 is a scalar, and F depends on G_i - G_j
                    # This is NOT diagonal in general.
            
            # Let me redo the matrix construction properly
            A = np.zeros((NG, NG), dtype=complex)
            for i in range(NG):
                Gi = G[i]
                Gi_k2 = (Gi[0] - kx)**2 + (Gi[1] - ky)**2
                for j in range(NG):
                    Gj = G[j]
                    dGx = int(Gi[0] - Gj[0])
                    dGy = int(Gi[1] - Gj[1])
                    fi = dGx + nmax
                    fj = dGy + nmax
                    if 0 <= fi < N and 0 <= fj < N:
                        Fval = F[fi, fj]
                    else:
                        Fval = 0
                    A[i, j] = Gi_k2 * Fval
            
            # Solve eigenvalue problem: A c = omega^2 c
            # A is Hermitian (since F[-G] = conj(F[G]) and the structure is real)
            try:
                eigvals = np.linalg.eigvalsh(A)
                # Take the 5 smallest positive eigenvalues
                eigvals = np.sort(eigvals)
                # Filter out near-zero or negative
                eigvals = eigvals[eigvals > 1e-10]
                if len(eigvals) >= 5:
                    omega4 = np.sqrt(eigvals[3])  # 4th band (0-indexed: 3)
                    omega5 = np.sqrt(eigvals[4])  # 5th band
                    max_omega4 = max(max_omega4, omega4)
                    min_omega5 = min(min_omega5, omega5)
            except:
                pass
        
        if min_omega5 == np.inf:
            return 0.0
        gap = min_omega5 - max_omega4
        midgap = (min_omega5 + max_omega4) / 2
        if midgap <= 0:
            return 0.0
        return gap / midgap
    
    # Initial design: try a few hand-designed candidates
    def make_square_rod(radius_frac):
        """Square dielectric rod."""
        cell = []
        for i in range(size):
            row = []
            for j in range(size):
                x = abs((i + 0.5) / size - 0.5)
                y = abs((j + 0.5) / size - 0.5)
                if x < radius_frac and y < radius_frac:
                    row.append(1)
                else:
                    row.append(0)
            cell.append(row)
        return cell
    
    def make_air_hole(dielectric_frac, hole_frac):
        """Dielectric background with square air hole."""
        cell = []
        for i in range(size):
            row = []
            for j in range(size):
                x = abs((i + 0.5) / size - 0.5)
                y = abs((j + 0.5) / size - 0.5)
                if x < hole_frac and y < hole_frac:
                    row.append(0)  # air hole
                else:
                    row.append(1)  # dielectric
            cell.append(row)
        return cell
    
    def make_cross(arm_width):
        """Cross-shaped dielectric element."""
        cell = []
        for i in range(size):
            row = []
            for j in range(size):
                x = abs((i + 0.5) / size - 0.5)
                y = abs((j + 0.5) / size - 0.5)
                if x < arm_width/2 or y < arm_width/2:
                    row.append(1)
                else:
                    row.append(0)
            cell.append(row)
        return cell
    
    # Try several candidates
    candidates = []
    candidates.append(make_square_rod(0.3))
    candidates.append(make_square_rod(0.35))
    candidates.append(make_square_rod(0.4))
    candidates.append(make_air_hole(0.7, 0.2))
    candidates.append(make_air_hole(0.7, 0.25))
    candidates.append(make_air_hole(0.7, 0.3))
    candidates.append(make_cross(0.3))
    candidates.append(make_cross(0.35))
    
    best_score = -1
    best_cell = candidates[0]
    
    for cell in candidates:
        F = compute_fourier(cell)
        score = solve_bands(F)
        if score > best_score:
            best_score = score
            best_cell = [row[:] for row in cell]
    
    # Local search: flip pixels in the 8x8 fundamental domain
    def get_fundamental(cell):
        """Get the 8x8 top-left quadrant."""
        return [cell[i][:8] for i in range(8)]
    
    def set_fundamental(cell, fund):
        """Set the full cell from the 8x8 fundamental domain."""
        new_cell = [[0]*size for _ in range(size)]
        for i in range(8):
            for j in range(8):
                val = fund[i][j]
                new_cell[i][j] = val
                new_cell[i][size-1-j] = val
                new_cell[size-1-i][j] = val
                new_cell[size-1-i][size-1-j] = val
                new_cell[j][i] = val
                new_cell[j][size-1-i] = val
                new_cell[size-1-j][i] = val
                new_cell[size-1-j][size-1-i] = val
        return new_cell
    
    # Simulated annealing on the fundamental domain
    fund = get_fundamental(best_cell)
    current_score = best_score
    best_score_sa = best_score
    best_fund = [row[:] for row in fund]
    
    temp = 0.1
    cooling = 0.995
    min_temp = 0.001
    
    while time.time() < deadline and temp > min_temp:
        # Pick a random pixel in the fundamental domain
        i = np.random.randint(0, 8)
        j = np.random.randint(0, 8)
        
        # Flip it
        fund[i][j] = 1 - fund[i][j]
        
        cell = set_fundamental(fund, fund)
        F = compute_fourier(cell)
        new_score = solve_bands(F)
        
        if new_score > current_score or np.random.random() < math.exp((new_score - current_score) / temp):
            current_score = new_score
            if new_score > best_score_sa:
                best_score_sa = new_score
                best_fund = [row[:] for row in fund]
        else:
            # Revert
            fund[i][j] = 1 - fund[i][j]
        
        temp *= cooling
    
    # Return the best cell found
    best_cell_final = set_fundamental(best_fund, best_fund)
    
    # Verify symmetry
    for i in range(size):
        for j in range(size):
            assert best_cell_final[i][j] == best_cell_final[size-1-i][size-1-j]
            assert best_cell_final[i][j] == best_cell_final[j][i]
            assert best_cell_final[i][j] == best_cell_final[size-1-i][j]
    
    return best_cell_final
