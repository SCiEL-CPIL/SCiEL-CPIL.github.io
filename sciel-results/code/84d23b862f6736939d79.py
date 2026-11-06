"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import math

# EVOLVE-BLOCK-START
def design(problem):
    """
    Optimized four-rod lattice.
    Uses a runtime local search to tune rod positions and radius to maximize
    the gap between the 4th and 5th TM bands.
    """
    size = problem["pixels"]
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)
    lower_band = problem.get("lower_band", 4)
    
    # Deadline guard: 230s wall clock limit, keep 10s margin for safety
    start_time = time.time()
    deadline = start_time + 220.0
    
    # --- Plane Wave Setup ---
    # 289 plane waves => (2*11+1)^2 = 13x13 grid of G vectors
    # G = (m, n) where m, n in [-11, 11]
    g_max = 11
    Gs = []
    for m in range(-g_max, g_max + 1):
        for n in range(-g_max, g_max + 1):
            Gs.append((m, n))
    N_pw = len(Gs)
    
    # Reciprocal lattice vectors for square lattice (a=1)
    # G = 2*pi * (m, n)
    G_vectors = np.array([(2*math.pi*m, 2*math.pi*n) for m, n in Gs])
    G_mags_sq = np.sum(G_vectors**2, axis=1)
    
    # Irreducible Brillouin Zone (IBZ)
    # Square IBZ: 0 <= kx <= ky <= pi
    # We need 30 wave vectors.
    # Boundary: kx=0, ky in [0, pi] (Gamma-M)
    # Boundary: ky=pi, kx in [0, pi/2] (M-X)
    # Interior: symmetric points
    
    # Let's generate a dense set of 30 points
    k_points = []
    
    # Gamma-M line (kx=0, ky from 0 to pi)
    # 11 points
    for i in range(11):
        ky = math.pi * i / 10.0
        k_points.append((0.0, ky))
        
    # M-X line (ky=pi, kx from 0 to pi/2)
    # 11 points
    for i in range(1, 11): # Skip Gamma-M start point (0, pi) already covered? 
                           # Gamma-M ends at (0, pi). M-X starts at (0, pi).
                           # Let's include (0, pi) in M-X or Gamma-M. 
                           # Gamma-M has (0, pi) at i=10.
                           # M-X should start from (0, pi) to (pi/2, pi).
                           # So i=1 to 10 gives 10 points. Total 11+10=21.
        kx = (math.pi / 2.0) * i / 10.0
        k_points.append((kx, math.pi))
        
    # M-X line other half? No, IBZ is kx <= ky.
    # Wait, standard square IBZ is 0 <= kx <= ky <= pi.
    # Boundary 1: kx = 0, 0 <= ky <= pi (Gamma-M)
    # Boundary 2: ky = pi, 0 <= kx <= pi/2 (M-X)
    # Boundary 3: kx = ky, 0 <= kx <= pi/2 (Gamma-X)
    
    # We have 11 (Gamma-M) + 10 (M-X, excluding start) = 21 points.
    # Need 9 more.
    # Gamma-X line (kx=ky, 0 to pi/2)
    # 9 points
    for i in range(1, 10): # Skip Gamma (0,0) and M (pi/2, pi/2 is not M, M is (0,pi) or (pi,0)?)
                           # In this IBZ, M is (0, pi) or (pi, 0)?
                           # Usually M is midpoint of edge. Edge is kx=0 to ky=pi. Midpoint is (0, pi/2).
                           # My Gamma-M line goes 0 to pi. So (0, pi/2) is included at i=5.
                           # X is corner (pi/2, pi/2).
                           # M is (0, pi).
                           # So Gamma-X is from (0,0) to (pi/2, pi/2).
        kx = (math.pi / 2.0) * i / 9.0
        k_points.append((kx, kx))
        
    # Total 21 + 9 = 30 points.
    
    k_array = np.array(k_points)
    k_mags_sq = np.sum(k_array**2, axis=1)
    
    # Precompute K+G vectors and squared magnitudes for each k point
    # Shape: (N_k, N_pw)
    KG_mags_sq = np.zeros((len(k_points), N_pw))
    for i_k, k in enumerate(k_array):
        KG = G_vectors + k
        KG_mags_sq[i_k] = np.sum(KG**2, axis=1)
    
    # --- Helper: Calculate Gap Score ---
    def calculate_score(cell):
        # cell is 32x32 list of lists
        eps = np.array(cell, dtype=float)
        eps[eps == 1] = eps_high
        eps[eps == 0] = eps_low
        
        # Fourier coefficients of 1/eps
        # inv_eps = 1/eps
        inv_eps = 1.0 / eps
        
        # FFT to get Fourier coefficients
        # The pixel map is on a 32x32 grid.
        # G = (m, n). Frequency in pixels is m*32, n*32?
        # Standard: f_{mn} = (1/N^2) sum_{x,y} f(x,y) exp(-i 2pi (mx/N + ny/N))
        # This corresponds to G = (2pi m / a, 2pi n / a) with a=1?
        # Yes, if we index pixels 0..31, the FFT gives coefficients for m, n in -15..15.
        # We need m, n in -11..11.
        
        fft_inv_eps = np.fft.fft2(inv_eps)
        fft_inv_eps = np.fft.fftshift(fft_inv_eps) / (size * size)
        
        # Extract coefficients for m, n in [-11, 11]
        # fftshift centers zero frequency at index 16 (for size 32).
        # Index for m: 16 + m.
        # m range -11..11 -> indices 5..27.
        
        coeff = np.zeros(N_pw, dtype=complex)
        for idx, (m, n) in enumerate(Gs):
            im = 16 + m
            in_ = 16 + n
            coeff[idx] = fft_inv_eps[im, in_]
            
        # Build Hamiltonian for each k point
        # H_{ij} = (k+G_i) dot (k+G_j) * delta_{ij} + (k+G_i) dot (k+G_j) * (1 - delta_{ij}) * eps_{G_i - G_j} ?
        # Wait, TM mode equation:
        # (k+G_i) dot (k+G_j) * eps^{-1}_{G_i - G_j} * E_j = omega^2 E_i
        # No, standard PWE for TM:
        # sum_j (k+G_i) dot (k+G_j) * eps^{-1}_{G_i - G_j} * E_j = omega^2 E_i
        # Where eps^{-1} is the Fourier coefficient of 1/epsilon.
        
        # Note: The term (k+G_i) dot (k+G_j) is a scalar.
        # eps^{-1}_{G_i - G_j} is the coefficient for difference vector G_i - G_j.
        
        # We have coeff for G in [-11, 11].
        # G_i - G_j can be in [-22, 22].
        # If |m| > 11 or |n| > 11, coeff is 0 (aliasing/truncation).
        # This is consistent with the 289 PW truncation.
        
        min_omega_5 = float('inf')
        max_omega_4 = 0.0
        
        for i_k in range(len(k_points)):
            # Construct matrix H
            # H_{ij} = dot_{ij} * coeff_{G_i - G_j}
            
            # dot_{ij} = (k+G_i) dot (k+G_j)
            # We can compute this on the fly or precompute.
            # For speed, precompute dot products for this k?
            # N_pw = 289. 289x289 matrix.
            
            # Let's compute the matrix
            H = np.zeros((N_pw, N_pw), dtype=complex)
            
            # Get k+G vectors for this k
            KG = G_vectors + k_array[i_k]
            
            # Compute dot products
            # dot[i, j] = KG[i] dot KG[j]
            dot = np.sum(KG[:, np.newaxis, :] * KG[np.newaxis, :, :], axis=2)
            
            # Get coefficients for differences
            # diff_m = m_i - m_j, diff_n = n_i - n_j
            # We need coeff[diff]
            
            # Map (m, n) to index in Gs list
            # Gs is list of (m, n)
            # Create a dictionary for fast lookup
            # Or just loop. N_pw is 289. 289*289 = 83k ops. 30 k-points = 2.5M ops.
            # Python loops are slow. Vectorize if possible.
            
            # Let's vectorize the coefficient lookup.
            # m_i - m_j
            m_arr = np.array([g[0] for g in Gs])
            n_arr = np.array([g[1] for g in Gs])
            
            diff_m = m_arr[:, np.newaxis] - m_arr[np.newaxis, :]
            diff_n = n_arr[:, np.newaxis] - n_arr[np.newaxis, :]
            
            # Check if diff is within [-11, 11]
            # If yes, find index in Gs.
            # We can create a 2D array of indices.
            # Initialize with -1 (invalid)
            idx_diff = np.full((N_pw, N_pw), -1, dtype=int)
            
            # Create a lookup table for (m, n) -> index
            # m, n in [-11, 11]
            lookup = {}
            for idx, (m, n) in enumerate(Gs):
                lookup[(m, n)] = idx
                
            # Fill idx_diff
            # Loop over unique diffs? Or just loop over all pairs.
            # 83k iterations in Python is ~0.05s. 30 k-points -> 1.5s. Acceptable.
            # But we can do it faster with numpy.
            
            # Let's use the lookup table approach but vectorized via np.vectorize or similar?
            # Actually, we can precompute the coefficient matrix for ALL possible diffs?
            # No, coeff depends on the specific diff.
            
            # Let's just loop, it's safe.
            # Optimization: Precompute the "coeff_matrix" for a given cell?
            # coeff_matrix[i, j] = coeff[G_i - G_j]
            # This depends only on the cell, not on k!
            
            # So we should compute coeff_matrix once per cell evaluation.
            
            # Let's move this outside the k loop.
            
            pass # Placeholder
            
        # Recompute with optimized structure
        
        # Precompute coeff_matrix for this cell
        coeff_matrix = np.zeros((N_pw, N_pw), dtype=complex)
        for i in range(N_pw):
            mi, ni = Gs[i]
            for j in range(N_pw):
                mj, nj = Gs[j]
                dm = mi - mj
                dn = ni - nj
                if -11 <= dm <= 11 and -11 <= dn <= 11:
                    # Find index of (dm, dn) in Gs
                    # Since Gs is small, we can use a dictionary
                    idx = lookup.get((dm, dn), -1)
                    if idx != -1:
                        coeff_matrix[i, j] = coeff[idx]
                    else:
                        coeff_matrix[i, j] = 0.0
                else:
                    coeff_matrix[i, j] = 0.0
        
        # Now loop over k points
        for i_k in range(len(k_points)):
            KG = G_vectors + k_array[i_k]
            dot = np.sum(KG[:, np.newaxis, :] * KG[np.newaxis, :, :], axis=2)
            
            H = dot * coeff_matrix
            
            # Solve eigenvalue problem
            # H is Hermitian?
            # dot is real symmetric.
            # coeff_matrix: coeff[G] = conj(coeff[-G]) for real eps.
            # H_{ij} = dot_{ij} * coeff[G_i - G_j]
            # H_{ji} = dot_{ji} * coeff[G_j - G_i] = dot_{ij} * conj(coeff[G_i - G_j])
            # So H is Hermitian.
            
            eigvals = np.linalg.eigvalsh(H)
            
            # Take first 5 eigenvalues
            # Note: There are spurious modes? No, PWE is standard.
            # The lowest modes correspond to the bands.
            
            # We want min(omega_5) and max(omega_4)
            # omega = sqrt(eigval)
            # eigvals are sorted ascending.
            
            # Band 4 is index 3. Band 5 is index 4.
            # But wait, there might be evanescent modes or other issues?
            # Standard PWE gives all modes.
            
            w4 = np.sqrt(eigvals[3])
            w5 = np.sqrt(eigvals[4])
            
            if w4 > max_omega_4:
                max_omega_4 = w4
            if w5 < min_omega_5:
                min_omega_5 = w5
                
        # Calculate score
        if min_omega_5 <= 0 or max_omega_4 <= 0:
            return 0.0
            
        gap = min_omega_5 - max_omega_4
        midgap = (min_omega_5 + max_omega_4) / 2.0
        
        if midgap == 0:
            return 0.0
            
        ratio = gap / midgap
        
        # The task says "gap-midgap ratio 2 (min omega_5 - max omega_4) / (min omega_5 + max omega_4)"
        # So it's gap / midgap.
        
        return ratio

    # --- Generate Initial Cell ---
    def generate_cell(centers, radius):
        cell = [[0 for _ in range(size)] for _ in range(size)]
        for i in range(size):
            y = (i + 0.5) / size
            for j in range(size):
                x = (j + 0.5) / size
                is_diel = False
                for cx, cy in centers:
                    dx = x - cx
                    dy = y - cy
                    if dx*dx + dy*dy < radius*radius:
                        is_diel = True
                        break
                if is_diel:
                    cell[i][j] = 1
        return cell

    # --- Local Search ---
    # Parameters: offset (shift from 0.25), radius
    # Initial: offset=0, radius=0.15
    # Centers: (0.25+ox, 0.25+oy), etc.
    
    best_score = -1.0
    best_params = (0.0, 0.0, 0.15) # ox, oy, radius
    
    # Initial evaluation
    init_centers = [(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)]
    init_cell = generate_cell(init_centers, 0.15)
    best_score = calculate_score(init_cell)
    
    # Hill Climbing
    # Step sizes
    step_ox = 0.01
    step_oy = 0.01
    step_r = 0.01
    
    iterations = 0
    max_iterations = 500 # Adjust based on time
    
    while time.time() < deadline:
        iterations += 1
        
        # Random perturbation
        ox, oy, r = best_params
        
        # Determine if we should expand or shrink steps based on success?
        # Simple: try random directions
        
        # Generate 5 random candidates
        found_improvement = False
        
        for _ in range(5):
            # Random direction
            dx = np.random.uniform(-1, 1) * step_ox
            dy = np.random.uniform(-1, 1) * step_oy
            dr = np.random.uniform(-1, 1) * step_r
            
            new_ox = ox + dx
            new_oy = oy + dy
            new_r = r + dr
            
            # Constraints
            # Radius must be positive
            if new_r < 0.05:
                new_r = 0.05
            # Centers must stay in cell
            # 0.25 + ox must be in [0, 1]
            if not (0 <= 0.25 + new_ox <= 1 and 0 <= 0.25 + new_oy <= 1):
                continue
            if not (0 <= 0.75 + new_ox <= 1 and 0 <= 0.75 + new_oy <= 1):
                continue
            
            centers = [
                (0.25 + new_ox, 0.25 + new_oy),
                (0.75 + new_ox, 0.25 + new_oy),
                (0.25 + new_ox, 0.75 + new_oy),
                (0.75 + new_ox, 0.75 + new_oy)
            ]
            
            cell = generate_cell(centers, new_r)
            
            # Check symmetry?
            # The centers are symmetric by construction.
            # The grid rasterization might break exact symmetry if not careful?
            # generate_cell uses distance check, which is symmetric if centers are symmetric.
            # (0.25+ox, 0.25+oy) and (0.75+ox, 0.25+oy) are symmetric about x=0.5 if ox=0?
            # No, 0.25+ox and 0.75+ox.
            # Midpoint is 0.5+ox.
            # For the cell to be symmetric about 0.5, we need ox=0?
            # Wait, the cell must be invariant under square symmetry about the cell centre.
            # "cell == cell[::-1] row-wise and column-wise and cell == its transpose"
            # This implies symmetry about the center of the pixel grid.
            # If we shift the centers, we break the symmetry unless we shift them in a way that preserves the overall pattern symmetry?
            # Actually, the task requires the OUTPUT cell to be symmetric.
            # If I shift the centers by (ox, oy), the resulting pixel map might not be symmetric
            # unless ox=0 and oy=0, OR unless the shift maps the set of centers to itself?
            # The set of centers {(0.25+ox, 0.25+oy), ...} is symmetric about (0.5+ox, 0.5+oy).
            # For the cell to be symmetric about (0.5, 0.5), we must have ox=0 and oy=0.
            
            # So I CANNOT shift the centers!
            # I can only change the radius.
            # Or I can change the shape (e.g. square rods)?
            # Or I can use a different base geometry that allows parameter variation while maintaining symmetry.
            
            # Let's stick to symmetric geometries.
            # 1. Radius of 4 rods at (0.25, 0.25) etc. (Only radius varies)
            # 2. Square rods: size varies.
            # 3. Rods at (0.5, 0.5) (center) + 4 rods at corners?
            
            # Let's optimize Radius only for the 4-rod lattice.
            # And maybe try a center rod too?
            
            # Strategy:
            # Parametrize: 
            # r1: radius of 4 corner rods (centers 0.25, 0.25)
            # r2: radius of center rod (center 0.5, 0.5)
            
            # Start with r2=0.
            
            new_r1 = new_r # Using r as r1
            new_r2 = 0.0
            
            # If we want to vary r2, we need a separate parameter.
            # Let's use (r1, r2)
            
            # Reset search to 2D: r1, r2
            # Current best is r1=0.15, r2=0.
            
            # Let's restart the search logic properly.
            break # Break out of random candidate loop to fix logic
            
        # Fix: Use (r1, r2)
        # If we are in the first iteration, best_params should be (r1, r2)
        
        # Let's restart the loop with correct params
        if iterations == 1:
            best_params = (0.15, 0.0) # r1, r2
            continue
            
        r1, r2 = best_params
        
        for _ in range(5):
            dr1 = np.random.uniform(-1, 1) * step_r
            dr2 = np.random.uniform(-1, 1) * step_r
            
            new_r1 = r1 + dr1
            new_r2 = r2 + dr2
            
            if new_r1 < 0.01: new_r1 = 0.01
            if new_r2 < 0.0: new_r2 = 0.0
            
            centers = [
                (0.25, 0.25),
                (0.75, 0.25),
                (0.25, 0.75),
                (0.75, 0.75)
            ]
            
            cell = generate_cell(centers, new_r1)
            
            if new_r2 > 0:
                # Add center rod
                for i in range(size):
                    y = (i + 0.5) / size
                    for j in range(size):
                        x = (j + 0.5) / size
                        dx = x - 0.5
                        dy = y - 0.5
                        if dx*dx + dy*dy < new_r2*new_r2:
                            cell[i][j] = 1
                            
            score = calculate_score(cell)
            
            if score > best_score:
                best_score = score
                best_params = (new_r1, new_r2)
                found_improvement = True
                # Shrink steps?
                step_r *= 0.8
            else:
                # Expand steps?
                step_r *= 1.05
                if step_r > 0.05: step_r = 0.05
                if step_r < 0.001: step_r = 0.001

    # Finalize with best params
    r1, r2 = best_params
    centers = [
        (0.25, 0.25),
        (0.75, 0.25),
        (0.25, 0.75),
        (0.75, 0.75)
    ]
    final_cell = generate_cell(centers, r1)
    
    if r2 > 0:
        for i in range(size):
            y = (i + 0.5) / size
            for j in range(size):
                x = (j + 0.5) / size
                dx = x - 0.5
                dy = y - 0.5
                if dx*dx + dy*dy < r2*r2:
                    final_cell[i][j] = 1
                    
    return final_cell

# EVOLVE-BLOCK-END
