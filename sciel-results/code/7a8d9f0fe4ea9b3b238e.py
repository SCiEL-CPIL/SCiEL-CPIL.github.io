"""Photonic-crystal unit cell design: maximise the TM band gap between bands 4 and 5."""
import numpy as np
import time
import math

def design(problem):
    """Design a symmetric 32x32 unit cell with a large TM band gap between bands 4 and 5."""
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before the 230s wall-clock kill

    size = problem["pixels"]  # 32
    eps_low = problem["eps_low"]   # 1
    eps_high = problem["eps_high"]  # 12
    lower_band = problem["lower_band"]  # 4
    polarization = problem.get("polarization", "TM")

    # ---------- Plane-wave solver setup ----------
    # Irreducible Brillouin zone for square lattice (a=1):
    # kx >= 0, ky >= 0, kx >= ky
    # We use a set of 30 wave vectors covering the boundary and interior.
    
    # Define the BZ boundary points (irreducible wedge):
    # M = (0.5, 0), X = (0.5, 0.5), and the line from Gamma=(0,0) to M, and Gamma to X.
    # Actually the irreducible BZ is: 0 <= kx <= 0.5, 0 <= ky <= 0.5, ky <= kx.
    # Key points: Gamma=(0,0), M=(0.5,0), X=(0.5,0.5)
    # We sample along Gamma->M, M->X, and Gamma->X, plus some interior points.
    
    k_points = []
    # Gamma to M: (t*0.5, 0) for t in [0, 1]
    for t in np.linspace(0, 1, 11):
        k_points.append((t * 0.5, 0.0))
    # M to X: (0.5, t*0.5) for t in [0, 1]
    for t in np.linspace(0, 1, 11):
        k_points.append((0.5, t * 0.5))
    # Gamma to X: (t*0.5, t*0.5) for t in [0, 1]
    for t in np.linspace(0, 1, 10):
        k_points.append((t * 0.5, t * 0.5))
    
    # Remove duplicates
    seen = set()
    unique_k_points = []
    for k in k_points:
        key = (round(k[0], 6), round(k[1], 6))
        if key not in seen:
            seen.add(key)
            unique_k_points.append(k)
    k_points = unique_k_points[:30]  # Take first 30 unique
    
    # ---------- Fourier coefficients of epsilon ----------
    # For a pixel map, we compute the Fourier coefficients exactly.
    # epsilon(G) = (1/N^2) * sum_{i,j} eps(x_i, y_j) * exp(-i G . (x_i, y_j))
    # where x_i = (i+0.5)/N - 0.5, y_j = (j+0.5)/N - 0.5, G = (2*pi*m, 2*pi*n)
    
    # G-vectors: we need enough to capture the pixel structure.
    # For a 32x32 pixel map, we use G-vectors up to some order.
    # The evaluator uses 289 plane waves, so G vectors from -12 to 12 (25x25=625? no, 289=17^2)
    # Actually 289 = 17^2, so G vectors from -8 to 8 (17 values).
    
    G_max = 8  # G vectors from -8 to 8, giving 17x17 = 289 plane waves
    G_values = list(range(-G_max, G_max + 1))
    G_vectors = [(2 * math.pi * gx, 2 * math.pi * gy) for gx in G_values for gy in G_values]
    N_pw = len(G_vectors)  # 289
    
    # Precompute pixel positions
    pixel_positions = []
    for i in range(size):
        x = (i + 0.5) / size - 0.5
        for j in range(size):
            y = (j + 0.5) / size - 0.5
            pixel_positions.append((x, y))
    
    def compute_eps_fourier(cell):
        """Compute the Fourier coefficients of the epsilon map."""
        cell_arr = np.array(cell, dtype=np.float64)
        eps_values = np.where(cell_arr == 1, eps_high, eps_low)
        
        # Compute Fourier coefficients
        # eps(G) = (1/N^2) * sum_{pixels} eps(pixel) * exp(-i G . pixel_pos)
        eps_fourier = np.zeros(N_pw, dtype=np.complex128)
        
        for idx, (gx, gy) in enumerate(G_vectors):
            Gx, Gy = gx, gy
            # Compute sum over all pixels
            total = 0.0
            for pidx, (x, y) in enumerate(pixel_positions):
                total += eps_values[pidx // size, pidx % size] * math.cos(Gx * x + Gy * y)
                # For imaginary part:
                # Actually, let's compute both real and imaginary parts
            # Let me redo this more carefully with numpy for speed
            
        # More efficient: use numpy
        eps_fourier = np.zeros(N_pw, dtype=np.complex128)
        for idx, (Gx, Gy) in enumerate(G_vectors):
            phase = np.array([Gx * x + Gy * y for (x, y) in pixel_positions])
            total = np.sum(eps_values.ravel() * np.exp(-1j * phase))
            eps_fourier[idx] = total / (size * size)
        
        return eps_fourier
    
    def solve_bands_at_k(kx, ky, eps_fourier):
        """Solve the TM band structure at wave vector (kx, ky)."""
        # For TM polarization, the eigenvalue problem is:
        # det[ (k + G)^2 * delta_{GG'} - (1/epsilon(G - G')) ] = 0
        # More precisely, the matrix elements are:
        # M[G, G'] = (k + G)^2 * delta_{G,G'} - (1/epsilon(G - G'))
        # where epsilon is the Fourier series of the dielectric function.
        
        # Actually, the standard formulation for TM:
        # (k + G)^2 * E(k + G) = (1/epsilon) convolved with E
        # In matrix form: sum_{G'} [(k+G)^2 delta_{G,G'} - 1/eps(G-G')] E(k+G') = omega^2 E(k+G)
        
        # Let's build the matrix
        k_vec = np.array([kx, ky])
        
        # Build the matrix
        M = np.zeros((N_pw, N_pw), dtype=np.complex128)
        
        # Precompute 1/eps for each G difference
        # eps(G - G') for all pairs
        # We need the inverse of the dielectric function in real space, then Fourier transform
        # Actually, for the standard plane-wave method:
        # The matrix is: M[G, G'] = (k + G)^2 delta_{G,G'} - (1/eps(G - G'))
        # where 1/eps is the Fourier series of 1/epsilon(x)
        
        # Let me compute the Fourier coefficients of 1/epsilon instead
        # 1/eps(x) = sum_G (1/eps)(G) exp(i G . x)
        # Then the matrix element is: (k+G)^2 delta - (1/eps)(G - G')
        
        return M
    
    # Let me restructure. I'll compute 1/eps Fourier coefficients.
    
    def compute_inv_eps_fourier(cell):
        """Compute Fourier coefficients of 1/epsilon(x)."""
        cell_arr = np.array(cell, dtype=np.float64)
        eps_values = np.where(cell_arr == 1, eps_high, eps_low)
        inv_eps_values = 1.0 / eps_values
        
        inv_eps_fourier = np.zeros(N_pw, dtype=np.complex128)
        for idx, (Gx, Gy) in enumerate(G_vectors):
            phase = np.array([Gx * x + Gy * y for (x, y) in pixel_positions])
            total = np.sum(inv_eps_values.ravel() * np.exp(-1j * phase))
            inv_eps_fourier[idx] = total / (size * size)
        
        return inv_eps_fourier
    
    def solve_tm_bands(kx, ky, inv_eps_fourier):
        """Solve TM eigenvalue problem at wave vector (kx, ky)."""
        k_vec = np.array([kx, ky])
        
        # Build matrix M[G, G'] = (k + G)^2 delta_{G,G'} - inv_eps(G - G')
        # We need inv_eps(G - G') for all pairs G, G'
        # G - G' ranges from -2*G_max to 2*G_max
        
        # Precompute inv_eps for all needed differences
        # The difference G - G' has components from -2*G_max to 2*G_max
        # But we only computed up to G_max. We need to extend or handle this.
        
        # Actually, for the Galerkin method with N_pw plane waves, we need:
        # M[G, G'] = (k + G)^2 delta - inv_eps(G - G')
        # where inv_eps is the Fourier coefficient at (G - G')
        
        # Since G and G' both range from -G_max to G_max, G - G' ranges from -2*G_max to 2*G_max
        # We need inv_eps for these larger G values.
        
        # Let me recompute inv_eps with a larger range
        # Or better: compute the matrix directly
        
        M = np.zeros((N_pw, N_pw), dtype=np.complex128)
        
        for i, Gi in enumerate(G_vectors):
            for j, Gj in enumerate(G_vectors):
                if i == j:
                    k_plus_G = k_vec + np.array(Gi)
                    M[i, j] = np.dot(k_plus_G, k_plus_G)
                else:
                    G_diff = (Gi[0] - Gj[0], Gi[1] - Gj[1])
                    # Need inv_eps at G_diff
                    # Compute on the fly or precompute
                    # For now, compute directly
                    # inv_eps(G_diff) = (1/N^2) sum_{pixels} (1/eps(pixel)) exp(-i G_diff . pixel_pos)
                    # This is expensive. Let me precompute.
                    pass
        
        # This is too slow. Let me precompute inv_eps for all needed G differences.
        # G_diff components range from -2*G_max to 2*G_max, so -16 to 16
        # That's 33x33 = 1089 values
        
        # Actually, let me just compute the matrix more efficiently using the fact that
        # inv_eps(G) is the Fourier coefficient.
        
        # Let me use a different approach: compute the matrix using numpy operations.
        
        # For each pair (i, j), G_diff = G_i - G_j
        # We need inv_eps(G_diff)
        
        # Let me precompute a dictionary of inv_eps for all possible G differences
        # G_diff_x, G_diff_y each range from -2*G_max to 2*G_max = -16 to 16
        
        # This is getting complex. Let me simplify by using a smaller G_max for the search.
        
        return None
    
    # Let me take a simpler approach: use a known good design and do local optimization.
    
    # Start with a few candidate designs and pick the best one.
    
    def make_cell(radius, offset=0.0, square=False, square_size=0.3):
        """Create a cell with a central feature."""
        cell = [[0] * size for _ in range(size)]
        for i in range(size):
            for j in range(size):
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                if square:
                    if abs(x) < square_size / 2 and abs(y) < square_size / 2:
                        cell[i][j] = 1
                else:
                    if x * x + y * y < radius * radius:
                        cell[i][j] = 1
        return cell
    
    # Try several candidate designs
    candidates = []
    
    # 1. Circular rod, various radii
    for r in [0.15, 0.2, 0.25, 0.3]:
        candidates.append(make_cell(r))
    
    # 2. Square rod, various sizes
    for s in [0.2, 0.25, 0.3, 0.35]:
        candidates.append(make_cell(0, square=True, square_size=s))
    
    # 3. Diamond (rotated square)
    for s in [0.2, 0.25, 0.3]:
        cell = [[0] * size for _ in range(size)]
        for i in range(size):
            for j in range(size):
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                if abs(x) + abs(y) < s / 2:
                    cell[i][j] = 1
        candidates.append(cell)
    
    # Evaluate each candidate using a quick band solver
    # For speed, use a smaller number of plane waves for evaluation
    
    def quick_score(cell):
        """Quickly score a cell using a reduced plane-wave basis."""
        # Use fewer G vectors for speed
        G_max_q = 5  # 11x11 = 121 plane waves
        G_values_q = list(range(-G_max_q, G_max_q + 1))
        G_vectors_q = [(2 * math.pi * gx, 2 * math.pi * gy) for gx in G_values_q for gy in G_values_q]
        N_pw_q = len(G_vectors_q)
        
        cell_arr = np.array(cell, dtype=np.float64)
        eps_values = np.where(cell_arr == 1, eps_high, eps_low)
        inv_eps_values = 1.0 / eps_values
        
        # Compute inv_eps Fourier coefficients for the needed G differences
        # G_diff ranges from -2*G_max_q to 2*G_max_q = -10 to 10
        # That's 21x21 = 441 values
        G_diff_values = list(range(-2 * G_max_q, 2 * G_max_q + 1))
        G_diff_vectors = [(gx, gy) for gx in G_diff_values for gy in G_diff_values]
        
        inv_eps_fourier = {}
        for Gx, Gy in G_diff_vectors:
            phase = np.array([Gx * x + Gy * y for (x, y) in pixel_positions])
            total = np.sum(inv_eps_values.ravel() * np.exp(-1j * phase))
            inv_eps_fourier[(Gx, Gy)] = total / (size * size)
        
        # Solve at a few k points
        k_eval_points = [(0.0, 0.0), (0.5, 0.0), (0.5, 0.5), (0.25, 0.0), (0.25, 0.25), (0.5, 0.25)]
        
        band4_max = 0.0
        band5_min = float('inf')
        
        for kx, ky in k_eval_points:
            # Build matrix
            M = np.zeros((N_pw_q, N_pw_q), dtype=np.complex128)
            
            for i, Gi in enumerate(G_vectors_q):
                for j, Gj in enumerate(G_vectors_q):
                    k_plus_G = np.array([kx + Gi[0], ky + Gi[1]])
                    if i == j:
                        M[i, j] = np.dot(k_plus_G, k_plus_G)
                    else:
                        G_diff = (Gi[0] - Gj[0], Gi[1] - Gj[1])
                        # Scale G_diff to the G_diff_vectors range
                        G_diff_scaled = (G_diff[0] / (2 * math.pi), G_diff[1] / (2 * math.pi))
                        # Find the closest G_diff_vectors entry
                        # Actually, G_diff is already in units of 2*pi
                        # G_diff_vectors are in units of 1 (multiplied by 2*pi in G_vectors)
                        # Let me fix this
                        pass
            
            # This is getting messy. Let me use a cleaner approach.
            # The G vectors in G_vectors_q are (2*pi*gx, 2*pi*gy)
            # The G difference is (2*pi*(gx_i - gx_j), 2*pi*(gy_i - gy_j))
            # So the index into inv_eps_fourier should be (gx_i - gx_j, gy_i - gy_j)
            
            M = np.zeros((N_pw_q, N_pw_q), dtype=np.complex128)
            
            for i, (Gxi, Gy_i) in enumerate(G_vectors_q):
                Gxi_scaled = Gxi / (2 * math.pi)
                Gy_i_scaled = Gy_i / (2 * math.pi)
                for j, (Gxj, Gy_j) in enumerate(G_vectors_q):
                    Gxj_scaled = Gxj / (2 * math.pi)
                    Gy_j_scaled = Gy_j / (2 * math.pi)
                    
                    k_plus_G = np.array([kx + Gxi, ky + Gy_i])
                    M[i, j] = -inv_eps_fourier.get((Gxi_scaled - Gxj_scaled, Gy_i_scaled - Gy_j_scaled), 0)
                    if i == j:
                        M[i, j] += np.dot(k_plus_G, k_plus_G)
            
            # Solve eigenvalue problem
            try:
                eigvals = np.linalg.eigvalsh(M)
                # Take the first several bands
                if len(eigvals) >= 5:
                    # omega^2 = eigvals, so omega = sqrt(eigvals)
                    # But eigvals can be negative for unphysical solutions
                    # Take the positive ones
                    pos_eigvals = eigvals[eigvals > 0]
                    if len(pos_eigvals) >= 5:
                        band4 = np.sqrt(pos_eigvals[3])
                        band5 = np.sqrt(pos_eigvals[4])
                        band4_max = max(band4_max, band4)
                        band5_min = min(band5_min, band5)
            except:
                pass
        
        if band5_min == float('inf'):
            return 0.0
        
        gap = 2 * (band5_min - band4_max) / (band5_min + band4_max)
        return gap
    
    # Evaluate all candidates
    best_cell = None
    best_score = -1.0
    
    for cell in candidates:
        score = quick_score(cell)
        if score > best_score:
            best_score = score
            best_cell = [row[:] for row in cell]
    
    # Now do local optimization on the best cell
    # Try flipping individual pixels to improve the score
    
    if time.time() > deadline - 30:
        return best_cell
    
    # Local search: try flipping pairs of pixels (to maintain symmetry)
    current_cell = [row[:] for row in best_cell]
    current_score = best_score
    
    max_iterations = 50
    iteration = 0
    
    while iteration < max_iterations and time.time() < deadline - 20:
        iteration += 1
        improved = False
        
        # Try flipping some pixels
        # To maintain symmetry, flip pixels in symmetric groups
        # For now, just try random flips and check symmetry
        
        for attempt in range(20):
            trial_cell = [row[:] for row in current_cell]
            
            # Pick a random pixel and flip it
            i = np.random.randint(0, size)
            j = np.random.randint(0, size)
            trial_cell[i][j] = 1 - trial_cell[i][j]
            
            # Enforce symmetry: mirror and transpose
            # Cell must be symmetric under: row reversal, col reversal, transpose
            # So if we flip (i, j), we must also flip:
            # (size-1-i, j), (i, size-1-j), (size-1-i, size-1-j)
            # And from transpose: (j, i), (j, size-1-i), (size-1-j, i), (size-1-j, size-1-i)
            
            def enforce_symmetry(cell):
                # Make cell symmetric under all four operations
                # Start from the top-left quadrant and fill in
                result = [[0] * size for _ in range(size)]
                for i in range(size // 2 + 1):
                    for j in range(size // 2 + 1):
                        val = cell[i][j]
                        # Four corners in the top-left quadrant map to:
                        # (i, j), (i, size-1-j), (size-1-i, j), (size-1-i, size-1-j)
                        result[i][j] = val
                        result[i][size - 1 - j] = val
                        result[size - 1 - i][j] = val
                        result[size - 1 - i][size - 1 - j] = val
                return result
            
            trial_cell = enforce_symmetry(trial_cell)
            
            score = quick_score(trial_cell)
            
            if score > current_score + 1e-6:
                current_cell = [row[:] for row in trial_cell]
                current_score = score
                improved = True
                break
        
        if not improved:
            break
    
    return current_cell

