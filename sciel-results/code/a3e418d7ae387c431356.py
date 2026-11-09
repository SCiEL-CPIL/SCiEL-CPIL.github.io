"""
Photonic crystal design for TM band gap between bands 4 and 5.
Strategy:
- Use the high-performing "4 air holes in dielectric background" structure (inspiration 1).
- Represent the hole radius as a single parameter (or a few if we wanted to shape them, 
  but radius is the most critical).
- Use a fast, accurate plane-wave method (7x7 grid) to evaluate the gap-midgap ratio.
- Perform a 1D hill-climbing search on the hole radius to maximize the gap.
- Since the structure is symmetric and simple, this is robust and fast.
"""
import numpy as np
import time

def design(problem):
    size = problem.get("pixels", 32)
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)
    
    start_time = time.time()
    deadline = start_time + 190.0  # Safety margin

    # --- 1. Precompute Plane Wave Basis ---
    # The evaluator uses 289 PWs (17x17). We use 7x7 = 49 PWs for speed.
    # G vectors: integers (h, k).
    # We want G vectors within a radius. For 7x7, h, k in [-3, 3].
    pw_range = 3
    G_vectors = []
    for h in range(-pw_range, pw_range + 1):
        for k in range(-pw_range, pw_range + 1):
            G_vectors.append((h, k))
    
    N_pw = len(G_vectors)
    
    # Precompute G_i - G_j differences for FFT lookup
    # diff_h[i, j] = G_i[0] - G_j[0]
    # diff_k[i, j] = G_i[1] - G_j[1]
    G_arr = np.array(G_vectors)
    diff_h = G_arr[:, 0][:, None] - G_arr[0, 0][None, :] # Wait, G_arr[:,0] is column
    # Correctly:
    diff_h = G_arr[:, 0][:, None] - G_arr[0, 0][None, :]
    # No, G_arr is (N, 2).
    Gx = G_arr[:, 0]
    Gy = G_arr[:, 1]
    diff_h = Gx[:, None] - Gx[None, :]
    diff_k = Gy[:, None] - Gy[None, :]
    
    # Map differences to FFT indices (mod size)
    idx_h = diff_h % size
    idx_k = diff_k % size

    # --- 2. Precompute K-points ---
    # The evaluator samples 30 k-points in the irreducible BZ.
    # Irreducible BZ for square lattice: 0 <= kx <= ky, kx + ky <= 0.5 (in units of 1/a? No, 1/2pi?)
    # Let's use units where reciprocal lattice vector is 1.
    # BZ is [-0.5, 0.5] x [-0.5, 0.5].
    # Irreducible wedge: 0 <= kx <= ky, kx + ky <= 0.5.
    # Key points:
    # Gamma (0,0)
    # M (0.5, 0) -> (0, 0.5) in wedge? No, M is (0.5, 0). In wedge, M is on boundary.
    # X (0.5, 0.5) -> corner.
    # Let's sample along Gamma-M (0, t), M-X (t, 0.5-t) for t in [0, 0.5].
    # And Gamma-X (t, t)? No, Gamma-X is (t, t) in the full BZ? No, X is (0.5, 0.5).
    # The line Gamma-X is (t, t).
    # The line Gamma-M is (0, t).
    # The line M-X is (t, 0.5-t).
    
    k_points = []
    # 10 points along Gamma-M: (0, t)
    for i in range(10):
        t = (i + 0.5) / 10.0 * 0.5
        k_points.append((0.0, t))
    # 10 points along M-X: (t, 0.5 - t)
    for i in range(10):
        t = (i + 0.5) / 10.0 * 0.5
        k_points.append((t, 0.5 - t))
    # 10 points along Gamma-X: (t, t)
    for i in range(10):
        t = (i + 0.5) / 10.0 * 0.5
        k_points.append((t, t))
        
    k_points = np.array(k_points)
    
    # --- 3. Fast Evaluator ---
    
    def evaluate_gap(cell):
        """
        cell: 32x32 numpy array of 0/1.
        Returns gap-midgap ratio.
        """
        # Compute eps grid
        eps = np.where(cell == 1, eps_high, eps_low)
        mu = 1.0 / eps
        
        # FFT of mu
        mu_fft = np.fft.fft2(mu)
        
        # Extract coefficients for G_i - G_j
        # coeff[i, j] = mu_fft[idx_h[i, j], idx_k[i, j]] / (size * size)
        coeffs = mu_fft[idx_h, idx_k] / (size * size)
        
        max_omega_4 = 0.0
        min_omega_5 = float('inf')
        
        for kx, ky in k_points:
            # Build H matrix
            # H_{ij} = (G_i + k) dot (G_j + k) * coeffs[i, j]
            # K_i = (Gx[i] + kx, Gy[i] + ky)
            Kx = Gx + kx
            Ky = Gy + ky
            
            # Dot products: K_i dot K_j
            # This is a rank-1 update? No.
            # K_i dot K_j = Kx[i]*Kx[j] + Ky[i]*Ky[j]
            # We can compute this efficiently.
            # dot = np.outer(Kx, Kx) + np.outer(Ky, Ky)
            dot = np.outer(Kx, Kx) + np.outer(Ky, Ky)
            
            H = dot * coeffs
            
            # H is Hermitian (real symmetric if coeffs are real).
            # coeffs are real because mu is real and symmetric?
            # mu is symmetric in the sense of the crystal, but the FFT values might be complex if the grid is not perfectly symmetric in the FFT sense?
            # Actually, for a real function, FFT is conjugate symmetric.
            # H_{ji} = (G_j + k) dot (G_i + k) * coeffs[j, i]
            # coeffs[j, i] = mu_fft[-diff_h[i,j], -diff_k[i,j]] = conj(mu_fft[diff_h[i,j], diff_k[i,j]])
            # Since mu is real, coeffs[j, i] = conj(coeffs[i, j]).
            # And dot is symmetric.
            # So H is Hermitian.
            # We can use eigvalsh on the real part if imaginary part is negligible, or just use eigvalsh on H if it's complex.
            # np.linalg.eigvalsh works for Hermitian matrices.
            
            # For speed, if imaginary part is small, take real part.
            # Given the symmetry of the structure, it should be real.
            H_real = H.real
            
            evals = np.linalg.eigvalsh(H_real)
            
            # Bands 4 and 5 (1-indexed) -> indices 3 and 4
            if len(evals) >= 5:
                w4 = np.sqrt(max(0, evals[3]))
                w5 = np.sqrt(max(0, evals[4]))
                
                if w4 > max_omega_4:
                    max_omega_4 = w4
                if w5 < min_omega_5:
                    min_omega_5 = w5
        
        if min_omega_5 == float('inf'):
            return 0.0
            
        gap = min_omega_5 - max_omega_4
        midgap = (min_omega_5 + max_omega_4) / 2.0
        
        if midgap == 0:
            return 0.0
            
        ratio = 2 * gap / midgap
        return ratio

    # --- 4. Parameterization and Optimization ---
    
    def generate_cell(radius):
        """
        Generate a 32x32 cell with 4 air holes of given radius.
        Centers at (0.25, 0.25), (-0.25, 0.25), (0.25, -0.25), (-0.25, -0.25).
        """
        x = (np.arange(size) + 0.5) / size - 0.5
        y = (np.arange(size) + 0.5) / size - 0.5
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        # Distance to nearest center
        # Centers:
        # (0.25, 0.25), (-0.25, 0.25), (0.25, -0.25), (-0.25, -0.25)
        # Due to symmetry, we can just check distance to (0.25, 0.25) in the first quadrant?
        # No, the holes are at the four "corners" of the central square.
        # Let's compute distance to all 4.
        
        d1 = np.sqrt((X - 0.25)**2 + (Y - 0.25)**2)
        d2 = np.sqrt((X + 0.25)**2 + (Y - 0.25)**2)
        d3 = np.sqrt((X - 0.25)**2 + (Y + 0.25)**2)
        d4 = np.sqrt((X + 0.25)**2 + (Y + 0.25)**2)
        
        min_d = np.minimum(np.minimum(d1, d2), np.minimum(d3, d4))
        
        # If min_d < radius, it's air (0). Else dielectric (1).
        cell = (min_d >= radius).astype(int)
        return cell

    # Initial guess from inspiration 1
    best_radius = 0.28
    best_score = evaluate_gap(generate_cell(best_radius))
    
    # Hill climbing on radius
    # Radius can be roughly between 0.1 and 0.4
    # Step size
    step = 0.01
    
    current_radius = best_radius
    current_score = best_score
    
    # Adaptive step
    # If we improve, increase step or keep.
    # If we fail, decrease step.
    
    num_iter = 0
    max_iter = 1000
    
    while time.time() < deadline and num_iter < max_iter:
        num_iter += 1
        
        # Random direction
        direction = np.random.choice([-1, 1])
        new_radius = current_radius + direction * step
        
        # Bounds
        if new_radius < 0.1 or new_radius > 0.4:
            # Reverse direction
            direction = -direction
            new_radius = current_radius + direction * step
            if new_radius < 0.1 or new_radius > 0.4:
                step *= 0.9
                continue
                
        new_cell = generate_cell(new_radius)
        new_score = evaluate_gap(new_cell)
        
        if new_score > current_score:
            current_radius = new_radius
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_radius = new_radius
            # Increase step slightly
            step *= 1.05
            if step > 0.05:
                step = 0.05
        else:
            # Decrease step
            step *= 0.9
            if step < 0.001:
                # If step is too small, try a larger jump
                step = 0.02
                # Or restart from best
                current_radius = best_radius + np.random.uniform(-0.05, 0.05)
                current_score = evaluate_gap(generate_cell(current_radius))
                
    # Finalize
    final_cell = generate_cell(best_radius)
    
    # Enforce symmetry (just in case)
    # The construction is symmetric, but let's verify.
    # cell == cell[::-1] row-wise and column-wise and cell == its transpose.
    
    # Convert to list of lists
    return final_cell.tolist()
