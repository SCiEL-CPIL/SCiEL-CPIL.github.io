"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import math
import scipy.optimize as optimize

# EVOLVE-BLOCK-START
def design(problem):
    """
    Designs a 32x32 unit cell for a TM band gap between bands 4 and 5.
    Strategy:
    1. Represent the cell via a small set of C4v-symmetric Fourier coefficients.
    2. Optimize these coefficients using Nelder-Mead to maximize the band gap.
    3. Threshold the resulting continuous field to binary 0/1.
    """
    size = problem["pixels"]  # 32
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)
    
    # Time management
    start_time = time.time()
    deadline = start_time + 190.0  # Safety margin for 200s limit

    # --- Fourier Basis Definition ---
    # We use a set of (h, k) indices for the Fourier series.
    # To maintain C4v symmetry, we must include orbits of (h, k).
    # For a square lattice, the fundamental domain of the reciprocal space
    # for the structure factor is similar to the real space.
    # We will select a few low-order harmonics that are important for the band gap.
    # (0,0) is the DC component (average density).
    # (1,0), (0,1) and their symmetric equivalents form the first shell.
    # (1,1), (1,-1) etc. form the second shell.
    
    # Let's define the independent Fourier coefficients.
    # A C4v symmetric function f(x,y) can be written as:
    # f(x,y) = sum_{h,k} c_{h,k} e^{i(hx+ky)}
    # Due to symmetry, c_{h,k} = c_{-h,-k} = c_{k,h} = c_{-k,-h} = ...
    # We can just define the real amplitudes for a set of unique orbits.
    
    # Orbits of (h,k) under C4v:
    # 1. (0,0)
    # 2. (1,0) -> includes (0,1), (-1,0), (0,-1)
    # 3. (1,1) -> includes (1,-1), (-1,1), (-1,-1)
    # 4. (2,0) -> includes (0,2), (-2,0), (0,-2)
    # 5. (2,1) -> includes (1,2), (-2,-1), ... (8 points)
    
    # We will optimize the amplitudes of these modes.
    # Let A0 be the DC term.
    # Let A1 be the amplitude for (1,0) orbit.
    # Let A2 be the amplitude for (1,1) orbit.
    # Let A3 be the amplitude for (2,0) orbit.
    # Let A4 be the amplitude for (2,1) orbit.
    # Let A5 be the amplitude for (3,0) orbit.
    
    # We also need to consider phases. For C4v, the phases are constrained.
    # For (1,0) orbit, the term is cos(x) + cos(y). Phase is 0 or pi.
    # For (1,1) orbit, the term is cos(x+y) + cos(x-y). Phase is 0 or pi.
    # So we can just use signed amplitudes.
    
    # Design variables: [A0, A1, A2, A3, A4, A5, A6, A7]
    # A0: DC (0 to 1)
    # A1: (1,0) amplitude
    # A2: (1,1) amplitude
    # A3: (2,0) amplitude
    # A4: (2,1) amplitude
    # A5: (3,0) amplitude
    # A6: (3,1) amplitude
    # A7: (4,0) amplitude
    
    def generate_cell_from_fourier(params):
        """
        Generate a 32x32 cell from Fourier parameters.
        params: [A0, A1, A2, A3, A4, A5, A6, A7]
        """
        A0, A1, A2, A3, A4, A5, A6, A7 = params
        
        # Create grid
        x = np.linspace(0, 2 * np.pi, size, endpoint=False)
        y = np.linspace(0, 2 * np.pi, size, endpoint=False)
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        # Sum of Fourier components
        # Note: The standard Fourier series for a periodic function on [0, 2pi)
        # f(x,y) = sum c_{h,k} e^{i(hx+ky)}
        # For real symmetric functions, we use cosines.
        # The coefficient in the sum needs to be scaled correctly to match the pixel values.
        # If we want the pixel values to be in [0,1], we need to normalize.
        
        # Let's construct the field:
        # f(x,y) = A0 + A1 * (cos(x) + cos(y)) / 2 + A2 * (cos(x+y) + cos(x-y)) / 2 + ...
        # The division by 2 ensures that if A1=1, the max contribution is 1.
        
        field = np.full((size, size), A0)
        
        # Orbit (1,0): cos(x) + cos(y)
        # Max value of (cos(x)+cos(y)) is 2. So divide by 2 to keep amplitude A1.
        field += A1 * (np.cos(X) + np.cos(Y)) / 2.0
        
        # Orbit (1,1): cos(x+y) + cos(x-y)
        field += A2 * (np.cos(X + Y) + np.cos(X - Y)) / 2.0
        
        # Orbit (2,0): cos(2x) + cos(2y)
        field += A3 * (np.cos(2 * X) + np.cos(2 * Y)) / 2.0
        
        # Orbit (2,1): cos(2x+y) + cos(2y+x) + cos(2x-y) + cos(2y-x)
        # There are 4 terms in this orbit for C4v?
        # (2,1) -> (1,2) -> (-2,-1) -> (-1,-2)
        # (2,-1) -> (-1,2) -> (-2,1) -> (1,-2)
        # Wait, C4v symmetry group has 8 elements.
        # The orbit of (2,1) has 8 distinct vectors: (2,1), (1,2), (-2,1), (-1,2), (-2,-1), (-1,-2), (2,-1), (1,-2).
        # The sum of cosines for these is:
        # 2*cos(2x+y) + 2*cos(2y+x) + 2*cos(2x-y) + 2*cos(2y-x)
        # Let's verify.
        # e^{i(2x+y)} + e^{-i(2x+y)} = 2cos(2x+y)
        # e^{i(x+2y)} + e^{-i(x+2y)} = 2cos(x+2y)
        # e^{i(2x-y)} + e^{-i(2x-y)} = 2cos(2x-y)
        # e^{i(y-2x)} + e^{-i(y-2x)} = 2cos(y-2x) = 2cos(2x-y)
        # So the sum is 2cos(2x+y) + 2cos(x+2y) + 4cos(2x-y).
        # This is getting complicated. Let's just sum over the explicit orbit vectors.
        
        def add_orbit(h, k, amp):
            """Add amplitude * sum of cos(h*x + k*y) for all (h',k') in orbit of (h,k)."""
            orbit = [
                (h, k), (k, h), (-h, k), (-k, h),
                (-h, -k), (-k, -h), (h, -k), (k, -h)
            ]
            # Remove duplicates if any (e.g., if h=0 or k=0 or h=k)
            unique_orbit = list(set(orbit))
            # Normalize by number of terms to keep amplitude scale consistent
            s = 0
            for hh, kk in unique_orbit:
                s += np.cos(hh * X + kk * Y)
            field += amp * s / len(unique_orbit)
            
        # We already handled (1,0) and (1,1) manually for speed, but let's use the helper for others
        # Actually, let's just use the helper for all to ensure correctness, but it's slow.
        # Let's stick to the manual ones for the first few and use helper for higher orders if needed.
        # For now, let's just use the manual ones which are most important.
        
        # Add higher orders using helper
        add_orbit(2, 0, A3) # Overwrite previous A3 calculation? No, I already added A3 manually. 
        # Let's remove the manual A3, A4 etc and just use the helper for all non-DC.
        # Re-initialize field
        field = np.full((size, size), A0)
        add_orbit(1, 0, A1)
        add_orbit(1, 1, A2)
        add_orbit(2, 0, A3)
        add_orbit(2, 1, A4)
        add_orbit(3, 0, A5)
        add_orbit(3, 1, A6)
        add_orbit(4, 0, A7)
        
        # Clip and threshold
        # The sum of cosines can range from -N to N.
        # We want the density to be in [0,1].
        # Let's normalize the field to [0,1] before thresholding?
        # Or just threshold at 0?
        # If A0 is the mean, and other terms are small, thresholding at A0 might be good.
        # Let's threshold at 0.5 * max_possible?
        # Simpler: Threshold at 0. If field > 0, it's dielectric.
        # But we need to ensure the fill factor is reasonable.
        # Let's threshold at 0.
        binary_cell = (field > 0).astype(int)
        
        # Enforce strict symmetry on the binary cell
        # The Fourier construction should be symmetric, but floating point errors might break it.
        # Also, the thresholding might break it slightly.
        # Average over symmetric points
        sym_cell = np.zeros_like(binary_cell)
        for i in range(size):
            for j in range(size):
                pts = [
                    (i, j), (size - 1 - i, j), (i, size - 1 - j), (size - 1 - i, size - 1 - j),
                    (j, i), (size - 1 - j, j), (j, size - 1 - i), (size - 1 - j, size - 1 - i)
                ]
                s = sum(binary_cell[pi, pj] for pi, pj in pts)
                sym_cell[i, j] = 1 if s >= 4 else 0
        return sym_cell

    # --- Fast Plane-Wave Evaluator ---
    # Use 9x9 plane waves (81 modes)
    pw_half = 4
    G_vectors = [(h, k) for h in range(-pw_half, pw_half + 1) 
                 for k in range(-pw_half, pw_half + 1)]
    N_pw = len(G_vectors)
    
    Gx_arr = np.array([g[0] for g in G_vectors], dtype=float)
    Gy_arr = np.array([g[1] for g in G_vectors], dtype=float)
    
    # Precompute G differences indices
    G_diff_idx_h = np.zeros((N_pw, N_pw), dtype=int)
    G_diff_idx_k = np.zeros((N_pw, N_pw), dtype=int)
    for i in range(N_pw):
        for j in range(N_pw):
            G_diff_idx_h[i, j] = (G_vectors[i][0] - G_vectors[j][0]) % size
            G_diff_idx_k[i, j] = (G_vectors[i][1] - G_vectors[j][1]) % size
            
    # K-points in irreducible BZ
    # Gamma (0,0), M (0.5, 0), X (0.5, 0.5)
    # Path: Gamma -> M -> X -> Gamma
    k_points = []
    # Gamma-M
    for t in [0.0, 0.25, 0.5]:
        k_points.append((0.0, t))
    # M-X
    for t in [0.0, 0.25, 0.5]:
        k_points.append((0.5 - t, 0.5 - t + 0.5)) # Wait, M is (0.5, 0), X is (0.5, 0.5).
    # Let's use standard high-symmetry points and a few interior points.
    k_points = [
        (0.0, 0.0), # Gamma
        (0.25, 0.0),
        (0.5, 0.0), # M
        (0.5, 0.25),
        (0.5, 0.5), # X
        (0.25, 0.25)
    ]
    
    def evaluate_gap(cell_grid):
        """
        Computes the approximate band gap between bands 4 and 5.
        """
        # Build inverse epsilon grid
        inv_eps = np.where(cell_grid == 1, 1.0/eps_high, 1.0/eps_low)
        
        # FFT for Fourier coefficients
        fft_1eps = np.fft.fft2(inv_eps)
        coeffs = fft_1eps[G_diff_idx_h, G_diff_idx_k] / (size * size)
        
        max_omega_4 = 0.0
        min_omega_5 = float('inf')
        
        for kx, ky in k_points:
            Kx = Gx_arr + kx
            Ky = Gy_arr + ky
            
            # H = (K_i · K_j) * coeffs_{G_i - G_j}
            dots = np.outer(Kx, Kx) + np.outer(Ky, Ky)
            H = dots * coeffs
            H_real = (H + H.conj().T) * 0.5  # Ensure Hermitian
            
            # Get eigenvalues
            eigenvalues = np.linalg.eigvalsh(H_real)
            
            if len(eigenvalues) >= 5:
                w4_sq = eigenvalues[3]
                w5_sq = eigenvalues[4]
                
                w4 = np.sqrt(max(0.0, w4_sq))
                w5 = np.sqrt(max(0.0, w5_sq))
                
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

    def objective(params):
        """Negative gap ratio for minimization."""
        if time.time() > deadline:
            return 0.0
        cell = generate_cell_from_fourier(params)
        score = evaluate_gap(cell)
        return -score

    # Initial guess
    # A0 = 0.5 (50% fill)
    # Other amplitudes start small
    x0 = np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    # Bounds
    # A0 in [0,1]
    # Others in [-1, 1]
    bounds = [(0, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1)]
    
    best_params = x0
    best_score = objective(x0)
    
    # Optimize
    try:
        result = optimize.minimize(objective, x0, method='Nelder-Mead', bounds=bounds, 
                                   options={'maxiter': 200, 'xatol': 1e-3, 'fatol': 1e-4})
        if result.fun < best_score:
            best_score = result.fun
            best_params = result.x
    except Exception as e:
        pass

    # Finalize
    final_cell = generate_cell_from_fourier(best_params)
    return final_cell.tolist()

# EVOLVE-BLOCK-END
