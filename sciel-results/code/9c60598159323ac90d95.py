"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def design(problem):
    """
    Design a symmetric 32x32 unit cell using Simulated Annealing with a fast
    Plane Wave Method approximation to maximize the TM band gap between bands 4 and 5.
    """
    size = problem["pixels"]
    eps_low = problem["eps_low"]
    eps_high = problem["eps_high"]
    
    # Pre-compute G vectors and weights for the fast solver
    # We use a small number of plane waves for speed
    N_g = 2 # Index range -2 to 2, total 5x5=25 plane waves
    G_vectors = []
    for m in range(-N_g, N_g + 1):
        for n in range(-N_g, N_g + 1):
            G_vectors.append((m, n))
            
    G_vectors = np.array(G_vectors, dtype=float)
    # Scale G vectors to match the lattice constant (normalized to 1)
    # The pixel grid is 32x32. The lattice constant a = 32 pixels.
    # In normalized units where a=1, G = 2*pi*(m, n).
    G_vectors *= 2 * np.pi

    # Pre-compute (k+G_m) . (k+G_n) for specific k points?
    # k points: Gamma (0,0), X (pi, 0), M (pi, pi) in normalized units (a=1)
    k_points = [
        np.array([0.0, 0.0]),
        np.array([np.pi, 0.0]),
        np.array([np.pi, np.pi])
    ]

    def compute_eps_coeffs(cell):
        """
        Compute Fourier coefficients of the epsilon map.
        cell is 32x32 list of 0s and 1s.
        1 -> eps_high, 0 -> eps_low.
        """
        eps_map = np.array(cell, dtype=float)
        eps_map = eps_map * (eps_high - eps_low) + eps_low
        
        # Compute FFT
        # We need coefficients for G = 2*pi*(m,n)
        # The FFT of a 32x32 array gives frequencies k=0..31.
        # We need to map m,n in [-2, 2] to FFT indices.
        # FFT index i corresponds to frequency i/32 * 2*pi? No, standard numpy fft.
        # f_k = sum eps_x exp(-i 2 pi k x / N)
        # We want c_mn = 1/N^2 sum eps_x exp(-i (G_m x + G_n y))
        # G_m = 2 pi m. So exponent is -i 2 pi m x.
        # This matches the DFT definition if we scale correctly.
        
        fft_coeffs = np.fft.fft2(eps_map)
        fft_coeffs /= (size * size)
        
        # Map G vectors (m,n) to FFT indices
        # FFT index i corresponds to frequency i.
        # We want m, n in [-N_g, N_g].
        # If m is negative, index is m + size.
        coeffs = np.zeros((len(G_vectors), len(G_vectors)), dtype=complex)
        for i, (m, n) in enumerate(G_vectors):
            idx_m = int(m) % size
            idx_n = int(n) % size
            coeffs[i, :] = fft_coeffs[idx_m, idx_n] # This is not quite right, we need to build the matrix
            
        # Actually, the matrix element is epsilon_{mn} where epsilon_{mn} is the Fourier coeff of epsilon(x) at G_m - G_n.
        # Let's compute the 2D Fourier series coefficients c_{mn} for m,n in range.
        c = np.zeros((len(G_vectors), len(G_vectors)), dtype=complex)
        for i, (mi, ni) in enumerate(G_vectors):
            for j, (mj, nj) in enumerate(G_vectors):
                dm = int(mi - mj)
                dn = int(ni - nj)
                # Get FFT coefficient for (dm, dn)
                idx_m = dm % size
                idx_n = dn % size
                c[i, j] = fft_coeffs[idx_m, idx_n]
                
        return c

    def estimate_gap(cell):
        """
        Estimate the gap between band 4 and 5 using the fast solver.
        """
        c = compute_eps_coeffs(cell)
        min_omega_5 = float('inf')
        max_omega_4 = 0.0
        
        for k in k_points:
            # Build matrix M
            # M_ij = c_ij * (k + G_i) . (k + G_j)
            # Note: c_ij is epsilon_{G_i - G_j}
            
            # Compute (k + G_i) . (k + G_j)
            # G_i is vector. k is vector.
            # Dot product of (k+G_i) and (k+G_j)
            
            # Vectorized dot product
            # (k+G_i) = k[0]+G_i[0], k[1]+G_i[1]
            # We need a matrix D where D[i,j] = (k+G_i) . (k+G_j)
            
            # k+G_i for all i
            KG = G_vectors + k
            # D = KG @ KG.T (since dot product of rows)
            # Wait, KG is (N, 2). KG @ KG.T is (N, N) where [i,j] = dot(KG[i], KG[j])
            D = KG @ KG.T
            
            M = c * D
            
            # M should be Hermitian. Symmetrize to remove numerical noise
            M = (M + M.conj().T) / 2.0
            
            # Solve eigenvalues
            # np.linalg.eigvalsh for Hermitian
            try:
                eigs = np.linalg.eigvalsh(M)
            except:
                return 0.0
                
            # eigs are omega^2
            # We need omega = sqrt(eigs)
            # The bands are ordered by omega.
            # Since omega^2 is monotonic with omega, ordering is same.
            # Gap between 4th and 5th band (index 3 and 4)
            if len(eigs) >= 5:
                w4 = np.sqrt(eigs[3])
                w5 = np.sqrt(eigs[4])
                # Update global max/min
                if w4 > max_omega_4:
                    max_omega_4 = w4
                if w5 < min_omega_5:
                    min_omega_5 = w5
                    
        # Calculate gap ratio
        if min_omega_5 == 0:
            return 0.0
        gap = 2 * (min_omega_5 - max_omega_4) / (min_omega_5 + max_omega_4)
        return gap

    # Simulated Annealing
    np.random.seed(42)
    
    # Initialize with a random symmetric pattern
    # We will generate a 16x16 block and mirror it
    half = size // 2
    best_cell = None
    best_score = -100
    
    # Start with a simple guess: square rod
    cell = [[0 for _ in range(size)] for _ in range(size)]
    # Create a square of side 10 in center
    for i in range(size):
        for j in range(size):
            if abs(i - 15.5) < 5 and abs(j - 15.5) < 5:
                cell[i][j] = 1
                
    best_cell = [row[:] for row in cell]
    best_score = estimate_gap(cell)
    
    current_score = best_score
    T = 1.0
    cooling = 0.995
    max_iters = 5000 # Adjust based on speed
    
    start_time = time.time()
    
    for it in range(max_iters):
        if time.time() - start_time > 110: # Leave 10s buffer
            break
            
        # Generate neighbor: flip a random pixel in the top-left quadrant
        # and mirror it
        ri = np.random.randint(0, half)
        rj = np.random.randint(0, half)
        
        # Original value
        old_val = cell[ri][rj]
        new_val = 1 - old_val
        
        # Apply to all symmetric positions
        # (ri, rj), (ri, size-1-rj), (size-1-ri, rj), (size-1-ri, size-1-rj)
        cell[ri][rj] = new_val
        cell[ri][size-1-rj] = new_val
        cell[size-1-ri][rj] = new_val
        cell[size-1-ri][size-1-rj] = new_val
        
        new_score = estimate_gap(cell)
        
        delta = new_score - current_score
        if delta > 0 or np.random.rand() < np.exp(delta / T):
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_cell = [row[:] for row in cell]
        else:
            # Revert
            cell[ri][rj] = old_val
            cell[ri][size-1-rj] = old_val
            cell[size-1-ri][rj] = old_val
            cell[size-1-ri][size-1-rj] = old_val
            
        T *= cooling
        
    return best_cell
# EVOLVE-BLOCK-END
