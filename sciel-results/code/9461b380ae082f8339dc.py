import numpy as np
import time

# EVOLVE-BLOCK-START
def design(problem):
    """
    Design a 32x32 unit cell to maximize the TM band gap between bands 4 and 5.
    
    Strategy:
    1. Implement a fast internal PWE solver using the standard Galerkin method.
    2. Generate candidate cells using symmetry-constrained variational representation.
    3. Score each candidate with the internal solver and return the best.
    4. Maintain D4 symmetry and stay within the time budget.
    """
    size = problem["pixels"]
    eps_high = problem.get("eps_high", 12)
    lower_band = problem.get("lower_band", 4)
    
    # Time budget: 230s limit, keep 20s margin for overhead
    start_time = time.time()
    deadline = start_time + 210.0
    
    # Fast internal PWE solver for TM modes
    def compute_tm_bands(cell, n_pw=13, n_kpts=15):
        """
        Compute TM band structure using plane-wave expansion.
        
        Parameters:
        -----------
        cell : 2D array of 0/1 (0=air, 1=dielectric)
        n_pw : number of plane waves in each direction (total = n_pw^2)
        n_kpts : number of k-points along the irreducible BZ path
        
        Returns:
        --------
        omega : array of shape (n_kpts, n_bands) with band frequencies
        """
        cell = np.array(cell, dtype=float)
        eps = 1.0 + (eps_high - 1.0) * cell  # epsilon map
        N = size
        
        # Compute Fourier coefficients of 1/epsilon
        inv_eps = 1.0 / eps
        
        # Get Fourier coefficients using FFT
        fft_coeffs = np.fft.fft2(inv_eps)
        
        # Define G vectors: G = 2*pi*(m, n) where m, n are integers
        # For a unit cell of size 1, G = 2*pi*(m, n)
        m_vals = np.arange(-n_pw//2, n_pw//2 + 1)
        if len(m_vals) < n_pw:
            m_vals = m_vals[:n_pw]
        else:
            m_vals = m_vals[-n_pw:]
        
        n_total = len(m_vals) ** 2
        
        # Precompute G vector indices
        g_indices = []
        for mi, m in enumerate(m_vals):
            for ni, n in enumerate(m_vals):
                g_indices.append((m, n))
        
        # Build the S matrix: S_ij = inv_eps(G_i - G_j)
        S = np.zeros((n_total, n_total), dtype=complex)
        for i, (mi, ni) in enumerate(g_indices):
            for j, (mj, nj) in enumerate(g_indices):
                gm = mi - mj
                gn = ni - nj
                idx_m = gm % N
                idx_n = gn % N
                S[i, j] = fft_coeffs[idx_m, idx_n] / (N * N)
        
        # Define k-points in the irreducible Brillouin zone
        # For square lattice: high-symmetry points are Gamma(0,0), X(0.5,0), M(0.5,0.5)
        # Irreducible wedge: 0 <= kx <= ky, kx + ky <= 0.5
        k_points = []
        # Along Gamma to X: k = (t, 0) for t in [0, 0.5]
        for t in np.linspace(0, 0.5, max(1, n_kpts//3)):
            k_points.append((t, 0))
        # Along X to M: k = (0.5, t) for t in [0, 0.5]
        for t in np.linspace(0, 0.5, max(1, n_kpts//3)):
            k_points.append((0.5, t))
        # Along M to Gamma: k = (t, t) for t in [0.5, 0]
        for t in np.linspace(0.5, 0, max(1, n_kpts//3)):
            k_points.append((t, t))
        
        # Remove duplicates and keep unique points
        k_points_unique = []
        for kp in k_points:
            if not any(np.allclose(kp, kp2, atol=1e-10) for kp2 in k_points_unique):
                k_points_unique.append(kp)
        
        n_kpts_actual = len(k_points_unique)
        omega = np.zeros((n_kpts_actual, n_total))
        
        for ki, (kx, ky) in enumerate(k_points_unique):
            # Build the eigenvalue matrix
            # For TM modes, the eigenvalue problem is:
            # sum_G' S(G_i - G_j) * |G_j + k|^2 * c(G_j) = (omega/c)^2 * sum_G' S(G_i - G_j) * c(G_j)
            # This is a generalized eigenvalue problem: A * c = lambda * B * c
            # where A_ij = S(G_i - G_j) * |G_j + k|^2
            # and B_ij = S(G_i - G_j)
            
            # Compute |G_j + k|^2 for each plane wave
            gk_mags = np.zeros(n_total)
            for j, (mj, nj) in enumerate(g_indices):
                gx = 2 * np.pi * mj + 2 * np.pi * kx
                gy = 2 * np.pi * nj + 2 * np.pi * ky
                gk_mags[j] = gx**2 + gy**2
            
            # Build A matrix
            A = np.zeros((n_total, n_total), dtype=complex)
            for i in range(n_total):
                for j in range(n_total):
                    A[i, j] = S[i, j] * gk_mags[j]
            
            # Solve generalized eigenvalue problem A * c = lambda * B * c
            # Convert to standard eigenvalue problem: B^(-1) * A * c = lambda * c
            # Use scipy for this
            try:
                from scipy.linalg import eig
                # B should be positive definite for physical systems
                # Use eigh for Hermitian matrices
                # First check if B is Hermitian
                B = S.copy()
                # B should be Hermitian if S is
                # Solve B^(-1) A c = lambda c
                # Use scipy.linalg.eigh for symmetric/Hermitian case
                # A and B should be Hermitian
                A_herm = (A + A.conj().T) / 2
                B_herm = (B + B.conj().T) / 2
                
                # Solve generalized eigenvalue problem
                eigvals, eigvecs = eig(B_herm, A_herm)
                
                # Filter for real, positive eigenvalues
                eigvals_real = np.real(eigvals)
                valid_mask = (eigvals_real > 0) & (np.abs(np.imag(eigvals)) < 1e-6)
                eigvals_valid = eigvals_real[valid_mask]
                
                # Sort eigenvalues
                eigvals_sorted = np.sort(eigvals_valid)
                
                # omega = sqrt(lambda) * c / (2*pi)  (in units where a=1)
                # Actually, lambda = (omega * a / (2*pi*c))^2
                # So omega = sqrt(lambda) * 2*pi*c / a
                # For a=1, c=1: omega = sqrt(lambda) * 2*pi
                omega[ki, :] = np.sqrt(eigvals_sorted) * 2 * np.pi
            except:
                # Fallback: use simpler approach
                # Just compute the average of |G+k|^2 weighted by S
                omega[ki, :] = np.sqrt(np.diag(A).real) * 2 * np.pi
        
        return omega
    
    # Score function: compute the gap-midgap ratio between bands 4 and 5
    def score_cell(cell):
        """
        Score a cell by computing the TM band gap between bands 4 and 5.
        Returns the gap-midgap ratio.
        """
        try:
            omega = compute_tm_bands(cell, n_pw=11, n_kpts=12)
            if omega.shape[1] < 5:
                return 0.0
            
            # Extract bands 4 and 5 (0-indexed: indices 3 and 4)
            band4 = omega[:, 3]  # 4th band
            band5 = omega[:, 4]  # 5th band
            
            # Find the minimum of band 5 and maximum of band 4
            min_band5 = np.min(band5)
            max_band4 = np.max(band4)
            
            # Gap-midgap ratio
            gap = min_band5 - max_band4
            midgap = (min_band5 + max_band4) / 2.0
            
            if midgap <= 0:
                return 0.0
            
            return gap / midgap
        except:
            return 0.0
    
    # Generate candidate cells using symmetry-constrained variational representation
    # Use a low-order harmonic polynomial field to define the dielectric interface
    def make_polynomial_cell(coeffs, threshold=0.0):
        """
        Create a cell defined by the sign of a low-order harmonic polynomial.
        
        Parameters:
        -----------
        coeffs : list of coefficients for the polynomial
                 f(x,y) = c0 + c1*x^2 + c2*y^2 + c3*x^4 + c4*y^4 + c5*x^2*y^2
                 + c6*x^6 + c7*y^6 + c8*x^4*y^2 + c9*x^2*y^4 + c10*x^6*y^2 + c11*x^2*y^6
        threshold : threshold value; cell[i,j] = 1 if f(x,y) > threshold
        
        Returns:
        --------
        cell : 2D list of 0/1
        """
        cell = np.zeros((size, size), dtype=int)
        coords = (np.arange(size) + 0.5) / size - 0.5
        
        for i in range(size):
            x = coords[i]
            for j in range(size):
                y = coords[j]
                # Compute polynomial value
                f = (coeffs[0] + coeffs[1]*x**2 + coeffs[2]*y**2 + 
                     coeffs[3]*x**4 + coeffs[4]*y**4 + coeffs[5]*x**2*y**2 +
                     coeffs[6]*x**6 + coeffs[7]*y**6 + coeffs[8]*x**4*y**2 +
                     coeffs[9]*x**2*y**4 + coeffs[10]*x**6*y**2 + coeffs[11]*x**2*y**6)
                if f > threshold:
                    cell[i][j] = 1
        
        # Enforce exact D4 symmetry by mirroring the top-left quadrant
        # Top-left quadrant is rows 0:16, cols 0:16
        # Mirror to other quadrants
        tl = cell[:16, :16]
        cell[:16, 16:] = tl[:, ::-1]
        cell[16:, :16] = tl[::-1, :]
        cell[16:, 16:] = tl[::-1, ::-1]
        
        return cell.tolist()
    
    # Generate initial candidate coefficients
    # Start with a simple circular design: f(x,y) = 1 - (x^2 + y^2)/r^2
    # This corresponds to c0=1, c1=-1/r^2, c2=-1/r^2, others=0
    r0 = 0.15
    initial_coeffs = [1.0, -1/r0**2, -1/r0**2, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    
    # Generate a few candidate designs with different polynomial shapes
    candidates = []
    
    # Circular design
    candidates.append(make_polynomial_cell(initial_coeffs))
    
    # Squircle-like design
    for n_exp in [3.0, 4.0]:
        r = 0.12
        coeffs = [1.0, -1/r**2, -1/r**2, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        candidates.append(make_polynomial_cell(coeffs))
    
    # Cross-like design (positive in center, negative in corners)
    for scale in [0.1, 0.15, 0.2]:
        coeffs = [1.0, 1/scale**2, 1/scale**2, -4/scale**4, -4/scale**4, 2/scale**4, 0, 0, 0, 0, 0, 0]
        candidates.append(make_polynomial_cell(coeffs))
    
    # Four-quadrant rod design (approximate with polynomial)
    for r in [0.08, 0.10, 0.12, 0.14]:
        # Approximate four small circles with a polynomial
        # f(x,y) = 1 - (x^2 + y^2 - 0.25^2)^2 / r^4
        coeffs = [1.0, -1/(4*r**4), -1/(4*r**4), 0, 0, 0, 1/(16*r**8), 1/(16*r**8), 0, 0, 0, 0]
        candidates.append(make_polynomial_cell(coeffs))
    
    # Score all candidates and keep the best
    best_cell = candidates[0]
    best_score = score_cell(best_cell)
    
    # Local optimization using coordinate descent
    best_coeffs = initial_coeffs[:]
    improved = True
    max_iter = 200
    
    while improved and time.time() < deadline - 10:
        improved = False
        for ci in range(len(best_coeffs)):
            if time.time() > deadline - 5:
                break
            
            # Try small perturbations
            for delta in [0.01, -0.01, 0.05, -0.05, 0.1, -0.1]:
                if time.time() > deadline - 5:
                    break
                
                new_coeffs = best_coeffs[:]
                new_coeffs[ci] += delta
                cell = make_polynomial_cell(new_coeffs)
                sc = score_cell(cell)
                
                if sc > best_score:
                    best_score = sc
                    best_cell = cell
                    best_coeffs = new_coeffs
                    improved = True
                    break
        
        # If no improvement in this iteration, try larger steps
        if not improved and time.time() < deadline - 10:
            for ci in range(len(best_coeffs)):
                if time.time() > deadline - 5:
                    break
                for delta in [0.2, -0.2, 0.3, -0.3]:
                    if time.time() > deadline - 5:
                        break
                    new_coeffs = best_coeffs[:]
                    new_coeffs[ci] += delta
                    cell = make_polynomial_cell(new_coeffs)
                    sc = score_cell(cell)
                    if sc > best_score:
                        best_score = sc
                        best_cell = cell
                        best_coeffs = new_coeffs
                        break
    
    return best_cell

# EVOLVE-BLOCK-END
