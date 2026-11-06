import numpy as np
import time
from scipy.linalg import eigh

def design(problem):
    t0 = time.time()
    deadline = t0 + 210.0  # safety margin before 230s kill
    size = problem["pixels"]
    eps_low = problem["eps_low"]
    eps_high = problem["eps_high"]
    lower_band = problem.get("lower_band", 4)
    
    def make_cell_from_spectrum(spectrum, threshold=0.5):
        """
        Reconstruct a 32x32 cell from a low-frequency Fourier spectrum.
        spectrum is a 2D array of complex coefficients.
        We enforce square symmetry by construction:
        The spectrum is defined on a grid that respects D4 symmetry.
        For a 32x32 grid, the D4 symmetric Fourier modes are those where
        (m, n) is equivalent to (n, m), (-m, n), (m, -n), etc.
        However, simply summing a symmetric set of coefficients will yield
        a real, symmetric function.
        
        Specifically, we will use a set of modes (m, n) where m, n >= 0.
        And we will ensure the coefficients satisfy c[m, n] = c[n, m] = c[-m, n] = ...
        Actually, for a real function, c[-m, -n] = conj(c[m, n]).
        For D4 symmetry, f(x, y) = f(y, x) = f(-x, y) = f(x, -y) ...
        This implies c[m, n] = c[n, m] and c[m, n] = c[-m, n] = c[m, -n] = c[-m, -n].
        Since f is real, c[-m, -n] = conj(c[m, n]).
        So c[m, n] must be real and c[m, n] = c[n, m].
        
        We will define the spectrum on a grid of size 2*max_k+1 x 2*max_k+1.
        And populate it symmetrically.
        """
        # spectrum is expected to be a 2D array, or a list of (m, n, coeff)
        # Here we assume spectrum is a 2D array of real values for m, n >= 0
        # and we will build the full complex FFT grid.
        
        # Let's define max_k based on the size of the input spectrum
        # If spectrum is a list of tuples, handle that.
        if isinstance(spectrum, list):
            # List of (m, n, coeff)
            max_k = max(max(abs(m), abs(n)) for m, n, _ in spectrum)
            fft_grid = np.zeros((2 * max_k + 1, 2 * max_k + 1), dtype=complex)
            for m, n, coeff in spectrum:
                fft_grid[m + max_k, n + max_k] = coeff
                # Enforce symmetries
                # c[n, m]
                fft_grid[n + max_k, m + max_k] = coeff
                # c[-m, n] -> index (max_k-m, max_k+n)
                fft_grid[max_k - m, max_k + n] = coeff
                # c[m, -n] -> index (max_k+m, max_k-n)
                fft_grid[max_k + m, max_k - n] = coeff
                # c[-m, -n] -> index (max_k-m, max_k-n)
                fft_grid[max_k - m, max_k - n] = coeff
        else:
            # 2D array of real coefficients for m, n in [0, max_k]
            max_k = spectrum.shape[0] // 2
            fft_grid = np.zeros((2 * max_k + 1, 2 * max_k + 1), dtype=complex)
            for m in range(spectrum.shape[0]):
                for n in range(spectrum.shape[1]):
                    coeff = spectrum[m, n]
                    if coeff == 0: continue
                    fft_grid[m + max_k, n + max_k] = coeff
                    fft_grid[n + max_k, m + max_k] = coeff
                    if m != 0:
                        fft_grid[max_k - m, max_k + n] = coeff
                        fft_grid[max_k - m, max_k - n] = coeff
                    if n != 0:
                        fft_grid[max_k + m, max_k - n] = coeff
                        if m != 0:
                            fft_grid[max_k - m, max_k - n] = coeff # already set

        # Now compute the IDFT to get the pixel values
        # Note: The grid size is (2*max_k+1). We need to resize to 32x32.
        # The physical size is 1x1. The frequency resolution is 1/(2*max_k+1).
        # To get 32 pixels, we need to sample the field at 32 points.
        # The field is periodic with period 1.
        # x_i = (i + 0.5) / 32 - 0.5
        # f(x, y) = sum_{m,n} c[m,n] exp(i 2 pi (m x + n y))
        
        x = (np.arange(size) + 0.5) / size - 0.5
        y = (np.arange(size) + 0.5) / size - 0.5
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        field = np.zeros((size, size), dtype=complex)
        # The fft_grid indices are centered at max_k.
        # So index i corresponds to mode m = i - max_k.
        for i in range(fft_grid.shape[0]):
            m = i - max_k
            for j in range(fft_grid.shape[1]):
                n = j - max_k
                c = fft_grid[i, j]
                if c != 0:
                    field += c * np.exp(1j * 2 * np.pi * (m * X + n * Y))
        
        field = np.real(field)
        # Normalize to 0/1
        # Threshold at 0
        cell = (field > 0).astype(np.int8)
        return cell

    def pwe_gap(cell, n_pw=10, k_points=None):
        """
        Compute gap-midgap ratio using PWE method.
        """
        try:
            # Build eps_inv grid
            eps_inv = np.ones((size, size))
            eps_inv[cell == 1] = 1.0 / eps_high
            fft_eps_inv = np.fft.fft2(eps_inv)
            
            # Plane wave grid
            n_max = n_pw
            m_vals = np.arange(-n_max, n_max + 1)
            n_vals = np.arange(-n_max, n_max + 1)
            M_grid, N_grid = np.meshgrid(m_vals, n_vals, indexing='ij')
            Gx = M_grid * 2 * np.pi / size
            Gy = N_grid * 2 * np.pi / size
            G_flat_x = Gx.ravel()
            G_flat_y = Gy.ravel()
            N_g = len(G_flat_x)
            
            # Precompute eps_inv for all difference vectors needed
            diff_range = 2 * n_max
            dm_vals = np.arange(-diff_range, diff_range + 1)
            dn_vals = np.arange(-diff_range, diff_range + 1)
            DM, DN = np.meshgrid(dm_vals, dn_vals, indexing='ij')
            eps_inv_diff = np.zeros((2 * diff_range + 1, 2 * diff_range + 1), dtype=complex)
            for idx_i, dm in enumerate(dm_vals):
                for idx_j, dn in enumerate(dn_vals):
                    u = dm % size
                    v = dn % size
                    eps_inv_diff[idx_i, idx_j] = fft_eps_inv[v, u]
            
            # K-points in irreducible BZ
            if k_points is None:
                k_points = []
                # Gamma to X: (k, 0) for k in [0, pi]
                for i in range(11):
                    k_points.append((np.pi * i / 10, 0))
                # X to M: (pi, k) for k in [0, pi]
                for i in range(1, 11):
                    k_points.append((np.pi, np.pi * i / 10))
                # M to Gamma: (k, k) for k in [0, pi]
                for i in range(1, 11):
                    k_points.append((np.pi * (1 - i / 10), np.pi * (1 - i / 10)))
            
            min_omega_5 = np.inf
            max_omega_4 = 0.0
            
            for kx, ky in k_points:
                k_vec = np.array([kx, ky])
                KG_x = k_vec[0] + G_flat_x
                KG_y = k_vec[1] + G_flat_y
                KG2 = KG_x ** 2 + KG_y ** 2
                
                # Build M matrix vectorized
                KG_dot = np.outer(KG_x, KG_x) + np.outer(KG_y, KG_y)
                
                # Get differences
                M_idx = M_grid.ravel()
                N_idx = N_grid.ravel()
                diff_M = M_idx[:, None] - M_idx[None, :]
                diff_N = N_idx[:, None] - N_idx[None, :]
                
                # Look up eps_inv_diff
                denom = np.zeros((N_g, N_g), dtype=complex)
                for i in range(N_g):
                    for j in range(N_g):
                        dm = diff_M[i, j]
                        dn = diff_N[i, j]
                        denom[i, j] = eps_inv_diff[dm + diff_range, dn + diff_range]
                
                # Avoid division by zero
                denom[denom == 0] = 1e-10
                
                M_mat = KG_dot / denom
                
                # Solve eigenvalue problem
                w2 = np.linalg.eigvalsh(M_mat)
                w2 = np.sort(np.real(w2))
                
                if len(w2) >= 5:
                    omega_4 = np.sqrt(max(w2[3], 0))
                    omega_5 = np.sqrt(max(w2[4], 0))
                    max_omega_4 = max(max_omega_4, omega_4)
                    min_omega_5 = min(min_omega_5, omega_5)
            
            if min_omega_5 == np.inf or max_omega_4 == 0:
                return 0.0
            gap = min_omega_5 - max_omega_4
            midgap = (min_omega_5 + max_omega_4) / 2
            if midgap <= 0:
                return 0.0
            return gap / midgap

        except Exception:
            return 0.0

    best_gap = -np.inf
    best_cell = None

    # Phase 1: Coarse search over spectral parameters
    # We will parameterize the spectrum by a few coefficients.
    # Let's try a few "basis" spectra and optimize their weights.
    # Basis 1: DC (all dielectric or air) - not useful for gap
    # Basis 2: Fundamental mode (1,0) and (0,1) -> Stripe pattern
    # Basis 3: (1,1) -> Diagonal stripes
    # Basis 4: (2,0), (0,2) -> Higher order stripes
    # Basis 5: (2,2)
    
    # Let's define a few candidate spectra and test them.
    # We will use a simple grid search over the coefficients.
    
    # Candidate A: Rod-like (Dominant low freq, positive DC)
    # Candidate B: Hole-like (Negative DC, positive low freq)
    # Candidate C: Ring-like (Mix of low and high freq)
    
    # Let's just generate a few specific spectral patterns.
    
    # Pattern 1: Circle approx (Gaussian in k-space)
    # Pattern 2: Square approx (Sinc in k-space)
    # Pattern 3: Ring (Bessel in k-space)
    
    # Since we can't easily define continuous spectra, we will define discrete ones.
    
    # Discrete Spectrum 1: "Rod"
    # Strong DC, weak (1,0), (0,1)
    spec_rod = np.zeros((5, 5))
    spec_rod[2, 2] = 1.0  # DC
    spec_rod[1, 2] = 0.1  # (1,0)
    spec_rod[2, 1] = 0.1  # (0,1)
    
    # Discrete Spectrum 2: "Hole"
    spec_hole = np.zeros((5, 5))
    spec_hole[2, 2] = -1.0  # DC
    spec_hole[1, 2] = 0.5  # (1,0)
    spec_hole[2, 1] = 0.5  # (0,1)
    
    # Discrete Spectrum 3: "Ring"
    # DC + (2,0) + (0,2) with negative sign?
    spec_ring = np.zeros((5, 5))
    spec_ring[2, 2] = 0.5
    spec_ring[0, 2] = -0.2
    spec_ring[2, 0] = -0.2
    spec_ring[1, 1] = 0.1
    spec_ring[1, 3] = 0.1
    spec_ring[3, 1] = 0.1
    spec_ring[3, 3] = 0.1
    
    # Discrete Spectrum 4: "Cross"
    spec_cross = np.zeros((5, 5))
    spec_cross[2, 2] = 0.5
    spec_cross[1, 2] = 0.5
    spec_cross[2, 1] = 0.5
    
    candidates = [spec_rod, spec_hole, spec_ring, spec_cross]
    
    # Also try the seed circle as a reference
    seed_cell = np.zeros((size, size), dtype=np.int8)
    for i in range(size):
        y = (i + 0.5) / size - 0.5
        for j in range(size):
            x = (j + 0.5) / size - 0.5
            if x * x + y * y < 0.2 ** 2:
                seed_cell[i, j] = 1
    gap_seed = pwe_gap(seed_cell, n_pw=8)
    if gap_seed > best_gap:
        best_gap = gap_seed
        best_cell = seed_cell

    # Test spectral candidates
    for spec in candidates:
        if time.time() > deadline:
            break
        # Try different thresholds
        for thresh in [-0.2, 0.0, 0.2]:
            if time.time() > deadline:
                break
            cell = make_cell_from_spectrum(spec, threshold=thresh)
            # Check if it's symmetric (it should be by construction, but just in case)
            # cell = (cell == cell[::-1, :]).all() and (cell == cell[:, ::-1]).all() and (cell == cell.T).all()
            gap = pwe_gap(cell, n_pw=8)
            if gap > best_gap:
                best_gap = gap
                best_cell = cell

    # Phase 2: Local search in spectral space
    if best_cell is not None and time.time() < deadline - 30:
        # We need to invert the best_cell to get a spectrum?
        # Or just perturb the best spectral candidate?
        # Let's keep track of the best spectrum.
        # For now, we will just do a random perturbation of the pixels of the best_cell
        # and check if the gap improves.
        # This is less elegant but robust.
        
        for _ in range(10):
            if time.time() > deadline:
                break
            cell = best_cell.copy()
            # Flip a few pixels randomly but maintain symmetry
            for _ in range(5):
                i = np.random.randint(0, size // 2)
                j = np.random.randint(0, size // 2)
                # Apply symmetrically
                val = 1 - cell[i, j]
                cell[i, j] = val
                cell[size - 1 - i, j] = val
                cell[i, size - 1 - j] = val
                cell[size - 1 - i, size - 1 - j] = val
                # Also transpose symmetry
                cell[j, i] = val
                cell[j, size - 1 - i] = val
                cell[size - 1 - j, i] = val
                cell[size - 1 - j, size - 1 - i] = val
            gap = pwe_gap(cell, n_pw=8)
            if gap > best_gap:
                best_gap = gap
                best_cell = cell

    if best_cell is None:
        best_cell = seed_cell

    return best_cell.tolist()
