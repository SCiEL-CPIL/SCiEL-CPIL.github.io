"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.linalg import eigh

# EVOLVE-BLOCK-START
def design(problem):
    """
    Design a 32x32 unit cell to maximize the TM band gap between bands 4 and 5.
    
    Strategy:
    1. Implement a fast internal PWE solver for TM modes.
    2. Generate candidate cells using four-quadrant rod configurations 
       with various shapes and sizes.
    3. Use the internal solver to score each candidate and return the best.
    4. Maintain D4 symmetry and stay within the time budget.
    """
    size = problem["pixels"]
    eps_high = problem.get("eps_high", 12)
    lower_band = problem.get("lower_band", 4)
    
    # Time budget: 230s limit, keep 20s margin for overhead
    start_time = time.time()
    deadline = start_time + 210.0
    
    def check_time():
        return time.time() > deadline - 5.0
    
    # Fast internal PWE solver for TM modes
    def compute_tm_gap(cell, n_pw=15, n_kpts=8):
        """
        Compute the TM band gap ratio between bands 4 and 5.
        
        Parameters:
        -----------
        cell : 2D array of 0/1 (0=air, 1=dielectric)
        n_pw : number of plane waves in each direction (total = n_pw^2)
        n_kpts : number of k-points along the irreducible BZ path
        
        Returns:
        --------
        gap_ratio : the gap-midgap ratio 2*(min omega_5 - max omega_4)/(min omega_5 + max omega_4)
        """
        cell = np.array(cell, dtype=float)
        eps = 1.0 + (eps_high - 1.0) * cell  # epsilon map
        inv_eps = 1.0 / eps
        
        N = size
        
        # Compute Fourier coefficients of 1/epsilon using FFT
        fft_coeffs = np.fft.fft2(inv_eps)
        
        def get_coeff(m, n):
            """Get Fourier coefficient for G = 2*pi*(m, n)"""
            idx_m = m % N
            idx_n = n % N
            return fft_coeffs[idx_m, idx_n] / (N * N)
        
        # Build the list of G vectors
        m_vals = np.arange(-n_pw//2, n_pw//2 + 1)
        if len(m_vals) < n_pw:
            m_vals = m_vals[-n_pw:]
        else:
            m_vals = m_vals[:n_pw]
        
        g_indices = []
        for mi in m_vals:
            for ni in m_vals:
                g_indices.append((mi, ni))
        
        n_total = len(g_indices)
        
        # Precompute the S matrix: S_ij = inv_eps(G_i - G_j)
        S = np.zeros((n_total, n_total), dtype=complex)
        for i, (mi, ni) in enumerate(g_indices):
            for j, (mj, nj) in enumerate(g_indices):
                gm = mi - mj
                gn = ni - nj
                S[i, j] = get_coeff(gm, gn)
        
        # Define k-points in the irreducible Brillouin zone
        # For square lattice: Gamma(0,0), X(0.5,0), M(0.5,0.5)
        # Irreducible wedge: 0 <= kx <= ky, kx + ky <= 0.5
        k_points = []
        # Along Gamma to X: k = (t, 0) for t in [0, 0.5]
        for t in np.linspace(0, 0.5, n_kpts//3 + 1):
            k_points.append((t, 0))
        # Along X to M: k = (0.5, t) for t in [0, 0.5]
        for t in np.linspace(0, 0.5, n_kpts//3 + 1):
            k_points.append((0.5, t))
        # Along M to Gamma: k = (t, t) for t in [0.5, 0]
        for t in np.linspace(0.5, 0, n_kpts//3 + 1):
            k_points.append((t, t))
        
        # Remove duplicates
        k_points_unique = []
        for kp in k_points:
            if not any(np.allclose(kp, kp2, atol=1e-10) for kp2 in k_points_unique):
                k_points_unique.append(kp)
        
        min_omega_5 = np.inf
        max_omega_4 = -np.inf
        
        for (kx, ky) in k_points_unique:
            # Build the eigenvalue matrix for this k-point
            # For TM modes, the eigenvalue problem is:
            # sum_G' S(G_i - G_j) * |G_j + k|^2 * c_Gj = (omega/c)^2 * sum_G' S(G_i - G_j) * c_Gj
            
            # This is a generalized eigenvalue problem: A c = lambda B c
            # where A_ij = S_ij * |G_j + k|^2 and B_ij = S_ij
            
            A = np.zeros((n_total, n_total), dtype=complex)
            B = np.zeros((n_total, n_total), dtype=complex)
            
            for i in range(n_total):
                for j in range(n_total):
                    gi_m = g_indices[i][0]
                    gi_n = g_indices[i][1]
                    gj_m = g_indices[j][0]
                    gj_n = g_indices[j][1]
                    
                    # |G_j + k|^2 = (gj_m + kx)^2 + (gj_n + ky)^2
                    # Note: k is in units of 1/(2*pi), so G = 2*pi*(m,n)
                    # Actually, let's work in units where 2*pi = 1 for simplicity
                    # G_j = (gj_m, gj_n) in units of 2*pi
                    # k = (kx, ky) in units of 1/(2*pi)
                    # So G_j + k = (gj_m + kx, gj_n + ky) in units of 2*pi
                    # |G_j + k|^2 = (gj_m + kx)^2 + (gj_n + ky)^2
                    
                    k_mag_sq = (gj_m + kx)**2 + (gj_n + ky)**2
                    A[i, j] = S[i, j] * k_mag_sq
                    B[i, j] = S[i, j]
            
            # Solve the generalized eigenvalue problem
            # Use scipy.linalg.eigh for Hermitian matrices
            # First, we need to ensure B is positive definite
            # For safety, add a small regularization
            B_reg = B + 1e-10 * np.eye(n_total)
            
            try:
                # Convert to real if possible
                A_real = np.real_if_close(A, tol=1000)
                B_real = np.real_if_close(B_reg, tol=1000)
                
                if np.isrealobj(A_real) and np.isrealobj(B_real):
                    eigenvalues, eigenvectors = eigh(A_real, B_real)
                else:
                    # Use scipy's generalized eigenvalue solver
                    from scipy.linalg import eig
                    eigenvalues, eigenvectors = eig(A, B_reg)
                    eigenvalues = np.real_if_close(eigenvalues, tol=1000)
                    if np.iscomplexobj(eigenvalues):
                        eigenvalues = np.real(eigenvalues)
                
                # Sort eigenvalues
                idx = np.argsort(eigenvalues)
                eigenvalues = eigenvalues[idx]
                
                # Get bands 4 and 5 (0-indexed: 3 and 4)
                if len(eigenvalues) >= 5:
                    omega_4 = np.sqrt(eigenvalues[3])  # omega = sqrt(lambda)
                    omega_5 = np.sqrt(eigenvalues[4])
                    
                    if omega_4 > max_omega_4:
                        max_omega_4 = omega_4
                    if omega_5 < min_omega_5:
                        min_omega_5 = omega_5
                    
            except Exception:
                continue
        
        # Compute the gap-midgap ratio
        if min_omega_5 > max_omega_4:
            gap = min_omega_5 - max_omega_4
            midgap = (min_omega_5 + max_omega_4) / 2.0
            if midgap > 0:
                return 2.0 * gap / (min_omega_5 + max_omega_4)
            else:
                return -1.0
        else:
            return -1.0
    
    # Generate candidate cells
    def make_four_quadrant_cell(rod_radius, exponent=2.0):
        """
        Create a cell with four rods centered in each quadrant.
        
        Parameters:
        -----------
        rod_radius : radius of each rod in units of the lattice constant
        exponent : superellipse exponent (2=circle, >2=squircle)
        
        Returns:
        --------
        cell : 2D list of 0/1
        """
        cell = np.zeros((size, size), dtype=int)
        coords = (np.arange(size) + 0.5) / size - 0.5
        
        # Rod centers in normalized coordinates [-0.5, 0.5]
        centers = [(-0.25, -0.25), (0.25, -0.25), (-0.25, 0.25), (0.25, 0.25)]
        
        for i in range(size):
            x = coords[i]
            for j in range(size):
                y = coords[j]
                for (cx, cy) in centers:
                    dx = x - cx
                    dy = y - cy
                    if rod_radius > 0:
                        val = abs(dx/rod_radius)**exponent + abs(dy/rod_radius)**exponent
                    else:
                        val = 0
                    if val <= 1.0 + 1e-9:
                        cell[i, j] = 1
                        break
        return cell.tolist()
    
    def make_central_cell(r, n=2.0):
        cell = np.zeros((size, size), dtype=int)
        coords = (np.arange(size) + 0.5) / size - 0.5
        for i in range(size):
            x = coords[i]
            for j in range(size):
                y = coords[j]
                val = abs(x/r)**n + abs(y/r)**n
                if val <= 1.0 + 1e-9:
                    cell[i, j] = 1
        return cell.tolist()
    
    # Generate candidates to test
    candidates = []
    
    # Four-quadrant circles with different radii
    for r in [0.08, 0.10, 0.12, 0.14, 0.16, 0.18]:
        candidates.append(make_four_quadrant_cell(r, 2.0))
    
    # Four-quadrant squircles
    for r in [0.10, 0.12, 0.14]:
        for n in [3.0, 4.0]:
            candidates.append(make_four_quadrant_cell(r, n))
    
    # Central circle as fallback
    candidates.append(make_central_cell(0.20, 2.0))
    
    # Score candidates and find the best
    best_score = -np.inf
    best_candidate = candidates[0]
    
    for idx, cand in enumerate(candidates):
        if check_time():
            break
        
        try:
            score = compute_tm_gap(cand, n_pw=13, n_kpts=8)
            if score > best_score:
                best_score = score
                best_candidate = cand
        except:
            continue
    
    return best_candidate

# EVOLVE-BLOCK-END
