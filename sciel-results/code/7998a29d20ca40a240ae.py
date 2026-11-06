"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

# EVOLVE-BLOCK-START
def design(problem):
    """
    Design a 32x32 unit cell to maximize the TM band gap between bands 4 and 5.
    Uses a pure NumPy PWE solver and a randomized hill-climbing search.
    """
    size = problem["pixels"]
    eps_high = problem.get("eps_high", 12)
    lower_band = problem.get("lower_band", 4)
    
    start_time = time.time()
    deadline = start_time + 210.0

    # --- PWE Solver ---
    def compute_tm_bands(cell, n_pw=9, n_kpts=12):
        """
        Compute TM band structure using plane-wave expansion.
        Returns (max_omega_lower, min_omega_upper) for bands lower_band and lower_band+1.
        """
        cell = np.array(cell, dtype=float)
        eps = np.where(cell == 1, eps_high, 1.0)
        inv_eps = 1.0 / eps
        
        # Compute Fourier coefficients of 1/epsilon
        # f(x,y) = sum c_mn exp(2pi i (mx+ny))
        # c_mn = (1/N^2) sum f(x,y) exp(-2pi i (mx+ny))
        # Using FFT: F = fft2(inv_eps). c_mn = F[m%N, n%N] / N^2
        N = size
        fft_coeffs = np.fft.fft2(inv_eps)
        
        def get_coeff(m, n):
            return fft_coeffs[m % N, n % N] / (N * N)

        # Plane waves: diamond cutoff
        # m, n in [-n_pw//2, n_pw//2]
        m_vals = np.arange(-n_pw//2, n_pw//2 + 1)
        if len(m_vals) < n_pw:
             # If n_pw is even, we might want to adjust, but for simplicity, keep square grid of G vectors
             # Actually, standard is to take all (m,n) such that m^2+n^2 <= R^2
             # For small n_pw, a square grid is fine.
             pass
             
        g_indices = []
        for m in m_vals:
            for n in m_vals:
                g_indices.append((m, n))
                
        n_total = len(g_indices)
        
        # Precompute S matrix: S[i,j] = 1/eps(G_i - G_j)
        S = np.zeros((n_total, n_total), dtype=float)
        for i, (mi, ni) in enumerate(g_indices):
            for j, (mj, nj) in enumerate(g_indices):
                gm = mi - mj
                gn = ni - nj
                S[i, j] = get_coeff(gm, gn).real

        # K-points in irreducible BZ
        # Square lattice BZ: [-pi, pi] x [-pi, pi] in units of 1/a?
        # Let's use units where G = 2pi(m,n). k is in units of 1/a.
        # BZ boundaries at k = +/- pi.
        # Irreducible wedge: 0 <= kx <= ky, kx + ky <= pi.
        # Vertices: Gamma(0,0), X(pi,0) -> equivalent to (0,pi), M(pi,pi).
        # Path: Gamma -> (0, pi) -> (pi, pi) -> Gamma.
        
        k_points = []
        n_seg = max(2, n_kpts // 3)
        
        # Gamma(0,0) to X(0, pi)
        for t in np.linspace(0, np.pi, n_seg + 1):
            k_points.append((0.0, t))
        # X(0, pi) to M(pi, pi)
        for t in np.linspace(0, np.pi, n_seg + 1):
            k_points.append((t, np.pi))
        # M(pi, pi) to Gamma(0,0)
        for t in np.linspace(np.pi, 0, n_seg + 1):
            k_points.append((t, t))
            
        # Remove duplicates
        k_points_unique = []
        for kp in k_points:
            if not any(np.allclose(kp, kp2, atol=1e-10) for kp2 in k_points_unique):
                k_points_unique.append(kp)
        
        max_omega_lower = 0.0
        min_omega_upper = np.inf
        
        n_bands_needed = lower_band + 1 # We need band index lower_band-1 (0-based) and lower_band
        
        for kx, ky in k_points_unique:
            # Build A and B
            # A[i,j] = S[i,j] * |G_j + k|^2
            # B[i,j] = S[i,j]
            # G_j = 2*pi*(m_j, n_j)
            # |G_j + k|^2 = (2*pi*m_j + kx)^2 + (2*pi*n_j + ky)^2
            
            A = np.zeros((n_total, n_total), dtype=float)
            for i in range(n_total):
                for j in range(n_total):
                    mj, nj = g_indices[j]
                    gx = 2.0 * np.pi * mj + kx
                    gy = 2.0 * np.pi * nj + ky
                    g_sq = gx*gx + gy*gy
                    A[i, j] = S[i, j] * g_sq
            
            B = S.copy()
            
            # Solve A c = lambda B c
            # Use np.linalg.eig on B^{-1} A or generalized solver if available.
            # Since scipy is not guaranteed, we do:
            # 1. Diagonalize B: B = V diag(w) V^T
            # 2. Transform to standard problem: (V^T A V) y = lambda (V^T B V) y = lambda diag(w) y
            # 3. So (V^T A V) y = lambda diag(w) y => (V^T A V) diag(w)^-0.5 = lambda diag(w)^-0.5 ...
            # Simpler: Solve det(A - lambda B) = 0.
            # For small matrices, np.linalg.eig on B_inv @ A is robust enough if B is invertible.
            # B is positive definite for non-zero epsilon.
            
            try:
                # Check condition of B
                if np.linalg.cond(B) > 1e10:
                    # Fallback to eig
                    pass
                else:
                    B_inv = np.linalg.inv(B)
                    M = B_inv @ A
                    eigenvalues = np.linalg.eigvals(M)
                    eigenvalues = np.real(eigenvalues)
                    eigenvalues = np.sort(eigenvalues)
                    
                    # Filter out negative eigenvalues (numerical noise)
                    eigenvalues = eigenvalues[eigenvalues > 1e-6]
                    
                    if len(eigenvalues) >= n_bands_needed:
                        omega_vals = np.sqrt(eigenvalues[:n_bands_needed])
                        
                        # Band 4 is index 3, Band 5 is index 4
                        o4 = omega_vals[lower_band - 1]
                        o5 = omega_vals[lower_band]
                        
                        if o4 > max_omega_lower:
                            max_omega_lower = o4
                        if o5 < min_omega_upper:
                            min_omega_upper = o5
            except:
                continue
        
        if min_omega_upper == np.inf:
            return 0.0, 0.0
            
        return max_omega_lower, min_omega_upper

    def evaluate_cell(cell):
        """Calculate the score for a given cell."""
        o_low, o_high = compute_tm_bands(cell, n_pw=9, n_kpts=12)
        if o_low == 0.0:
            return -np.inf
        gap = o_high - o_low
        midgap = (o_high + o_low) / 2.0
        if midgap <= 0:
            return -np.inf
        return 2.0 * gap / midgap

    def enforce_d4(cell):
        """Ensure the cell has D4 symmetry."""
        cell = np.array(cell)
        # Fold the cell to the top-left quadrant (16x16)
        # If any pixel in the orbit is 1, the representative is 1?
        # Or just average? For binary, let's say if majority is 1, it's 1.
        # Actually, for search, we define the quadrant and expand.
        pass # Not needed if we generate from quadrant

    def expand_quadrant(q):
        """Expand a 16x16 quadrant to a 32x32 D4-symmetric cell."""
        q = np.array(q)
        cell = np.zeros((32, 32), dtype=int)
        
        # Top-left: q
        cell[0:16, 0:16] = q
        # Top-right: flip horizontal
        cell[0:16, 16:32] = np.fliplr(q)
        # Bottom-left: flip vertical
        cell[16:32, 0:16] = np.flipud(q)
        # Bottom-right: flip both
        cell[16:32, 16:32] = np.fliplr(np.flipud(q))
        
        return cell.tolist()

    # --- Search Strategy ---
    
    # Initialize candidates
    # 1. Central disk
    def make_central_disk(radius):
        q = np.zeros((16, 16), dtype=int)
        for i in range(16):
            for j in range(16):
                x = (j + 0.5) / 32.0
                y = (i + 0.5) / 32.0
                # Center of full cell is (0.5, 0.5). Center of quadrant is (0.25, 0.25) in full cell coords?
                # No, the quadrant maps to [0, 0.5] x [0, 0.5].
                # The full cell center is (0.5, 0.5).
                # Distance from center (0.5, 0.5) to (x,y) in quadrant:
                dx = abs(x - 0.5)
                dy = abs(y - 0.5)
                if dx*dx + dy*dy <= radius*radius:
                    q[i,j] = 1
        return q

    # 2. Four disks (one in each quadrant)
    def make_four_disks(radius):
        q = np.zeros((16, 16), dtype=int)
        for i in range(16):
            for j in range(16):
                x = (j + 0.5) / 32.0
                y = (i + 0.5) / 32.0
                # Center of this quadrant's disk is at (0.25, 0.25) in full cell coords
                dx = x - 0.25
                dy = y - 0.25
                if dx*dx + dy*dy <= radius*radius:
                    q[i,j] = 1
        return q

    # 3. Random
    def make_random():
        q = np.random.randint(0, 2, (16, 16))
        return q

    candidates = []
    # Add some structured seeds
    for r in [0.1, 0.15, 0.2, 0.25]:
        candidates.append(make_central_disk(r))
    for r in [0.05, 0.1, 0.15]:
        candidates.append(make_four_disks(r))
    
    # Add some random seeds
    for _ in range(5):
        candidates.append(make_random())
    
    best_score = -np.inf
    best_quadrant = None
    best_cell = None
    
    # Evaluate initial candidates
    for q in candidates:
        if time.time() > deadline - 10:
            break
        cell = expand_quadrant(q)
        score = evaluate_cell(cell)
        if score > best_score:
            best_score = score
            best_quadrant = q.copy()
            best_cell = cell

    # Hill Climbing
    if best_quadrant is not None:
        current_q = best_quadrant.copy()
        current_score = best_score
        no_improve_count = 0
        max_no_improve = 50
        
        while time.time() < deadline - 20 and no_improve_count < max_no_improve:
            # Pick a random pixel in the quadrant to flip
            i = random.randint(0, 15)
            j = random.randint(0, 15)
            
            # Flip
            current_q[i, j] = 1 - current_q[i, j]
            
            cell = expand_quadrant(current_q)
            score = evaluate_cell(cell)
            
            if score > current_score:
                current_score = score
                if score > best_score:
                    best_score = score
                    best_quadrant = current_q.copy()
                    best_cell = cell
                no_improve_count = 0
            else:
                # Revert
                current_q[i, j] = 1 - current_q[i, j]
                no_improve_count += 1
                
            # Occasionally try a larger move (flip 2 pixels)
            if random.random() < 0.1:
                i2 = random.randint(0, 15)
                j2 = random.randint(0, 15)
                current_q[i2, j2] = 1 - current_q[i2, j2]
                cell = expand_quadrant(current_q)
                score = evaluate_cell(cell)
                if score > current_score:
                    current_score = score
                    if score > best_score:
                        best_score = score
                        best_quadrant = current_q.copy()
                        best_cell = cell
                    no_improve_count = 0
                else:
                    current_q[i2, j2] = 1 - current_q[i2, j2]
                    no_improve_count += 1

    return best_cell

# EVOLVE-BLOCK-END
