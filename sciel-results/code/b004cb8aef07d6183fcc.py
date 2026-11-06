"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def design(problem):
    """
    Design a symmetric 32x32 photonic crystal unit cell to maximize the TM band gap
    between bands 4 and 5. Uses a low-frequency Fourier parameterization with
    square symmetry, then performs a local search in coefficient space.
    """
    size = problem["pixels"]  # 32
    eps_low = problem["eps_low"]   # 1
    eps_high = problem["eps_high"] # 12
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110  # 110s out of 120s budget
    
    # We'll work in the 8x8 quadrant (top-left), since the cell has 4-fold symmetry.
    # The full cell is constructed by reflecting this quadrant.
    # For a 32x32 cell with 4-fold symmetry about the center:
    # The cell can be described by its top-left 16x16 quadrant, but we need to be careful
    # about the center. Actually, for a 32x32 cell, the symmetry group D4 means:
    # cell[i][j] == cell[31-i][j] == cell[i][31-j] == cell[31-i][31-j]
    # and cell[i][j] == cell[j][i]
    # So we only need to specify the top-left 16x16 quadrant, but with the constraint
    # that the "center line" pixels are shared.
    
    # Actually, let's think more carefully. For a 32x32 cell:
    # Rows 0-15 are the top half, rows 16-31 are the bottom half (reflected).
    # Within the top half, columns 0-15 are the left half, columns 16-31 are the right half.
    # So we need to specify cell[0:16, 0:16], and the rest is determined by symmetry.
    # But wait: cell[i][j] = cell[31-i][j] means row 15 maps to row 16.
    # And cell[i][j] = cell[i][31-j] means col 15 maps to col 16.
    # So the "center" is between pixels 15 and 16.
    
    # Let's use a simpler approach: parameterize the 16x16 top-left quadrant,
    # and construct the full 32x32 cell by reflection.
    
    def quadrant_to_full(q):
        """Convert a 16x16 quadrant to a full 32x32 symmetric cell."""
        q = np.array(q, dtype=int)
        # Full cell: top-left is q, top-right is q flipped horizontally,
        # bottom-left is q flipped vertically, bottom-right is q flipped both.
        top = np.hstack([q, q[:, ::-1]])
        bottom = top[::-1, :]
        return np.vstack([top, bottom])
    
    def full_to_quadrant(cell):
        """Extract the top-left 16x16 quadrant from a full 32x32 cell."""
        return np.array(cell[:16, :16], dtype=int)
    
    # Fast band gap evaluation using plane wave method
    # For speed, we'll use a simplified but reasonably accurate method.
    
    def compute_gap(cell_2d):
        """
        Compute the gap-midgap ratio between TM bands 4 and 5.
        Uses a plane-wave expansion with a truncated basis.
        For speed, we use a smaller basis but multiple k-points.
        """
        N = 32
        # Get dielectric permittivity
        eps = np.where(np.array(cell_2d) == 1, eps_high, eps_low)
        
        # For TM polarization, we solve:
        # (1/eps(k)) * |k|^2 * a = omega^2 * a
        # where eps(k) is the inverse dielectric in Fourier space
        
        # Use a plane wave basis with cutoff
        # For speed, use a moderate number of plane waves
        npw = 13  # half-width of Fourier grid, so total (2*npw+1)^2 = 729 waves
        # Actually, let's use npw=9 for speed: (19)^2 = 361 waves
        
        # Let's use npw = 8: (17)^2 = 289 waves, which matches the evaluator
        npw = 8
        G = []
        for mx in range(-npw, npw+1):
            for my in range(-npw, npw+1):
                G.append((2*np.pi*mx, 2*np.pi*my))
        G = np.array(G)
        nG = len(G)
        
        # Compute inverse dielectric Fourier coefficients
        # 1/eps(r) in real space, then FFT to get Fourier coefficients
        inv_eps = 1.0 / eps
        # Upsample to avoid aliasing
        inv_eps_up = np.kron(inv_eps, np.ones((4,4)))
        inv_eps_up = inv_eps_up / 16.0
        
        # Compute Fourier coefficients
        inv_eps_F = np.fft.fft2(inv_eps_up)
        inv_eps_F = np.fft.fftshift(inv_eps_F)
        # Scale to get the right Fourier series coefficients
        # The Fourier series is: f(r) = sum_G f_G exp(iG.r)
        # where f_G = (1/A) int f(r) exp(-iG.r) dr
        # For our discrete grid, we need to normalize properly
        
        # Actually, let's use a direct approach for the Fourier coefficients
        # of the pixelated dielectric.
        
        # For a pixelated structure, the Fourier coefficient of 1/eps is:
        # (1/eps)_G = (1/N^2) sum_{i,j} (1/eps_{ij}) exp(-i G . r_{ij})
        # where r_{ij} = (i+0.5, j+0.5)/N (in lattice units)
        
        # But this is expensive for each G. Let's use FFT instead.
        
        # For the FFT approach:
        # f_G = (1/N^2) * FFT(f) evaluated at the right frequencies
        # Since our grid is uniform, the FFT gives us the coefficients directly.
        
        # The FFT of a function on a grid of size N gives:
        # F[m,n] = sum_{i,j} f[i,j] exp(-2*pi*i*(m*i + n*j)/N)
        # This corresponds to G = (2*pi*m/N, 2*pi*n/N) in lattice units... 
        # wait, the lattice constant is 1, so G = (2*pi*m, 2*pi*n) for the reciprocal lattice.
        # But the FFT gives frequencies in cycles per sample, so we need to be careful.
        
        # Let me just compute the Fourier coefficients directly for the needed G vectors.
        # This is O(N^2 * nG) which is 32*32*289 = 294,912 operations, which is fast.
        
        x = (np.arange(N) + 0.5) / N
        y = (np.arange(N) + 0.5) / N
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        inv_eps_coeffs = np.zeros(nG, dtype=complex)
        for idx, (gx, gy) in enumerate(G):
            inv_eps_coeffs[idx] = np.sum(inv_eps * np.exp(-1j * (gx * X + gy * Y))) / (N * N)
        
        # Now solve the eigenvalue problem for several k-points in the irreducible BZ
        # Irreducible BZ for square lattice: 0 <= ky <= kx, 0 <= kx <= pi
        # Key points: Gamma (0,0), M (pi,0), X (pi/2, pi/2), and some interior points
        
        k_points = [
            (0.0, 0.0),      # Gamma
            (np.pi, 0.0),    # M
            (np.pi/2, np.pi/2),  # X
            (np.pi/2, 0.0),  # Midpoint of Gamma-M
            (np.pi/2, np.pi/4),
            (3*np.pi/8, np.pi/8),
            (np.pi/4, np.pi/4),
        ]
        
        max_omega4 = 0.0
        min_omega5 = np.inf
        
        for kx, ky in k_points:
            # Build the eigenvalue matrix
            # For TM: H(k) = diag(|k+G|^2) * A(k) * diag(|k+G'|^2)
            # where A(k)_{G,G'} = (1/eps)_{G-G'}
            # Actually, the standard formulation is:
            # sum_{G'} (k+G)·(k+G') / eps_{G-G'} * a_{G'} = omega^2 * a_G
            
            kvec = np.array([kx, ky])
            k_plus_G = kvec + G  # shape (nG, 2)
            
            # Build the matrix
            # M[G, G'] = (k+G)·(k+G') / eps_{G-G'}
            # But this is not quite right. The correct formulation for TM is:
            # omega^2 * a_G = sum_{G'} (k+G)^2 / eps_{G-G'} * a_{G'}
            # Wait, I need to be more careful.
            
            # For TM polarization (Ez nonzero), the equation is:
            # (1/eps(r)) * (-nabla^2) * Ez = omega^2 * Ez
            # In plane wave basis:
            # sum_{G'} (k+G)^2 * (1/eps)_{G-G'} * a_{G'} = omega^2 * a_G
            # Wait, that's not right either. Let me think again.
            
            # The Helmholtz equation for TM is:
            # -1/eps(r) * nabla^2 Ez(r) = omega^2 Ez(r)
            # In Fourier space, this becomes:
            # sum_{G'} (k+G)^2 * (1/eps)_{G-G'} * a_{G'} = omega^2 * a_G
            # Hmm, actually the standard formulation uses:
            # The operator is (1/eps) * (-nabla^2), which in Fourier space is:
            # (1/eps)_{G-G'} * |k+G|^2
            # So the matrix element is: M[G, G'] = |k+G|^2 * (1/eps)_{G-G'}
            
            # Let me build this matrix
            kG_sq = np.sum(k_plus_G ** 2, axis=1)  # |k+G|^2 for each G
            
            # Build the (1/eps) matrix in the plane wave basis
            # (1/eps)_{G-G'} = inv_eps_coeffs[index of (G-G')]
            # We need to find the index of each difference G-G'
            
            # For efficiency, precompute the index mapping
            G_dict = {}
            for idx, (gx, gy) in enumerate(G):
                G_dict[(int(gx / (2*np.pi)), int(gy / (2*np.pi)))] = idx
            
            M = np.zeros((nG, nG), dtype=complex)
            for i in range(nG):
                for j in range(nG):
                    diff = (int(G[i][0] / (2*np.pi)) - int(G[j][0] / (2*np.pi)),
                            int(G[i][1] / (2*np.pi)) - int(G[j][1] / (2*np.pi)))
                    if diff in G_dict:
                        M[i, j] = kG_sq[i] * inv_eps_coeffs[G_dict[diff]]
            
            # Solve the generalized eigenvalue problem (it's actually a standard eigenvalue problem)
            # Since M is Hermitian (for real eps), we can use eigh
            M = (M + M.conj().T) / 2  # Ensure Hermitian
            eigvals = np.linalg.eigvalsh(M)
            eigvals = np.real(eigvals)
            omega = np.sqrt(np.maximum(eigvals, 0))
            
            # Find the 4th and 5th bands (0-indexed: 3 and 4)
            if len(omega) >= 5:
                omega4 = omega[3]  # 4th band
                omega5 = omega[4]  # 5th band
                max_omega4 = max(max_omega4, omega4)
                min_omega5 = min(min_omega5, omega5)
        
        gap = min_omega5 - max_omega4
        midgap = (min_omega5 + max_omega4) / 2
        if midgap <= 0:
            return 0.0
        return gap / midgap * 2  # gap-midgap ratio
    
    # Initial design: a centered dielectric rod
    q_init = np.zeros((16, 16), dtype=int)
    cx, cy = 7.5, 7.5  # Center of the 16x16 quadrant
    for i in range(16):
        for j in range(16):
            dx = i - cx
            dy = j - cy
            if dx*dx + dy*dy < 3.5**2:
                q_init[i, j] = 1
    
    cell_init = quadrant_to_full(q_init)
    score_init = compute_gap(cell_init)
    
    best_score = score_init
    best_cell = cell_init.copy()
    
    # Local search: flip individual pixels in the quadrant and accept if score improves
    # Use a simple hill-climbing approach with random restarts
    
    rng = np.random.RandomState(42)
    current_q = q_init.copy()
    current_score = score_init
    
    # Precompute some useful structures
    quadrant_indices = []
    for i in range(16):
        for j in range(16):
            quadrant_indices.append((i, j))
    
    # Local search
    step = 0
    while time.time() < deadline:
        # Pick a random pixel in the quadrant to flip
        idx = rng.randint(len(quadrant_indices))
        i, j = quadrant_indices[idx]
        
        # Flip the pixel
        current_q[i, j] = 1 - current_q[i, j]
        cell_new = quadrant_to_full(current_q)
        score_new = compute_gap(cell_new)
        
        if score_new > current_score:
            current_score = score_new
            if score_new > best_score:
                best_score = score_new
                best_cell = cell_new.copy()
        else:
            # Revert
            current_q[i, j] = 1 - current_q[i, j]
        
        step += 1
        # Check time periodically
        if step % 10 == 0:
            if time.time() > deadline - 5:
                break
    
    # Convert back to list of lists
    result = best_cell.tolist()
    return result
# EVOLVE-BLOCK-END
