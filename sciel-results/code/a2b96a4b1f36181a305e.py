import math
import time
import numpy as np

def design(problem):
    """
    Design a symmetric 32x32-pixel unit cell to maximize the TM band gap
    between bands 4 and 5 using a low-frequency Fourier parameterization
    with local optimization in coefficient space.
    """
    size = problem["pixels"]  # 32
    eps_low = problem["eps_low"]   # 1
    eps_high = problem["eps_high"] # 12
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 115  # 115s out of 120s budget (with margin)
    
    # We'll work with the 16x16 top-left quadrant due to 4-fold symmetry
    # For a 32x32 cell with D4 symmetry:
    # cell[i][j] == cell[31-i][j] == cell[i][31-j] == cell[31-i][31-j]
    # and cell[i][j] == cell[j][i]
    # So we specify the 16x16 top-left quadrant and reflect it.
    
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
    def compute_gap(cell_2d):
        """
        Compute the gap-midgap ratio between TM bands 4 and 5.
        Uses a plane-wave expansion with 289 basis functions.
        """
        N = 32
        eps = np.where(np.array(cell_2d) == 1, eps_high, eps_low)
        
        # Use 289 plane waves: (2*8+1)^2 = 17^2 = 289
        npw = 8
        G = []
        for mx in range(-npw, npw+1):
            for my in range(-npw, npw+1):
                G.append((2*np.pi*mx, 2*np.pi*my))
        G = np.array(G)
        nG = len(G)
        
        # Compute inverse dielectric Fourier coefficients
        inv_eps = 1.0 / eps
        x = (np.arange(N) + 0.5) / N
        y = (np.arange(N) + 0.5) / N
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        inv_eps_coeffs = np.zeros(nG, dtype=complex)
        for idx, (gx, gy) in enumerate(G):
            inv_eps_coeffs[idx] = np.sum(inv_eps * np.exp(-1j * (gx * X + gy * Y))) / (N * N)
        
        # Key k-points in the irreducible Brillouin zone
        # Irreducible BZ: 0 <= ky <= kx, 0 <= kx <= pi
        k_points = [
            (0.0, 0.0),           # Gamma
            (np.pi, 0.0),         # M
            (np.pi/2, np.pi/2),   # X
            (np.pi/2, 0.0),       # Midpoint Gamma-M
            (np.pi/2, np.pi/4),
            (3*np.pi/8, np.pi/8),
            (np.pi/4, np.pi/4),
            (np.pi/3, np.pi/6),
            (2*np.pi/5, np.pi/5),
            (np.pi/2, np.pi/6),
        ]
        
        max_omega4 = 0.0
        min_omega5 = np.inf
        
        for kx, ky in k_points:
            kvec = np.array([kx, ky])
            k_plus_G = kvec + G  # shape (nG, 2)
            
            # Build the eigenvalue matrix for TM polarization
            # H[G, G'] = |k+G|^2 * (1/eps)_{G-G'}
            # But we need to be careful with the formulation
            
            # The correct formulation for TM is:
            # omega^2 * a_G = sum_{G'} (k+G)^2 * (1/eps)_{G-G'} * a_{G'}
            # This is not quite right. Let me use the standard formulation.
            
            # For TM: (1/eps(r)) * (-nabla^2) Ez = omega^2 Ez
            # In plane wave basis:
            # sum_{G'} (k+G)^2 * (1/eps)_{G-G'} * a_{G'} = omega^2 * a_G
            
            # Wait, this is still not right. The operator (1/eps)(-nabla^2) 
            # in Fourier space involves convolution with 1/eps.
            
            # The correct matrix element is:
            # H[G, G'] = (k+G)·(k+G') * (1/eps)_{G-G'}
            # No, that's for TE. For TM it's:
            # H[G, G'] = (k+G)^2 * (1/eps)_{G-G'}
            
            # Actually, I think the standard formulation is:
            # For TM: omega^2 a_G = sum_{G'} (k+G)^2 * (1/eps)_{G-G'} * a_{G'}
            # This can be written as: omega^2 a = D * A * a
            # where D = diag((k+G)^2) and A is the matrix with elements (1/eps)_{G-G'}
            
            # Let me construct this properly.
            
            # First, compute (k+G)^2 for each G
            kG_sq = np.sum(k_plus_G**2, axis=1)  # shape (nG,)
            
            # Build the A matrix: A[G, G'] = (1/eps)_{G-G'}
            # We need to find the index of G-G' in our G list
            G_dict = {}
            for idx, (gx, gy) in enumerate(G):
                G_dict[(gx, gy)] = idx
            
            A = np.zeros((nG, nG), dtype=complex)
            for i in range(nG):
                for j in range(nG):
                    G_diff = G[i] - G[j]
                    key = (G_diff[0], G_diff[1])
                    if key in G_dict:
                        A[i, j] = inv_eps_coeffs[G_dict[key]]
                    # If G_diff is not in our basis, the coefficient is 0
                    # (this is the truncation error)
            
            # The eigenvalue matrix is: H = A * diag(kG_sq)
            # Or equivalently: H[i, j] = A[i, j] * kG_sq[j]
            H = A * kG_sq[np.newaxis, :]
            
            # Solve the eigenvalue problem
            # H is not Hermitian in general, but for our case it should be
            # Let's make it Hermitian by averaging with its conjugate transpose
            H = (H + H.conj().T) / 2
            
            # Get eigenvalues
            eigvals = np.linalg.eigvalsh(H)
            eigvals = np.sort(eigvals.real)
            
            # Take the 5th and 6th bands (indices 4 and 5, 0-indexed)
            # But we need to make sure we have at least 6 bands
            if len(eigvals) >= 6:
                omega4 = eigvals[4]
                omega5 = eigvals[5]
                max_omega4 = max(max_omega4, omega4)
                min_omega5 = min(min_omega5, omega5)
        
        if min_omega5 == np.inf or max_omega4 == 0:
            return 0.0
        
        gap = min_omega5 - max_omega4
        midgap = (min_omega5 + max_omega4) / 2
        
        if midgap <= 0:
            return 0.0
        
        return gap / midgap
    
    # Initialize with a 2x2 array of rods
    # Centers at (0.25, 0.25), (0.25, 0.75), (0.75, 0.25), (0.75, 0.75)
    # Radius 0.14 in lattice units
    
    def initial_design():
        """Create initial 2x2 rod design."""
        radius = 0.14
        centers = [
            (0.25, 0.25),
            (0.25, 0.75),
            (0.75, 0.25),
            (0.75, 0.75)
        ]
        
        cell = np.zeros((size, size), dtype=int)
        for i in range(size):
            y = (i + 0.5) / size
            for j in range(size):
                x = (j + 0.5) / size
                for cx, cy in centers:
                    dx = x - cx
                    dy = y - cy
                    if dx*dx + dy*dy < radius*radius:
                        cell[i, j] = 1
                        break
        
        return cell
    
    # Start with initial design
    best_cell = initial_design()
    best_score = compute_gap(best_cell)
    
    # Local search in the 16x16 quadrant space
    # We'll flip individual pixels and see if it improves the score
    
    quadrant = full_to_quadrant(best_cell)
    best_quadrant = quadrant.copy()
    
    # Perform local search
    step_count = 0
    max_steps = 5000
    
    while time.time() < deadline and step_count < max_steps:
        step_count += 1
        
        # Try flipping a random pixel in the quadrant
        i = np.random.randint(0, 16)
        j = np.random.randint(0, 16)
        
        # Flip the pixel
        quadrant[i, j] = 1 - quadrant[i, j]
        
        # Construct full cell and evaluate
        full_cell = quadrant_to_full(quadrant)
        score = compute_gap(full_cell)
        
        if score > best_score:
            best_score = score
            best_quadrant = quadrant.copy()
        else:
            # Revert
            quadrant[i, j] = 1 - quadrant[i, j]
    
    # Return the best cell found
    best_full_cell = quadrant_to_full(best_quadrant)
    
    # Convert to nested list
    return best_full_cell.tolist()
