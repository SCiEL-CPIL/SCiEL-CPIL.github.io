import math
import time
import numpy as np
from scipy import signal as sig

def design(problem):
    """
    Design a symmetric 32x32-pixel unit cell to maximize the TM band gap
    between bands 4 and 5 using wavelet-coefficient parameterization
    with gradient-free optimization in coefficient space.
    """
    size = problem["pixels"]  # 32
    eps_low = problem["eps_low"]   # 1
    eps_high = problem["eps_high"] # 12
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110  # 110s out of 120s budget (with margin)
    
    def quadrant_to_full(q):
        """Convert a 16x16 quadrant to a full 32x32 symmetric cell."""
        q = np.array(q, dtype=int)
        top = np.hstack([q, q[:, ::-1]])
        bottom = top[::-1, :]
        return np.vstack([top, bottom])
    
    def full_to_quadrant(cell):
        """Extract the top-left 16x16 quadrant from a full 32x32 cell."""
        return np.array(cell[:16, :16], dtype=int)
    
    def compute_gap(cell_2d):
        """
        Compute the gap-midgap ratio between TM bands 4 and 5.
        Uses a plane-wave expansion with 289 basis functions.
        """
        N = 32
        eps = np.where(np.array(cell_2d) == 1, eps_high, eps_low)
        
        npw = 8
        G = []
        for mx in range(-npw, npw+1):
            for my in range(-npw, npw+1):
                G.append((2*np.pi*mx, 2*np.pi*my))
        G = np.array(G)
        nG = len(G)
        
        inv_eps = 1.0 / eps
        x = (np.arange(N) + 0.5) / N
        y = (np.arange(N) + 0.5) / N
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        inv_eps_coeffs = np.zeros(nG, dtype=complex)
        for idx, (gx, gy) in enumerate(G):
            inv_eps_coeffs[idx] = np.sum(inv_eps * np.exp(-1j * (gx * X + gy * Y))) / (N * N)
        
        k_points = [
            (0.0, 0.0),
            (np.pi, 0.0),
            (np.pi/2, np.pi/2),
            (np.pi/2, 0.0),
            (np.pi/2, np.pi/4),
            (3*np.pi/8, np.pi/8),
            (np.pi/4, np.pi/4),
            (np.pi/3, np.pi/6),
            (2*np.pi/5, np.pi/5),
            (np.pi/2, np.pi/6),
        ]
        
        max_omega4 = 0.0
        min_omega5 = np.inf
        
        # Precompute G difference indices
        G_dict = {}
        for idx, (gx, gy) in enumerate(G):
            G_dict[(gx, gy)] = idx
        
        for kx, ky in k_points:
            kvec = np.array([kx, ky])
            k_plus_G = kvec + G
            kG_sq = np.sum(k_plus_G**2, axis=1)
            
            # Build A matrix
            A = np.zeros((nG, nG), dtype=complex)
            for i in range(nG):
                for j in range(nG):
                    G_diff = G[i] - G[j]
                    key = (G_diff[0], G_diff[1])
                    if key in G_dict:
                        A[i, j] = inv_eps_coeffs[G_dict[key]]
            
            H = A * kG_sq[np.newaxis, :]
            H = (H + H.conj().T) / 2
            
            eigvals = np.linalg.eigvalsh(H)
            eigvals = np.sort(eigvals.real)
            
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
    
    # Wavelet-based parameterization
    # We'll use Daubechies wavelets at multiple scales to represent the 16x16 quadrant
    # The idea: represent the quadrant as a sum of scaled/shifted wavelets,
    # then threshold to binary. This allows smooth variation of structural features.
    
    # Create a set of wavelet basis functions on the 16x16 quadrant
    # Using db2 (Daubechies order 2) wavelets at scales 1-4
    
    def create_wavelet_basis(n=16):
        """Create a set of wavelet basis functions for the n x n quadrant."""
        from scipy.signal import convolve2d
        
        # db2 wavelet filter (length 4)
        # Normalized db2 low-pass: [0.482962913145, 0.836516303738, 0.224143868042, -0.129409522551]
        db2_lo = np.array([0.482962913145, 0.836516303738, 0.224143868042, -0.129409522551])
        db2_hi = np.array([-0.129409522551, 0.224143868042, -0.836516303738, 0.482962913145])
        
        basis = []
        
        # Add coarse-scale features (low frequency)
        # These capture overall shape (like a blurred rod pattern)
        for scale in [4, 8]:
            # Create a Gaussian-like blob at different positions
            for cx in [4, 8, 12]:
                for cy in [4, 8, 12]:
                    X, Y = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
                    blob = np.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * scale**2))
                    basis.append(blob / np.max(blob))
        
        # Add mid-scale features using wavelet-like patterns
        # Horizontal and vertical edge detectors at different scales
        for scale in [2, 4]:
            # Horizontal edges
            for pos in range(0, n - scale, scale):
                h_edge = np.zeros((n, n))
                h_edge[pos:pos+scale, :] = 1.0
                h_edge[pos+scale:pos+2*scale, :] = -1.0
                if pos + 2*scale <= n:
                    basis.append(h_edge)
            
            # Vertical edges
            for pos in range(0, n - scale, scale):
                v_edge = np.zeros((n, n))
                v_edge[:, pos:pos+scale] = 1.0
                v_edge[:, pos+scale:pos+2*scale] = -1.0
                if pos + 2*scale <= n:
                    basis.append(v_edge)
        
        # Add diagonal features
        for scale in [2, 4]:
            for offset in range(-n, n, scale):
                diag = np.zeros((n, n))
                for i in range(n):
                    j = i + offset
                    if 0 <= j < n:
                        diag[i, j] = 1.0
                        if 0 <= j+scale < n:
                            diag[i, j+scale] = -1.0
                if np.any(diag):
                    basis.append(diag)
        
        # Add ring patterns (annuli)
        for cx in [4, 8, 12]:
            for cy in [4, 8, 12]:
                for r_inner in [1, 2, 3]:
                    for r_outer in [r_inner+1, r_inner+2, r_inner+3]:
                        X, Y = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
                        R = np.sqrt((X - cx)**2 + (Y - cy)**2)
                        ring = ((R >= r_inner) & (R < r_outer)).astype(float)
                        if np.any(ring):
                            basis.append(ring / np.max(ring))
        
        # Normalize all basis functions
        basis = np.array(basis)
        for i in range(len(basis)):
            norm = np.linalg.norm(basis[i])
            if norm > 0:
                basis[i] /= norm
        
        return basis
    
    # Create the wavelet basis
    basis = create_wavelet_basis(16)
    n_basis = len(basis)
    
    # Initial coefficients: start with a 2x2 rod pattern
    def initial_coefficients():
        """Initialize coefficients to represent a 2x2 rod pattern."""
        coeffs = np.zeros(n_basis)
        
        # Find which basis functions best represent a 2x2 rod pattern
        target = np.zeros((16, 16))
        for cx, cy in [(4, 4), (4, 12), (12, 4), (12, 12)]:
            X, Y = np.meshgrid(np.arange(16), np.arange(16), indexing='ij')
            target += np.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * 2.5**2))
        target = (target > 0.3).astype(float)
        
        # Project target onto basis
        for i in range(n_basis):
            coeffs[i] = np.sum(target * basis[i])
        
        return coeffs
    
    def coeffs_to_quadrant(coeffs, threshold=0.0):
        """Convert wavelet coefficients to a binary 16x16 quadrant."""
        # Reconstruct the continuous field
        field = np.zeros((16, 16))
        for i in range(n_basis):
            field += coeffs[i] * basis[i]
        
        # Threshold to binary
        quadrant = (field > threshold).astype(int)
        return quadrant
    
    # Start with initial design
    best_coeffs = initial_coefficients()
    best_quadrant = coeffs_to_quadrant(best_coeffs)
    best_full_cell = quadrant_to_full(best_quadrant)
    best_score = compute_gap(best_full_cell)
    
    # Optimization: Nelder-Mead style search in coefficient space
    # with adaptive step sizes
    
    def evaluate_coeffs(coeffs):
        """Evaluate the score for given coefficients."""
        quadrant = coeffs_to_quadrant(coeffs)
        full_cell = quadrant_to_full(quadrant)
        return compute_gap(full_cell)
    
    # Nelder-Mead optimization
    def nelder_mead(coeffs0, f, max_iter=200, tol=1e-4):
        """Simple Nelder-Mead optimization."""
        n = len(coeffs0)
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5
        
        # Create simplex
        simplex = [coeffs0.copy()]
        for i in range(n):
            pt = coeffs0.copy()
            pt[i] += 0.1 if coeffs0[i] == 0 else 0.1
            simplex.append(pt)
        
        # Evaluate initial simplex
        fvals = [f(pt) for pt in simplex]
        
        best_idx = np.argmin(fvals)
        best_val = fvals[best_idx]
        best_pt = simplex[best_idx].copy()
        
        for iteration in range(max_iter):
            if time.time() > deadline:
                break
            
            # Sort simplex
            order = np.argsort(fvals)
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]
            
            # Check convergence
            if abs(fvals[-1] - fvals[0]) < tol:
                break
            
            # Centroid of all but worst point
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            worst = simplex[-1]
            xr = centroid + alpha * (centroid - worst)
            fr = f(xr)
            
            if fr < fvals[0]:
                # Expansion
                xe = centroid + gamma * (centroid - worst)
                fe = f(xe)
                if fe < fr:
                    simplex[-1] = xe
                    fvals[-1] = fe
                else:
                    simplex[-1] = xr
                    fvals[-1] = fr
            elif fr < fvals[-2]:
                simplex[-1] = xr
                fvals[-1] = fr
            else:
                # Contraction
                if fr < fvals[-1]:
                    xc = centroid + rho * (worst - centroid)
                    fc = f(xc)
                    if fc <= fr:
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            fvals[i] = f(simplex[i])
                else:
                    # Shrink
                    for i in range(1, len(simplex)):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        fvals[i] = f(simplex[i])
            
            # Track best
            cur_best_idx = np.argmin(fvals)
            if fvals[cur_best_idx] < best_val:
                best_val = fvals[cur_best_idx]
                best_pt = simplex[cur_best_idx].copy()
        
        return best_pt, best_val
    
    # Run Nelder-Mead with a limited number of coefficients
    # Use only the most important coefficients (those with largest initial magnitude)
    importance = np.abs(best_coeffs)
    top_k = min(30, n_basis)
    top_indices = np.argsort(importance)[::-1][:top_k]
    
    # Project to top-k coefficients
    def evaluate_top_k(coeffs_topk):
        """Evaluate score using only top-k coefficients."""
        full_coeffs = np.zeros(n_basis)
        full_coeffs[top_indices] = coeffs_topk
        return evaluate_coeffs(full_coeffs)
    
    initial_topk = best_coeffs[top_indices].copy()
    
    # Run optimization
    opt_result, opt_score = nelder_mead(
        initial_topk, 
        evaluate_top_k, 
        max_iter=100,
        tol=1e-5
    )
    
    if opt_score < best_score:
        # Use the better of the two
        best_coeffs[top_indices] = initial_topk
    else:
        best_coeffs[top_indices] = opt_result
    
    # Final evaluation
    best_quadrant = coeffs_to_quadrant(best_coeffs)
    best_full_cell = quadrant_to_full(best_quadrant)
    final_score = compute_gap(best_full_cell)
    
    # If the wavelet approach didn't improve, fall back to pixel-level local search
    if final_score <= best_score:
        # Do a few rounds of pixel flipping to refine
        quadrant = full_to_quadrant(best_full_cell)
        for _ in range(50):
            if time.time() > deadline:
                break
            i = np.random.randint(0, 16)
            j = np.random.randint(0, 16)
            quadrant[i, j] = 1 - quadrant[i, j]
            full_cell = quadrant_to_full(quadrant)
            score = compute_gap(full_cell)
            if score > best_score:
                best_score = score
                best_full_cell = full_cell
            else:
                quadrant[i, j] = 1 - quadrant[i, j]
    
    # Return the best cell found
    best_full_cell = quadrant_to_full(full_to_quadrant(best_full_cell))
    
    # Convert to nested list
    return best_full_cell.tolist()
