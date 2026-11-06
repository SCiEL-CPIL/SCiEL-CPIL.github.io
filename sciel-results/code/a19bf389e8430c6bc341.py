import math
import time
import numpy as np

def design(problem):
    """Design a 32x32 D4-symmetric unit cell for maximum TM 4-5 band gap.
    
    Uses a 2x2 array of circular dielectric rods with local optimization,
    followed by pixel-level refinement for fine-tuning.
    """
    size = problem.get("pixels", 32)
    eps_low = problem.get("eps_low", 1)
    eps_high = problem.get("eps_high", 12)
    
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    def quadrant_to_full(q):
        """Convert a 16x16 quadrant to a full 32x32 symmetric cell."""
        q = np.array(q, dtype=int)
        top = np.hstack([q, q[:, ::-1]])
        bottom = top[::-1, :]
        return np.vstack([top, bottom])
    
    def make_cell_from_params(radius, cx, cy):
        """Create a 32x32 cell with 4 circular rods at (cx,cy), (cx,1-cy), (1-cx,cy), (1-cx,1-cy)."""
        cell = np.zeros((size, size), dtype=int)
        centers = [
            (cx, cy),
            (cx, 1.0 - cy),
            (1.0 - cx, cy),
            (1.0 - cx, 1.0 - cy)
        ]
        for i in range(size):
            y = (i + 0.5) / size
            for j in range(size):
                x = (j + 0.5) / size
                for ccx, ccy in centers:
                    dx = x - ccx
                    dy = y - ccy
                    if dx*dx + dy*dy < radius*radius:
                        cell[i, j] = 1
                        break
        return cell
    
    def compute_gap(cell_2d):
        """Compute the gap-midgap ratio between TM bands 4 and 5."""
        eps = np.where(cell_2d == 1, eps_high, eps_low)
        
        # 17x17 = 289 plane waves
        npw = 8
        G = []
        for mx in range(-npw, npw+1):
            for my in range(-npw, npw+1):
                G.append((2*np.pi*mx, 2*np.pi*my))
        G = np.array(G)
        nG = len(G)
        
        # Compute inverse dielectric Fourier coefficients
        inv_eps = 1.0 / eps
        x = (np.arange(size) + 0.5) / size
        y = (np.arange(size) + 0.5) / size
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        inv_eps_coeffs = np.zeros(nG, dtype=complex)
        for idx, (gx, gy) in enumerate(G):
            inv_eps_coeffs[idx] = np.sum(inv_eps * np.exp(-1j * (gx * X + gy * Y))) / (size * size)
        
        # Key k-points in the irreducible Brillouin zone
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
            (np.pi/2, np.pi/3),
            (3*np.pi/4, np.pi/4),
            (np.pi/2, np.pi/8),
            (5*np.pi/8, np.pi/8),
            (np.pi/4, 0.0),
        ]
        
        max_omega4 = 0.0
        min_omega5 = float('inf')
        
        for kx, ky in k_points:
            kvec = np.array([kx, ky])
            k_plus_G = kvec + G  # shape (nG, 2)
            kG_sq = np.sum(k_plus_G**2, axis=1)  # shape (nG,)
            
            # Build the A matrix: A[G, G'] = (1/eps)_{G-G'}
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
            
            # The eigenvalue matrix is: H = A * diag(kG_sq)
            H = A * kG_sq[np.newaxis, :]
            
            # Make Hermitian
            H = (H + H.conj().T) / 2
            
            # Get eigenvalues
            try:
                eigvals = np.linalg.eigvalsh(H)
                eigvals = np.sort(eigvals.real)
            except:
                continue
            
            if len(eigvals) >= 5:
                # Bands 4 and 5 (0-indexed: 3 and 4)
                omega4 = eigvals[3]
                omega5 = eigvals[4]
                if omega4 > max_omega4:
                    max_omega4 = omega4
                if omega5 < min_omega5:
                    min_omega5 = omega5
        
        if min_omega5 == float('inf') or max_omega4 == 0:
            return 0.0
        
        gap = min_omega5 - max_omega4
        midgap = (min_omega5 + max_omega4) / 2
        if midgap <= 0:
            return 0.0
        
        return gap / midgap
    
    # Phase 1: Optimize circular rod parameters
    best_score = -float('inf')
    best_cell = None
    
    # Initial guess
    radius = 0.14
    cx = 0.25
    cy = 0.25
    
    # Local search over radius, cx, cy
    param_ranges = {
        'radius': (0.08, 0.22),
        'cx': (0.18, 0.32),
        'cy': (0.18, 0.32)
    }
    
    # Coarse grid search
    n_steps = 8
    for r_idx in range(n_steps):
        if time.time() > deadline - 30:
            break
        r = param_ranges['radius'][0] + r_idx * (param_ranges['radius'][1] - param_ranges['radius'][0]) / (n_steps - 1)
        for cx_idx in range(n_steps):
            if time.time() > deadline - 30:
                break
            cx_val = param_ranges['cx'][0] + cx_idx * (param_ranges['cx'][1] - param_ranges['cx'][0]) / (n_steps - 1)
            for cy_idx in range(n_steps):
                if time.time() > deadline - 30:
                    break
                cy_val = param_ranges['cy'][0] + cy_idx * (param_ranges['cy'][1] - param_ranges['cy'][0]) / (n_steps - 1)
                
                cell = make_cell_from_params(r, cx_val, cy_val)
                score = compute_gap(cell)
                
                if score > best_score:
                    best_score = score
                    best_cell = cell.copy()
                    radius, cx, cy = r, cx_val, cy_val
    
    if best_cell is None:
        # Fallback
        best_cell = make_cell_from_params(0.14, 0.25, 0.25)
    
    # Phase 2: Pixel-level refinement
    if time.time() < deadline - 10:
        # Try flipping individual pixels in the quadrant and keep improvements
        improved = True
        max_iter = 500
        iter_count = 0
        
        while improved and iter_count < max_iter and time.time() < deadline - 10:
            improved = False
            iter_count += 1
            
            # Work with the quadrant
            quad = best_cell[:16, :16].copy()
            
            for i in range(16):
                if time.time() > deadline - 10:
                    break
                for j in range(16):
                    if time.time() > deadline - 10:
                        break
                    
                    # Try flipping this pixel
                    old_val = quad[i, j]
                    quad[i, j] = 1 - old_val
                    
                    full_cell = quadrant_to_full(quad)
                    score = compute_gap(full_cell)
                    
                    if score > best_score + 1e-6:
                        best_score = score
                        best_cell = full_cell.copy()
                        improved = True
                    else:
                        quad[i, j] = old_val  # Revert
    
    # Convert to nested list
    result = best_cell.tolist()
    
    # Ensure D4 symmetry
    for i in range(size):
        for j in range(size):
            # Horizontal flip
            result[i][j] = max(result[i][j], result[i][size-1-j])
            # Vertical flip  
            result[i][j] = max(result[i][j], result[size-1-i][j])
            # Both flips
            result[i][j] = max(result[i][j], result[size-1-i][size-1-j])
            # Transpose
            result[i][j] = max(result[i][j], result[j][i])
    
    # Final symmetry enforcement
    for i in range(size):
        for j in range(size):
            val = 0
            for ri in [i, size-1-i]:
                for rj in [j, size-1-j, i, size-1-i]:
                    val = max(val, result[ri][rj])
            # Also check transpose
            for ri in [i, size-1-i]:
                for rj in [j, size-1-j]:
                    val = max(val, result[rj][ri])
            result[i][j] = val
    
    return result
