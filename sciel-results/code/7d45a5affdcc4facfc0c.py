import math
import time
import numpy as np

def design(problem):
    """
    Design a symmetric 32x32-pixel unit cell to maximize the TM band gap
    between bands 4 and 5.
    
    Strategy:
    1. Restrict search to D4-symmetric rectangular blocks (dielectric in air or air in dielectric).
    2. Exhaustively evaluate all valid width/height combinations.
    3. Use a fast plane-wave solver to compute the gap ratio.
    """
    size = problem["pixels"]  # 32
    eps_low = problem["eps_low"]   # 1
    eps_high = problem["eps_high"] # 12
    
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Plane wave basis: 289 waves -> (2*8+1)^2
    npw = 8
    # Generate G vectors
    G_list = []
    for mx in range(-npw, npw + 1):
        for my in range(-npw, npw + 1):
            G_list.append((2 * np.pi * mx, 2 * np.pi * my))
    G = np.array(G_list)
    nG = len(G)
    
    # Pre-compute indices for G differences
    # We need to map (gx, gy) to index in G_list
    # Since G vectors are on a grid, we can compute index directly
    # idx = (mx + npw) * (2*npw + 1) + (my + npw)
    def get_g_index(gx, gy):
        mx = round(gx / (2 * np.pi))
        my = round(gy / (2 * np.pi))
        if -npw <= mx <= npw and -npw <= my <= npw:
            return (mx + npw) * (2 * npw + 1) + (my + npw)
        return -1

    # K-points in irreducible BZ (0 <= ky <= kx <= pi)
    # We sample a grid of points to ensure we catch the gap edges
    k_points = []
    # High symmetry
    k_points.append((0.0, 0.0))       # Gamma
    k_points.append((np.pi, 0.0))     # M
    k_points.append((np.pi/2, np.pi/2)) # X
    # Along Gamma-M (y=0)
    k_points.append((np.pi/4, 0.0))
    k_points.append((np.pi/2, 0.0))
    # Along Gamma-X (x=y)
    k_points.append((np.pi/4, np.pi/4))
    # Interior
    k_points.append((3*np.pi/8, np.pi/8))
    k_points.append((np.pi/2, np.pi/4))
    k_points.append((np.pi/2, np.pi/6))
    k_points.append((3*np.pi/8, np.pi/6))
    k_points.append((np.pi/2, np.pi/3)) # Note: ky <= kx, so pi/3 <= pi/2 is ok
    
    def compute_gap_ratio(cell_2d):
        """
        Compute gap-midgap ratio for bands 4 and 5.
        """
        if time.time() > deadline:
            return -1.0
            
        eps = np.where(cell_2d == 1, eps_high, eps_low)
        inv_eps = 1.0 / eps
        
        # Compute Fourier coefficients of 1/eps
        # inv_eps_coeffs[G] = 1/N^2 * sum_{i,j} inv_eps[i,j] * exp(-i G . r_ij)
        # r_ij = (i+0.5, j+0.5) / N
        N = size
        x = (np.arange(N) + 0.5) / N
        y = (np.arange(N) + 0.5) / N
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        inv_eps_coeffs = np.zeros(nG, dtype=complex)
        for idx in range(nG):
            gx, gy = G[idx]
            inv_eps_coeffs[idx] = np.sum(inv_eps * np.exp(-1j * (gx * X + gy * Y))) / (N * N)
        
        max_omega4 = 0.0
        min_omega5 = np.inf
        
        for kx, ky in k_points:
            if time.time() > deadline:
                return -1.0
                
            kvec = np.array([kx, ky])
            k_plus_G = kvec + G
            kG_sq = np.sum(k_plus_G**2, axis=1)
            
            # Build H matrix
            # H[i, j] = (1/eps)_{G_i - G_j} * kG_sq[j]
            # We can construct the convolution matrix A where A[i,j] = inv_eps_coeffs[G_i - G_j]
            # Then H = A * diag(kG_sq)
            
            # Efficient construction of A
            # A is circulant-like but in 2D.
            # We can just loop over i, j or use broadcasting if we had a 2D array for coeffs.
            # Let's use a 2D array for coeffs for easier indexing.
            # coeffs_2d[mx, my] where mx,my in [-npw, npw]
            coeffs_2d = np.zeros((2*npw+1, 2*npw+1), dtype=complex)
            for idx in range(nG):
                gx, gy = G[idx]
                mx = int(round(gx / (2 * np.pi)))
                my = int(round(gy / (2 * np.pi)))
                coeffs_2d[mx + npw, my + npw] = inv_eps_coeffs[idx]
            
            # Extract G_x and G_y components
            Gx = G[:, 0]
            Gy = G[:, 1]
            
            # We need inv_eps_coeffs[G_i - G_j]
            # G_i - G_j has components (Gx_i - Gx_j, Gy_i - Gy_j)
            # Let's compute the differences
            diff_Gx = Gx[:, np.newaxis] - Gx[np.newaxis, :]
            diff_Gy = Gy[:, np.newaxis] - Gy[np.newaxis, :]
            
            # Convert diff to indices in coeffs_2d
            # diff_Gx is multiple of 2*pi
            diff_mx = diff_Gx / (2 * np.pi)
            diff_my = diff_Gy / (2 * np.pi)
            
            # Check if within bounds
            # If out of bounds, coefficient is 0
            # coeffs_2d index is mx + npw
            idx_i = diff_mx + npw
            idx_j = diff_my + npw
            
            # Initialize A
            A = np.zeros((nG, nG), dtype=complex)
            
            # Mask valid indices
            valid = (idx_i >= 0) & (idx_i < 2*npw+1) & (idx_j >= 0) & (idx_j < 2*npw+1)
            
            # Fill A
            # We need to be careful with integer rounding if floating point errors exist
            # But G vectors are exact multiples of 2*pi, so differences are exact integers in units of 2*pi
            idx_i_int = np.round(idx_i).astype(int)
            idx_j_int = np.round(idx_j).astype(int)
            
            A[valid] = coeffs_2d[idx_i_int[valid], idx_j_int[valid]]
            
            # H = A * diag(kG_sq)
            # H[i, j] = A[i, j] * kG_sq[j]
            H = A * kG_sq[np.newaxis, :]
            
            # H should be Hermitian. Symmetrize to be safe.
            H = (H + H.conj().T) / 2
            
            # Solve eigenvalue problem
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

    best_score = -np.inf
    best_cell = None
    
    # Search Space: Rectangular Blocks
    # 1. Dielectric Block in Air (1s in center)
    # 2. Air Hole in Dielectric (0s in center, 1s elsewhere)
    
    # Widths and Heights must be even to maintain symmetry about the center of 32x32
    # Center is between 15 and 16.
    # A block of width W centered: starts at 16 - W/2, ends at 16 + W/2 - 1
    # W must be even.
    possible_widths = list(range(2, size + 1, 2))
    
    for w in possible_widths:
        for h in possible_widths:
            if time.time() > deadline:
                break
                
            # Construct Cell 1: Dielectric Block
            cell1 = np.zeros((size, size), dtype=int)
            start_i = size // 2 - h // 2
            end_i = start_i + h
            start_j = size // 2 - w // 2
            end_j = start_j + w
            cell1[start_i:end_i, start_j:end_j] = 1
            
            score1 = compute_gap_ratio(cell1)
            if score1 > best_score:
                best_score = score1
                best_cell = cell1.copy()
                
            # Construct Cell 2: Air Hole (Inverse)
            cell2 = 1 - cell1
            
            score2 = compute_gap_ratio(cell2)
            if score2 > best_score:
                best_score = score2
                best_cell = cell2.copy()
                
        if time.time() > deadline:
            break
            
    if best_cell is None:
        # Fallback: default design
        cell = np.zeros((size, size), dtype=int)
        cell[8:24, 8:24] = 1
        best_cell = cell
        
    return best_cell.tolist()
