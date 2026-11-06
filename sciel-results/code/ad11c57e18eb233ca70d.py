import numpy as np
import time
import copy

def design(problem):
    size = problem["pixels"]
    eps_low = problem["eps_low"]
    eps_high = problem["eps_high"]
    
    # 1. Setup G vectors and k-points
    # 289 plane waves: Gx, Gy in [-13, 13]
    G_range = np.arange(-13, 14)
    Gx, Gy = np.meshgrid(G_range, G_range, indexing='ij')
    G_vecs = np.column_stack((Gx.ravel(), Gy.ravel()))
    N_G = len(G_vecs)
    
    # 30 k-points in irreducible BZ
    # BZ: kx, ky in [-pi, pi]. Irreducible: 0 <= ky <= kx <= pi
    k_points = []
    # Gamma
    k_points.append([0, 0])
    # X (pi, 0)
    k_points.append([np.pi, 0])
    # M (pi, pi)
    k_points.append([np.pi, np.pi])
    # Along kx axis
    k_points.append([np.pi/2, 0])
    # Along diagonal
    k_points.append([np.pi/2, np.pi/2])
    # Along ky axis (symmetric to kx)
    # More points along boundaries
    for i in range(1, 5):
        k = np.pi * i / 5
        k_points.append([k, 0])
        k_points.append([k, k])
        k_points.append([np.pi, k])
    # Remove duplicates
    unique_k_points = []
    for k in k_points:
        if not any(np.allclose(k, uk) for uk in unique_k_points):
            unique_k_points.append(k)
    k_points = np.array(unique_k_points[:30]) # Ensure 30 points
    
    # Fast k-points subset
    fast_k_indices = [0, 1, 2, 3, 4] # Gamma, X, M, X/2, M/2
    
    # 2. Pixel coordinates
    # r_ij = (i - 15.5)/32, (j - 15.5)/32
    # Note: The unit cell is 1x1.
    x_coords = (np.arange(size) + 0.5) / size - 0.5
    y_coords = (np.arange(size) + 0.5) / size - 0.5
    X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
    R_vecs = np.column_stack((X.ravel(), Y.ravel()))
    
    # Precompute exp(-i G . R) for all G and all pixels?
    # G (289x2), R (1024x2)
    # Dot product: 289 x 1024
    # This is 296k complex exponentials.
    # exp(-1j * (Gx*Rx + Gy*Ry))
    # We can compute this once.
    G_R_dot = np.dot(G_vecs, R_vecs.T) # 289 x 1024
    exp_G_R = np.exp(-1j * G_R_dot) # 289 x 1024
    
    # 3. Independent pixels
    # Indices i, j in [16, 31] with i >= j
    indep_indices = []
    for i in range(16, 32):
        for j in range(16, i + 1):
            indep_indices.append((i, j))
    N_indep = len(indep_indices)
    
    # Map independent pixel (i,j) to all symmetric pixels
    # We need to know which pixels in the 32x32 grid correspond to which independent pixel.
    # Symmetry:
    # (i, j) -> (i, j)
    # (i, 31-j)
    # (31-i, j)
    # (31-i, 31-j)
    # (j, i)
    # (j, 31-i)
    # (31-j, i)
    # (31-j, 31-i)
    
    # Let's build a map: pixel_index -> independent_index
    # And for each independent_index, list of pixel_indices it controls
    pixel_to_indep = {}
    indep_to_pixels = [[] for _ in range(N_indep)]
    
    for idx, (i, j) in enumerate(indep_indices):
        # Generate all symmetric counterparts
        coords = set()
        coords.add((i, j))
        coords.add((i, 31-j))
        coords.add((31-i, j))
        coords.add((31-i, 31-j))
        coords.add((j, i))
        coords.add((j, 31-i))
        coords.add((31-j, i))
        coords.add((31-j, 31-i))
        
        for ci, cj in coords:
            p_idx = ci * size + cj
            if p_idx not in pixel_to_indep:
                pixel_to_indep[p_idx] = idx
                indep_to_pixels[idx].append(p_idx)
            else:
                # Check if consistent
                if pixel_to_indep[p_idx] != idx:
                    # This shouldn't happen if logic is correct
                    pass
    
    # 4. Initialization
    # Start with a cross shape
    cell = [[0]*size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            # Cross: dielectric if i in [14,17] or j in [14,17]
            if 14 <= i <= 17 or 14 <= j <= 17:
                cell[i][j] = 1
    
    # Calculate initial rho_G
    rho_G = np.zeros(N_G, dtype=complex)
    for i in range(size):
        for j in range(size):
            if cell[i][j] == 1:
                p_idx = i * size + j
                # rho_G += (1/1024) * exp_G_R[:, p_idx]
                rho_G += exp_G_R[:, p_idx] / 1024.0
    
    # 5. Evaluation Functions
    def compute_bands(rho_G, k_points_subset):
        # Returns max omega_4 and min omega_5
        max_w4 = 0
        min_w5 = np.inf
        
        # epsilon_G = delta_G0 + 11 * rho_G
        eps_G = np.zeros(N_G, dtype=complex)
        eps_G[0] = 1.0 # G=0
        eps_G += 11.0 * rho_G
        
        for k in k_points_subset:
            # Build D matrix (diagonal)
            # k_G = k + G
            K_G = k + G_vecs # (289, 2)
            D_diag = np.sum(K_G**2, axis=1) # (289,)
            
            # Build epsilon matrix
            # eps_mat[G1, G2] = eps_G[G1 - G2]
            # We need to map G indices to differences
            # G indices are 0..288. G[0] = (-13, -13), G[1]=(-13, -12)...
            # We can precompute a mapping of (Gx, Gy) to index
            G_map = {}
            for idx, g in enumerate(G_vecs):
                G_map[(int(g[0]), int(g[1]))] = idx
            
            eps_mat = np.zeros((N_G, N_G), dtype=complex)
            for i in range(N_G):
                gi = G_vecs[i]
                for j in range(N_G):
                    gj = G_vecs[j]
                    diff = (int(gi[0] - gj[0]), int(gi[1] - gj[1]))
                    if diff in G_map:
                        idx_diff = G_map[diff]
                        eps_mat[i, j] = eps_G[idx_diff]
            
            # Solve generalized eigenvalue problem: D c = w^2 eps_mat c
            # => eps_mat^{-1} D c = w^2 c
            # Use Cholesky of eps_mat (it is SPD)
            try:
                L = np.linalg.cholesky(eps_mat)
                # L L^H = eps_mat
                # y = L^H c => c = L^{-H} y
                # D L^{-H} y = w^2 L L^H L^{-H} y = w^2 L y
                # L^{-1} D L^{-H} y = w^2 y
                L_inv = np.linalg.inv(L)
                L_inv_H = L_inv.conj().T
                H = L_inv @ np.diag(D_diag) @ L_inv_H
                H = (H + H.conj().T) / 2 # Symmetrize
                eigvals = np.linalg.eigvalsh(H)
                eigvals = np.sort(eigvals)
                # We need 5 smallest
                if len(eigvals) >= 5:
                    w4 = np.sqrt(eigvals[3])
                    w5 = np.sqrt(eigvals[4])
                    max_w4 = max(max_w4, w4)
                    min_w5 = min(min_w5, w5)
            except np.linalg.LinAlgError:
                # If Cholesky fails, use dense solve
                try:
                    eigvals, eigvecs = np.linalg.eig(np.linalg.solve(eps_mat, np.diag(D_diag)))
                    eigvals = np.sort(np.real(eigvals))
                    if len(eigvals) >= 5:
                        w4 = np.sqrt(eigvals[3])
                        w5 = np.sqrt(eigvals[4])
                        max_w4 = max(max_w4, w4)
                        min_w5 = min(min_w5, w5)
                except:
                    pass
        
        if min_w5 == np.inf:
            return 0.0
        gap = min_w5 - max_w4
        midgap = (min_w5 + max_w4) / 2.0
        if midgap == 0:
            return 0.0
        return gap / midgap

    # 6. Optimization Loop
    best_score = compute_bands(rho_G, k_points[fast_k_indices])
    best_cell = copy.deepcopy(cell)
    current_score = best_score
    start_time = time.time()
    deadline = start_time + 220.0 # 10s margin
    
    # Precompute exp_G_R for independent pixels for fast update
    # When flipping pixel (i,j), we update rho_G by +/- exp_G_R[:, p_idx] / 1024
    # We need to know the sign (adding or removing)
    
    while time.time() < deadline:
        # Pick random independent pixel
        idx = np.random.randint(N_indep)
        i, j = indep_indices[idx]
        p_idx = i * size + j
        
        # Determine if we are adding or removing
        if cell[i][j] == 1:
            # Remove
            rho_G -= exp_G_R[:, p_idx] / 1024.0
            cell[i][j] = 0
            # Update symmetric pixels in cell (for consistency, though rho_G is updated)
            for pi, pj in indep_to_pixels[idx]:
                cell[pi][pj] = 0
            
            new_score = compute_bands(rho_G, k_points[fast_k_indices])
            
            if new_score > current_score:
                current_score = new_score
                if new_score > best_score:
                    # Verify with full k-points
                    full_score = compute_bands(rho_G, k_points)
                    if full_score > best_score:
                        best_score = full_score
                        best_cell = copy.deepcopy(cell)
            else:
                # Revert
                cell[i][j] = 1
                rho_G += exp_G_R[:, p_idx] / 1024.0
                for pi, pj in indep_to_pixels[idx]:
                    cell[pi][pj] = 1
        else:
            # Add
            rho_G += exp_G_R[:, p_idx] / 1024.0
            cell[i][j] = 1
            for pi, pj in indep_to_pixels[idx]:
                cell[pi][pj] = 1
            
            new_score = compute_bands(rho_G, k_points[fast_k_indices])
            
            if new_score > current_score:
                current_score = new_score
                if new_score > best_score:
                    full_score = compute_bands(rho_G, k_points)
                    if full_score > best_score:
                        best_score = full_score
                        best_cell = copy.deepcopy(cell)
            else:
                # Revert
                cell[i][j] = 0
                rho_G -= exp_G_R[:, p_idx] / 1024.0
                for pi, pj in indep_to_pixels[idx]:
                    cell[pi][pj] = 0

    return best_cell
