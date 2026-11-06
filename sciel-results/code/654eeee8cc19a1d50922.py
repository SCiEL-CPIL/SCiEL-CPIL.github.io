import numpy as np
import time
import itertools

def design(problem):
    """
    Design a symmetric 32x32-pixel unit cell to maximize the TM band gap
    between the 4th and 5th bands.
    """
    size = problem["pixels"] # 32
    eps_low = problem.get("eps_low", 1)
    eps_high = problem.get("eps_high", 12)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin for safety
    
    # 1. Define the 4x4 block structure
    # The 32x32 cell is divided into 4x4 blocks of 8x8 pixels.
    # Symmetry constraints:
    # B00 == B03 == B30 == B33
    # B01 == B02 == B31 == B32
    # B10 == B20 == B13 == B23
    # B11 == B12 == B21 == B22
    
    # We need to define 4 unique 8x8 blocks: B00, B01, B10, B11
    # Each block must be symmetric under 180-degree rotation to ensure the whole cell is symmetric?
    # Actually, the whole cell symmetry is about the center (16,16).
    # If we mirror the blocks correctly, the global symmetry is satisfied.
    # However, for the blocks themselves to be compatible with the global symmetry, 
    # B00 (top-left) must be the 180-degree rotation of B33 (bottom-right). 
    # Since B00 == B33, B00 must be 180-degree symmetric.
    # Similarly, B01 (top-mid-left) is mirrored to B02 (top-mid-right). 
    # And B10 (left-mid-top) is mirrored to B13 (left-mid-bottom).
    # B11 (center) must be 180-degree symmetric.
    
    # Let's generate a library of candidate 8x8 blocks.
    # To ensure global symmetry, we will generate blocks that are 180-degree symmetric.
    # A block is 180-degree symmetric if block[i][j] == block[7-i][7-j].
    
    def generate_block_library():
        blocks = []
        # We iterate over the first 32 pixels of an 8x8 block (half of 64)
        # Actually, 180-degree symmetry means (i,j) determines (7-i, 7-j).
        # There are 64/2 = 32 independent pixels.
        # 2^32 is too large to enumerate.
        # Instead, we use parametric shapes for the 8x8 blocks.
        
        # Shape 1: Empty
        blocks.append(np.zeros((8, 8), dtype=int))
        
        # Shape 2: Full
        blocks.append(np.ones((8, 8), dtype=int))
        
        # Shape 3: Circle centered at (3.5, 3.5) with various radii
        for r in [1.0, 1.5, 2.0, 2.5, 3.0]:
            block = np.zeros((8, 8), dtype=int)
            for i in range(8):
                for j in range(8):
                    # Center of pixel is (i+0.5, j+0.5)
                    # Distance from center (3.5, 3.5)
                    dx = (i + 0.5) - 3.5
                    dy = (j + 0.5) - 3.5
                    if dx*dx + dy*dy <= r*r:
                        block[i, j] = 1
            blocks.append(block)
            
        # Shape 4: Square centered at (3.5, 3.5) with various sizes
        for s in [1.0, 1.5, 2.0, 2.5, 3.0]:
            block = np.zeros((8, 8), dtype=int)
            for i in range(8):
                for j in range(8):
                    dx = (i + 0.5) - 3.5
                    dy = (j + 0.5) - 3.5
                    if abs(dx) <= s and abs(dy) <= s:
                        block[i, j] = 1
            blocks.append(block)
            
        # Shape 5: Cross (plus sign)
        for w in [1.0, 1.5, 2.0]:
            block = np.zeros((8, 8), dtype=int)
            for i in range(8):
                for j in range(8):
                    dx = (i + 0.5) - 3.5
                    dy = (j + 0.5) - 3.5
                    if abs(dx) <= w or abs(dy) <= w:
                        block[i, j] = 1
            blocks.append(block)
            
        # Shape 6: X shape
        for w in [1.0, 1.5, 2.0]:
            block = np.zeros((8, 8), dtype=int)
            for i in range(8):
                for j in range(8):
                    dx = (i + 0.5) - 3.5
                    dy = (j + 0.5) - 3.5
                    if abs(dx - dy) <= w or abs(dx + dy) <= w:
                        block[i, j] = 1
            blocks.append(block)

        return blocks

    block_library = generate_block_library()
    
    # 2. Fast PWE Solver
    # We use a reduced basis of plane waves for speed.
    # N_pw = 9 (9x9 = 81 PWs) is a good balance.
    N_pw = 9
    G_list = [(n, m) for n in range(-N_pw // 2, N_pw // 2 + 1) 
              for m in range(-N_pw // 2, N_pw // 2 + 1)]
    Gx = np.array([g[0] for g in G_list])
    Gy = np.array([g[1] for g in G_list])
    N_G = len(G_list)
    
    # Precompute pixel positions
    i_idx, j_idx = np.indices((size, size))
    x_pix = (i_idx + 0.5) / size
    y_pix = (j_idx + 0.5) / size
    x_flat = x_pix.flatten()
    y_flat = y_pix.flatten()
    
    # Key points in the irreducible Brillouin Zone
    # Gamma (0,0), M (0.5, 0.5), X (0.5, 0)
    # The gap is often determined by the minimum of band 5 and maximum of band 4.
    # We sample a few points.
    k_points = [
        (0.0, 0.0), # Gamma
        (0.5, 0.5), # M
        (0.5, 0.0), # X
        (0.25, 0.25), # Mid Gamma-M
        (0.25, 0.0), # Mid Gamma-X
    ]
    
    def compute_gap(cell):
        """
        Compute the gap between band 4 and 5 for the given cell.
        cell is a 32x32 numpy array of 0s and 1s.
        """
        # Calculate 1/eps for each pixel
        eps_pix = np.where(cell == 1, eps_high, eps_low)
        inv_eps_pix = 1.0 / eps_pix
        inv_eps_flat = inv_eps_pix.flatten()
        
        # Compute Fourier coefficients of 1/eps
        # eps_inv_G = (1/A^2) * sum_{pix} inv_eps(pix) * exp(-i G . r_pix)
        phases = Gx[:, None] * x_flat[None, :] + Gy[:, None] * y_flat[None, :]
        eps_inv_G = np.sum(inv_eps_flat[None, :] * np.exp(-1j * phases), axis=1) / (size * size)
        
        # Map G vectors to indices
        g_map = {(g[0], g[1]): i for i, g in enumerate(G_list)}
        
        min_gap = 100.0
        
        # For each k point, solve the eigenvalue problem
        for kx, ky in k_points:
            kg_sq = (kx + Gx)**2 + (ky + Gy)**2
            
            # Build Matrix M
            # M[i, j] = eps_inv(G_i - G_j) * kg_sq[j]
            # We need to construct this matrix efficiently.
            # Since G_list is a regular grid, we can use broadcasting if we had a 2D array of eps_inv_G.
            # But eps_inv_G is a 1D array indexed by (n,m).
            # Let's create a 2D lookup table for eps_inv_G based on differences.
            # diff_n = Gx[i] - Gx[j]
            # diff_m = Gy[i] - Gy[j]
            
            # Precompute diff indices
            # diff_n range: -8..8
            # diff_m range: -8..8
            # G_list covers -4..4.
            # So for differences outside -4..4, we don't have the coefficient directly in eps_inv_G?
            # Wait, eps_inv_G is computed for G vectors in G_list (-4..4).
            # The convolution requires eps_inv(G_i - G_j).
            # If G_i - G_j is not in G_list, we don't have that coefficient.
            # This is a truncation error. To be more accurate, we should compute eps_inv_G for a larger range.
            # Let's re-compute eps_inv_G for a larger range of G vectors, say -8..8.
            
            # Recompute eps_inv_G for larger range
            N_pw_large = 17 # -8..8
            G_list_large = [(n, m) for n in range(-N_pw_large // 2, N_pw_large // 2 + 1) 
                            for m in range(-N_pw_large // 2, N_pw_large // 2 + 1)]
            Gx_l = np.array([g[0] for g in G_list_large])
            Gy_l = np.array([g[1] for g in G_list_large])
            
            phases_l = Gx_l[:, None] * x_flat[None, :] + Gy_l[:, None] * y_flat[None, :]
            eps_inv_G_l = np.sum(inv_eps_flat[None, :] * np.exp(-1j * phases_l), axis=1) / (size * size)
            g_map_l = {(g[0], g[1]): i for i, g in enumerate(G_list_large)}
            
            # Now build M using the larger coefficient set
            M = np.zeros((N_G, N_G), dtype=complex)
            for i in range(N_G):
                for j in range(N_G):
                    diff_n = Gx[i] - Gx[j]
                    diff_m = Gy[i] - Gy[j]
                    key = (diff_n, diff_m)
                    if key in g_map_l:
                        idx_diff = g_map_l[key]
                        M[i, j] = eps_inv_G_l[idx_diff] * kg_sq[j]
                    else:
                        # If diff is out of range of large set, it's very small, ignore
                        M[i, j] = 0
                        
            eigvals = np.linalg.eigvalsh(M)
            eigvals = np.sort(eigvals)
            
            if len(eigvals) >= 5:
                w4 = np.sqrt(max(0, eigvals[3]))
                w5 = np.sqrt(max(0, eigvals[4]))
                gap = w5 - w4
                if gap < min_gap:
                    min_gap = gap
                    
        return min_gap

    # 3. Optimization
    # We have 4 unique blocks: B00, B01, B10, B11.
    # We will try a grid search over the block library.
    # Library size is ~20. 20^4 = 160,000 combinations.
    # Each evaluation takes ~50ms (est). 160,000 * 0.05s = 8000s. Too slow.
    # We need to reduce the search space or use a faster evaluator.
    
    # Strategy:
    # 1. Pre-filter blocks based on fill fraction.
    # 2. Use a smaller subset of k-points for the search (e.g., only Gamma and M).
    # 3. Use a smaller N_pw for the search (e.g., 7x7=49 PWs).
    
    N_pw_search = 7
    G_list_search = [(n, m) for n in range(-N_pw_search // 2, N_pw_search // 2 + 1) 
                     for m in range(-N_pw_search // 2, N_pw_search // 2 + 1)]
    Gx_s = np.array([g[0] for g in G_list_search])
    Gy_s = np.array([g[1] for g in G_list_search])
    N_G_s = len(G_list_search)
    
    # Recompute eps_inv_G for search range
    N_pw_large_search = 13 # -6..6
    G_list_large_s = [(n, m) for n in range(-N_pw_large_search // 2, N_pw_large_search // 2 + 1) 
                      for m in range(-N_pw_large_search // 2, N_pw_large_search // 2 + 1)]
    Gx_ls = np.array([g[0] for g in G_list_large_s])
    Gy_ls = np.array([g[1] for g in G_list_large_s])
    
    # We need eps_inv_G_l for each candidate cell.
    # This is the bottleneck.
    
    # Let's try a greedy approach or a limited random search.
    # We will select a subset of blocks to test.
    
    # Filter blocks by fill fraction
    block_fills = [np.mean(b) for b in block_library]
    # Keep blocks with fill fraction between 0.2 and 0.8
    candidate_blocks = [b for b, f in zip(block_library, block_fills) if 0.2 <= f <= 0.8]
    
    if not candidate_blocks:
        candidate_blocks = block_library
        
    # If still too many, sample
    if len(candidate_blocks) > 10:
        # Pick 10 diverse blocks
        # Sort by fill fraction and pick evenly
        indices = np.linspace(0, len(candidate_blocks)-1, 10).astype(int)
        candidate_blocks = [candidate_blocks[i] for i in indices]
        
    best_cell = None
    best_gap = -1
    
    # Random search over combinations of 4 blocks
    num_trials = 200
    for t in range(num_trials):
        if time.time() > deadline:
            break
            
        # Pick 4 random blocks from candidates
        b00 = np.random.choice(candidate_blocks)
        b01 = np.random.choice(candidate_blocks)
        b10 = np.random.choice(candidate_blocks)
        b11 = np.random.choice(candidate_blocks)
        
        # Construct 32x32 cell
        cell = np.zeros((size, size), dtype=int)
        
        # Place blocks
        # B00 at (0,0), (0,24), (24,0), (24,24)
        cell[0:8, 0:8] = b00
        cell[0:8, 24:32] = b00
        cell[24:32, 0:8] = b00
        cell[24:32, 24:32] = b00
        
        # B01 at (0,8), (0,16), (24,8), (24,16)
        cell[0:8, 8:16] = b01
        cell[0:8, 16:24] = b01
        cell[24:32, 8:16] = b01
        cell[24:32, 16:24] = b01
        
        # B10 at (8,0), (16,0), (8,24), (16,24)
        cell[8:16, 0:8] = b10
        cell[16:24, 0:8] = b10
        cell[8:16, 24:32] = b10
        cell[16:24, 24:32] = b10
        
        # B11 at (8,8), (8,16), (16,8), (16,16)
        cell[8:16, 8:16] = b11
        cell[8:16, 16:24] = b11
        cell[16:24, 8:16] = b11
        cell[16:24, 16:24] = b11
        
        # Evaluate gap using fast solver
        # We need to compute eps_inv_G for this cell
        eps_pix = np.where(cell == 1, eps_high, eps_low)
        inv_eps_pix = 1.0 / eps_pix
        inv_eps_flat = inv_eps_pix.flatten()
        
        # Compute Fourier coefficients for search range
        phases_s = Gx_ls[:, None] * x_flat[None, :] + Gy_ls[:, None] * y_flat[None, :]
        eps_inv_G_s = np.sum(inv_eps_flat[None, :] * np.exp(-1j * phases_s), axis=1) / (size * size)
        g_map_s = {(g[0], g[1]): i for i, g in enumerate(G_list_large_s)}
        
        min_gap = 100.0
        # Evaluate at Gamma and M only for speed
        for kx, ky in [(0.0, 0.0), (0.5, 0.5)]:
            kg_sq = (kx + Gx_s)**2 + (ky + Gy_s)**2
            
            M = np.zeros((N_G_s, N_G_s), dtype=complex)
            for i in range(N_G_s):
                for j in range(N_G_s):
                    diff_n = Gx_s[i] - Gx_s[j]
                    diff_m = Gy_s[i] - Gy_s[j]
                    key = (diff_n, diff_m)
                    if key in g_map_s:
                        idx_diff = g_map_s[key]
                        M[i, j] = eps_inv_G_s[idx_diff] * kg_sq[j]
                        
            eigvals = np.linalg.eigvalsh(M)
            eigvals = np.sort(eigvals)
            
            if len(eigvals) >= 5:
                w4 = np.sqrt(max(0, eigvals[3]))
                w5 = np.sqrt(max(0, eigvals[4]))
                gap = w5 - w4
                if gap < min_gap:
                    min_gap = gap
                    
        if min_gap > best_gap:
            best_gap = min_gap
            best_cell = cell.copy()
            
    # If we found a good cell, refine it with a more accurate solver?
    # Or just return it.
    
    if best_cell is None:
        # Fallback: Simple circle
        cell = np.zeros((size, size), dtype=int)
        for i in range(size):
            x = (i + 0.5) / size - 0.5
            for j in range(size):
                y = (j + 0.5) / size - 0.5
                if x*x + y*y < 0.2**2:
                    cell[i, j] = 1
        best_cell = cell

    return best_cell.tolist()
