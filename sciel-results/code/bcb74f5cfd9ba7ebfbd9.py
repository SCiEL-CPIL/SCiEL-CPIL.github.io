"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

# EVOLVE-BLOCK-START
def _symmetrize(cell):
    """
    Enforce C4v symmetry (invariant under 90 deg rotation and reflection).
    The problem states: cell == cell[::-1] row-wise and column-wise and cell == its transpose.
    This implies symmetry about the center.
    """
    n = len(cell)
    # Convert to numpy for speed
    c = np.array(cell, dtype=int)
    
    # Apply symmetries
    # 1. Reflection about vertical axis: c[i, j] == c[i, n-1-j]
    # 2. Reflection about horizontal axis: c[i, j] == c[n-1-i, j]
    # 3. Transpose: c[i, j] == c[j, i]
    
    # We can just average or take the max/min? 
    # The problem says "invariant", so if we generate it, it should be.
    # If we modify pixels, we must ensure consistency.
    # Let's define the fundamental domain: i <= n//2, j <= n//2.
    # For n=32, center is between 15 and 16.
    # Indices 0..15 map to 31..16.
    # Symmetry:
    # c[i, j] = c[n-1-i, j] = c[i, n-1-j] = c[n-1-i, n-1-j]
    # Also c[i, j] = c[j, i]
    
    # To enforce, we iterate over the fundamental domain (top-left quadrant including diagonals/axes)
    # and copy to the other 3 quadrants.
    
    c_out = np.zeros_like(c)
    
    for i in range(n // 2):
        for j in range(n // 2):
            val = c[i, j]
            # Top-Left
            c_out[i, j] = val
            # Top-Right
            c_out[i, n-1-j] = val
            # Bottom-Left
            c_out[n-1-i, j] = val
            # Bottom-Right
            c_out[n-1-i, n-1-j] = val
            
            # Transpose symmetry: c[i, j] == c[j, i]
            # This links (i, j) in TL to (j, i) in TL.
            # So we must ensure c[i, j] == c[j, i].
            # We can enforce this by only defining values for i <= j and copying.
            
    # The above loop doesn't handle the transpose constraint between (i,j) and (j,i) if i!=j.
    # Let's do it properly:
    # Define values for i in [0, n//2), j in [i, n//2).
    # Then:
    # c[i, j] = v
    # c[j, i] = v  (Transpose)
    # c[n-1-i, n-1-j] = v (180 rot)
    # c[n-1-j, n-1-i] = v (180 rot of transpose)
    # c[i, n-1-j] = v (Ref Y)
    # c[n-1-i, j] = v (Ref X)
    # c[j, n-1-i] = v (Ref Y of transpose)
    # c[n-1-j, i] = v (Ref X of transpose)
    
    c_out = np.zeros((n, n), dtype=int)
    for i in range(n // 2):
        for j in range(i, n // 2):
            val = c[i, j]
            # 8 symmetric positions
            c_out[i, j] = val
            c_out[j, i] = val
            c_out[i, n-1-j] = val
            c_out[n-1-j, i] = val
            c_out[n-1-i, n-1-j] = val
            c_out[n-1-j, n-1-i] = val
            c_out[n-1-i, j] = val
            c_out[j, n-1-i] = val
            
    # Handle the middle rows/cols if n is even?
    # n=32 is even.
    # i ranges 0..15.
    # j ranges i..15.
    # This covers the top-left 16x16 block.
    # The mappings cover the whole 32x32 grid.
    
    return c_out.tolist()

def _fast_tm_bands(cell, n_pw=21, k_points=None):
    """
    Approximate TM band solver using plane waves.
    Returns gap ratio score.
    """
    size = len(cell)
    eps = np.array(cell, dtype=float)
    # 1 = dielectric (12), 0 = air (1)
    eps[eps == 1] = 12.0
    eps[eps == 0] = 1.0
    
    # Compute Fourier coefficients of 1/eps
    inv_eps = 1.0 / eps
    inv_eps_fft = np.fft.fft2(inv_eps)
    inv_eps_fft /= (size * size)
    
    if k_points is None:
        # Sample Gamma, M, X, and interior points
        # Irreducible BZ for square lattice: kx, ky >= 0, kx + ky <= 0.5 (if normalized to 1)
        # Actually, standard square BZ is [-0.5, 0.5] x [-0.5, 0.5].
        # Irreducible sector: kx >= ky >= 0, kx + ky <= 0.5.
        # Let's sample points in this region.
        k_points = []
        # Gamma
        k_points.append((0.0, 0.0))
        # X (0.5, 0)
        k_points.append((0.5, 0.0))
        # M (0.5, 0.5) - wait, M is (0.5, 0.5) in the full BZ?
        # Square BZ boundary: |kx|<=0.5, |ky|<=0.5.
        # M is (0.5, 0.5). X is (0.5, 0).
        # The irreducible triangle is Gamma(0,0) to X(0.5,0) to M(0.5,0.5)?
        # No, the square lattice BZ is a square. The irreducible region is 1/8 of it.
        # Vertices: (0,0), (0.5, 0), (0.5, 0.5) is not in the same triangle if we use kx>=ky.
        # The line kx=ky goes from (0,0) to (0.5, 0.5).
        # The line kx+ky=0.5 goes from (0.5, 0) to (0, 0.5).
        # So the irreducible sector is bounded by kx=0, ky=kx, and kx+ky=0.5?
        # No, for a square lattice, the reciprocal lattice is square.
        # The Wigner-Seitz cell is a square.
        # The irreducible BZ is 1/8 of the square.
        # Triangle with vertices (0,0), (0.5, 0), (0.5, 0.5) is 1/2 of the square? No.
        # (0,0) to (0.5, 0) to (0.5, 0.5) to (0, 0.5) is the quadrant.
        # Split by kx=ky.
        # So vertices: (0,0), (0.5, 0), (0.5, 0.5) is wrong.
        # The line ky=kx intersects the boundary kx=0.5 at (0.5, 0.5).
        # So the triangle is (0,0), (0.5, 0), (0.5, 0.5).
        # Wait, is (0.5, 0.5) in the BZ? Yes, it's the corner M.
        # Is the region kx >= ky, kx <= 0.5, ky <= 0.5 the irreducible sector?
        # Yes, if we also consider the reflection kx<->ky.
        # So we sample kx in [0, 0.5], ky in [0, kx].
        
        for kx in [0.0, 0.25, 0.5]:
            for ky in [0.0, kx/2, kx]:
                if ky <= 0.5:
                    k_points.append((kx, ky))
        
        # Add some interior points
        k_points.append((0.3, 0.1))
        k_points.append((0.4, 0.2))
        
    min_omega_5 = float('inf')
    max_omega_4 = 0.0
    
    half = n_pw // 2
    indices = np.arange(-half, half + 1)
    Gx, Gy = np.meshgrid(indices, indices, indexing='ij')
    G_vecs = np.stack([Gx.flatten(), Gy.flatten()], axis=1)
    
    for kx, ky in k_points:
        K = np.array([kx, ky])
        K_plus_G = G_vecs + K
        dots = np.sum(K_plus_G[:, np.newaxis, :] * K_plus_G[np.newaxis, :, :], axis=2)
        
        # Get inverse epsilon coefficients
        Gx_2d, Gy_2d = Gx, Gy
        Dx = Gx_2d[:, None] - Gx_2d[None, :]
        Dy = Gy_2d[:, None] - Gy_2d[None, :]
        
        idx_x = Dx % size
        idx_y = Dy % size
        coeffs = inv_eps_fft[idx_y, idx_x]
        
        H = dots * coeffs
        evals = np.linalg.eigvalsh(H)
        
        # Filter out zero/negative eigenvalues (numerical noise)
        # The first few eigenvalues correspond to low frequency bands.
        # We need the 4th and 5th bands.
        # In a 32x32 cell with 17x17 PWs, we have 289 modes.
        # The bands are indexed 0, 1, 2, 3, 4...
        # We take index 3 and 4.
        
        # Ensure we have enough eigenvalues
        if len(evals) < 5:
            continue
            
        omega_4_sq = evals[3]
        omega_5_sq = evals[4]
        
        if omega_4_sq < 0 or omega_5_sq < 0:
            continue
            
        omega_4 = np.sqrt(omega_4_sq)
        omega_5 = np.sqrt(omega_5_sq)
        
        if omega_5 < min_omega_5:
            min_omega_5 = omega_5
        if omega_4 > max_omega_4:
            max_omega_4 = omega_4
            
    if min_omega_5 == float('inf') or max_omega_4 == 0.0:
        return -1.0
        
    score = 2 * (min_omega_5 - max_omega_4) / (min_omega_5 + max_omega_4)
    return score

def _random_symmetric_cell(size, density=0.5):
    """Generate a random symmetric cell."""
    cell = np.zeros((size, size), dtype=int)
    # Define fundamental domain: i in [0, size//2), j in [i, size//2)
    for i in range(size // 2):
        for j in range(i, size // 2):
            val = 1 if random.random() < density else 0
            # Set 8 symmetric positions
            cell[i, j] = val
            cell[j, i] = val
            cell[i, size-1-j] = val
            cell[size-1-j, i] = val
            cell[size-1-i, size-1-j] = val
            cell[size-1-j, size-1-i] = val
            cell[size-1-i, j] = val
            cell[j, size-1-i] = val
    return cell.tolist()

def _local_search(cell, size, start_time, deadline, max_iters=100):
    """
    Hill climbing: flip one pixel (and its symmetric partners) if it improves the score.
    """
    current_cell = [row[:] for row in cell]
    current_score = _fast_tm_bands(current_cell, n_pw=17) # Use smaller basis for speed in loop
    
    if current_score < 0:
        current_score = -1.0

    # Precompute symmetric partners for each pixel in fundamental domain
    partners = {}
    for i in range(size // 2):
        for j in range(i, size // 2):
            p = [
                (i, j), (j, i),
                (i, size-1-j), (size-1-j, i),
                (size-1-i, size-1-j), (size-1-j, size-1-i),
                (size-1-i, j), (j, size-1-i)
            ]
            # Remove duplicates
            p = list(set(p))
            partners[(i, j)] = p
            
    keys = list(partners.keys())
    
    for _ in range(max_iters):
        if time.time() > deadline:
            break
            
        best_neighbor_score = current_score
        best_neighbor_cell = None
        best_neighbor_key = None
        
        # Try flipping a random subset of keys to save time
        random.shuffle(keys)
        for key in keys[:20]: # Try 20 random moves
            p = partners[key]
            # Flip
            for (ri, cj) in p:
                current_cell[ri][cj] = 1 - current_cell[ri][cj]
                
            new_score = _fast_tm_bands(current_cell, n_pw=17)
            
            # Unflip
            for (ri, cj) in p:
                current_cell[ri][cj] = 1 - current_cell[ri][cj]
                
            if new_score > best_neighbor_score:
                best_neighbor_score = new_score
                best_neighbor_cell = [row[:] for row in current_cell] # Save the flipped state? No, we unflipped.
                # We need to re-apply the flip to the best cell later.
                best_neighbor_key = key
                
        if best_neighbor_cell is not None:
            # Apply the best move
            p = partners[best_neighbor_key]
            for (ri, cj) in p:
                current_cell[ri][cj] = 1 - current_cell[ri][cj]
            current_score = best_neighbor_score
            
    return current_cell, current_score

def design(problem):
    """
    Search for the best shape among a set of candidates.
    """
    start_time = time.time()
    deadline = start_time + 110 # 110s budget, leaving 10s margin for overhead/return
    size = problem["pixels"]
    
    best_cell = None
    best_score = -float('inf')
    
    # 1. Try a few parametric shapes first (fast)
    # Circle hole
    cell = np.zeros((size, size), dtype=int)
    cx, cy = np.meshgrid(np.arange(size), np.arange(size), indexing='ij')
    dx = cx - (size/2 - 0.5)
    dy = cy - (size/2 - 0.5)
    dist = np.sqrt(dx**2 + dy**2)
    
    for r in [5, 8, 10, 12]:
        cell[:] = 1
        cell[dist < r] = 0
        cell_list = cell.tolist()
        cell_list = _symmetrize(cell_list) # Ensure perfect symmetry
        score = _fast_tm_bands(cell_list, n_pw=17)
        if score > best_score:
            best_score = score
            best_cell = cell_list
            
    # 2. Random search with local optimization
    for _ in range(5): # 5 random starts
        if time.time() > deadline:
            break
            
        # Random density
        density = random.uniform(0.3, 0.7)
        cell = _random_symmetric_cell(size, density)
        
        # Local search
        cell, score = _local_search(cell, size, start_time, deadline, max_iters=50)
        
        if score > best_score:
            best_score = score
            best_cell = cell
            
    # 3. If still no good, try a checkerboard-like pattern
    if best_score < 0.04: # Seed is ~0.043
        cell = np.zeros((size, size), dtype=int)
        for i in range(size):
            for j in range(size):
                if (i + j) % 2 == 0:
                    cell[i, j] = 1
        cell_list = cell.tolist()
        cell_list = _symmetrize(cell_list)
        score = _fast_tm_bands(cell_list, n_pw=17)
        if score > best_score:
            best_score = score
            best_cell = cell_list

    if best_cell is None:
        # Fallback: All air (should give negative score, but valid structure)
        best_cell = [[0 for _ in range(size)] for _ in range(size)]
        
    # Final check and symmetrization
    best_cell = _symmetrize(best_cell)
    
    return best_cell
# EVOLVE-BLOCK-END
