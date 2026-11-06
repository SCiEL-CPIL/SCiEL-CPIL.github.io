import numpy as np
import time

def design(problem):
    """
    Optimizes a 32x32 pixel unit cell to maximize the 4-5 TM band gap.
    Uses D4 symmetry and local search.
    """
    size = problem["pixels"] # 32
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)
    
    start_time = time.time()
    deadline = start_time + 115.0 # 15s margin

    # 1. Setup G-vectors (289 plane waves)
    # The evaluator uses 289 plane waves. We assume they are the lowest 289 G vectors.
    # Indices -16 to 16 would be 33x33=1089, so it's a subset.
    # Usually, it's a circle of G vectors with |G| <= G_max.
    # Let's generate G vectors and sort by magnitude.
    g_range = np.arange(-20, 21)
    gx, gy = np.meshgrid(g_range, g_range, indexing='ij')
    g_mags = gx**2 + gy**2
    flat_gx = gx.flatten()
    flat_gy = gy.flatten()
    flat_mags = g_mags.flatten()
    sorted_indices = np.argsort(flat_mags)
    n_pw = 289
    basis_idx = sorted_indices[:n_pw]
    Gx = flat_gx[basis_idx]
    Gy = flat_gy[basis_idx]
    N = n_pw

    # 2. Setup K-points
    # 30 points in irreducible BZ.
    # We will use a representative subset for local search: Gamma (0,0), X (1,0), M (1,1)
    # and maybe some intermediate points.
    # For the final score, we must use all 30.
    # Let's define the 30 points as in the seed, but we can also just use the high symmetry points
    # if the gap is determined there. The seed used a grid.
    
    # Let's use the grid from the seed for full evaluation.
    k_points_full = []
    for i in range(6):
        kx = i / 5.0
        for j in range(5):
            ky = kx * (j / 4.0)
            k_points_full.append((kx * np.pi, ky * np.pi))
            
    # Subset for local search: Gamma, X, M
    k_points_fast = [
        (0.0, 0.0),
        (np.pi, 0.0),
        (np.pi, np.pi)
    ]

    # Pre-compute k+G vectors for fast evaluation
    # kGx_fast[i, j] = kx_i + Gx_j
    kGx_fast = np.array([[kx + Gx for kx, ky in k_points_fast]])
    kGy_fast = np.array([[ky + Gy for kx, ky in k_points_fast]])
    
    kGx_full = np.array([[kx + Gx for kx, ky in k_points_full]])
    kGy_full = np.array([[ky + Gy for kx, ky in k_points_full]])

    # Pre-compute G difference indices for eta lookup
    # We need eta(Gi - Gj)
    # Gx_i - Gx_j, Gy_i - Gy_j
    # We can compute the FFT of the eta field and index it.
    # eta_fft is 64x64.
    # Index ix = (Gx_i - Gx_j + 32) % 64
    # Index iy = (Gy_i - Gy_j + 32) % 64
    
    # Pre-compute the index matrices for the differences
    # This is 289x289.
    Gx_diff = Gx[:, None] - Gx[None, :]
    Gy_diff = Gy[:, None] - Gy[None, :]
    ix_diff = (Gx_diff + 32) % 64
    iy_diff = (Gy_diff + 32) % 64

    def evaluate_cell(cell, use_fast_k=False):
        """
        Evaluates the band gap for a given 32x32 cell.
        Returns the gap-midgap ratio.
        """
        if time.time() > deadline:
            return -1.0
            
        # Compute eta pixel map
        # eta = 1/eps
        # eps = 12 if cell==1, 1 if cell==0
        eta_pixel = np.ones((size, size), dtype=np.float64)
        eta_pixel[cell == 1] = 1.0 / eps_high
        
        # FFT
        eta_padded = np.zeros((64, 64), dtype=np.float64)
        eta_padded[:size, :size] = eta_pixel
        eta_fft = np.fft.fft2(eta_padded)
        eta_fft = np.fft.fftshift(eta_fft)
        
        # Select k points
        if use_fast_k:
            kGx = kGx_fast
            kGy = kGy_fast
            n_k = len(k_points_fast)
        else:
            kGx = kGx_full
            kGy = kGy_full
            n_k = len(k_points_full)
            
        min_w5 = np.inf
        max_w4 = -np.inf
        
        for k in range(n_k):
            kx = kGx[k]
            ky = kGy[k]
            
            # Construct Matrix A
            # A_ij = eta(Gi - Gj) * (kx_i * kx_j + ky_i * ky_j)
            
            # Get eta matrix
            # eta_fft is indexed by [iy, ix]
            eta_matrix = eta_fft[iy_diff, ix_diff]
            
            # Dot products
            dot_x = kx[:, None] * kx[None, :]
            dot_y = ky[:, None] * ky[None, :]
            dot_total = dot_x + dot_y
            
            A = eta_matrix * dot_total
            
            # Solve eigenvalues
            # A is Hermitian
            try:
                eigs = np.linalg.eigvalsh(A)
            except:
                return -1.0
                
            # Get 4th and 5th bands (indices 3 and 4)
            w4 = np.sqrt(eigs[3])
            w5 = np.sqrt(eigs[4])
            
            if w5 < min_w5:
                min_w5 = w5
            if w4 > max_w4:
                max_w4 = w4
                
        if min_w5 == np.inf or max_w4 == -np.inf:
            return -1.0
            
        gap = min_w5 - max_w4
        if gap <= 0:
            return -1.0
        else:
            return 2 * gap / (min_w5 + max_w4)

    def make_cross_cell(arm_width, arm_length, center_size):
        """
        Creates a D4 symmetric cross shape.
        """
        cell = np.zeros((size, size), dtype=np.int8)
        # Work in the top-right quadrant (16x16)
        # Center of cell is (16, 16) in 32x32?
        # Pixel centers are 0.5, 1.5, ..., 31.5.
        # Symmetry center is between 15.5 and 16.5?
        # The problem says "invariant under square symmetry group about the cell centre".
        # For a 32x32 grid, the center is at (16, 16) in continuous coordinates?
        # Pixels 0..31. Center is 15.5?
        # Let's check the symmetry condition: cell == cell[::-1] and cell == cell.T
        # This implies symmetry about the center of the grid.
        # For even N, the center is between pixels N/2-1 and N/2.
        # So we can define the top-left 16x16 and mirror it.
        
        # Let's define the quadrant 0..15 (16x16)
        # y: 0..15, x: 0..15
        # The cross is centered at (16, 16) in the full grid?
        # If we mirror 0..15 to 16..31, the seam is at 15.5/16.5.
        
        # Let's define the shape in the full grid directly to be safe, using D4 symmetry.
        # y in 0..31, x in 0..31.
        # Center is 15.5, 15.5.
        # Distance from center: dy = |y - 15.5|, dx = |x - 15.5|
        
        ys, xs = np.mgrid[0:size, 0:size]
        dx = np.abs(xs - 15.5)
        dy = np.abs(ys - 15.5)
        
        # Cross:
        # Horizontal bar: dy < arm_width/2 AND dx < arm_length/2
        # Vertical bar: dx < arm_width/2 AND dy < arm_length/2
        # Union of the two.
        
        # Ensure it fits in the quadrant
        max_range = 16.0
        
        h_bar = (dy < arm_width/2.0) & (dx < arm_length/2.0)
        v_bar = (dx < arm_width/2.0) & (dy < arm_length/2.0)
        
        cell = (h_bar | v_bar).astype(np.int8)
        
        # Clip to size
        return cell

    def make_rounded_cross_cell(arm_width, arm_length, corner_radius):
        """
        Cross with rounded outer corners.
        """
        cell = np.zeros((size, size), dtype=np.int8)
        ys, xs = np.mgrid[0:size, 0:size]
        dx = np.abs(xs - 15.5)
        dy = np.abs(ys - 15.5)
        
        h_bar = (dy < arm_width/2.0) & (dx < arm_length/2.0)
        v_bar = (dx < arm_width/2.0) & (dy < arm_length/2.0)
        
        # Corner cuts:
        # Top-right corner of the horizontal bar:
        # Center of cut is at (arm_length/2 - r, arm_width/2 - r) relative to center?
        # Let's just use a distance check for the "end caps".
        # End of H bar: x = +/- arm_length/2. y in [-arm_width/2, arm_width/2].
        # We want to round the tips.
        # Tip is at (arm_length/2, y).
        # Distance from (arm_length/2 - r, 0) should be < r?
        
        # Simpler: Just use the union and then apply a "rounding" by checking distance to the ideal cross boundary?
        # Or just stick to sharp crosses for now, they are often optimal.
        
        return (h_bar | v_bar).astype(np.int8)

    # Initial candidates
    candidates = []
    
    # Sharp Crosses
    for aw in [2, 3, 4, 5, 6]:
        for al in [6, 8, 10, 12, 14]:
            if al <= aw: continue
            candidates.append(make_cross_cell(aw, al, 0))
            
    # Circles (from seed)
    for r in np.arange(3.0, 7.5, 0.5):
        cell = np.zeros((size, size), dtype=np.int8)
        ys, xs = np.mgrid[0:size, 0:size]
        # 2x2 array of circles
        centers = [(7.5, 7.5), (7.5, 23.5), (23.5, 7.5), (23.5, 23.5)]
        for cx, cy in centers:
            dist2 = (xs - cx)**2 + (ys - cy)**2
            cell[dist2 < r**2] = 1
        candidates.append(cell)

    best_score = -np.inf
    best_cell = None

    # Evaluate initial candidates with full k-points
    for cell in candidates:
        if time.time() > deadline - 30: # Leave time for local search
            break
        score = evaluate_cell(cell, use_fast_k=False)
        if score > best_score:
            best_score = score
            best_cell = cell.copy()
            
    if best_cell is None:
        # Fallback
        cell = np.zeros((size, size), dtype=np.int8)
        cell[10:22, 10:22] = 1
        return cell.tolist()

    # Local Search
    # Use fast k-points for guidance
    # Flip pixels in the top-right quadrant (0..15, 0..15)
    # Due to D4 symmetry, flipping one pixel flips 4 (or 1 if on axis).
    # We iterate over the quadrant.
    
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        # Iterate over all pixels in the quadrant
        for y in range(16):
            for x in range(16):
                if time.time() > deadline - 10:
                    break
                
                # Try flipping this pixel
                # The pixel affects 4 positions:
                # (y, x), (y, 31-x), (31-y, x), (31-y, 31-x)
                # If x==y or x+y==31, it might be fewer, but for 32x32, 
                # x in 0..15, y in 0..15.
                # x != 31-x always (since x!=15.5)
                # y != 31-y always
                # So always 4 distinct pixels unless on the diagonal of the quadrant?
                # No, the symmetry is about the center (15.5, 15.5).
                # (y, x) -> (31-y, 31-x) is 180 rot.
                # (y, x) -> (y, 31-x) is x-reflection.
                # So yes, 4 distinct pixels for any (y,x) in 0..15.
                
                # Save old values
                old_vals = best_cell[y, x], best_cell[y, 31-x], best_cell[31-y, x], best_cell[31-y, 31-x]
                
                # Flip
                new_val = 1 - best_cell[y, x]
                best_cell[y, x] = new_val
                best_cell[y, 31-x] = new_val
                best_cell[31-y, x] = new_val
                best_cell[31-y, 31-x] = new_val
                
                # Evaluate with fast k-points
                new_score = evaluate_cell(best_cell, use_fast_k=True)
                
                if new_score > best_score: # Compare to current best fast score?
                    # We need to maintain a fast_score for the current best_cell
                    # Let's evaluate the current best_cell with fast k-points first to have a baseline
                    # This is expensive. 
                    # Better: Keep a `current_fast_score` variable.
                    # If new_score > current_fast_score, accept.
                    pass # Logic below
                
                # Restore if not accepted
                # We need a proper loop structure.
                
                # Let's restructure:
                # 1. Compute current_fast_score
                # 2. Try flip
                # 3. Compute new_fast_score
                # 4. If new > current, accept and update best_score (full) and current_fast_score
                # 5. Else restore
                
                # This requires computing current_fast_score once per iteration? No, it's the same cell.
                # So we can compute it once before the loop.
                
                # Break out of this flawed logic and rewrite the loop properly.
                break
            break # Break out of y loop to rewrite
    
    # Rewrite Local Search Properly
    current_fast_score = evaluate_cell(best_cell, use_fast_k=True)
    best_full_score = best_score
    
    while time.time() < deadline - 15:
        improved = False
        for y in range(16):
            if time.time() > deadline - 15:
                break
            for x in range(16):
                if time.time() > deadline - 15:
                    break
                
                # Flip
                new_val = 1 - best_cell[y, x]
                old_val = best_cell[y, x]
                
                # Apply flip
                best_cell[y, x] = new_val
                best_cell[y, 31-x] = new_val
                best_cell[31-y, x] = new_val
                best_cell[31-y, 31-x] = new_val
                
                new_fast_score = evaluate_cell(best_cell, use_fast_k=True)
                
                if new_fast_score > current_fast_score:
                    # Accept
                    current_fast_score = new_fast_score
                    # Update full score
                    full_score = evaluate_cell(best_cell, use_fast_k=False)
                    if full_score > best_full_score:
                        best_full_score = full_score
                    improved = True
                else:
                    # Revert
                    best_cell[y, x] = old_val
                    best_cell[y, 31-x] = old_val
                    best_cell[31-y, x] = old_val
                    best_cell[31-y, 31-x] = old_val
                    
    # Final check
    if best_full_score > best_score:
        best_score = best_full_score
        
    # If we still have time, do a few more random restarts?
    # Or just return the best found.
    
    return best_cell.tolist()

# Fallback if design fails or returns None
if __name__ == "__main__":
    # Not used in evaluator, but good for testing
    pass
