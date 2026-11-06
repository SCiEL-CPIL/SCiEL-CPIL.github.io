import numpy as np
import time
import sys

# Constants for the problem
SIZE = 32
EPS_LOW = 1.0
EPS_HIGH = 12.0
LOWER_BAND = 4
UPPER_BAND = 5

def compute_gap(cell, n_pw=13, k_points=None):
    """
    Computes the gap between the 4th and 5th TM bands using a simplified 
    plane-wave method.
    
    cell: 32x32 array of 0s and 1s.
    n_pw: Number of plane waves in x (and y) direction. Odd number for symmetry.
    k_points: List of (kx, ky) tuples.
    """
    if k_points is None:
        # Irreducible BZ points: Gamma (0,0), M (pi,0), X (pi/2, pi/2)
        # We sample a few points to get a robust estimate
        # Standard irreducible wedge for square lattice: kx >= 0, ky >= 0, kx >= ky
        # Points: Gamma(0,0), X(pi/2, 0) -> wait, BZ is [-pi, pi]^2 if lattice constant is 1?
        # Problem says "unit cell", usually L=1. G = 2*pi.
        # Irreducible BZ for square lattice: 0 <= ky <= kx, kx <= pi/2? No.
        # First BZ: -pi <= kx,ky <= pi.
        # Irreducible: 0 <= ky <= kx <= pi/2.
        # Key points: Gamma(0,0), X(pi/2, 0) is on boundary? No, X is (pi, 0) in some conventions?
        # If L=1, reciprocal lattice vectors are 2*pi. BZ boundaries at pi.
        # X point is usually (pi, 0). M point is (pi, pi). Gamma is (0,0).
        # Let's use a grid of points in the irreducible wedge.
        kx_vals = [0.0, np.pi/4, np.pi/2]
        ky_vals = [0.0, np.pi/4, np.pi/2]
        pts = []
        for kx in kx_vals:
            for ky in ky_vals:
                if ky <= kx:
                    pts.append((kx, ky))
        # Add M point (pi, pi) is outside 0..pi/2?
        # Wait, if L=1, G=2*pi. BZ is [-pi, pi].
        # Irreducible triangle: (0,0) to (pi,0) to (pi,pi)? No.
        # Symmetry group C4v.
        # Wedge: 0 <= theta <= pi/4.
        # Points: (0,0), (pi/2, 0) [X?], (pi/2, pi/2) [M?], (pi, 0) [X?].
        # Let's just use a dense enough set of points in the wedge 0<=ky<=kx<=pi.
        # Actually, standard high symmetry points: Gamma(0,0), X(pi,0), M(pi,pi).
        # Let's sample a few along Gamma-X and Gamma-M and X-M.
        k_points = [
            (0.0, 0.0),          # Gamma
            (np.pi/2, 0.0),      # Mid Gamma-X
            (np.pi, 0.0),        # X
            (np.pi, np.pi),      # M
            (np.pi/2, np.pi/2),  # Mid Gamma-M
            (3*np.pi/4, 0.0),
            (np.pi, np.pi/2),
            (np.pi/2, np.pi/4)
        ]
    
    # Construct permittivity grid
    eps = np.zeros((SIZE, SIZE))
    eps[cell == 1] = EPS_HIGH
    eps[cell == 0] = EPS_LOW
    
    # Inverse permittivity for TM modes
    inv_eps = 1.0 / eps
    
    # Fourier coefficients of inv_eps
    N = n_pw
    indices = np.arange(-N//2, N//2 + 1)
    
    # Compute Fourier coefficients of inv_eps using FFT
    # We need C[m, n] = sum_{i,j} inv_eps[i,j] exp(-i 2*pi (m*i + n*j)/SIZE) / SIZE^2
    # np.fft.fft2 gives coefficients for frequencies 0..SIZE-1.
    # We need to map negative frequencies.
    
    fft_inv = np.fft.fft2(inv_eps)
    # Normalize
    fft_inv = fft_inv / (SIZE * SIZE)
    
    # Create C_inv array of size N x N
    # C_inv[p, q] corresponds to m=indices[p], n=indices[q]
    C_inv = np.zeros((N, N), dtype=complex)
    
    for p, m in enumerate(indices):
        for q, n in enumerate(indices):
            # Map m, n to FFT indices
            # FFT index k corresponds to frequency k if k < SIZE/2, else k - SIZE
            # We want frequency m.
            m_fft = m % SIZE
            n_fft = n % SIZE
            C_inv[p, q] = fft_inv[m_fft, n_fft]
            
    max_omega_4 = 0.0
    min_omega_5 = float('inf')
    
    # Precompute G vectors
    Gx = 2 * np.pi * indices
    Gy = 2 * np.pi * indices
    
    for kx, ky in k_points:
        # Construct matrix M
        # M[i, j] = ( (k - G_i) . (k - G_j) ) / |k - G_i|^2 * C_inv[G_i - G_j]
        
        # Vectorize over i for efficiency
        # k - G_i
        kx_vec = kx - Gx
        ky_vec = ky - Gy
        denom = kx_vec**2 + ky_vec**2
        
        # Avoid division by zero
        # If denom is 0, the term is 0 (numerator is also 0 for i=j? No, numerator is (k-Gi).(k-Gj). If k=Gi, num=0. So term is 0.)
        # We can set inv_denom = 0 where denom is 0.
        inv_denom = np.zeros_like(denom)
        mask = denom > 1e-12
        inv_denom[mask] = 1.0 / denom[mask]
        
        # Matrix of (k - G_i) dot (k - G_j)
        # This is (kx_i * kx_j + ky_i * ky_j)
        # Can be written as outer product of kx_vec and kx_vec + outer product of ky_vec and ky_vec
        num_matrix = np.outer(kx_vec, kx_vec) + np.outer(ky_vec, ky_vec)
        
        # Scale rows by inv_denom[i]
        M_num = num_matrix * inv_denom[:, np.newaxis]
        
        # Now multiply by C_inv[G_i - G_j]
        # G_i - G_j = (2*pi*(indices[i]-indices[j]), 2*pi*(indices[i]-indices[j]))
        # We need C_inv[p, q] where indices[p] = indices[i] - indices[j]
        
        # Build index matrix for C_inv
        # diff_m[i, j] = indices[i] - indices[j]
        # We need to map this to p, q in C_inv.
        # C_inv is indexed by p, q corresponding to indices[p], indices[q].
        # So we need to find p such that indices[p] = diff_m[i, j].
        
        # Since indices are contiguous integers from -N//2 to N//2,
        # diff_m is in range [-N+1, N-1].
        # We can create a lookup.
        
        # Let's just loop over i, j for the C_inv part to avoid complex indexing logic errors, 
        # but we already vectorized the geometric part.
        # Actually, let's just build the full M matrix in a loop for simplicity and safety, 
        # it's only 13x13 = 169 ops per k-point. 8 k-points. 1344 ops. Negligible.
        
        M = np.zeros((N, N), dtype=complex)
        for i in range(N):
            if not mask[i]:
                continue
            for j in range(N):
                dm = indices[i] - indices[j]
                dn = indices[i] - indices[j] # Square lattice, same index difference
                
                # Find p, q in C_inv
                # indices is [-N//2, ..., N//2]
                # p = dm + N//2
                # q = dn + N//2
                p = dm + N//2
                q = dn + N//2
                
                if 0 <= p < N and 0 <= q < N:
                    c_val = C_inv[p, q]
                else:
                    c_val = 0.0
                
                M[i, j] = M_num[i, j] * c_val
        
        # Solve (I - M) c = omega^2 c
        # eigvals of (I - M)
        I_mat = np.eye(N, dtype=complex)
        A = I_mat - M
        
        # A should be Hermitian
        # Use eigvalsh
        try:
            eigvals = np.linalg.eigvalsh(A)
        except:
            eigvals = np.linalg.eigvals(A)
            eigvals = np.real(eigvals)
            
        eigvals = np.sort(eigvals)
        
        if len(eigvals) >= 5:
            # omega^2 = eigvals
            # omega = sqrt(eigvals)
            # We need 4th and 5th bands (indices 3 and 4)
            w4_sq = eigvals[3]
            w5_sq = eigvals[4]
            
            # Ensure positive
            if w4_sq < 0: w4_sq = 0
            if w5_sq < 0: w5_sq = 0
            
            w4 = np.sqrt(w4_sq)
            w5 = np.sqrt(w5_sq)
            
            max_omega_4 = max(max_omega_4, w4)
            min_omega_5 = min(min_omega_5, w5)
            
    if min_omega_5 == float('inf'):
        return -1.0
        
    gap = 2 * (min_omega_5 - max_omega_4) / (min_omega_5 + max_omega_4)
    return gap

def get_symmetric_neighbors(i, j, size=32):
    """
    Get all pixels that must be flipped together to maintain C4v symmetry.
    i, j are indices in 0..size-1.
    """
    # Center is at (size/2 - 0.5)? No, pixel centers are (0.5, 0.5) to (size-0.5, size-0.5).
    # Symmetry about cell center.
    # Cell center is (size/2, size/2) in pixel coordinates if we consider pixel (0,0) top-left?
    # Let's assume pixel (i,j) has center (i+0.5, j+0.5).
    # Center of cell is (16, 16) for size 32.
    # Reflections:
    # 1. Identity: (i, j)
    # 2. Rotate 90: (j, size-1-i) ? 
    #    Let's check. Center (16,16).
    #    Vector from center: (i+0.5-16, j+0.5-16) = (i-15.5, j-15.5).
    #    Rotate 90 CCW: (-(j-15.5), i-15.5) = (15.5-j, i-15.5).
    #    New pixel: (15.5-j+0.5, i-15.5+0.5) = (16-j, i-15) -> (15-j, i-15)?
    #    Let's use 0-based indices.
    #    Map (i,j) -> (j, size-1-i) is a 90 deg rotation?
    #    Test: (0,0) -> (0, 31). Top-left to Top-right? No, that's reflection.
    #    Rotation 90 CCW of (0,0) around (15.5, 15.5) should be (31, 0).
    #    Formula: i' = j, j' = size-1-i.
    #    (0,0) -> (0, 31). This is Top-Left to Top-Right. That's a reflection across vertical axis? No.
    #    Let's stick to the definition: cell == cell[::-1] row-wise (Vertical Reflection), cell == cell[:, ::-1] (Horizontal Reflection), cell == cell.T (Transpose).
    #    If a cell is invariant under these, it is C4v symmetric.
    #    So if we flip (i,j), we must flip:
    #    (i, j)
    #    (size-1-i, j)  # Vertical reflection
    #    (i, size-1-j)  # Horizontal reflection
    #    (size-1-i, size-1-j) # 180 rotation (or V+H)
    #    Wait, Transpose symmetry implies (i,j) == (j,i).
    #    So the orbit of (i,j) under C4v is:
    #    (i, j)
    #    (j, i)
    #    (size-1-i, j)
    #    (size-1-i, size-1-j)
    #    (j, size-1-i)
    #    (size-1-j, i)
    #    (size-1-j, size-1-i)
    #    (i, size-1-j)
    
    # There are at most 8 symmetric pixels.
    pixels = set()
    i1, j1 = i, j
    i2, j2 = j, i
    i3, j3 = size-1-i, j
    i4, j4 = size-1-i, size-1-j
    i5, j5 = j, size-1-i
    i6, j6 = size-1-j, i
    i7, j7 = size-1-j, size-1-i
    i8, j8 = i, size-1-j
    
    pixels.add((i1, j1))
    pixels.add((i2, j2))
    pixels.add((i3, j3))
    pixels.add((i4, j4))
    pixels.add((i5, j5))
    pixels.add((i6, j6))
    pixels.add((i7, j7))
    pixels.add((i8, j8))
    
    return list(pixels)

def apply_symmetric_flip(cell, i, j, size=32):
    """
    Flips the pixel at (i,j) and all its symmetric partners.
    Returns a new cell.
    """
    new_cell = cell.copy()
    pixels = get_symmetric_neighbors(i, j, size)
    current_val = cell[i, j]
    new_val = 1 - current_val
    for pi, pj in pixels:
        new_cell[pi, pj] = new_val
    return new_cell

def design(problem):
    size = problem["pixels"]
    start_time = time.time()
    time_limit = 200.0 # Safety margin for 230s limit
    
    # Initial candidates
    candidates = []
    
    # 1. Rotated Square (Dielectric)
    # Distance from center < radius
    x = np.arange(size)
    y = np.arange(size)
    X, Y = np.meshgrid(x, y, indexing='ij')
    # Center at (size/2, size/2)
    cx, cy = size/2.0, size/2.0
    # Rotated square: |x-cx| + |y-cy| < R? No, that's diamond.
    # Square aligned with axes: max(|x-cx|, |y-cy|) < R
    # Rotated 45 deg: |x-cx| + |y-cy| < R
    # Let's try Diamond (rotated square)
    R = size * 0.35
    dist = np.abs(X - cx) + np.abs(Y - cy)
    cell_diamond = (dist < R).astype(int)
    candidates.append(cell_diamond)
    
    # 2. Rotated Square (Air in Dielectric)
    cell_diamond_inv = 1 - cell_diamond
    candidates.append(cell_diamond_inv)
    
    # 3. Square aligned with axes
    R_sq = size * 0.35
    dist_sq = np.maximum(np.abs(X - cx), np.abs(Y - cy))
    cell_sq = (dist_sq < R_sq).astype(int)
    candidates.append(cell_sq)
    
    # 4. Square aligned with axes (Air)
    cell_sq_inv = 1 - cell_sq
    candidates.append(cell_sq_inv)
    
    # 5. Random symmetric
    np.random.seed(42)
    cell_rand = np.zeros((size, size), dtype=int)
    # Fill upper triangle and mirror
    for i in range(size):
        for j in range(i, size):
            if np.random.rand() > 0.5:
                # Flip orbit
                pass # We'll just generate a random field and symmetrize
    # Simpler: Generate random, then enforce symmetry
    rand_field = np.random.rand(size, size)
    # Symmetrize
    sym_field = rand_field
    sym_field = (sym_field + sym_field[::-1, :]) / 2.0
    sym_field = (sym_field + sym_field[:, ::-1]) / 2.0
    sym_field = (sym_field + sym_field.T) / 2.0
    cell_rand = (sym_field > 0.5).astype(int)
    candidates.append(cell_rand)
    
    best_cell = candidates[0]
    best_gap = -10.0
    
    # Evaluate all candidates
    for c in candidates:
        g = compute_gap(c)
        if g > best_gap:
            best_gap = g
            best_cell = c.copy()
            
    # Local Search
    # We will perform hill climbing.
    # To avoid getting stuck in local minima, we can do multiple restarts or simulated annealing.
    # Given time, let's do simple hill climbing with random restarts if stuck.
    
    current_cell = best_cell.copy()
    current_gap = best_gap
    no_improve_count = 0
    max_no_improve = 50 # Restart if no improvement in 50 steps
    
    while time.time() - start_time < time_limit:
        # Pick a random pixel to flip
        i = np.random.randint(size)
        j = np.random.randint(size)
        
        new_cell = apply_symmetric_flip(current_cell, i, j, size)
        new_gap = compute_gap(new_cell)
        
        if new_gap > current_gap:
            current_cell = new_cell
            current_gap = new_gap
            no_improve_count = 0
            
            if current_gap > best_gap:
                best_gap = current_gap
                best_cell = current_cell.copy()
        else:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                # Restart from best found so far with some noise?
                # Or just reset counter to continue?
                # Let's reset counter to allow more exploration
                no_improve_count = 0
                # Optionally, perturb the current cell significantly
                # For now, just continue.
                
    return best_cell.tolist()

if __name__ == "__main__":
    problem = {"pixels": 32, "eps_low": 1, "eps_high": 12, "lower_band": 4, "polarization": "TM"}
    cell = design(problem)
    print("Cell shape:", np.array(cell).shape)
    print("Symmetry check:")
    c = np.array(cell)
    print("Row reverse:", np.array_equal(c, c[::-1]))
    print("Col reverse:", np.array_equal(c, c[:, ::-1]))
    print("Transpose:", np.array_equal(c, c.T))
    print("Gap:", compute_gap(c))
