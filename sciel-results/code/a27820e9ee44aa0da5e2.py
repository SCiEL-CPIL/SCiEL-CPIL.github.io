import numpy as np
import time
import math
import random
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigvalsh

def design(problem):
    size = problem.get("pixels", 32)
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)
    
    start_time = time.time()
    deadline = start_time + 110.0  # 110s budget, 20s safety margin

    # 1. Setup Fast Band Solver
    # 289 plane waves (17x17 grid of G vectors)
    half_pw = 8
    G_vecs = []
    for m in range(-half_pw, half_pw + 1):
        for n in range(-half_pw, half_pw + 1):
            G_vecs.append((m, n))
    G_vecs = np.array(G_vecs)
    nG = len(G_vecs)
    
    # Precompute k-points.
    # We use a dense grid in the irreducible BZ to ensure we capture min/max bands.
    # The evaluator uses 30 points, but we use a slightly denser set for robustness during search.
    k_points = []
    n_k = 12
    for i in range(n_k + 1):
        for j in range(n_k + 1):
            if i + j <= n_k:
                s = i / n_k
                t = j / n_k
                k_points.append((s * np.pi, t * np.pi))
    k_points = np.array(k_points)
    
    # Precompute kG2 for all k-points
    # kG2[i, j] = |k_i + G_j|^2
    kG2_cache = np.zeros((len(k_points), nG))
    for i, (kx, ky) in enumerate(k_points):
        kG = np.stack([kx + G_vecs[:, 0], ky + G_vecs[:, 1]], axis=1)
        kG2_cache[i] = kG[:, 0]**2 + kG[:, 1]**2

    def compute_score(cell):
        """
        Compute the 4-5 TM band gap score for a given cell (32x32 array of 0/1).
        Returns the gap-midgap ratio.
        Uses a fast sparse matrix construction for the eigenvalue problem.
        """
        # Convert to epsilon grid
        eps_grid = eps_low + (eps_high - eps_low) * np.array(cell, dtype=float)
        
        # FFT of epsilon for the convolution matrix
        eps_F = np.fft.fft2(eps_grid) / (size * size)
        inv_eps_F = 1.0 / eps_F
        
        min_omega5 = np.inf
        max_omega4 = 0.0
        
        for i_k, (kx, ky) in enumerate(k_points):
            kG2 = kG2_cache[i_k]
            
            # The matrix A is: A_{ij} = (k+G_i) * (k+G_j) * (1/eps)_{G_i - G_j}
            # We can construct this efficiently.
            # Let D = diag(kG2). Let C be the convolution matrix where C_{ij} = (1/eps)_{G_i - G_j}.
            # Then A = D @ C.
            # Since C is a circulant-like matrix (Toeplitz in 2D sense for the grid), 
            # but G vectors are on a square grid, C is actually a block-circulant or can be applied via FFT.
            # However, for eigsh, we need the matrix or a matvec.
            # Constructing the full 289x289 matrix is fast enough if we do it once per cell.
            
            # Construct C: C_{mn, pq} = inv_eps_F[G_m - G_p, G_n - G_q]
            # Indices: 0..16 correspond to G indices -8..8
            # diff index = i - j. If i-j is -8..8, map to 0..16.
            # G_i - G_j = (m_i - m_j, n_i - n_j).
            
            # We can build C using the structure:
            # C is the matrix representation of the linear operator L(v) = IFFT( inv_eps_F * FFT(v) )
            # Wait, the operator in the eigenproblem A u = lambda u is:
            # sum_j (k+G_i) (k+G_j) (1/eps)_{G_i-G_j} u_j = lambda u_i
            # This is (D C) u = lambda u.
            # C is the matrix of the operator "multiply by 1/eps in Fourier space".
            # C_{ij} = (1/eps)_{G_i - G_j}.
            
            # Let's build C explicitly. 289x289 is small.
            # Map G indices to 0..16
            # G_vecs is sorted by m then n? 
            # m in -8..8, n in -8..8.
            # Index = (m+8)*17 + (n+8)
            
            # Precompute the mapping from (dm, dn) to index offset?
            # Actually, we can just build C using loops or vectorization.
            # Given the speed requirement, we will build C once per cell.
            
            # Vectorized construction of C:
            # G_i = (m_i, n_i), G_j = (m_j, n_j)
            # dm = m_i - m_j, dn = n_i - n_j
            # We need inv_eps_F[dm, dn] where indices are shifted by 8.
            # inv_eps_F is 32x32. We need values for dm, dn in -8..8.
            # Note: inv_eps_F[k, l] corresponds to Fourier coeff of (k, l).
            # The convolution kernel is (1/eps)(G_i - G_j).
            
            # Let's extract the 17x17 block of inv_eps_F centered at 0?
            # No, the indices G_i - G_j range from -16 to 16?
            # G_i is in [-8, 8]. G_j is in [-8, 8].
            # Difference is in [-16, 16].
            # But inv_eps_F is computed from a 32x32 grid.
            # The Fourier coefficients of a 32x32 grid are defined for indices -16..15 (or 0..31).
            # The plane wave expansion uses G vectors. The convolution theorem applies.
            # The coefficient (1/eps)_{G} is the Fourier series coefficient of the periodic function 1/eps(x).
            # For a pixelated 1/eps, the Fourier coefficients are non-zero for all G.
            # However, the Galerkin method truncates the basis to G in [-8, 8].
            # The matrix element C_{ij} uses (1/eps)_{G_i - G_j}.
            # Since G_i, G_j are in [-8, 8], G_i - G_j is in [-16, 16].
            # We need the Fourier coefficients of 1/eps at these specific wave vectors.
            # The FFT of the 32x32 grid gives coefficients for k = 0..31.
            # k = -16 corresponds to 16. k = -8 corresponds to 24.
            # So we can map any integer k to the FFT array index: idx = k % 32.
            
            # Let's build C.
            # i, j are 0..288.
            # m_i = (i // 17) - 8
            # n_i = (i % 17) - 8
            # m_j = (j // 17) - 8
            # n_j = (j % 17) - 8
            # dm = m_i - m_j
            # dn = n_i - n_j
            # C[i, j] = inv_eps_F[dm % 32, dn % 32]
            
            # This is O(N^2) = 289^2 ~ 83k operations. Very fast.
            
            m_indices = (np.arange(nG) // 17) - 8
            n_indices = (np.arange(nG) % 17) - 8
            
            # dm[i, j] = m_i - m_j
            dm = m_indices[:, None] - m_indices[None, :]
            dn = n_indices[:, None] - n_indices[None, :]
            
            # Map to FFT indices
            dm_idx = dm % size
            dn_idx = dn % size
            
            C = inv_eps_F[dm_idx, dn_idx]
            
            # A = D @ C, where D = diag(kG2)
            # A = kG2[:, None] * C
            
            A = kG2[:, None] * C
            
            # Solve for lowest 6 eigenvalues
            # A is symmetric?
            # C_{ij} = (1/eps)_{G_i - G_j}. Since 1/eps is real, (1/eps)_{-G} = (1/eps)_G^*.
            # For real eps, (1/eps) is real, so C is symmetric?
            # C_{ji} = (1/eps)_{G_j - G_i} = (1/eps)_{-(G_i - G_j)} = (1/eps)_{G_i - G_j} (since real).
            # So C is symmetric. D is diagonal real.
            # A = D C is NOT symmetric in general.
            # However, the eigenvalues are real because the problem is Hermitian in the weighted inner product?
            # Actually, the standard PW method for TM leads to a generalized eigenvalue problem or a symmetric one if formulated correctly.
            # The equation is: (k+G) (k+G') (1/eps)_{G-G'} E_{G'} = omega^2 E_G
            # This matrix is not symmetric.
            # We should use eig (dense) or eigsh with which='SA' (smallest algebraic) if we trust it.
            # Since 289 is small, dense eig is fast and robust.
            
            try:
                vals = np.linalg.eigvalsh(A) # If A is symmetric. But it's not.
                # Let's check symmetry.
                # If eps is real, C is symmetric. A = D C.
                # A_{ij} = k_i C_{ij}. A_{ji} = k_j C_{ji} = k_j C_{ij}.
                # A_{ij} != A_{ji} unless k_i = k_j.
                # So A is non-symmetric.
                # We must use eig.
                vals = np.linalg.eigvals(A)
                vals = np.sort(vals.real)
                
                if len(vals) < 5:
                    return -1.0
                
                omega4 = np.sqrt(max(0, vals[3]))
                omega5 = np.sqrt(max(0, vals[4]))
                
                if omega5 < min_omega5:
                    min_omega5 = omega5
                if omega4 > max_omega4:
                    max_omega4 = omega4
                    
            except Exception:
                return -1.0
        
        if min_omega5 == np.inf or max_omega4 == 0.0:
            return -1.0
            
        if min_omega5 <= max_omega4:
            return -1.0
            
        return 2.0 * (min_omega5 - max_omega4) / (min_omega5 + max_omega4)

    # 2. Stochastic Optimization (Simulated Annealing / Hill Climbing)
    # We optimize the top-left 16x16 quadrant, then mirror to ensure symmetry.
    # Symmetry: cell == cell[::-1] row-wise, col-wise, and transpose.
    # This means the cell is symmetric under:
    # 1. Reflection across x=15.5 (vertical center)
    # 2. Reflection across y=15.5 (horizontal center)
    # 3. Rotation 90 deg (transposition)
    # This implies the cell is determined by its top-left 16x16 quadrant?
    # Let's check.
    # If cell is symmetric under transpose and reflections,
    # cell[i, j] = cell[j, i]
    # cell[i, j] = cell[31-i, j]
    # cell[i, j] = cell[i, 31-j]
    # So cell[i, j] = cell[31-i, 31-j]
    # The independent region is i in 0..15, j in 0..15.
    # Yes, 16x16 = 256 bits.
    
    # Initialize with a random pattern or a simple shape
    best_cell = None
    best_score = -1.0
    
    # Initial guess: 4 circles
    def generate_initial_cell():
        cell = np.zeros((size, size), dtype=int)
        ys, xs = np.mgrid[0:size, 0:size]
        centers = [(8, 8), (23, 8), (8, 23), (23, 23)]
        for cx, cy in centers:
            dx = xs - cx
            dy = ys - cy
            mask = (dx**2 + dy**2) <= 20 # radius ~4.5
            cell[mask] = 1
        return cell

    current_cell = generate_initial_cell()
    current_score = compute_score(current_cell)
    
    if current_score > best_score:
        best_score = current_score
        best_cell = current_cell.copy()
        
    # Annealing parameters
    T = 1.0
    cooling_rate = 0.99
    min_T = 0.01
    max_steps = 20000 # Adjust based on time
    
    # Precompute symmetry mapping?
    # To flip a bit in the 16x16 quadrant, we need to flip 4 bits in the full 32x32 cell
    # (or 2 if on axis, or 1 if center, but center is 15.5 so no single pixel center).
    # Actually, the 16x16 quadrant maps to 4 pixels in the full cell:
    # (i, j), (31-i, j), (i, 31-j), (31-i, 31-j)
    # Since i,j in 0..15, these are all distinct.
    
    # List of all 256 pixel positions in the quadrant
    pixels_to_flip = [(i, j) for i in range(16) for j in range(16)]
    
    step = 0
    while time.time() < deadline and T > min_T:
        # Pick a random pixel in the quadrant
        idx = random.randint(0, 255)
        i, j = pixels_to_flip[idx]
        
        # Flip the 4 symmetric pixels
        old_val = current_cell[i, j]
        new_val = 1 - old_val
        
        # Apply flip
        current_cell[i, j] = new_val
        current_cell[31-i, j] = new_val
        current_cell[i, 31-j] = new_val
        current_cell[31-i, 31-j] = new_val
        
        new_score = compute_score(current_cell)
        
        # Accept/Reject
        if new_score > current_score or random.random() < math.exp((new_score - current_score) / T):
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_cell = current_cell.copy()
        else:
            # Revert
            current_cell[i, j] = old_val
            current_cell[31-i, j] = old_val
            current_cell[i, 31-j] = old_val
            current_cell[31-i, 31-j] = old_val
            
        T *= cooling_rate
        step += 1
        
        # Break if time is running out
        if time.time() > deadline - 5:
            break

    # Final Safety: If no valid cell was found (unlikely), return the initial one
    if best_cell is None:
        best_cell = generate_initial_cell()

    # Ensure the returned cell is a list of lists of ints
    return best_cell.tolist()
