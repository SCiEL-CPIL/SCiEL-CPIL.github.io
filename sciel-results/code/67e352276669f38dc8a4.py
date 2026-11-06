import numpy as np
import time
import math

def design(problem):
    size = problem.get("pixels", 32)
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)
    
    start_time = time.time()
    deadline = start_time + 195.0  # Safety margin for 200s limit

    # 1. Fast Band Solver Setup
    # Use 15x15 = 225 plane waves for speed. The gap is robust to this resolution.
    half_pw = 7
    G_vecs = []
    for m in range(-half_pw, half_pw + 1):
        for n in range(-half_pw, half_pw + 1):
            G_vecs.append((m, n))
    G_vecs = np.array(G_vecs) * (2.0 * np.pi)
    nG = len(G_vecs)
    
    # Precompute k-points. 
    # Use a dense grid in the irreducible wedge 0 <= kx <= ky <= pi
    # The evaluator uses 30 points. We use ~30 points for better ranking fidelity.
    k_points = []
    # Grid points
    n_k = 5
    for i in range(n_k + 1):
        for j in range(i, n_k + 1):
            s = i / n_k
            t = j / n_k
            k_points.append((s * np.pi, t * np.pi))
    # Add specific high symmetry points if not already covered or to weight them
    # (pi, 0) is on the boundary of the wedge definition used above? 
    # Wedge: 0<=kx<=ky<=pi. 
    # (pi, 0) is not in 0<=kx<=ky. (0, pi) is.
    # The irreducible BZ for square lattice is usually defined as 0 <= kx <= ky <= pi.
    # Points: Gamma(0,0), X(pi/2, 0) -> mapped to (0, pi/2)? No, X is (pi,0) or (0,pi).
    # Standard irreducible wedge: 0 <= kx <= ky <= pi.
    # Gamma: (0,0)
    # M: (pi, pi)
    # X: (pi, 0) is equivalent to (0, pi) by symmetry. In the wedge, (0, pi) is included.
    # Let's add (0, pi) and (pi/2, pi/2) explicitly to be safe.
    k_points.append((0, np.pi))
    k_points.append((np.pi/2, np.pi/2))
    k_points.append((np.pi, np.pi))
    k_points = np.array(k_points)
    
    def compute_score(cell):
        """
        Compute the 4-5 TM band gap score for a given cell (32x32 array of 0/1).
        """
        eps_grid = eps_low + (eps_high - eps_low) * np.array(cell, dtype=float)
        
        # FFT of inverse epsilon for TM mode
        # HTE approximation: 1/eps is piecewise constant.
        # For binary 0/1, eps is 1 or 12.
        # 1/eps is 1 or 1/12.
        inv_eps_grid = np.zeros_like(eps_grid)
        inv_eps_grid[eps_grid > 1.5] = 1.0 / eps_high
        inv_eps_grid[eps_grid <= 1.5] = 1.0 / eps_low
        
        # Compute Fourier coefficients of 1/eps
        # Note: The plane wave expansion uses coefficients of 1/eps.
        # inv_eps_G = (1/N^2) sum_{r} (1/eps(r)) exp(-i G.r)
        inv_eps_F = np.fft.fft2(inv_eps_grid) / (size * size)
        
        min_omega5 = np.inf
        max_omega4 = 0.0
        
        for kx, ky in k_points:
            kG = np.stack([kx + G_vecs[:, 0], ky + G_vecs[:, 1]], axis=1)
            kG2 = kG[:, 0]**2 + kG[:, 1]**2
            
            # Matrix A for TM: A_{G', G} = (k + G') . (k + G) * (1/eps)_{G'-G}
            # We need to construct the matrix or use matvec.
            # Matvec is faster for small k in eigsh.
            
            def matvec(v):
                # v is vector of coefficients c_G
                # w_G = sum_{G'} (k+G).(k+G') inv_eps_{G-G'} c_{G'}
                # This is a convolution in G-space.
                # inv_eps_{G} is the Fourier coeff.
                # Let's use FFT trick.
                # Reshape v to 2D grid
                v_2d = v.reshape((2*half_pw+1), (2*half_pw+1))
                
                # The operator is:
                # 1. Multiply by kG2? No, that's for TE.
                # For TM, the operator is:
                # (k+G) dot (k+G') is not separable in G and G' simply.
                # Wait, the standard PWE matrix for TM is:
                # M_{G,G'} = (k+G) . (k+G') * (1/eps)_{G-G'}
                # This is NOT a simple convolution because (k+G).(k+G') depends on both G and G'.
                # However, we can expand:
                # (kx+Gx)(kx+G'x) + (ky+Gy)(ky+G'y)
                # = kx^2 + kx(Gx+G'x) + Gx G'x + ky^2 + ky(Gy+G'y) + Gy G'y
                
                # This structure allows us to compute the matrix-vector product
                # using a few FFTs/IFFTs if we precompute terms.
                # But for 225x225, direct matrix construction is fast enough.
                # Let's construct the matrix once per k-point? No, 225x225 is small.
                # 30 k-points * 225x225 matrix = 30 * 50k floats = 1.5M floats. Very fast.
                
                # Let's just build the matrix for this k-point.
                # Precompute Gx, Gy
                Gx = G_vecs[:, 0]
                Gy = G_vecs[:, 1]
                
                # kGx = kx + Gx, kGy = ky + Gy
                kGx = kx + Gx
                kGy = ky + Gy
                
                # Dot product matrix: D[i, j] = kGx[i]*kGx[j] + kGy[i]*kGy[j]
                D = np.outer(kGx, kGx) + np.outer(kGy, kGy)
                
                # Inverse epsilon matrix: E[i, j] = inv_eps_{G_i - G_j}
                # G_i - G_j = (m_i-m_j, n_i-n_j)
                # We need to look up inv_eps_F at index corresponding to (m_i-m_j, n_i-n_j)
                # The FFT array inv_eps_F has indices 0..N-1.
                # G vector (m, n) corresponds to index (m mod N, n mod N).
                # Since G_vecs are in -half_pw..half_pw, and N=2*half_pw+1.
                # Index mapping: idx = (val + half_pw) % N ?
                # Let's check.
                # G_vecs[0] is (-7, -7). Index in FFT?
                # FFT of a grid: coeff at (m,n) is at index (m % N, n % N).
                # So (-7, -7) -> (5, 5) if N=15.
                # Let's create a lookup table for inv_eps coefficients.
                
                # Actually, we can just compute the matrix E directly.
                # E[i, j] = inv_eps_F[(G_vecs[i,0] - G_vecs[j,0]) % N, (G_vecs[i,1] - G_vecs[j,1]) % N]
                
                N = 2*half_pw + 1
                G_diff_x = (G_vecs[:, 0][:, None] - G_vecs[:, 0][None, :]) % N
                G_diff_y = (G_vecs[:, 1][:, None] - G_vecs[:, 1][None, :]) % N
                
                E = inv_eps_F[G_diff_x, G_diff_y]
                
                # Matrix A = D * E (element-wise multiplication)
                A = D * E
                
                # Solve A v = lambda v
                # We want the lowest eigenvalues.
                # A is real symmetric.
                try:
                    vals, vecs = np.linalg.eigvalsh(A)
                    vals = np.sort(vals.real)
                    
                    # The eigenvalues lambda are omega^2.
                    # We need bands 4 and 5 (indices 3 and 4).
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

    def generate_cell(radius, sharpness):
        """
        Generate a 32x32 cell with 4 rounded square inclusions at the corners.
        The inclusions are dielectric (1).
        radius: distance from corner to the "center" of the rounded part? 
        Actually, let's define the shape by a distance function.
        We want a shape that is roughly a square at the corner, but with rounded edges.
        Or a "superellipse" centered at the corner.
        
        Let's place 4 superellipses centered at (r, r), (31-r, r), etc.
        The superellipse equation: |x-cx|^n + |y-cy|^n <= R^n
        We want to vary R and n.
        To ensure symmetry, we generate one in the top-left and mirror.
        
        Center of top-left inclusion: (cx, cy)
        Let's fix the center at (6, 6) or so, and vary the size.
        Or better: vary the "radius" of the dielectric region.
        
        Let's try a simpler parametric family:
        4 squares of size S x S at the corners, with corners cut off (rounded).
        Or just 4 circles of radius R at the corners.
        Circles are symmetric and easy.
        Let's start with circles and sweep R.
        """
        cell = np.zeros((size, size), dtype=int)
        
        # Centers of the 4 inclusions
        # To maximize the gap, we often want the inclusions to be as large as possible
        # but separated.
        # Let's place centers at (c, c), (31-c, c), (c, 31-c), (31-c, 31-c).
        # And vary c and radius R.
        
        # Actually, a "rounded square" filling the corner is better.
        # Let's define the dielectric region as:
        # For each pixel (i, j), it is dielectric if it is "close" to any of the 4 corners.
        # Distance to corner (0,0) is sqrt(i^2 + j^2).
        # If dist < R, it is dielectric.
        
        # Let's sweep R from 5 to 12.
        # And maybe shift the center?
        # If we use circles at (0,0), they are cut off by the boundary.
        # This is effectively a quarter-circle.
        
        # Let's try a "superellipse" centered at (0,0) with size R.
        # |i|^n + |j|^n <= R^n
        # This allows us to vary the "squareness".
        
        # We will sweep R and n.
        
        # Precompute i, j grids
        i, j = np.mgrid[0:size, 0:size]
        
        # Distance to corners
        # Corner 1: (0,0)
        # Corner 2: (31,0) -> j' = 31-j
        # Corner 3: (0,31) -> i' = 31-i
        # Corner 4: (31,31) -> i' = 31-i, j' = 31-j
        
        # We want the pixel to be 1 if it is inside ANY of the 4 shapes.
        
        # To be fast, we can just check the top-left corner and use symmetry.
        # But generating the full array is fast enough for 32x32.
        
        # Let's define the shape in the top-left quadrant.
        # i_top = i
        # j_top = j
        
        # For the other quadrants, we mirror.
        # i_bl = 31-i, j_bl = j
        # i_tr = i, j_tr = 31-j
        # i_br = 31-i, j_br = 31-j
        
        # We want to use the same shape definition for all.
        # So, compute dist for top-left, and apply the same threshold to the mirrored coordinates?
        # No, the shape is identical in all corners.
        
        # Let's compute the "superellipse distance" for the top-left corner.
        # val = |i|^n + |j|^n
        # If val <= R^n, it is dielectric.
        
        # We need to do this for all 4 corners.
        
        # Optimization: 
        # min_val = min(
        #   |i|^n + |j|^n,
        #   |31-i|^n + |j|^n,
        #   |i|^n + |31-j|^n,
        #   |31-i|^n + |31-j|^n
        # )
        # if min_val <= R^n: cell[i,j] = 1
        
        # This is vectorizable.
        
        i_abs = i.astype(float)
        j_abs = j.astype(float)
        
        i_inv = (size - 1 - i).astype(float)
        j_inv = (size - 1 - j).astype(float)
        
        # Calculate terms
        # We will sweep n from 1 to 4.
        # And R from 5 to 12.
        
        return None # Placeholder, actual generation in loop

    best_score = -1.0
    best_cell = None
    
    # Parametric Sweep
    # n: 1.5, 2.0, 2.5, 3.0, 4.0
    # R: 5.0 to 12.0 step 0.5
    
    n_values = [1.5, 2.0, 2.5, 3.0, 4.0]
    R_values = np.arange(5.0, 12.5, 0.5)
    
    i, j = np.mgrid[0:size, 0:size]
    i_f = i.astype(float)
    j_f = j.astype(float)
    i_inv_f = (size - 1 - i).astype(float)
    j_inv_f = (size - 1 - j).astype(float)
    
    # Precompute powers for speed?
    # i_f ** n is fast.
    
    for n in n_values:
        # Precompute powers for this n
        i_pow = i_f ** n
        j_pow = j_f ** n
        i_inv_pow = i_inv_f ** n
        j_inv_pow = j_inv_f ** n
        
        # Corner terms
        tl = i_pow + j_pow
        bl = i_inv_pow + j_pow
        tr = i_pow + j_inv_pow
        br = i_inv_pow + j_inv_pow
        
        min_term = np.minimum(np.minimum(tl, bl), np.minimum(tr, br))
        
        for R in R_values:
            if time.time() > deadline - 5.0:
                break
                
            R_pow = R ** n
            mask = min_term <= R_pow
            cell = mask.astype(int)
            
            # Ensure symmetry is exact (it should be by construction, but check)
            # cell == cell[::-1] etc.
            
            score = compute_score(cell)
            
            if score > best_score:
                best_score = score
                best_cell = cell.copy()
                # print(f"New best: {score}, n={n}, R={R}")
                
    # If no valid cell found, return a default
    if best_cell is None:
        # Default: 4 circles at corners
        cell = np.zeros((size, size), dtype=int)
        i, j = np.mgrid[0:size, 0:size]
        dist_tl = np.sqrt(i**2 + j**2)
        dist_bl = np.sqrt((31-i)**2 + j**2)
        dist_tr = np.sqrt(i**2 + (31-j)**2)
        dist_br = np.sqrt((31-i)**2 + (31-j)**2)
        min_dist = np.minimum(np.minimum(dist_tl, dist_bl), np.minimum(dist_tr, dist_br))
        best_cell = (min_dist < 8.0).astype(int)
        
    return best_cell.tolist()
