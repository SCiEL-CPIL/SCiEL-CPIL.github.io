"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np

def design(problem):
    """
    Design a 32x32 unit cell using local pixel-flip hill climbing
    with a fast approximate band-structure surrogate.
    """
    start_time = time.time()
    deadline = start_time + 200.0  # Safety margin before 230s limit
    
    size = problem.get("pixels", 32)
    eps_high = problem.get("eps_high", 12.0)
    eps_low = problem.get("eps_low", 1.0)
    
    # --- Fast Surrogate Band Structure ---
    def compute_surrogate_gap(cell):
        """
        Compute an approximate gap ratio using a small plane-wave basis.
        This is much faster than the full 289-PW method used by the evaluator.
        """
        # Convert cell to numpy array for speed
        cell_np = np.array(cell, dtype=np.float64)
        
        # Compute inverse epsilon map
        # epsilon = eps_low + (eps_high - eps_low) * cell
        eps_map = eps_low + (eps_high - eps_low) * cell_np
        inv_eps_map = 1.0 / eps_map
        
        # Discrete Fourier Transform of 1/epsilon
        # We need Fourier coefficients for a small set of reciprocal lattice vectors
        # Use a small number of plane waves for speed
        # Let's use a 5x5 grid of plane waves (N=5) -> 25 PWs
        # This is a trade-off: less accurate than 17x17 (289 PWs) but much faster
        
        N_pw = 5  # 5x5 grid of plane waves
        # Reciprocal lattice vectors: G = (m, n) * 2*pi/a
        # We work in units where a=1, so G = (m, n)
        m_vals = np.arange(-N_pw//2, N_pw//2 + 1)
        n_vals = np.arange(-N_pw//2, N_pw//2 + 1)
        
        # Compute Fourier coefficients of 1/epsilon
        # c_G = (1/N^2) * sum_{i,j} inv_eps(i,j) * exp(-i * G . r_ij)
        # We can use FFT for this, but since we only need specific G, 
        # direct sum or a small FFT might be okay.
        # Actually, let's use numpy FFT to get all coefficients up to N_pw.
        
        # Pad the 32x32 grid to a size that is a multiple of N_pw? No, FFT works on 32x32.
        # The Fourier coefficients of the pixel map are given by:
        # c_G = (1/32^2) * sum_{i,j} f(i,j) exp(-i 2pi (m i/32 + n j/32))
        # This is exactly (1/32^2) * FFT2(f) evaluated at (m, n).
        
        fft_inv_eps = np.fft.fft2(inv_eps_map) / (size * size)
        
        # Extract the needed coefficients
        # G = (m, n). The index in FFT is (m mod 32, n mod 32).
        # For m in [-2, 2], n in [-2, 2]:
        # If m >= 0, index is m. If m < 0, index is 32 + m.
        
        c_G = {}
        for m in m_vals:
            for n in n_vals:
                idx_m = m % size
                idx_n = n % size
                c_G[(m, n)] = fft_inv_eps[idx_m, idx_n]
        
        # Build the Hamiltonian matrix for a given k-point
        # H_{G, G'} = (k + G) . (k + G') * c_{G-G'}  (for TM mode, roughly)
        # Actually, the standard PWE Hamiltonian for TM is:
        # H_{G, G'} = (k+G) x (k+G') dot product? No.
        # For TM (H field in plane, Ez out of plane):
        # The equation is: [ (k+G)^2 ] E_G = omega^2 sum_{G'} (1/eps)_{G-G'} E_{G'}
        # So the matrix to solve is:
        # M_{G, G'} = delta_{G,G'} (k+G)^2 - omega^2 c_{G-G'} = 0
        # Or generalized eigenvalue problem:
        # A v = lambda B v
        # Where A_{G,G'} = (k+G)^2 delta_{G,G'} and B_{G,G'} = c_{G-G'}
        
        # Let's define the k-points.
        # Irreducible BZ for square lattice: 0 <= kx <= ky, kx+ky <= pi/a (in units 2pi/a=1? No, usually k is in rad/m)
        # Let's use dimensionless k where a=1. G = 2pi (m, n).
        # So k is in units of 2pi/a.
        # Key k-points:
        # Gamma: (0, 0)
        # M: (0.5, 0.5)  -> (pi, pi) in rad/m if a=1? No.
        # Let's stick to the convention: G = 2pi (m, n).
        # k-points in the BZ boundary:
        # Gamma: (0, 0)
        # X: (0.5, 0)
        # M: (0.5, 0.5)
        # We should also check a point along Gamma-M, e.g., (0.25, 0.25)
        
        k_points = [
            (0.0, 0.0),   # Gamma
            (0.5, 0.0),   # X
            (0.5, 0.5),   # M
            (0.25, 0.25)  # Interior
        ]
        
        # Precompute G vectors and (k+G)^2
        G_vecs = []
        for m in m_vals:
            for n in n_vals:
                G_vecs.append((2*math.pi*m, 2*math.pi*n))
        
        N = len(G_vecs)
        
        min_omega_5 = float('inf')
        max_omega_4 = 0.0
        
        for kx, ky in k_points:
            # Build matrices A and B
            A = np.zeros((N, N))
            B = np.zeros((N, N), dtype=complex)
            
            for i, (Gx, Gy) in enumerate(G_vecs):
                kx_G = kx + Gx
                ky_G = ky + Gy
                A[i, i] = kx_G**2 + ky_G**2
                
                for j, (Gx_p, Gy_p) in enumerate(G_vecs):
                    G_diff = (m_vals[i % len(m_vals)] - m_vals[j % len(m_vals)], 
                              n_vals[i % len(n_vals)] - n_vals[j % len(n_vals)])
                    # Wait, m_vals and n_vals are 1D arrays.
                    # G_vecs index i corresponds to (m_vals[i//len(n_vals)], n_vals[i%len(n_vals)])
                    # Let's restructure G_vecs generation to keep track of m,n
                    pass
            
            # Let's redo the loop structure for clarity and correctness
            A = np.zeros((N, N))
            B = np.zeros((N, N), dtype=complex)
            
            # Map index to (m, n)
            idx_to_mn = []
            for m in m_vals:
                for n in n_vals:
                    idx_to_mn.append((m, n))
            
            for i in range(N):
                m_i, n_i = idx_to_mn[i]
                Gx_i = 2*math.pi*m_i
                Gy_i = 2*math.pi*n_i
                kx_G = kx + Gx_i
                ky_G = ky + Gy_i
                A[i, i] = kx_G**2 + ky_G**2
                
                for j in range(N):
                    m_j, n_j = idx_to_mn[j]
                    G_diff_m = m_i - m_j
                    G_diff_n = n_i - n_j
                    # Check if (G_diff_m, G_diff_n) is in our c_G dictionary
                    # If not, it's 0 (or we need to compute it, but we assumed small G)
                    # Since m_vals is small, the difference is small, so it should be in c_G
                    if (G_diff_m, G_diff_n) in c_G:
                        B[i, j] = c_G[(G_diff_m, G_diff_n)]
                    else:
                        B[i, j] = 0.0
            
            # Solve generalized eigenvalue problem: A v = omega^2 B v
            # Use numpy.linalg.eig for small matrices
            try:
                eigvals, eigvecs = np.linalg.eig(np.linalg.inv(B) @ A)
                # Filter real eigenvalues
                real_eigvals = np.real(eigvals)
                real_eigvals = real_eigvals[real_eigvals > 0]
                real_eigvals.sort()
                
                if len(real_eigvals) >= 5:
                    # omega = sqrt(eigval)
                    omega_4 = np.sqrt(real_eigvals[3])
                    omega_5 = np.sqrt(real_eigvals[4])
                    max_omega_4 = max(max_omega_4, omega_4)
                    min_omega_5 = min(min_omega_5, omega_5)
            except:
                return -1.0  # Invalid or error
            
        if max_omega_4 == 0.0 or min_omega_5 == float('inf'):
            return -1.0
            
        gap = min_omega_5 - max_omega_4
        midgap = (min_omega_5 + max_omega_4) / 2.0
        if midgap == 0:
            return -1.0
        return 2.0 * gap / midgap

    # --- Initial Design ---
    # Start with a four-rod design, but maybe with a different radius or shape
    # Let's start with a slightly larger radius or a different configuration
    cell = [[0 for _ in range(size)] for _ in range(size)]
    
    # Four rods at (±0.25, ±0.25), radius 0.15
    radius = 0.15
    centers = [
        (-0.25, -0.25),
        ( 0.25, -0.25),
        (-0.25,  0.25),
        ( 0.25,  0.25)
    ]
    
    for i in range(size):
        y = (i + 0.5) / size - 0.5
        for j in range(size):
            x = (j + 0.5) / size - 0.5
            inside = False
            for cx, cy in centers:
                dist_sq = (x - cx)**2 + (y - cy)**2
                if dist_sq < radius**2:
                    inside = True
                    break
            cell[i][j] = 1 if inside else 0

    # Enforce symmetry
    # The construction is symmetric, but let's ensure it
    for i in range(size // 2):
        for j in range(size):
            # Vertical flip
            if cell[i][j] != cell[size - 1 - i][j]:
                # Keep the one with 1? Or average? 
                # For binary, if one is 1 and other 0, we have a conflict.
                # Since it's symmetric by construction, this shouldn't happen.
                # If it does, we can just set both to the value of the first.
                pass
            # Horizontal flip
            if cell[i][j] != cell[i][size - 1 - j]:
                pass
            # Transpose
            if cell[i][j] != cell[j][i]:
                pass

    best_cell = [row[:] for row in cell]
    best_score = compute_surrogate_gap(best_cell)
    
    # --- Hill Climbing ---
    # We will flip pixels that preserve symmetry.
    # Symmetry group: C4v.
    # Orbit of a pixel (i, j):
    # (i, j), (size-1-i, j), (i, size-1-j), (size-1-i, size-1-j)
    # (j, i), (size-1-j, i), (j, size-1-i), (size-1-j, size-1-i)
    # If (i,j) is on an axis or diagonal, the orbit is smaller.
    
    # Generate list of unique orbits
    visited = set()
    orbits = []
    for i in range(size):
        for j in range(size):
            if (i, j) in visited:
                continue
            
            orbit = set()
            # Generate all symmetric counterparts
            points = [
                (i, j),
                (size - 1 - i, j),
                (i, size - 1 - j),
                (size - 1 - i, size - 1 - j),
                (j, i),
                (size - 1 - j, i),
                (j, size - 1 - i),
                (size - 1 - j, size - 1 - i)
            ]
            for p in points:
                orbit.add(p)
                visited.add(p)
            
            orbits.append(orbit)
    
    # Shuffle orbits for randomization
    import random
    random.seed(42)
    random.shuffle(orbits)
    
    # Hill climbing loop
    improved = True
    while improved and time.time() < deadline - 10.0:
        improved = False
        for orbit in orbits:
            if time.time() > deadline - 10.0:
                break
            
            # Try flipping all pixels in the orbit
            # First, check if they are all 0 or all 1
            val = cell[list(orbit)[0][0]][list(orbit)[0][1]]
            all_same = all(cell[i][j] == val for i, j in orbit)
            
            if not all_same:
                # This shouldn't happen if we maintain symmetry, but just in case
                continue
            
            # Flip them
            new_val = 1 - val
            for i, j in orbit:
                cell[i][j] = new_val
            
            # Evaluate
            score = compute_surrogate_gap(cell)
            
            if score > best_score:
                best_score = score
                best_cell = [row[:] for row in cell]
                improved = True
            else:
                # Revert
                for i, j in orbit:
                    cell[i][j] = val
    
    return best_cell

# EVOLVE-BLOCK-END
