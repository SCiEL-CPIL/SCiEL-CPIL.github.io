import time
import math
import numpy as np
from scipy.optimize import minimize

def design(problem):
    start_time = time.time()
    deadline = start_time + 210  # 20s margin before 230s kill
    
    size = problem["pixels"]
    eps_low = problem["eps_low"]
    eps_high = problem["eps_high"]
    
    # 1. Define the parametric space
    # We use C4v invariant Fourier terms.
    # Symmetry requirements:
    # 1. cell[i][j] == cell[31-i][31-j] (inversion about center)
    # 2. cell[i][j] == cell[j][i] (transpose)
    # 
    # In Fourier space, this restricts us to specific harmonics.
    # For a function f(x,y) on [0,1]x[0,1] with these symmetries:
    # f(x,y) = f(1-x, 1-y) = f(y,x)
    # 
    # Let's use a basis of functions:
    # phi_0 = 1 (mean)
    # phi_1 = cos(2*pi*x)*cos(2*pi*y) + cos(2*pi*y)*cos(2*pi*x) (symmetric part of (2,2) if we consider x,y swap)
    # Actually, simpler:
    # The fundamental domain is the triangle 0 <= x <= y <= 0.5.
    # We can define the permittivity as a sum of basis functions that are symmetric under C4v.
    
    # Basis functions:
    # b0 = 1
    # b1 = cos(2*pi*x)*cos(2*pi*y)  (This is invariant under x->1-x, y->1-y and x<->y? 
    #    cos(2pi(1-x))cos(2pi(1-y)) = cos(2pi x)cos(2pi y). Yes.
    #    cos(2pi y)cos(2pi x) = same. Yes.)
    # b2 = cos(4*pi*x)*cos(4*pi*y)
    # b3 = cos(2*pi*x)*cos(4*pi*y) + cos(4*pi*x)*cos(2*pi*y) (Symmetric sum)
    # b4 = cos(4*pi*x)*cos(2*pi*y) + cos(2*pi*x)*cos(4*pi*y) (Same as b3? No, wait.
    #    Let's check x<->y swap on b3:
    #    cos(2pi y)cos(4pi x) + cos(4pi y)cos(2pi x). This is symmetric.
    #    So b3 is a valid basis function.
    # b4 = cos(4*pi*x)*cos(4*pi*y) is b2.
    # b5 = cos(6*pi*x)*cos(6*pi*y)
    # b6 = cos(2*pi*x)*cos(6*pi*y) + cos(6*pi*x)*cos(2*pi*y)
    
    # Let's use the first 5 coefficients: [c0, c1, c2, c3, c4]
    # c0: mean permittivity level (0 to 1 scale)
    # c1: fundamental harmonic
    # c2: second harmonic (4,4)
    # c3: mixed harmonic (2,4) + (4,2)
    # c4: third harmonic (6,6)
    
    # We will define the continuous permittivity map eps(x,y) in [0,1] (normalized)
    # Then map to 0/1 pixels by thresholding.
    
    # Precompute grid
    x = (np.arange(size) + 0.5) / size
    y = (np.arange(size) + 0.5) / size
    XX, YY = np.meshgrid(x, y)
    
    # Precompute basis functions on the grid
    # b0 = 1
    # b1 = cos(2pi XX) * cos(2pi YY)
    # b2 = cos(4pi XX) * cos(4pi YY)
    # b3 = cos(2pi XX) * cos(4pi YY) + cos(4pi XX) * cos(2pi YY)
    # b4 = cos(6pi XX) * cos(6pi YY)
    
    b0 = np.ones((size, size))
    b1 = np.cos(2 * math.pi * XX) * np.cos(2 * math.pi * YY)
    b2 = np.cos(4 * math.pi * XX) * np.cos(4 * math.pi * YY)
    b3 = np.cos(2 * math.pi * XX) * np.cos(4 * math.pi * YY) + \
         np.cos(4 * math.pi * XX) * np.cos(2 * math.pi * YY)
    b4 = np.cos(6 * math.pi * XX) * np.cos(6 * math.pi * YY)
    
    basis_funcs = [b0, b1, b2, b3, b4]
    
    def coeffs_to_cell(coeffs):
        # coeffs: [c0, c1, c2, c3, c4]
        # Continuous value v = sum c_i * b_i
        # We want v in [0,1] to represent "dielectric-ness"
        # Let's normalize v such that c0=0.5, others=0 gives 0.5 everywhere.
        # Then threshold at 0.5.
        
        v = np.zeros((size, size))
        for i, c in enumerate(coeffs):
            v += c * basis_funcs[i]
            
        # Shift so that mean is 0.5? 
        # If c0 is the mean, and we want to control volume fraction, 
        # it's better to let c0 vary and threshold at 0.5.
        # But if c0 is large, everything becomes 1.
        # Let's constrain c0 to be around 0.5.
        
        # Threshold
        cell = (v > 0.5).astype(int)
        
        # Enforce strict symmetry (just in case of floating point errors in basis)
        cell = (cell + cell[::-1, ::-1]) // 2
        cell = (cell + cell.T) // 2
        
        return cell

    # 2. Fast Evaluator
    # Use a small number of plane waves for speed during optimization.
    # The evaluator uses 289 (17x17). We can use 13x13=169 or 11x11=121.
    # Let's use N=11 (121 planes) for a good balance of speed and accuracy.
    
    N_pw = 11
    pw_list = []
    for nx in range(-N_pw//2, N_pw//2 + 1):
        for ny in range(-N_pw//2, N_pw//2 + 1):
            pw_list.append((nx, ny))
    pw_list = np.array(pw_list)
    N_pw_actual = len(pw_list)
    
    # K-points in irreducible BZ
    # Gamma (0,0), M (0.5, 0), X (0.5, 0.5)
    # Let's sample a few points along the boundary and interior.
    k_points = [
        (0.0, 0.0),
        (0.25, 0.0),
        (0.5, 0.0),
        (0.5, 0.25),
        (0.5, 0.5),
        (0.25, 0.25),
        (0.125, 0.0),
        (0.375, 0.0),
        (0.5, 0.125),
        (0.5, 0.375)
    ]
    
    def compute_eps_inv_fourier(cell):
        eps = np.where(cell == 1, eps_high, eps_low)
        inv_eps = 1.0 / eps
        fft_inv_eps = np.fft.fft2(inv_eps)
        fft_inv_eps /= (size * size)
        fft_inv_eps = np.fft.fftshift(fft_inv_eps)
        return fft_inv_eps

    def solve_bands(cell, kx, ky, num_bands=5):
        eps_inv_fft = compute_eps_inv_fourier(cell)
        k_vectors = pw_list + np.array([kx, ky])
        
        H = np.zeros((N_pw_actual, N_pw_actual), dtype=complex)
        # Precompute k_diff_sq and indices for speed? 
        # For N=11, 121x121 matrix. 14641 elements. 
        # Vectorized construction is better.
        
        # Create index arrays for k_vectors
        kx_arr = k_vectors[:, 0]
        ky_arr = k_vectors[:, 1]
        
        # Differences
        dkx = kx_arr[:, None] - kx_arr[None, :]
        dky = ky_arr[:, None] - ky_arr[None, :]
        dk_sq = dkx**2 + dky**2
        
        # Get coefficients
        # We need eps_inv_fft at indices (round(dkx), round(dky))
        # Since kx, ky are floats, dkx, dky are floats.
        # But k_vectors are integers + k. 
        # Wait, pw_list are integers. kx, ky are floats.
        # So dkx = (nx_i + kx) - (nx_j + kx) = nx_i - nx_j. Integer!
        # So we can just use integer indices.
        
        idx_x = np.round(dkx).astype(int)
        idx_y = np.round(dky).astype(int)
        
        # Map to FFT indices
        # fftshifted: index 0 is -N//2.
        # So fft_idx = idx + N//2.
        fft_idx_x = idx_x + N_pw//2
        fft_idx_y = idx_y + N_pw//2
        
        # Check bounds
        mask = (fft_idx_x >= 0) & (fft_idx_x < size) & (fft_idx_y >= 0) & (fft_idx_y < size)
        
        # Initialize coeff matrix with 0
        coeff_mat = np.zeros((N_pw_actual, N_pw_actual), dtype=complex)
        
        # Fill valid entries
        valid_x = fft_idx_x[mask]
        valid_y = fft_idx_y[mask]
        coeff_mat[mask] = eps_inv_fft[valid_y, valid_x]
        
        # H_ij = coeff_ij * dk_sq_ij
        H = coeff_mat * dk_sq
        
        # Solve
        vals, vecs = np.linalg.eigh(H)
        vals = vals[vals > 1e-6]
        
        if len(vals) < num_bands:
            return None, None
            
        return vals[:num_bands], vecs[:, :num_bands]

    def evaluate_gap(cell):
        max_omega_4 = 0
        min_omega_5 = float('inf')
        
        for kx, ky in k_points:
            vals, _ = solve_bands(cell, kx, ky, num_bands=5)
            if vals is None:
                continue
            omegas = np.sqrt(vals)
            max_omega_4 = max(max_omega_4, omegas[3])
            min_omega_5 = min(min_omega_5, omegas[4])
            
        if min_omega_5 == float('inf'):
            return -1.0
            
        gap = min_omega_5 - max_omega_4
        midgap = (min_omega_5 + max_omega_4) / 2.0
        if midgap == 0:
            return 0.0
        return gap / midgap

    # 3. Optimization
    # Objective function for scipy
    def objective(coeffs):
        if time.time() > deadline - 5:
            return 0.0
        cell = coeffs_to_cell(coeffs)
        score = evaluate_gap(cell)
        return -score  # Minimize negative score

    # Initial guess
    # Start with a symmetric pattern that often works well:
    # High dielectric in corners, low in center? Or vice versa?
    # Let's try a few initial points.
    
    initial_guesses = [
        [0.5, 0.0, 0.0, 0.0, 0.0],  # Uniform
        [0.5, 0.2, 0.0, 0.0, 0.0],  # Slight fundamental
        [0.5, -0.2, 0.0, 0.0, 0.0], # Inverted fundamental
        [0.5, 0.0, 0.2, 0.0, 0.0],  # Second harmonic
        [0.5, 0.0, -0.2, 0.0, 0.0],
        [0.5, 0.1, 0.1, 0.0, 0.0],
        [0.5, -0.1, 0.1, 0.0, 0.0],
    ]
    
    best_score = -1.0
    best_coeffs = initial_guesses[0]
    best_cell = coeffs_to_cell(best_coeffs)
    
    for guess in initial_guesses:
        if time.time() > deadline - 10:
            break
            
        # Use Nelder-Mead
        try:
            res = minimize(objective, guess, method='Nelder-Mead', 
                           options={'xatol': 1e-3, 'fatol': 1e-3, 'maxiter': 50})
            if res.fun < -best_score:
                best_score = -res.fun
                best_coeffs = res.x
                best_cell = coeffs_to_cell(best_coeffs)
        except Exception:
            pass

    # Local refinement on the best cell found by the parametric search
    # Use a few steps of the pixel-based hill climbing from the current program
    # to fix any pixel-level issues.
    
    cell = best_cell.copy()
    current_score = evaluate_gap(cell)
    
    # Simple local search: flip random pixels and check
    # But we must maintain symmetry.
    # Symmetry group C4v. Fundamental domain is 1/8 of the cell.
    # Size 32x32. Fundamental domain is roughly 4x16 or similar?
    # Actually, for 32x32, the fundamental domain under C4v is 1/8 of the area = 128 pixels.
    # We can iterate over the fundamental domain.
    
    # Define fundamental domain indices
    # 0 <= x <= 0.5, 0 <= y <= x
    # In grid indices:
    # i (row, y), j (col, x)
    # y = (i+0.5)/32, x = (j+0.5)/32
    # Condition: y <= x => i <= j
    # And x <= 0.5 => j <= 15 (since 15.5/32 < 0.5, 16.5/32 > 0.5)
    # Wait, 32/2 = 16. Pixels 0..15 are < 0.5.
    # So j in 0..15.
    # i in 0..j.
    
    fd_indices = []
    for j in range(16):
        for i in range(j + 1):
            fd_indices.append((i, j))
            
    # For each pixel in FD, try flipping it.
    # If it improves the score, keep it.
    
    for _ in range(20): # Max 20 passes over FD
        if time.time() > deadline - 10:
            break
            
        improved = False
        for i, j in fd_indices:
            if time.time() > deadline - 5:
                break
                
            # Try flipping
            # Original value
            orig_val = cell[i, j]
            new_val = 1 - orig_val
            
            # Apply flip to symmetric positions
            # Positions: (i,j), (j,i), (31-i, 31-j), (31-j, 31-i)
            # And their transposes?
            # The orbit of (i,j) under C4v:
            # (i, j)
            # (j, i)
            # (31-i, 31-j)
            # (31-j, 31-i)
            
            cell[i, j] = new_val
            cell[j, i] = new_val
            cell[31-i, 31-j] = new_val
            cell[31-j, 31-i] = new_val
            
            # Check if any of these are the same pixel (center or diagonal)
            # If i==j, then (i,j) and (j,i) are same.
            # If i==31-j, etc.
            # The assignment is idempotent for the same value, so it's safe.
            
            new_score = evaluate_gap(cell)
            
            if new_score > current_score:
                current_score = new_score
                best_score = max(best_score, current_score)
                improved = True
            else:
                # Revert
                cell[i, j] = orig_val
                cell[j, i] = orig_val
                cell[31-i, 31-j] = orig_val
                cell[31-j, 31-i] = orig_val
                
    # Final result
    # Ensure the cell is a list of lists of ints
    return cell.tolist()

