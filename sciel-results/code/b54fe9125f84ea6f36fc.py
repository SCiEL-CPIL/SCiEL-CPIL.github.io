import numpy as np
from scipy.optimize import minimize
import time
import sys

# EVOLVE-BLOCK-START

def _generate_cell(w_out, w_in):
    """
    Generate a 32x32 symmetric cell.
    1 = dielectric, 0 = air.
    w_out: width of the outer dielectric square (centered).
    w_in: width of the inner air square (centered).
    """
    size = 32
    # Convert to integer widths to ensure symmetry
    # The center is between 15 and 16.
    # A square of width W centered at 16 covers [16 - W/2, 16 + W/2).
    # Indices are 0..31.
    # For symmetry, W should be even.
    w_out = int(round(w_out))
    w_in = int(round(w_in))
    
    # Ensure valid ranges
    w_out = max(0, min(32, w_out))
    w_in = max(0, min(w_out, w_in))
    
    cell = np.zeros((size, size), dtype=int)
    
    # Define the dielectric region: A square of size w_out
    # Start index for dielectric
    start_out = 16 - w_out // 2
    end_out = start_out + w_out
    # Clip to valid indices
    start_out = max(0, start_out)
    end_out = min(32, end_out)
    
    # Fill with 1
    cell[start_out:end_out, start_out:end_out] = 1
    
    # Define the air hole region: A square of size w_in
    if w_in > 0:
        start_in = 16 - w_in // 2
        end_in = start_in + w_in
        start_in = max(0, start_in)
        end_in = min(32, end_in)
        cell[start_in:end_in, start_in:end_in] = 0
        
    return cell

def _evaluate_bands(cell):
    """
    Compute the TM band gap ratio for the 4th and 5th bands.
    Uses a plane-wave method.
    """
    size = 32
    eps_low = 1.0
    eps_high = 12.0
    
    # Fourier coefficients of 1/epsilon
    # For TM modes, the operator involves 1/eps.
    # We need the Fourier series of 1/eps(x,y).
    # 1/eps is 1 where cell is 0, and 1/12 where cell is 1.
    
    eps_inv = np.zeros((size, size))
    eps_inv[cell == 0] = 1.0
    eps_inv[cell == 1] = 1.0 / eps_high
    
    # Compute Fourier coefficients
    # N_pw = 289 implies a 17x17 grid of G vectors.
    # G vectors: (m, n) where m, n in [-8, 8].
    G_max = 8
    G_vectors = []
    for m in range(-G_Max, G_Max + 1):
        for n in range(-G_Max, G_Max + 1):
            G_vectors.append((m, n))
            
    N_pw = len(G_vectors)
    
    # Fourier coefficients c_{mn} = 1/L^2 sum_{x,y} f(x,y) exp(-i 2pi (mx/L + ny/L))
    # Here L=1 (normalized). x, y are in [0,1).
    # Pixel centers: (i+0.5)/size, (j+0.5)/size.
    
    # Precompute exponentials
    # f(x,y) = sum c_{mn} exp(i 2pi (mx+ny))
    # c_{mn} = sum_{i,j} eps_inv[i,j] * exp(-i 2pi (m(i+0.5)/size + n(j+0.5)/size)) / size^2
    
    # Vectorized computation of c
    # x = (i+0.5)/size, y = (j+0.5)/size
    i_idx = np.arange(size)
    j_idx = np.arange(size)
    x_coords = (i_idx + 0.5) / size
    y_coords = (j_idx + 0.5) / size
    
    # Create meshgrid for x, y
    X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
    
    # c_{mn}
    # We can compute this for all m, n at once using matrix multiplication or broadcasting.
    # exp(-i 2pi m X) is a matrix of size (size, size) for each m.
    # Let's just loop over G vectors, it's only 289.
    
    c_coeffs = np.zeros(N_pw, dtype=complex)
    for k, (m, n) in enumerate(G_vectors):
        # exp(-i 2pi (m X + n Y))
        phase = np.exp(-1j * 2 * np.pi * (m * X + n * Y))
        c_coeffs[k] = np.sum(eps_inv * phase) / (size * size)
        
    # Now solve the eigenvalue problem for each k-point.
    # The TM eigenvalue equation:
    # sum_N' [ (G + k) x (G' + k) - delta_{GG'} (G+k)^2 ] / eps_{G-G'} * A_{N'} = omega^2 A_N
    # Actually, the standard form is:
    # sum_N' [ (G_N + k) \cdot (G_N' + k) - (G_N + k)^2 \delta_{NN'} ] c_{N-N'} A_{N'} = omega^2 A_N
    # Wait, the operator is:
    # ( (G+k) x (G+k) ) / eps_{G-G'}
    # For TM, the field is scalar. The equation is:
    # sum_N' [ (G_N + k) \cdot (G_N' + k) - (G_N + k)^2 \delta_{NN'} ] c_{N-N'} A_{N'} = omega^2 A_N
    # This is a generalized eigenvalue problem or can be transformed.
    # Actually, it's a standard eigenvalue problem for the matrix M:
    # M_{NN'} = [ (G_N + k) \cdot (G_N' + k) - (G_N + k)^2 \delta_{NN'} ] c_{N-N'}
    # Note: c_{N-N'} is the Fourier coefficient of 1/eps.
    # We need c_{m,n} for differences in G indices.
    # G_N = (2pi m_N, 2pi n_N).
    # Difference G_N - G_N' = (2pi (m_N - m_N'), 2pi (n_N - n_N')).
    # We have c_coeffs for indices -8..8. The difference can be -16..16.
    # We need to compute c for differences up to 16.
    # Let's recompute c_coeffs for a larger range if needed, or note that c_{m,n} for |m|,|n| > 8 might be needed.
    # G_vectors range: -8 to 8. Diff range: -16 to 16.
    # We only computed up to 8. We need up to 16.
    # Let's recompute c for range -16 to 16.
    
    G_max_diff = 16
    c_diff = {}
    for m in range(-G_Max_diff, G_Max_diff + 1):
        for n in range(-G_Max_diff, G_Max_diff + 1):
            phase = np.exp(-1j * 2 * np.pi * (m * X + n * Y))
            c_diff[(m, n)] = np.sum(eps_inv * phase) / (size * size)
            
    # K-points in the irreducible Brillouin zone.
    # Square BZ: kx, ky in [-pi, pi].
    # Irreducible: 0 <= ky <= kx <= pi.
    # Boundary: (0,0), (pi,0), (pi,pi), (pi/2, 0), (pi/2, pi/2), etc.
    # The prompt says "30 wave vectors covering the boundary and interior".
    # Let's generate a grid of 30 points.
    
    k_points = []
    # Boundary points
    k_points.append((0, 0))
    k_points.append((np.pi, 0))
    k_points.append((np.pi, np.pi))
    k_points.append((np.pi/2, 0))
    k_points.append((np.pi/2, np.pi/2))
    k_points.append((np.pi, np.pi/2))
    
    # Interior points (grid)
    # Let's just add more points along the lines and in the triangle.
    # To be safe and fast, we'll use a fixed set of 30 points.
    # 5x5 grid in the triangle 0<=ky<=kx<=pi?
    # Total points in triangle: ~15.
    # Let's just use a dense grid of 30 points.
    
    # Generate 30 points
    # 10 points on M-X line (pi,0) to (pi/2,0) ? No, M is (pi,pi), X is (pi,0).
    # Gamma (0,0), X (pi,0), M (pi,pi).
    # Path Gamma-X-M-Gamma.
    # 10 points Gamma->X
    for i in range(10):
        t = i / 9.0
        k_points.append((t * np.pi, 0))
    # 10 points X->M
    for i in range(1, 10):
        t = i / 9.0
        k_points.append((np.pi, t * np.pi))
    # 10 points M->Gamma
    for i in range(1, 10):
        t = i / 9.0
        k_points.append((t * np.pi, t * np.pi))
        
    # Remove duplicates
    unique_k = []
    for k in k_points:
        if k not in unique_k:
            unique_k.append(k)
            
    # We need the lowest 5 bands at each k-point.
    min_omega_5 = float('inf')
    max_omega_4 = 0.0
    
    # Precompute G vectors as a matrix
    G_mat = np.array(G_vectors) * 2 * np.pi
    
    for kx, ky in unique_k:
        k_vec = np.array([kx, ky])
        
        # Construct the matrix M
        # M_{ij} = ( (G_i + k) dot (G_j + k) - (G_i + k)^2 delta_ij ) * c_{G_i - G_j}
        
        # (G_i + k)
        Gk = G_mat + k_vec
        
        # Dot products
        # Gk_dot = Gk @ Gk.T  (N x N)
        # Gk_sq = np.sum(Gk**2, axis=1) (N,)
        
        Gk_dot = np.dot(Gk, Gk.T)
        Gk_sq = np.sum(Gk**2, axis=1)
        
        # Matrix of differences
        # diff_m = m_i - m_j
        # diff_n = n_i - n_j
        m_vals = G_mat[:, 0] / (2 * np.pi)
        n_vals = G_mat[:, 1] / (2 * np.pi)
        
        diff_m = m_vals[:, None] - m_vals[None, :]
        diff_n = n_vals[:, None] - n_vals[None, :]
        
        # Get c coefficients
        # We need c_{diff_m, diff_n}
        # c_diff is a dict. Vectorizing dict lookup is slow.
        # Let's create a lookup matrix for c_diff.
        # Range -16 to 16.
        # Index mapping: idx = (val + 16)
        
        c_lookup = np.zeros((33, 33), dtype=complex)
        for m in range(-16, 17):
            for n in range(-16, 17):
                c_lookup[m + 16, n + 16] = c_diff.get((m, n), 0)
                
        c_mat = c_lookup[diff_m + 16, diff_n + 16]
        
        # M = (Gk_dot - np.diag(Gk_sq)) * c_mat
        # Note: The term is (G_i+k) dot (G_j+k) - (G_i+k)^2 delta_ij
        # This is exactly Gk_dot - diag(Gk_sq)
        
        M = (Gk_dot - np.diag(Gk_sq)) * c_mat
        
        # Solve eigenvalue problem
        # M is Hermitian?
        # c_{-m,-n} = c_{m,n}^*
        # M_{ji} = (G_j+k dot G_i+k - ...) c_{G_j-G_i}
        # M_{ij}^* = (G_i+k dot G_j+k - ...) c_{G_i-G_j}^* = (...) c_{G_j-G_i}
        # So M is Hermitian.
        
        # Use eigh
        try:
            vals, _ = np.linalg.eigh(M)
            # Filter real parts
            vals = np.real(vals)
            # Sort
            vals = np.sort(vals)
            # Take 5 lowest positive eigenvalues?
            # The bands are indexed 1, 2, 3, 4, 5.
            # Usually the lowest eigenvalues correspond to the lowest bands.
            # We need omega_4 and omega_5.
            # omega^2 = eigenvalue.
            # So omega = sqrt(eigenvalue).
            
            if len(vals) >= 5:
                # Ensure positive
                vals_pos = vals[vals > 1e-6]
                if len(vals_pos) >= 5:
                    o4 = np.sqrt(vals_pos[3])
                    o5 = np.sqrt(vals_pos[4])
                    
                    if o4 < max_omega_4:
                        max_omega_4 = o4
                    if o5 < min_omega_5:
                        min_omega_5 = o5
        except Exception:
            pass
            
    if min_omega_5 == float('inf') or max_omega_4 == 0:
        return -1.0
        
    gap = 2 * (min_omega_5 - max_omega_4)
    midgap = min_omega_5 + max_omega_4
    ratio = gap / midgap
    return ratio

def design(problem):
    start_time = time.time()
    deadline = start_time + 100.0 # 100s safety margin
    
    best_score = -1.0
    best_cell = None
    
    # Initial guesses for (w_out, w_in)
    # w_out: 0-32, w_in: 0-w_out
    # Symmetry: even numbers preferred.
    
    candidates = [
        (20, 0),   # Large rod
        (12, 0),   # Medium rod
        (8, 0),    # Small rod
        (32, 16),  # Frame
        (24, 8),   # Cross-like / Thick frame
        (28, 12),  # Another frame
        (16, 4),   # Rod with hole
        (20, 4),   # Rod with small hole
    ]
    
    # Evaluate initial candidates
    for w_out, w_in in candidates:
        cell = _generate_cell(w_out, w_in)
        score = _evaluate_bands(cell)
        if score > best_score:
            best_score = score
            best_cell = cell
            
    # Local optimization
    # Use Nelder-Mead on the best few candidates
    for w_out, w_in in candidates:
        if time.time() > deadline - 20:
            break
            
        # Only optimize if it's a promising region
        # We'll just optimize the best one found so far, and maybe a couple others.
        
        # Define objective function to minimize (negative score)
        def objective(params):
            w_out_opt = params[0]
            w_in_opt = params[1]
            
            # Bounds
            if w_out_opt < 0 or w_out_opt > 32 or w_in_opt < 0 or w_in_opt > w_out_opt:
                return 1.0 # Bad score
            
            cell = _generate_cell(w_out_opt, w_in_opt)
            score = _evaluate_bands(cell)
            return -score
        
        # Initial guess
        x0 = np.array([w_out, w_in])
        
        try:
            res = minimize(objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 50, 'xatol': 1e-4, 'fatol': 1e-4})
            
            if res.success:
                w_out_opt, w_in_opt = res.x
                cell = _generate_cell(w_out_opt, w_in_opt)
                score = _evaluate_bands(cell)
                if score > best_score:
                    best_score = score
                    best_cell = cell
        except Exception:
            pass
            
    # If time remains, try a random search in the parameter space
    if best_cell is None:
        best_cell = _generate_cell(20, 0)
        
    # Convert numpy array to nested list
    return best_cell.tolist()

# EVOLVE-BLOCK-END
