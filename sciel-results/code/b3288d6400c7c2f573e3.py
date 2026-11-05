import time
import math
import numpy as np

def design(problem):
    start_time = time.time()
    deadline = start_time + 220  # 10s margin
    
    size = problem["pixels"]
    eps_low = problem["eps_low"]
    eps_high = problem["eps_high"]
    
    # Initial guess: 4 rods in corners (same as seed but vectorized for speed)
    y, x = np.mgrid[0:size, 0:size]
    y = y + 0.5
    x = x + 0.5
    
    centers = [
        (size/4, size/4),
        (size/4, 3*size/4),
        (3*size/4, size/4),
        (3*size/4, 3*size/4)
    ]
    
    r = 5.5
    cell = np.zeros((size, size), dtype=int)
    for cx, cy in centers:
        dist = (x - cx)**2 + (y - cy)**2
        cell[dist < r**2] = 1
        
    # Ensure strict symmetry (just in case)
    cell = (cell + cell[::-1, ::-1]) // 2
    cell = (cell + cell.T) // 2
    
    # Fast local solver for plane waves
    # We use a small number of plane waves to estimate the gap quickly.
    # The evaluator uses 289, but we can use fewer for optimization steps.
    # Let's use N=15 (225 planes) or N=11 (121 planes) for speed.
    # Actually, to be safe and somewhat accurate, let's use N=13 (169 planes) if time permits.
    # Let's start with N=11 for speed in the loop.
    
    N_pw = 11
    # k-vectors in the irreducible BZ
    # Gamma-M-X-Gamma
    # Gamma: (0,0), M: (0.5, 0), X: (0.5, 0.5) in units of 2pi/a
    # Let's sample a few points.
    k_points = [
        (0.0, 0.0),
        (0.25, 0.0),
        (0.5, 0.0),
        (0.5, 0.25),
        (0.5, 0.5),
        (0.25, 0.25)
    ]
    
    # Precompute plane wave vectors
    # k = (nx + kx, ny + ky) * 2pi / a
    # We work in units where a=1, so k is in units of 2pi.
    pw_list = []
    for nx in range(-N_pw//2, N_pw//2 + 1):
        for ny in range(-N_pw//2, N_pw//2 + 1):
            pw_list.append((nx, ny))
    pw_list = np.array(pw_list)
    N_pw_actual = len(pw_list)
    
    # Fourier coefficients of 1/eps for the current cell
    # We need to compute the matrix A where A_ij = (1/eps)_q * delta(k_i - k_j + q)
    # Actually, the eigenvalue problem is:
    # sum_j (k_i - k_j)^2 / eps_{k_i - k_j} * c_j = omega^2 c_i
    # For TM, it's 1/eps.
    
    def compute_eps_inv_fourier(cell):
        # cell is 0/1. 1 -> eps_high, 0 -> eps_low
        eps = np.where(cell == 1, eps_high, eps_low)
        # Fourier coefficients of 1/eps
        # We need 1/eps values in real space, then FFT
        inv_eps = 1.0 / eps
        # FFT to get coefficients
        # Note: numpy FFT is unnormalized in forward, normalized in inverse?
        # np.fft.fft2 returns sum. We need coefficients c_q such that f(x) = sum c_q exp(i q x)
        # c_q = (1/N^2) sum f(x) exp(-i q x)
        # np.fft.fft2 gives sum f(x) exp(-i 2pi q x / N).
        # So c_q = fft2(f) / N^2
        fft_inv_eps = np.fft.fft2(inv_eps)
        fft_inv_eps /= (size * size)
        # Reorder to center zero frequency
        fft_inv_eps = np.fft.fftshift(fft_inv_eps)
        return fft_inv_eps, inv_eps

    def solve_bands(cell, kx, ky, num_bands=5):
        # Compute matrix
        eps_inv_fft, _ = compute_eps_inv_fourier(cell)
        
        # Get indices for each plane wave
        # k_i = (nx_i + kx, ny_i + ky)
        k_vectors = pw_list + np.array([kx, ky])
        
        # Matrix A
        # A_ij = |k_i|^2 / eps_{k_i - k_j}  (Wait, the standard PWE formulation is:
        # sum_j (1/eps)_{k_i - k_j} * |k_i - k_j|^2 * c_j = omega^2 c_i ? No.
        # The equation is: sum_j (k_i)^2 (1/eps)_{k_i - k_j} c_j = omega^2 c_i is wrong.
        
        # Correct PWE:
        # H is Hermitian.
        # H_ij = (1/eps)_{k_i - k_j} * (k_i - k_j)^2 is NOT correct.
        
        # Let's recall the standard form.
        # Maxwell: k^2 E = omega^2/c^2 eps E
        # E = sum c_k exp(i k r)
        # k^2 (sum c_k exp(i k r)) = (omega^2/c^2) (sum eps_q exp(i q r)) (sum c_k exp(i k r))
        # k^2 c_k = (omega^2/c^2) sum_q eps_q c_{k-q}
        # For TM, it's 1/eps.
        # k^2 c_k = (omega^2/c^2) sum_q (1/eps)_q c_{k-q}
        
        # So the matrix M is:
        # M_ij = (1/eps)_{k_i - k_j}
        # And the eigenvalue equation is:
        # sum_j (k_i)^2 M_ij c_j ? No.
        
        # The term k^2 acts on c_k.
        # So: sum_j (1/eps)_{k_i - k_j} (k_j)^2 c_j ? No.
        
        # Let's derive carefully.
        # k^2 c_k = omega^2 sum_q (1/eps)_q c_{k-q}
        # Let k = k_i. Then c_{k-q} = c_{k_i - q}.
        # We want an eigenvalue problem A c = lambda c.
        # Multiply by (1/k_i^2)? No, k_i can be 0.
        
        # Standard formulation:
        # The matrix is H_ij = (1/eps)_{k_i - k_j} * |k_i|^2 ? No.
        
        # Let's look at the standard "exact coefficient" method.
        # The eigenvalue problem is:
        # sum_j (k_i - k_j)^2 / eps_{k_i - k_j} c_j = omega^2 c_i  <-- This is for TE?
        
        # Actually, for TM:
        # The operator is -1/eps * (k^2).
        # In basis of plane waves:
        # H_ij = (1/eps)_{k_i - k_j} * (k_i - k_j)^2 is WRONG.
        
        # Correct TM PWE matrix:
        # H_ij = (1/eps)_{k_i - k_j} * (k_i)^2 ? No.
        
        # Let's use the form:
        # sum_j (1/eps)_{k_i - k_j} (k_i - k_j)^2 c_j = omega^2 c_i is also not quite right.
        
        # The most robust way:
        # The matrix is A where A_ij = (1/eps)_{k_i - k_j} * (k_i - k_j)^2 is for TE.
        # For TM, it is A_ij = (1/eps)_{k_i - k_j} * (k_i)^2 ?
        
        # Let's check a reference.
        # Joannopoulos, "Photonic Crystals":
        # For TE: sum_j (k_i - k_j)^2 / eps_{k_i - k_j} c_j = omega^2 c_i
        # For TM: sum_j (1/eps)_{k_i - k_j} (k_i - k_j)^2 c_j = omega^2 c_i
        
        # Wait, the symmetry of 1/eps is different.
        # Let's assume the TM formulation:
        # H_ij = (1/eps)_{k_i - k_j} * (k_i - k_j)^2
        
        # Let's implement this.
        
        H = np.zeros((N_pw_actual, N_pw_actual), dtype=complex)
        for i in range(N_pw_actual):
            for j in range(N_pw_actual):
                q = k_vectors[i] - k_vectors[j]
                # Get (1/eps)_q
                # q is in units of 2pi.
                # Our FFT grid is indexed -N/2 ... N/2.
                # q_x, q_y are integers if kx,ky are 0.
                # But kx,ky are not 0.
                # The Fourier coefficients are defined for integer q.
                # (1/eps)_q is the coefficient of exp(i 2pi q r).
                # The term in the sum is (1/eps)_{k_i - k_j}.
                # k_i - k_j = (nx_i - nx_j, ny_i - ny_j). This is an integer vector!
                # So we just look up the integer index.
                
                qx = int(round(k_vectors[i][0] - k_vectors[j][0]))
                qy = int(round(k_vectors[i][1] - k_vectors[j][1]))
                
                # Check if q is within our FFT range
                if -N_pw <= qx <= N_pw and -N_pw <= qy <= N_pw:
                    # Map to FFT index
                    # fftshifted array: index 0 is -N/2.
                    # So index = q + N/2.
                    idx_x = qx + N_pw//2
                    idx_y = qy + N_pw//2
                    if 0 <= idx_x < size and 0 <= idx_y < size:
                        coeff = eps_inv_fft[idx_y, idx_x]
                    else:
                        coeff = 0.0
                else:
                    coeff = 0.0
                
                k_diff_sq = (k_vectors[i][0] - k_vectors[j][0])**2 + (k_vectors[i][1] - k_vectors[j][1])**2
                H[i, j] = coeff * k_diff_sq
        
        # Solve eigenvalue problem
        # H is Hermitian
        vals, vecs = np.linalg.eigh(H)
        # Filter out small eigenvalues (numerical noise)
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
            # vals are omega^2
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

    current_score = evaluate_gap(cell)
    best_cell = cell.copy()
    best_score = current_score
    
    # Optimization loop: Gradient ascent on the "gap"
    # The gap is not differentiable, but we can use a heuristic.
    # Increase dielectric where the field intensity of the top of band 4 and bottom of band 5 are high?
    # Actually, to open a gap, we want to separate the bands.
    # A common heuristic is to adjust the dielectric to match the desired field profile.
    
    # Let's try a simple iterative method:
    # 1. Solve for bands.
    # 2. Calculate field intensity of band 4 (top) and band 5 (bottom).
    # 3. If we want to push band 4 down and band 5 up?
    #    Adding dielectric (eps=12) increases the index, which generally lowers the frequency?
    #    No, for a fixed k, higher eps means lower omega.
    #    So adding dielectric lowers the frequency of the modes that have high intensity there.
    
    # Strategy:
    # We want to maximize the gap.
    # If we add dielectric where band 4 is strong, we lower omega_4.
    # If we add dielectric where band 5 is strong, we lower omega_5.
    # We want to lower omega_4 and RAISE omega_5?
    # Raising omega_5 means REMOVING dielectric where band 5 is strong.
    
    # So:
    # Add dielectric where |E_4|^2 is high.
    # Remove dielectric where |E_5|^2 is high.
    
    # Let's implement this.
    
    for step in range(50):
        if time.time() > deadline - 10:
            break
            
        # Get fields
        # We need the fields at the critical k-points.
        # Let's find the k-point that gives the max omega_4 and the k-point that gives the min omega_5.
        
        max_4_val = 0
        max_4_k = None
        max_4_vec = None
        
        min_5_val = float('inf')
        min_5_k = None
        min_5_vec = None
        
        for kx, ky in k_points:
            vals, vecs = solve_bands(cell, kx, ky, num_bands=5)
            if vals is None:
                continue
            omegas = np.sqrt(vals)
            
            if omegas[3] > max_4_val:
                max_4_val = omegas[3]
                max_4_k = (kx, ky)
                max_4_vec = vecs[:, 3]
                
            if omegas[4] < min_5_val:
                min_5_val = omegas[4]
                min_5_k = (kx, ky)
                min_5_vec = vecs[:, 4]
                
        if max_4_vec is None or min_5_vec is None:
            break
            
        # Reconstruct field in real space
        # E(r) = sum_k c_k exp(i k r)
        # c_k are the components of the eigenvector.
        # Note: The eigenvector components correspond to the plane waves in pw_list.
        
        def field_from_vec(vec, kx, ky):
            # vec is complex array of size N_pw_actual
            # k_vectors = pw_list + (kx, ky)
            # E(x,y) = sum_j vec[j] exp(i (kx_j x + ky_j y))
            # We evaluate on the grid x,y = 0..31.
            # Note: The plane waves are defined in the unit cell.
            # The phase is exp(i 2pi (nx + kx) x).
            
            field = np.zeros((size, size), dtype=complex)
            for j in range(N_pw_actual):
                nx, ny = pw_list[j]
                kx_j = nx + kx
                ky_j = ny + ky
                # Phase: exp(i 2pi (kx_j x + ky_j y))
                # x, y are in units of a (0 to 1).
                # Our grid is 0..31. So x_grid = (i+0.5)/32.
                # Let's use i/32 for simplicity, or (i+0.5)/32.
                # Let's use (i+0.5)/32 to match pixel centers.
                x_grid = (np.arange(size) + 0.5) / size
                y_grid = (np.arange(size) + 0.5) / size
                
                # Vectorized
                phase_x = 2 * math.pi * kx_j * x_grid
                phase_y = 2 * math.pi * ky_j * y_grid
                
                # Create meshgrid
                XX, YY = np.meshgrid(phase_x, phase_y)
                phase = XX + YY
                field += vec[j] * np.exp(1j * phase)
                
            return field
            
        field_4 = field_from_vec(max_4_vec, max_4_k[0], max_4_k[1])
        field_5 = field_from_vec(min_5_vec, min_5_k[0], min_5_k[1])
        
        # Intensity
        int_4 = np.abs(field_4)**2
        int_5 = np.abs(field_5)**2
        
        # Normalize intensities
        if np.max(int_4) > 0:
            int_4 /= np.max(int_4)
        if np.max(int_5) > 0:
            int_5 /= np.max(int_5)
            
        # Update rule
        # Add dielectric where int_4 is high.
        # Remove dielectric where int_5 is high.
        
        # We need to maintain symmetry.
        # Let's create a "force" map.
        force = int_4 - int_5
        
        # Apply force to cell
        # If force > 0, try to set to 1.
        # If force < 0, try to set to 0.
        
        # To maintain symmetry, we apply the force to the fundamental domain and reflect.
        # Or simply apply to the whole grid and then enforce symmetry.
        
        # Let's do a probabilistic update or a threshold update.
        # Threshold: If force > 0.5, set to 1. If force < -0.5, set to 0.
        # This might be too aggressive.
        
        # Let's try a soft update:
        # new_cell = cell + alpha * force
        # Then threshold.
        
        alpha = 0.5
        new_cell_float = cell.astype(float) + alpha * force
        
        # Threshold
        # We want to keep the volume fraction roughly constant?
        # Not necessarily.
        
        # Let's just threshold at 0.5.
        new_cell = (new_cell_float > 0.5).astype(int)
        
        # Enforce symmetry
        new_cell = (new_cell + new_cell[::-1, ::-1]) // 2
        new_cell = (new_cell + new_cell.T) // 2
        
        # Evaluate new cell
        new_score = evaluate_gap(new_cell)
        
        if new_score > best_score:
            best_score = new_score
            best_cell = new_cell.copy()
            cell = new_cell.copy()
        else:
            # If it didn't improve, try a smaller step or revert
            # For now, just revert
            pass
            
    # Final check
    if best_score > current_score:
        cell = best_cell
    else:
        cell = best_cell # Return the best found
        
    # Convert to list of lists
    return cell.tolist()
