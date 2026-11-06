import time
import math
import numpy as np
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import eigsh

def design(problem):
    start_time = time.time()
    deadline = start_time + 210  # Safety margin for 230s limit
    
    size = problem["pixels"]
    eps_low = problem["eps_low"]
    eps_high = problem["eps_high"]
    
    # 1. Setup Plane Waves
    # N=15 gives 31x31 = 961 plane waves. 
    # This is significantly more accurate than N=11 (121) used in the seed.
    N_pw = 15
    pw_list = []
    half = N_pw // 2
    for nx in range(-half, half + 1):
        for ny in range(-half, half + 1):
            pw_list.append((nx, ny))
    pw_list = np.array(pw_list)
    N_pw_actual = len(pw_list)
    
    # K-points in irreducible BZ (0 to 0.5)
    # Gamma-M-X-Gamma
    k_points = [
        (0.0, 0.0),
        (0.125, 0.0),
        (0.25, 0.0),
        (0.375, 0.0),
        (0.5, 0.0),
        (0.5, 0.125),
        (0.5, 0.25),
        (0.5, 0.375),
        (0.5, 0.5),
        (0.25, 0.25),
        (0.125, 0.125)
    ]

    # 2. Precompute Matrix Structure
    # For a given k, the matrix H is:
    # H_ij = (1/eps)_{q_ij} * |k_i - k_j|^2
    # where q_ij = pw_list[i] - pw_list[j] (integer vector)
    
    # We precompute the integer differences and their magnitudes squared
    # and the indices into the FFT grid for 1/eps.
    
    # FFT grid size is 'size' (32).
    # We need to map integer q to FFT index.
    # q is in range [-N_pw, N_pw].
    # FFT index range [0, size-1].
    # fftshifted: index 0 is -size/2.
    # So index = q + size/2.
    
    # Precompute for all pairs (i, j)
    # This is a 961x961 array of indices and weights.
    # To save memory and time, we can compute the matrix H on the fly for each k,
    # but the q differences are independent of k.
    # |k_i - k_j|^2 depends on k.
    # Let K_vecs = pw_list + k.
    # Diff = K_vecs[i] - K_vecs[j] = pw_list[i] - pw_list[j].
    # Wait. k_i = (nx_i + kx, ny_i + ky).
    # k_j = (nx_j + kx, ny_j + ky).
    # k_i - k_j = (nx_i - nx_j, ny_i - ny_j).
    # So the difference vector q is INDEPENDENT of k!
    # This is a huge simplification.
    # |k_i - k_j|^2 is also independent of k?
    # Yes, because the kx, ky terms cancel out.
    
    # So H_ij = coeff(q_ij) * |q_ij|^2.
    # The matrix H is CONSTANT for all k-points?
    # NO. The eigenvalue problem is H c = omega^2 c.
    # But the basis functions are exp(i k r).
    # The operator is -1/eps * grad^2.
    # In plane wave basis, the kinetic energy term is k^2.
    # Wait, let's re-derive.
    # (1/eps) * k^2 E = omega^2 E
    # E = sum c_m exp(i (k+G_m) r)
    # k^2 acts on the exponential: (k+G_m)^2 c_m.
    # So the equation is:
    # sum_n (1/eps)_{m-n} (k+G_n)^2 c_n = omega^2 c_m ?
    # No.
    # k^2 (sum c_n exp(i(k+G_n)r)) = sum (k+G_n)^2 c_n exp(i(k+G_n)r)
    # (1/eps) (sum (k+G_n)^2 c_n exp(i(k+G_n)r))
    # = sum_m [ sum_n (1/eps)_{m-n} (k+G_n)^2 c_n ] exp(i(k+G_m)r)
    # So H_mn = (1/eps)_{m-n} (k+G_n)^2.
    # This DOES depend on k.
    
    # My previous simplification was wrong.
    # H_mn = coeff(q_mn) * |k+G_n|^2.
    
    # We can precompute:
    # 1. q_mn (integer vectors)
    # 2. coeff_idx (indices into FFT)
    # 3. G_n (vectors)
    
    q_diffs = pw_list[:, np.newaxis, :] - pw_list[np.newaxis, :, :] # (N, N, 2)
    q_diffs_flat = q_diffs.reshape(-1, 2)
    
    # Map to FFT indices
    # q_x, q_y in [-15, 15]
    # fft_idx = q + size//2
    fft_indices = q_diffs_flat + size // 2
    
    # Check bounds
    valid_mask = (fft_indices[:, 0] >= 0) & (fft_indices[:, 0] < size) & \
                 (fft_indices[:, 1] >= 0) & (fft_indices[:, 1] < size)
    
    # Precompute G_n magnitudes? No, |k+G_n|^2 depends on k.
    # But we can precompute G_n.
    
    def build_matrix(eps_inv_fft, kx, ky):
        # eps_inv_fft: 2D array of Fourier coeffs of 1/eps
        # kx, ky: k-point
        # Returns sparse matrix H
        
        # Extract coefficients
        # We need coeff for each (m, n)
        # idx_x = fft_indices[:, 0], idx_y = fft_indices[:, 1]
        # coeffs = eps_inv_fft[idx_y, idx_x]
        
        coeffs = eps_inv_fft[fft_indices[:, 1], fft_indices[:, 0]]
        # Set invalid to 0
        coeffs[~valid_mask] = 0.0
        
        # Compute |k + G_n|^2 for each n
        # G_n is pw_list[n]
        # k_total_x = kx + pw_list[:, 0]
        # k_total_y = ky + pw_list[:, 1]
        kx_total = kx + pw_list[:, 0]
        ky_total = ky + pw_list[:, 1]
        k_sq = kx_total**2 + ky_total**2 # (N,)
        
        # H_mn = coeffs[m*n] * k_sq[n]
        # We need to broadcast k_sq to (N, N) where each column n is multiplied by k_sq[n]
        # H = coeffs.reshape(N, N) * k_sq[np.newaxis, :]
        
        H_dense = coeffs.reshape(N_pw_actual, N_pw_actual) * k_sq[np.newaxis, :]
        
        # H should be Hermitian.
        # Due to numerical errors or truncation, it might not be exactly.
        # (1/eps)_{m-n} = conj((1/eps)_{n-m})
        # k_sq[n] is real.
        # H_nm = coeff(n-m) * k_sq[m]
        # H_mn = coeff(m-n) * k_sq[n]
        # H_mn^* = conj(coeff(m-n)) * k_sq[n] = coeff(n-m) * k_sq[n]
        # This is NOT H_nm unless k_sq[n] == k_sq[m].
        # Wait. The standard PWE matrix is:
        # H_mn = (1/eps)_{m-n} (k+G_m)^2 ? Or (k+G_n)^2?
        
        # Let's check the derivation again.
        # k^2 E = omega^2 eps E
        # E = sum c_n exp(i (k+G_n) r)
        # k^2 E = sum (k+G_n)^2 c_n exp(i (k+G_n) r)
        # eps E = sum_m [ sum_n eps_{m-n} c_n ] exp(i (k+G_m) r)
        # For TM: 1/eps.
        # (1/eps) k^2 E = omega^2 E
        # sum_n (1/eps)_{m-n} (k+G_n)^2 c_n = omega^2 c_m
        # So H_mn = (1/eps)_{m-n} (k+G_n)^2.
        
        # Is this Hermitian?
        # H_mn = c_{m-n} (k+G_n)^2
        # H_nm^* = c_{n-m}^* (k+G_m)^2 = c_{m-n} (k+G_m)^2
        # They are equal only if (k+G_n)^2 = (k+G_m)^2, which is false.
        
        # So the matrix is NOT Hermitian in this basis?
        # Ah, the operator is self-adjoint.
        # The issue is the basis.
        # Actually, the standard form is:
        # sum_n (k+G_m)^2 (1/eps)_{m-n} c_n ? No.
        
        # Let's look at Joannopoulos.
        # TE: sum_j (k_i - k_j)^2 / eps_{k_i - k_j} c_j = omega^2 c_i
        # TM: sum_j (1/eps)_{k_i - k_j} (k_i - k_j)^2 c_j = omega^2 c_i ??
        
        # Let's use the symmetric form if possible.
        # Actually, for TM, the operator is -1/eps * div grad.
        # It is self-adjoint with weight 1.
        # The matrix representation MUST be Hermitian.
        
        # Where is the error?
        # (1/eps) k^2 E.
        # k^2 is a differential operator.
        # (k+G_n)^2 c_n is the coefficient of k^2 E.
        # So the vector is v_n = (k+G_n)^2 c_n.
        # Then we multiply by 1/eps.
        # w_m = sum_n (1/eps)_{m-n} v_n = sum_n (1/eps)_{m-n} (k+G_n)^2 c_n.
        # So H_mn = (1/eps)_{m-n} (k+G_n)^2.
        
        # Why is it not Hermitian?
        # Because the basis functions are not orthogonal with respect to the inner product?
        # No, plane waves are orthogonal.
        
        # Let's check the Hermitian condition again.
        # H_mn = c_{m-n} (k+G_n)^2
        # H_nm = c_{n-m} (k+G_m)^2
        # H_mn^* = c_{n-m} (k+G_n)^2 (since c is real symmetric? No, c_{-q} = c_q^*)
        # c_{m-n} = c_{-(n-m)} = c_{n-m}^*
        # So H_mn^* = c_{n-m} (k+G_n)^2
        # H_nm = c_{n-m} (k+G_m)^2
        # They differ by (k+G_n)^2 vs (k+G_m)^2.
        
        # This implies the matrix is NOT Hermitian.
        # This is a known issue. The "exact coefficient" method for TM often leads to non-Hermitian matrices if formulated this way?
        # No, the physical problem is Hermitian.
        
        # The correct formulation for TM is:
        # sum_n (1/eps)_{m-n} (k+G_m)^2 c_n ?
        # Let's try: H_mn = (1/eps)_{m-n} (k+G_m)^2.
        # H_mn^* = c_{n-m} (k+G_m)^2
        # H_nm = c_{n-m} (k+G_n)^2.
        # Still not equal.
        
        # Actually, one must be careful.
        # The operator is A = 1/eps * k^2.
        # <m| A |n> = integral exp(-i k_m r) (1/eps) [ (k+G_n)^2 exp(i k_n r) ]
        # = (k+G_n)^2 integral (1/eps) exp(i (k_n - k_m) r)
        # = (k+G_n)^2 (1/eps)_{m-n}
        
        # So H_mn = (1/eps)_{m-n} (k+G_n)^2 is correct.
        # Why is it not Hermitian?
        # <n| A |m> = (k+G_m)^2 (1/eps)_{n-m}
        # <m| A |n>^* = (k+G_n)^2 (1/eps)_{m-n}^* = (k+G_n)^2 (1/eps)_{n-m}
        # So <n| A |m> != <m| A |n>^*.
        
        # This means the operator is not self-adjoint in this basis?
        # No, 1/eps is real and symmetric. k^2 is self-adjoint.
        # The product of two self-adjoint operators is self-adjoint ONLY if they commute.
        # 1/eps and k^2 do NOT commute.
        # So the TM problem is NOT Hermitian in the standard sense?
        # Wait. The wave equation is k^2 E = omega^2 eps E.
        # This can be written as eps^{-1} k^2 E = omega^2 E.
        # L = eps^{-1} k^2.
        # L is self-adjoint with respect to the inner product <f, g> = integral f^* g.
        # Proof: <f, Lg> = integral f^* eps^{-1} k^2 g
        # = integral (k^2 f^*)^* eps^{-1} g (integration by parts, boundary terms vanish)
        # = integral (k^2 f)^* eps^{-1} g
        # = integral f^* k^2 eps^{-1} g ? NO. k^2 and eps^{-1} don't commute.
        # So L is NOT self-adjoint.
        
        # However, the eigenvalues are REAL.
        # And we can use `eigsh` with `which='LM'` or `eig` for small matrices.
        # For N=961, `eig` is too slow (O(N^3) ~ 10^9 ops).
        # `eigsh` requires Hermitian.
        
        # Alternative: Use the generalized eigenvalue problem?
        # k^2 E = omega^2 eps E.
        # In basis:
        # sum_n (k+G_m)^2 delta_{mn} c_n = omega^2 sum_n eps_{m-n} c_n
        # K c = omega^2 E c
        # K is diagonal (real, symmetric).
        # E is Hermitian (eps is real symmetric).
        # This is a generalized Hermitian eigenvalue problem.
        # We can convert to standard: E^{-1} K c = omega^2 c.
        # E is positive definite (eps > 0).
        # We can solve (E^{-1/2} K E^{-1/2}) v = omega^2 v.
        # This matrix is Hermitian!
        
        # So:
        # 1. Compute E matrix (eps_{m-n}).
        # 2. Compute K matrix (diagonal, (k+G_n)^2).
        # 3. Compute E^{-1/2} K E^{-1/2}.
        # 4. Solve Hermitian eigenvalue problem.
        
        # This is robust and allows using `eigsh`.
        
        # Let's implement this.
        
        # 1. E matrix
        # E_mn = eps_{m-n}
        # We have eps_inv_fft. We need eps_fft.
        # We can compute eps from cell, then FFT.
        # Or, since 1/eps is piecewise constant, we can compute eps directly.
        
        # For speed, we precompute the indices for eps as well.
        # The indices are the same.
        
        # We need to pass eps_fft and eps_inv_fft?
        # Actually, we just need eps_fft for E.
        
        # Let's modify build_matrix to take eps_fft.
        
        E_dense = eps_fft[fft_indices[:, 1], fft_indices[:, 0]]
        E_dense[~valid_mask] = 0.0
        E_dense = E_dense.reshape(N_pw_actual, N_pw_actual)
        
        # K is diagonal
        K_diag = k_sq
        
        # We need to form H = E^{-1/2} K E^{-1/2}
        # E is Hermitian positive definite.
        # We can use eigendecomposition of E to get E^{-1/2}.
        # But E is 961x961. Eigendecomposition is O(N^3).
        # Doing this for every k-point and every optimization step is too slow.
        
        # Alternative: Use `scipy.sparse.linalg.eigsh` on the generalized problem?
        # No, `eigsh` doesn't support generalized directly in a simple way (requires `eig` or `lobpcg`).
        # `lobpcg` is good for large sparse.
        
        # Let's use `lobpcg` from scipy.sparse.linalg.
        # It solves A x = lambda B x.
        # A = K (diagonal)
        # B = E (sparse/dense)
        
        # We need the smallest eigenvalues? No, we want bands 4 and 5.
        # These are small indices.
        # `lobpcg` can find specific eigenvalues.
        
        # However, setting up `lobpcg` for each k-point might be slow if convergence is slow.
        # But N=961 is small enough that we can maybe use dense `eig` if we optimize?
        # 961^3 ~ 8.8 * 10^8. In Python/NumPy, this is ~1-2 seconds.
        # We have ~11 k-points. 11 * 2s = 22s per evaluation.
        # We have ~200s budget. So ~9 evaluations.
        # This is too few for optimization.
        
        # We need a faster solver.
        # `eigsh` on the Hermitian matrix H = E^{-1/2} K E^{-1/2}.
        # We can approximate E^{-1/2} or use a different formulation.
        
        # Actually, for a 2D problem with N=15, the matrix is banded?
        # No, eps is local, so E is banded?
        # E_mn = eps_{m-n}. If eps has finite support in Fourier space, E is banded.
        # But eps is a pixel map, so it has infinite Fourier support (truncated by FFT).
        # So E is dense.
        
        # Let's try `eigsh` with `which='SA'` (smallest algebraic) or `which='SM'`.
        # But we need the 5th eigenvalue.
        # `eigsh` with `k=5` finds the 5 smallest.
        
        # We need to construct H = E^{-1/2} K E^{-1/2}.
        # We can compute E^{-1/2} once per cell (not per k-point)?
        # No, E depends on cell, not k.
        # So for a fixed cell, E is fixed.
        # We can compute E^{-1/2} once.
        # Then for each k, we form H_k = E^{-1/2} K_k E^{-1/2}.
        # K_k is diagonal.
        # H_k is dense.
        # We can use `eigsh` on H_k.
        
        # Cost:
        # 1. Compute E (O(N^2)).
        # 2. Compute E^{-1/2} (O(N^3) once per cell).
        # 3. For each k: Form H_k (O(N^2)), Solve (O(N^2) or O(N^3) depending on method).
        
        # If we do this for 9 evaluations, it might work.
        # But we need more evaluations for optimization.
        
        # Let's reduce N_pw to 13 (169x169 = 289? No, 27x27=729).
        # N=13 -> 27x27 = 729.
        # 729^3 ~ 3.8 * 10^8. ~0.5s.
        # 11 k-points * 0.5s = 5.5s per eval.
        # 200s / 5.5s ~ 36 evaluations.
        # This is feasible for a simple optimization.
        
        # Let's stick with N=13 for safety.
        
    N_pw = 13
    pw_list = []
    half = N_pw // 2
    for nx in range(-half, half + 1):
        for ny in range(-half, half + 1):
            pw_list.append((nx, ny))
    pw_list = np.array(pw_list)
    N_pw_actual = len(pw_list)
    
    q_diffs = pw_list[:, np.newaxis, :] - pw_list[np.newaxis, :, :]
    q_diffs_flat = q_diffs.reshape(-1, 2)
    fft_indices = q_diffs_flat + size // 2
    valid_mask = (fft_indices[:, 0] >= 0) & (fft_indices[:, 0] < size) & \
                 (fft_indices[:, 1] >= 0) & (fft_indices[:, 1] < size)
    
    def compute_eps_fft(cell):
        eps = np.where(cell == 1, eps_high, eps_low)
        fft_eps = np.fft.fft2(eps) / (size * size)
        fft_eps = np.fft.fftshift(fft_eps)
        return fft_eps

    def evaluate_cell(cell):
        eps_fft = compute_eps_fft(cell)
        
        # Compute E matrix
        E_dense = eps_fft[fft_indices[:, 1], fft_indices[:, 0]]
        E_dense[~valid_mask] = 0.0
        E_dense = E_dense.reshape(N_pw_actual, N_pw_actual)
        
        # Ensure Hermitian
        E_dense = (E_dense + E_dense.conj().T) / 2
        
        # Compute E^{-1/2}
        # Use eigendecomposition
        vals, vecs = np.linalg.eigh(E_dense)
        # Clip small values
        vals = np.clip(vals, 1e-6, None)
        inv_sqrt_vals = 1.0 / np.sqrt(vals)
        E_inv_sqrt = (vecs * inv_sqrt_vals) @ vecs.conj().T
        
        max_omega_4 = 0
        min_omega_5 = float('inf')
        
        for kx, ky in k_points:
            kx_total = kx + pw_list[:, 0]
            ky_total = ky + pw_list[:, 1]
            k_sq = kx_total**2 + ky_total**2
            
            # H = E^{-1/2} K E^{-1/2}
            # K is diagonal
            # H = E_inv_sqrt @ diag(k_sq) @ E_inv_sqrt
            # To avoid forming diag, we can do:
            # H = E_inv_sqrt * k_sq[np.newaxis, :] * E_inv_sqrt
            # Wait. (A D B)_{ij} = A_{ik} D_{kk} B_{kj}.
            # If D is diagonal, (A D)_{ik} = A_{ik} D_{kk}.
            # So H = (E_inv_sqrt * k_sq[np.newaxis, :]) @ E_inv_sqrt
            
            H = (E_inv_sqrt * k_sq[np.newaxis, :]) @ E_inv_sqrt
            H = (H + H.conj().T) / 2 # Ensure Hermitian
            
            # Solve for 5 smallest eigenvalues
            # Use eigsh
            try:
                eigvals, eigvecs = eigsh(H, k=5, which='SA')
            except:
                # Fallback to dense if eigsh fails
                eigvals, eigvecs = np.linalg.eigh(H)
                eigvals = eigvals[:5]
                eigvecs = eigvecs[:, :5]
                
            omegas = np.sqrt(eigvals)
            
            if omegas[3] > max_omega_4:
                max_omega_4 = omegas[3]
            if omegas[4] < min_omega_5:
                min_omega_5 = omegas[4]
                
        if min_omega_5 == float('inf'):
            return -1.0
            
        gap = min_omega_5 - max_omega_4
        midgap = (min_omega_5 + max_omega_4) / 2.0
        if midgap == 0:
            return 0.0
        return gap / midgap

    # 3. Optimization
    # Start with a symmetric rod pattern
    y, x = np.mgrid[0:size, 0:size]
    y = y + 0.5
    x = x + 0.5
    
    centers = [
        (size/4, size/4),
        (size/4, 3*size/4),
        (3*size/4, size/4),
        (3*size/4, 3*size/4)
    ]
    
    r = 5.0
    cell = np.zeros((size, size), dtype=int)
    for cx, cy in centers:
        dist = (x - cx)**2 + (y - cy)**2
        cell[dist < r**2] = 1
        
    cell = (cell + cell[::-1, ::-1]) // 2
    cell = (cell + cell.T) // 2
    
    best_cell = cell.copy()
    best_score = evaluate_cell(cell)
    
    # Local search: Perturb radius and position slightly?
    # Or use the field intensity heuristic from the seed but with the accurate solver.
    
    # Let's try a few random symmetric perturbations
    for step in range(20):
        if time.time() > deadline - 10:
            break
            
        # Perturb the cell
        # Flip some pixels in a symmetric way
        mask = np.random.rand(size, size) < 0.1
        # Enforce symmetry
        mask = (mask + mask[::-1, ::-1]) // 2
        mask = (mask + mask.T) // 2
        
        new_cell = cell.copy()
        # Flip where mask is 1
        new_cell[mask] = 1 - new_cell[mask]
        
        new_score = evaluate_cell(new_cell)
        
        if new_score > best_score:
            best_score = new_score
            best_cell = new_cell.copy()
            cell = new_cell.copy()
        else:
            # If no improvement, try adjusting radius
            # This is a simple heuristic
            pass
            
    # Final refinement: Adjust radius of the main features
    # Identify the main dielectric regions
    # This is complex. Let's just return the best found.
    
    return best_cell.tolist()

