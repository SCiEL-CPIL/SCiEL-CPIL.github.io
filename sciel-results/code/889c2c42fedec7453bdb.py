"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _build_cell_from_fourier(coeffs, size=32):
    """
    Build a 32x32 cell from Fourier coefficients.
    coeffs: list of (m, n, value) where epsilon(x,y) = 1 + sum value * cos(2*pi*m*x) * cos(2*pi*n*y)
    We enforce symmetry by only using even m, n or appropriate symmetries.
    For C4v and inversion symmetry (which the task implies by 'symmetric... about cell centre'),
    we can use a basis of cos(2*pi*m*x)*cos(2*pi*n*y).
    """
    x = np.linspace(-0.5, 0.5, size, endpoint=False)
    X, Y = np.meshgrid(x, x)
    eps = np.ones((size, size))
    for m, n, val in coeffs:
        eps += val * np.cos(2 * np.pi * m * X) * np.cos(2 * np.pi * n * Y)
    # Threshold to 0/1
    # We want a binary map. A simple threshold at the mean or 0.5 of the range?
    # The problem says "1 = dielectric with epsilon 12, 0 = air".
    # The evaluator uses the pixel map to compute exact Fourier coefficients.
    # So we just need to return a 0/1 grid.
    # We can threshold the continuous field.
    # Let's threshold at the median or a fixed value to control fill fraction.
    # Actually, the optimization should ideally control the fill fraction too.
    # But for now, let's just threshold at 0.
    # Wait, if eps = 1 + sum..., the values can be anywhere.
    # Let's threshold at the mean of the continuous field to keep ~50% fill?
    # Or better, let the optimizer find the right shape.
    # Since the evaluator uses the *pixels*, the intermediate continuous field is just a parameterization.
    # Let's threshold at 0.
    cell = (eps > 0).astype(int)
    return cell.tolist()

def _fast_band_structure(cell, n_pw=15, n_kpts=5):
    """
    Fast approximate plane-wave solver for gradient estimation.
    This is a simplified version of the exact solver to allow many evaluations.
    Uses a smaller number of plane waves and k-points.
    """
    size = len(cell)
    eps = np.array(cell, dtype=float)
    # Compute Fourier coefficients of 1/eps for TM mode
    # In TM, the eigenvalue problem is for H_z, and the operator involves 1/eps.
    # The matrix elements are <G+k | 1/eps | G'+k>
    # We need the Fourier series of 1/eps.
    inv_eps = 1.0 / (1 + 11 * eps)  # eps is 0 or 1, so 1/eps is 1 or 1/12
    # Compute Fourier coefficients up to n_pw
    # G vectors: (m, n) for -n_pw <= m,n <= n_pw
    Gs = []
    for m in range(-n_pw, n_pw + 1):
        for n in range(-n_pw, n_pw + 1):
            Gs.append((m, n))
    Gs = np.array(Gs)
    n_G = len(Gs)
    
    # Compute Fourier coefficients of inv_eps
    # c_G = (1/A) integral inv_eps(x,y) exp(-i G . r) d^2r
    # We can compute this via FFT on the grid
    # Pad the grid to be large enough? Or just use the 32x32 grid directly?
    # The grid is 32x32. The fundamental frequency is 1.
    # G = (m, n) corresponds to frequency m, n.
    # We can use np.fft.fft2 on the 32x32 grid.
    # But we need frequencies up to n_pw.
    # The Nyquist frequency for 32 points is 16.
    # So n_pw should be <= 16.
    if n_pw > 16:
        n_pw = 16
    # Recompute Gs if n_pw was reduced
    Gs = []
    for m in range(-n_pw, n_pw + 1):
        for n in range(-n_pw, n_pw + 1):
            Gs.append((m, n))
    Gs = np.array(Gs)
    n_G = len(Gs)
    
    # FFT of inv_eps
    # Shift so that zero frequency is in the center
    inv_eps_shifted = np.fft.fftshift(inv_eps)
    fft_inv_eps = np.fft.fft2(inv_eps_shifted)
    # Normalize by N^2
    fft_inv_eps /= (size * size)
    
    # Map G indices to FFT indices
    # FFT index for frequency m: if m >= 0, index is m, else index is size + m
    # But we used fftshift, so index 0 is frequency -size/2 ... size/2 - 1
    # Let's just compute the coefficients directly by integration to be safe, or use the FFT mapping.
    # Mapping: freq m in [-16, 15] corresponds to index m + 16 in the shifted FFT.
    # Our Gs are in [-n_pw, n_pw].
    # Let's build a matrix of coefficients C of shape (n_G, n_G) where C[i,j] = <G_i | 1/eps | G_j>
    # For a periodic function, the matrix elements are just the Fourier coefficients if the basis is plane waves.
    # <G+k | 1/eps | G'+k> = c_{G-G'}
    # So we need c_G for G = G_i - G_j.
    
    # Precompute c_G for all needed G differences
    # Max difference is 2*n_pw
    max_diff = 2 * n_pw
    c_G = {}
    for m in range(-max_diff, max_diff + 1):
        for n in range(-max_diff, max_diff + 1):
            # Find the index in the shifted FFT
            idx_m = (m + size // 2) % size
            idx_n = (n + size // 2) % size
            c_G[(m, n)] = fft_inv_eps[idx_m, idx_n]
    
    # K-points in the irreducible BZ: (0,0), (0.5, 0), (0.5, 0.5)
    # And maybe (0, 0.25) or (0.25, 0) for interior?
    # The task says "30 wave vectors covering the boundary and interior".
    # For speed, let's use just the 3 high-symmetry points.
    kpts = [(0.0, 0.0), (0.5, 0.0), (0.5, 0.5)]
    
    # We want to find the 4th and 5th bands.
    # We need to solve the eigenvalue problem at each k-point.
    # The matrix is (n_G x n_G).
    # We only need the smallest few eigenvalues.
    
    min_omega_5 = np.inf
    max_omega_4 = -np.inf
    
    for kx, ky in kpts:
        # Build the matrix M
        # M[i, j] = c_{G_i - G_j} * |G_j + k|^2
        # Wait, the operator is (G+k)^2 / eps(G+k) ? No.
        # The TM equation is: (G+k)^2 H_{G+k} = sum_{G'} (1/eps)_{G-G'} H_{G'+k}
        # So the matrix is A_{i,j} = (1/eps)_{G_i - G_j}
        # And the eigenvalue is lambda = (G_j + k)^2 ? No, it's a generalized eigenvalue problem?
        # No, it's a standard eigenvalue problem if we multiply by (G+k)^2?
        # Actually, the equation is:
        # (G+k)^2 H_{G+k} = sum_{G'} c_{G-G'} H_{G'+k}
        # This is not a standard matrix eigenvalue problem because (G+k)^2 is diagonal but different for each row.
        # We can rewrite it as:
        # sum_{G'} [ c_{G-G'} delta_{G,G'} - (G+k)^2 delta_{G,G'} ] H_{G'+k} = 0 ? No.
        # Let's use the standard form:
        # sum_{G'} c_{G-G'} H_{G'+k} - (G+k)^2 H_{G+k} = 0
        # This is a generalized eigenvalue problem? No, it's a standard eigenvalue problem if we define the matrix properly.
        # Actually, it's:
        # M H = lambda H, where M_{i,j} = c_{G_i - G_j}
        # And lambda = (G_i + k)^2 ? No, lambda is the same for all i.
        # So we have:
        # sum_j M_{i,j} H_j = (G_i + k)^2 H_i
        # This is a generalized eigenvalue problem: M H = D H, where D is diagonal with (G_i+k)^2.
        # Or: M^{-1} D H = H ? No.
        # We can solve D^{-1} M H = lambda H, where lambda = 1/(G+k)^2 ? No.
        # Let's just solve the generalized eigenvalue problem using scipy.
        
        M = np.zeros((n_G, n_G), dtype=complex)
        D = np.zeros(n_G)
        for i in range(n_G):
            G_i = Gs[i]
            D[i] = (G_i[0] + kx)**2 + (G_i[1] + ky)**2
            for j in range(n_G):
                G_j = Gs[j]
                G_diff = (G_i[0] - G_j[0], G_i[1] - G_j[1])
                M[i, j] = c_G[G_diff]
        
        # Solve D^{-1} M H = lambda H
        # This is equivalent to M H = lambda D H
        # We can use scipy.linalg.eig or eigsh
        # Since D is diagonal and positive, we can transform to a standard problem.
        # Let H' = D^{1/2} H. Then M D^{-1/2} H' = lambda D^{1/2} H'
        # D^{-1/2} M D^{-1/2} H' = lambda H'
        
        D_sqrt = np.sqrt(D)
        D_inv_sqrt = 1.0 / D_sqrt
        M_std = M * D_inv_sqrt[None, :] * D_inv_sqrt[:, None]
        M_std = (M_std + M_std.conj().T) / 2  # Ensure Hermitian
        
        # Get the smallest eigenvalues
        # We need the 4th and 5th smallest.
        # n_G is about (2*15+1)^2 = 1024.
        # eigsh can find the smallest few.
        try:
            from scipy.sparse.linalg import eigsh
            M_sparse = M_std
            # eigsh requires the matrix to be sparse or a linear operator.
            # For 1024x1024, dense eig might be faster?
            # Let's use np.linalg.eigvalsh for dense.
            # 1024x1024 dense eigvalsh is fast.
            evals = np.linalg.eigvalsh(M_std)
            # The eigenvalues of the standard problem are lambda = 1/omega^2 ?
            # Wait, the original equation was:
            # sum_j c_{G-G'} H_{G'} = (G+k)^2 H_G
            # Let H = D^{-1/2} H'.
            # sum_j c_{G-G'} D_j^{-1/2} H'_j = D_i^{1/2} H'_i
            # D_i^{-1/2} sum_j c_{G-G'} D_j^{-1/2} H'_j = H'_i
            # So the eigenvalues of D^{-1/2} M D^{-1/2} are 1/omega^2.
            # So omega^2 = 1/eval.
            # We want the 4th and 5th smallest omega, which corresponds to the 4th and 5th largest 1/omega^2.
            # So we sort evals in descending order.
            evals_sorted = np.sort(evals)[::-1]
            # The first few are the largest 1/omega^2, so smallest omega.
            # omega_4^2 = 1/evals_sorted[3]
            # omega_5^2 = 1/evals_sorted[4]
            # But we need to be careful: the bands are ordered by omega.
            # So the 4th band is the 4th smallest omega, which is the 4th largest 1/omega^2.
            # Yes.
            
            # We need to find the gap between band 4 and 5.
            # This depends on the k-point.
            # We want to maximize the minimum gap over the BZ.
            # So for each k-point, we find omega_4 and omega_5.
            # Then we track the max of omega_4 and min of omega_5 over all k-points.
            
            if len(evals_sorted) >= 5:
                omega_4 = np.sqrt(1.0 / evals_sorted[3])
                omega_5 = np.sqrt(1.0 / evals_sorted[4])
                max_omega_4 = max(max_omega_4, omega_4)
                min_omega_5 = min(min_omega_5, omega_5)
        except Exception:
            pass
    
    if min_omega_5 == np.inf or max_omega_4 == -np.inf:
        return 0.0
    
    gap = min_omega_5 - max_omega_4
    midgap = (min_omega_5 + max_omega_4) / 2
    if midgap == 0:
        return 0.0
    return gap / midgap

def design(problem):
    size = problem["pixels"]
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Initial population of Fourier coefficient sets
    # Each set is a list of (m, n, value)
    # We use a fixed set of (m, n) pairs to keep the dimensionality low.
    # Symmetry: C4v.
    # We can use m, n >= 0.
    # Let's pick a few low-order harmonics.
    harmonics = [
        (0, 0), (1, 0), (0, 1), (1, 1), (2, 0), (0, 2), (2, 1), (1, 2), (2, 2)
    ]
    
    # Population size
    pop_size = 50
    population = []
    for _ in range(pop_size):
        coeffs = []
        for m, n in harmonics:
            val = np.random.uniform(-1, 1)
            coeffs.append((m, n, val))
        population.append(coeffs)
    
    best_score = -np.inf
    best_coeffs = None
    
    for gen in range(100):
        if time.time() > deadline:
            break
        
        scores = []
        for coeffs in population:
            cell = _build_cell_from_fourier(coeffs, size)
            score = _fast_band_structure(cell)
            scores.append(score)
            if score > best_score:
                best_score = score
                best_coeffs = coeffs
        
        # Selection: keep the top half
        idx = np.argsort(scores)[::-1]
        population = [population[i] for i in idx[:pop_size // 2]]
        
        # Reproduction: create new population by mutating the best ones
        new_population = list(population)
        while len(new_population) < pop_size:
            parent = population[np.random.randint(len(population))]
            child = []
            for m, n, val in parent:
                if np.random.random() < 0.1:
                    new_val = val + np.random.normal(0, 0.1)
                    new_val = np.clip(new_val, -2, 2)
                else:
                    new_val = val
                child.append((m, n, new_val))
            new_population.append(child)
        
        population = new_population
    
    if best_coeffs is None:
        # Fallback to the initial rod
        cell = []
        for i in range(size):
            x = (i + 0.5) / size - 0.5
            cell.append([1 if x * x + ((j + 0.5) / size - 0.5) ** 2 < 0.2 ** 2 else 0 for j in range(size)])
        return cell
    
    return _build_cell_from_fourier(best_coeffs, size)
# EVOLVE-BLOCK-END
