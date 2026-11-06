import time
import math
import numpy as np

# EVOLVE-BLOCK-START
def _make_symmetric_cell(size, base):
    """
    Enforce full square (C4v) symmetry: cell == cell[::-1] row/col and cell == cell.T.
    Majority vote over the 8 orbit positions.
    """
    b = np.array(base, dtype=np.int8)
    N = size
    cell = np.zeros((N, N), dtype=np.int8)
    processed = np.zeros((N, N), dtype=bool)
    for i in range(N):
        for j in range(N):
            if processed[i, j]:
                continue
            orbit = [
                (i, j), (N - 1 - i, j), (j, i), (N - 1 - i, N - 1 - j),
                (i, N - 1 - j), (j, N - 1 - i), (N - 1 - j, i), (N - 1 - j, N - 1 - i),
            ]
            unique_orbit = list(set(orbit))
            vals = [b[r, c] for r, c in unique_orbit]
            val = 1 if sum(vals) >= len(vals) / 2.0 else 0
            for r, c in unique_orbit:
                cell[r, c] = val
                processed[r, c] = True
    return cell.tolist()


def _compute_tm_bands(cell, size, n_pw, eps_high, eps_low, k_points):
    """
    Compute TM band gap ratio using plane-wave method.
    Returns (gap_ratio, max_band4, min_band5).
    """
    N = size
    # Reciprocal lattice vectors
    # The basis is G = m * (2pi/L) * x_hat + n * (2pi/L) * y_hat
    # We use dimensionless k and G in units of 2pi/L.
    # The matrix element A_{mn} = (k+G_m) . (k+G_n) * epsilon^{-1}_{G_m - G_n}
    
    G_range = range(-n_pw, n_pw + 1)
    Gs = np.array([(mx, my) for mx in G_range for my in G_range], dtype=float)
    
    # Fourier coefficients of 1/epsilon
    cell_arr = np.array(cell, dtype=float)
    eps_arr = np.where(cell_arr > 0.5, eps_high, eps_low)
    inv_eps_arr = 1.0 / eps_arr
    inv_eps_fft = np.fft.fft2(inv_eps_arr)
    # Normalize FFT? 
    # The standard PW method uses:
    # epsilon^{-1}_{G-G'} = (1/N^2) * sum_{i,j} epsilon^{-1}(r_{ij}) exp(-i (G-G') . r_{ij})
    # np.fft.fft2 computes sum_{i,j} f_{ij} exp(-2pi i (m i/N + n j/N))
    # So inv_eps_fft[m, n] is N^2 * epsilon^{-1}_{G-G'} where G-G' corresponds to (m, n) in FFT index.
    # Wait, standard FFT definition: F_k = sum_n f_n exp(-2pi i k n / N)
    # Inverse: f_n = (1/N) sum_k F_k exp(2pi i k n / N)
    # We need c_{G} = (1/N^2) sum_{r} f(r) exp(-i G . r)
    # This is exactly (1/N^2) * FFT(f) if we map G indices correctly.
    # G index in FFT: m = 0..N-1. G vector = (m - N//2, n - N//2) * (2pi/N) ?
    # Let's use the direct mapping.
    # Gs are in units of 2pi/L.
    # FFT index k corresponds to frequency k.
    # For N=32, k=0..31.
    # G = (mx, my).
    # The coefficient c_{mx, my} is inv_eps_fft[mx_mod, my_mod] / N^2.
    
    c_inv_eps = inv_eps_fft / (N * N)
    
    max_band4 = 0.0
    min_band5 = np.inf
    
    for kx, ky in k_points:
        k = np.array([kx, ky])
        kGs = k + Gs
        # Compute dot products (k+G_m) . (k+G_n)
        # This is a matrix of shape (n_pw*2+1)^2
        # Let M = len(Gs)
        # Dot_{mn} = (kx+Gm_x)(kx+Gn_x) + (ky+Gm_y)(ky+Gn_y)
        
        # Vectorized dot product
        # kGs is M x 2
        # We want outer product of components
        # D = kGs @ kGs.T
        D = np.einsum('mi,ni->mn', kGs, kGs)
        
        # Compute matrix A
        # A_{mn} = D_{mn} * c_{G_m - G_n}
        # G_m - G_n = (mx_m - mx_n, my_m - my_n)
        # We need to index c_inv_eps with these differences.
        # c_inv_eps is N x N.
        # Difference d = G_m - G_n.
        # d_x is in range [-2*n_pw, 2*n_pw].
        # We can map d_x to FFT index: (d_x % N).
        
        dmx = Gs[:, 0][:, None] - Gs[:, 0][None, :]
        dmy = Gs[:, 1][:, None] - Gs[:, 1][None, :]
        
        # Map to FFT indices
        idx_x = dmx % N
        idx_y = dmy % N
        
        c_mat = c_inv_eps[idx_x, idx_y]
        
        A = D * c_mat
        
        # Ensure Hermitian
        A = (A + A.conj().T) / 2.0
        
        # Solve eigenvalue problem
        # We only need the first 5-6 bands
        # np.linalg.eigvalsh is faster for all, but we can use scipy if available? 
        # NumPy is standard. 
        # For M=289, eigvalsh is fast.
        eigvals = np.linalg.eigvalsh(A)
        eigvals = np.real(eigvals)
        
        # Filter positive eigenvalues
        pos_eigvals = eigvals[eigvals > 1e-8]
        
        if len(pos_eigvals) >= 5:
            # Bands are 1-indexed in problem, 0-indexed in array.
            # Band 4 is index 3, Band 5 is index 4.
            b4 = pos_eigvals[3]
            b5 = pos_eigvals[4]
            
            max_band4 = max(max_band4, b4)
            min_band5 = min(min_band5, b5)
            
    if min_band5 == np.inf or max_band4 == 0:
        return -1.0, 0.0, 0.0
        
    gap = min_band5 - max_band4
    midgap = 0.5 * (min_band5 + max_band4)
    
    if midgap <= 0:
        return -1.0, max_band4, min_band5
        
    ratio = 2.0 * gap / midgap
    return ratio, max_band4, min_band5


def _generate_k_points():
    """
    Generate k-points covering the irreducible Brillouin zone.
    BZ is [0, pi] x [0, pi]. Irreducible triangle: 0 <= ky <= kx <= pi.
    """
    pts = []
    # Gamma
    pts.append((0.0, 0.0))
    # M point (pi, pi)
    pts.append((np.pi, np.pi))
    # X point (pi, 0)
    pts.append((np.pi, 0.0))
    # Points along Gamma-M
    for t in np.linspace(0.1, 0.9, 5):
        pts.append((t * np.pi, t * np.pi))
    # Points along Gamma-X
    for t in np.linspace(0.1, 0.9, 5):
        pts.append((t * np.pi, 0.0))
    # Points along X-M
    for t in np.linspace(0.1, 0.9, 5):
        pts.append((np.pi, t * np.pi))
    # Interior points
    pts.append((0.5 * np.pi, 0.25 * np.pi))
    pts.append((0.75 * np.pi, 0.25 * np.pi))
    pts.append((0.5 * np.pi, 0.5 * np.pi))
    return pts


def _generate_initial_candidates(size, rng):
    """
    Generate diverse initial candidates.
    """
    candidates = []
    center = size / 2.0 - 0.5
    
    def from_mask(mask):
        return _make_symmetric_cell(size, mask)
    
    def circle_mask(val, radius, cx=0.0, cy=0.0):
        m = np.zeros((size, size), dtype=np.int8)
        for i in range(size):
            for j in range(size):
                x = (i - center) - cx
                y = (j - center) - cy
                if x * x + y * y <= radius * radius:
                    m[i, j] = val
        return m
    
    def superpose(masks):
        m = np.zeros((size, size), dtype=np.int8)
        for mm in masks:
            m = np.maximum(m, mm)
        return m
    
    # 1. Single central rod
    for r in [3.0, 4.0, 5.0, 6.0]:
        candidates.append(from_mask(circle_mask(1, r)))
        
    # 2. Air hole in dielectric
    for r in [3.0, 4.0, 5.0]:
        m = np.ones((size, size), dtype=np.int8)
        for i in range(size):
            for j in range(size):
                x = (i - center); y = (j - center)
                if x * x + y * y <= r * r:
                    m[i, j] = 0
        candidates.append(from_mask(m))
        
    # 3. Rods at corners (symmetric)
    for d in [5.0, 6.0, 7.0]:
        for r in [1.5, 2.0, 2.5]:
            m = superpose([circle_mask(1, r, d, d), circle_mask(1, r, -d, -d),
                           circle_mask(1, r, d, -d), circle_mask(1, r, -d, d)])
            candidates.append(from_mask(m))
            
    # 4. Rods on axes
    for d in [5.0, 6.0]:
        for r in [1.5, 2.0]:
            m = superpose([circle_mask(1, r, d, 0), circle_mask(1, r, -d, 0),
                           circle_mask(1, r, 0, d), circle_mask(1, r, 0, -d)])
            candidates.append(from_mask(m))
                           
    # 5. Random symmetric patterns
    for _ in range(10):
        m = rng.integers(0, 2, (size, size), dtype=np.int8)
        candidates.append(from_mask(m))
        
    return candidates


def design(problem):
    size = problem["pixels"]
    eps_high = problem.get("eps_high", 12.0)
    eps_low = problem.get("eps_low", 1.0)
    
    start_time = time.time()
    deadline = start_time + 200.0 # 20s margin
    
    # Use n_pw=12 (25x25 basis) for balance of speed and accuracy
    # 25^2 = 625 eigenvalues.
    # If too slow, reduce to 10 (21x21=441).
    # Let's try 12 first.
    n_pw = 12 
    k_points = _generate_k_points()
    
    rng = np.random.default_rng(42)
    
    def now():
        return time.time()
        
    def evaluate(cell):
        try:
            score, b4, b5 = _compute_tm_bands(cell, size, n_pw, eps_high, eps_low, k_points)
            return score
        except Exception:
            return -1.0
            
    # Initialize population
    candidates = _generate_initial_candidates(size, rng)
    
    best_cell = None
    best_score = -1.0
    second_best_cell = None
    second_best_score = -1.0
    
    for cell in candidates:
        if now() > deadline:
            break
        score = evaluate(cell)
        if score > best_score:
            if best_cell is not None:
                second_best_cell = best_cell
                second_best_score = best_score
            best_cell = cell
            best_score = score
        elif score > second_best_score:
            second_best_cell = cell
            second_best_score = score
            
    if best_cell is None:
        # Fallback: central rod
        m = np.zeros((size, size), dtype=np.int8)
        c = size / 2.0 - 0.5
        for i in range(size):
            for j in range(size):
                if (i - c) ** 2 + (j - c) ** 2 <= 4.0:
                    m[i, j] = 1
        return _make_symmetric_cell(size, m)
        
    # Hill Climbing with Random Restarts
    cur_cell = best_cell
    cur_score = best_score
    cur_arr = np.array(cur_cell, dtype=np.int8)
    
    # Precompute orbit indices for fast mutation
    # We will flip random orbits
    orbits = []
    processed = np.zeros((size, size), dtype=bool)
    for i in range(size):
        for j in range(size):
            if processed[i, j]:
                continue
            orbit = [
                (i, j), (size - 1 - i, j), (j, i), (size - 1 - i, size - 1 - j),
                (i, size - 1 - j), (j, size - 1 - i), (size - 1 - j, i), (size - 1 - j, size - 1 - i),
            ]
            unique_orbit = list(set(orbit))
            orbits.append(unique_orbit)
            for r, c in unique_orbit:
                processed[r, c] = True
                
    max_iterations = 500
    for it in range(max_iterations):
        if now() > deadline:
            break
            
        # Mutation: flip 1-3 random orbits
        n_flips = rng.integers(1, 4)
        mut_arr = cur_arr.copy()
        for _ in range(n_flips):
            orbit_idx = rng.integers(0, len(orbits))
            orbit = orbits[orbit_idx]
            # Flip all pixels in this orbit
            for r, c in orbit:
                mut_arr[r, c] = 1 - mut_arr[r, c]
                
        mut_cell = _make_symmetric_cell(size, mut_arr)
        mut_score = evaluate(mut_cell)
        
        if mut_score > cur_score:
            cur_cell = mut_cell
            cur_arr = mut_arr
            cur_score = mut_score
            if cur_score > best_score:
                best_score = cur_score
                best_cell = cur_cell
        else:
            # If stuck for a while, restart from best
            # Simple strategy: if no improvement in last 20 steps, restart
            # We don't track steps exactly, so we do a random restart periodically
            if it % 50 == 0:
                # Perturb best cell
                mut_arr = np.array(best_cell, dtype=np.int8)
                for _ in range(3):
                    orbit_idx = rng.integers(0, len(orbits))
                    orbit = orbits[orbit_idx]
                    for r, c in orbit:
                        mut_arr[r, c] = 1 - mut_arr[r, c]
                cur_cell = _make_symmetric_cell(size, mut_arr)
                cur_arr = mut_arr
                cur_score = evaluate(cur_cell)
                
    return best_cell
# EVOLVE-BLOCK-END
