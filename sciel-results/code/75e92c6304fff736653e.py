import numpy as np
import time
from scipy.sparse.linalg import eigsh
from scipy.sparse import LinearOperator
import random

def design(problem):
    size = problem.get("pixels", 32)
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)
    
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Precompute G vectors: 17x17 grid = 289 plane waves
    half_pw = 8
    G_vecs = []
    for m in range(-half_pw, half_pw + 1):
        for n in range(-half_pw, half_pw + 1):
            G_vecs.append((m, n))
    G_vecs = np.array(G_vecs, dtype=float) * (2.0 * np.pi)
    nG = len(G_vecs)
    
    # K-points in the irreducible BZ (triangle from Gamma(0,0) to X(pi,0) to M(pi,pi))
    # Use a dense set to approximate the evaluator's 30 points
    k_points = []
    n_k = 12
    for i in range(n_k):
        for j in range(n_k):
            s = i / n_k
            t = j / n_k
            if s + t <= 1.0:
                k_points.append((s * np.pi, t * np.pi))
    k_points = np.array(k_points)
    
    def compute_eps_fourier(cell):
        """Compute 2D FFT of the dielectric constant grid."""
        eps_grid = eps_low + (eps_high - eps_low) * np.array(cell, dtype=float)
        eps_F = np.fft.fft2(eps_grid) / (size * size)
        return eps_F
    
    def solve_bands_at_k(cell, kx, ky, eps_F):
        """Solve for the lowest 6 TM eigenvalues at a given k-point."""
        kG = np.stack([kx + G_vecs[:, 0], ky + G_vecs[:, 1]], axis=1)
        kG2 = kG[:, 0]**2 + kG[:, 1]**2
        inv_eps_F = 1.0 / eps_F
        
        def matvec(v):
            w = kG2 * v
            u = np.fft.fft2(w.reshape(17, 17))
            z = inv_eps_F * u
            y = np.fft.ifft2(z)
            return y.flatten().real
        
        try:
            A_op = LinearOperator((nG, nG), matvec=matvec, dtype=complex)
            vals, _ = eigsh(A_op, k=6, which='SA', tol=1e-7, maxiter=500)
            vals = np.sort(vals.real)
            return vals
        except Exception:
            return None
    
    def compute_gap_score(cell):
        """Compute the 4-5 TM band gap score."""
        eps_F = compute_eps_fourier(cell)
        
        min_omega5 = np.inf
        max_omega4 = 0.0
        
        for kx, ky in k_points:
            vals = solve_bands_at_k(cell, kx, ky, eps_F)
            if vals is None or len(vals) < 5:
                return -1.0
            
            # eigenvalues are (omega/c)^2, so omega = sqrt(eigenvalue)*c, but ratio is scale-invariant
            lambda4 = max(0.0, vals[3])
            lambda5 = max(0.0, vals[4])
            
            omega4 = np.sqrt(lambda4)
            omega5 = np.sqrt(lambda5)
            
            if omega5 < min_omega5:
                min_omega5 = omega5
            if omega4 > max_omega4:
                max_omega4 = omega4
        
        if min_omega5 == np.inf or max_omega4 == 0.0:
            return -1.0
        
        if min_omega5 <= max_omega4:
            return -1.0
        
        return 2.0 * (min_omega5 - max_omega4) / (min_omega5 + max_omega4)
    
    def generate_moire_cell(m1, n1, m2, n2, threshold=0.0, phase1=0.0, phase2=0.0):
        """
        Generate a symmetric unit cell from the interference of two square-wave gratings.
        The pattern is: f(x,y) = sgn(sin(2*pi*m1*x + 2*pi*n1*y + phase1)) + sgn(sin(2*pi*m2*x + 2*pi*n2*y + phase2))
        Dielectric (1) where f > threshold.
        """
        x = np.linspace(0.5/size, 1 - 0.5/size, size)
        y = np.linspace(0.5/size, 1 - 0.5/size, size)
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        w1 = 2*np.pi*(m1*X + n1*Y) + phase1
        w2 = 2*np.pi*(m2*X + n2*Y) + phase2
        
        # Square wave: sign of sine
        s1 = np.sign(np.sin(w1))
        s2 = np.sign(np.sin(w2))
        # Handle zero: sign(0) = 0, which is fine
        s1 = np.where(s1 == 0, 1, s1)  # Break ties consistently
        s2 = np.where(s2 == 0, 1, s2)
        
        f = s1 + s2
        
        # Threshold: dielectric where f >= threshold
        # f ranges from -2 to 2
        # threshold=0: 50% fill when symmetric
        cell = (f >= threshold).astype(int)
        
        # Enforce D4 symmetry: the moire pattern may not be perfectly symmetric
        # due to discretization. Force it by averaging over the symmetry group.
        # D4: identity, rot90, rot180, rot270, reflect_x, reflect_y, reflect_diag, reflect_anti_diag
        # For a 32x32 grid, we can just check and enforce.
        # Actually, let's enforce by taking the majority vote over the 8 symmetry transforms.
        
        def rotate90(c):
            return np.rot90(c, 1)
        
        def reflect_x(c):  # flip rows
            return c[::-1, :]
        
        def reflect_y(c):  # flip columns
            return c[:, ::-1]
        
        transforms = [
            cell,
            rotate90(cell),
            rotate90(rotate90(cell)),
            rotate90(rotate90(rotate90(cell))),
            reflect_x(cell),
            reflect_y(cell),
            reflect_x(rotate90(cell)),  # anti-diagonal
            reflect_y(rotate90(cell))   # diagonal
        ]
        
        # Average all transforms
        avg = np.mean(transforms, axis=0)
        sym_cell = (avg >= 0.5).astype(int)
        
        return sym_cell.tolist()
    
    def generate_superellipse_cell(cx, cy, a, b, p):
        """Generate a cell from a superellipse pill and its symmetric images."""
        cell = np.zeros((size, size), dtype=int)
        
        centers = [
            (cx, cy), (cx, 1-cy), (1-cx, cy), (1-cx, 1-cy),
            (cy, cx), (cy, 1-cx), (1-cy, cx), (1-cy, 1-cx)
        ]
        unique_centers = list(set(tuple(np.round(c, 10)) for c in centers))
        
        for (ucx, ucy) in unique_centers:
            for i in range(size):
                for j in range(size):
                    x = (j + 0.5) / size
                    y = (i + 0.5) / size
                    dx = x - ucx
                    dy = y - ucy
                    if a > 0 and b > 0:
                        val = (abs(dx/a)**p + abs(dy/b)**p)
                    else:
                        val = np.inf
                    if val <= 1.0:
                        cell[i, j] = 1
                        
        return cell.tolist()
    
    def generate_hole_cell(cx, cy, r, hole=True):
        """Generate a cell with a central hole or rod."""
        cell = np.ones((size, size), dtype=int)
        if hole:
            cell[:] = 1  # Dielectric everywhere
            for i in range(size):
                for j in range(size):
                    x = (j + 0.5) / size - cx
                    y = (i + 0.5) / size - cy
                    if x*x + y*y < r*r:
                        cell[i, j] = 0
        return cell.tolist()
    
    # === Search Strategy ===
    # 1. Start with known good geometries
    # 2. Explore moire patterns
    # 3. Refine best with local pixel flips
    
    best_score = -1.0
    best_cell = None
    
    # Initial candidates: known good designs
    # 4-rod design
    init_params = [0.25, 0.25, 0.15, 0.15, 2.0]
    init_cell = generate_superellipse_cell(*init_params)
    init_score = compute_gap_score(init_cell)
    if init_score > best_score:
        best_score = init_score
        best_cell = init_cell
    
    # Central hole design
    hole_cell = generate_hole_cell(0.5, 0.5, 0.25, hole=True)
    hole_score = compute_gap_score(hole_cell)
    if hole_score > best_score:
        best_score = hole_score
        best_cell = hole_cell
    
    # Explore moire patterns
    # Counter-rotating pairs that enhance (2,0) and (2,2) coefficients
    moire_params_list = [
        # (m1, n1, m2, n2, threshold)
        (2, 0, 0, 2, 0.0),
        (2, 0, 0, 2, 0.5),
        (2, 0, 0, 2, -0.5),
        (2, 2, -2, 2, 0.0),
        (2, 2, 2, -2, 0.0),
        (1, 2, -2, 1, 0.0),
        (2, 1, 1, -2, 0.0),
        (2, 0, 2, 2, 0.0),
        (2, 0, -2, 2, 0.0),
        (0, 2, 2, 0, 0.0),
        (1, 1, 1, -1, 0.0),
        (1, 1, -1, 1, 0.0),
        (3, 0, 0, 3, 0.0),
        (3, 1, 1, -3, 0.0),
    ]
    
    for m1, n1, m2, n2, t in moire_params_list:
        if time.time() > deadline - 30:
            break
        cell = generate_moire_cell(m1, n1, m2, n2, t)
        score = compute_gap_score(cell)
        if score > best_score:
            best_score = score
            best_cell = cell
    
    # Random moire exploration
    n_random = 80
    for i in range(n_random):
        if time.time() > deadline - 20:
            break
        
        m1 = random.choice([1, 2, 3, 4])
        n1 = random.choice([0, 1, 2, 3, 4])
        # Ensure counter-rotating: (m2, n2) = (-n1, m1) or similar
        m2 = -n1
        n2 = m1
        t = random.uniform(-1.0, 1.0)
        
        cell = generate_moire_cell(m1, n1, m2, n2, t)
        score = compute_gap_score(cell)
        if score > best_score:
            best_score = score
            best_cell = cell
    
    # Local refinement: pixel flips on the best cell
    if best_cell is not None:
        # Try flipping individual pixels (respecting symmetry)
        # D4 symmetry: flipping a pixel requires flipping all its symmetric counterparts
        n_refine = 50
        for i in range(n_refine):
            if time.time() > deadline - 5:
                break
            
            cell_arr = np.array(best_cell, dtype=int)
            # Pick a random pixel
            pi = random.randint(0, size-1)
            pj = random.randint(0, size-1)
            
            # Find all symmetric counterparts
            # D4 transformations:
            # 1: (i, j)
            # rot90: (j, size-1-i)
            # rot180: (size-1-i, size-1-j)
            # rot270: (size-1-j, i)
            # reflect_x: (size-1-i, j)
            # reflect_y: (i, size-1-j)
            # reflect_diag: (j, i)
            # reflect_anti: (size-1-j, size-1-i)
            
            symmetric_pixels = [
                (pi, pj),
                (pj, size-1-pi),
                (size-1-pi, size-1-pj),
                (size-1-pj, pi),
                (size-1-pi, pj),
                (pi, size-1-pj),
                (pj, pi),
                (size-1-pj, size-1-pi)
            ]
            unique_pixels = list(set(symmetric_pixels))
            
            # Flip these pixels
            cell_arr_copy = cell_arr.copy()
            for (ii, jj) in unique_pixels:
                cell_arr_copy[ii, jj] = 1 - cell_arr_copy[ii, jj]
            
            new_cell = cell_arr_copy.tolist()
            new_score = compute_gap_score(new_cell)
            
            if new_score > best_score:
                best_score = new_score
                best_cell = new_cell
    
    # If no valid cell, fallback
    if best_cell is None:
        best_cell = generate_superellipse_cell(0.25, 0.25, 0.15, 0.15, 2.0)
    
    return best_cell
