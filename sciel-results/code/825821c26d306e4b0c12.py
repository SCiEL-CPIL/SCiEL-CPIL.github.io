import numpy as np
import time
import math

def design(problem):
    size = problem.get("pixels", 32)
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)
    
    start_time = time.time()
    deadline = start_time + 100.0  # Leave 20s margin for safety

    # 1. Setup Fast Band Solver
    # 289 plane waves (17x17 grid of G vectors)
    half_pw = 8
    G_vecs = []
    for m in range(-half_pw, half_pw + 1):
        for n in range(-half_pw, half_pw + 1):
            G_vecs.append((m, n))
    G_vecs = np.array(G_vecs) * (2.0 * np.pi)
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
    
    def compute_score(cell):
        """
        Compute the 4-5 TM band gap score for a given cell (32x32 array of 0/1).
        Returns the gap-midgap ratio.
        """
        # Convert to epsilon grid
        eps_grid = eps_low + (eps_high - eps_low) * np.array(cell, dtype=float)
        
        # FFT of epsilon for the convolution matrix
        eps_F = np.fft.fft2(eps_grid) / (size * size)
        inv_eps_F = 1.0 / eps_F
        
        min_omega5 = np.inf
        max_omega4 = 0.0
        
        for kx, ky in k_points:
            kG = np.stack([kx + G_vecs[:, 0], ky + G_vecs[:, 1]], axis=1)
            kG2 = kG[:, 0]**2 + kG[:, 1]**2
            
            # Define the matrix-vector product for A
            def matvec(v):
                # 1. w = D v
                w = kG2 * v
                # 2. u = FFT(w)
                u = np.fft.fft2(w.reshape(17, 17))
                # 3. z = (1/eps_F) * u
                z = inv_eps_F * u
                # 4. y = IFFT(z)
                y = np.fft.ifft2(z)
                return y.flatten().real
            
            # Solve for lowest 6 eigenvalues
            try:
                from scipy.sparse.linalg import eigsh
                from scipy.sparse import LinearOperator
                A_op = LinearOperator((nG, nG), matvec=matvec, dtype=complex)
                vals, vecs = eigsh(A_op, k=6, which='SA', tol=1e-5)
                vals = np.sort(vals.real)
                
                if len(vals) < 5:
                    return -1.0
                
                # lambda = (omega/c)^2. We set c=1.
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

    def generate_superellipse_cell(n, radius_scale):
        """
        Generate a 32x32 cell with 4 superelliptical rods.
        n: shape exponent (1=diamond, 2=circle, inf=square)
        radius_scale: effective radius
        """
        cell = np.zeros((size, size), dtype=int)
        
        # Centers at (8,8), (23,8), (8,23), (23,23)
        centers = [(8, 8), (23, 8), (8, 23), (23, 23)]
        
        # Precompute coordinates
        ys, xs = np.mgrid[0:size, 0:size]
        
        for cx, cy in centers:
            dx = xs - cx
            dy = ys - cy
            
            # Superellipse equation: |dx/r|^n + |dy/r|^n <= 1
            # We want to fill pixels where this is true.
            # To avoid division by zero issues and handle n->inf, we can use:
            # (|dx|^n + |dy|^n)^(1/n) <= r
            # Or simply: |dx|^n + |dy|^n <= r^n
            
            # If n is very large, it becomes a square.
            # If n is 2, it's a circle.
            # If n is 1, it's a diamond.
            
            # We use the form: (|dx|/r)^n + (|dy|/r)^n <= 1
            # This is numerically stable if r > 0.
            
            # Handle n=inf case (square)
            if n > 100:
                mask = (np.abs(dx) <= radius_scale) & (np.abs(dy) <= radius_scale)
            else:
                # Calculate distance
                # Use log-sum-exp trick for stability if n is large, but n<=20 is fine
                term_x = np.power(np.abs(dx), n)
                term_y = np.power(np.abs(dy), n)
                dist_n = term_x + term_y
                radius_n = np.power(radius_scale, n)
                mask = dist_n <= radius_n
            
            cell[mask] = 1
        
        # Enforce symmetry (should be symmetric by construction, but ensure binary)
        return cell

    # 2. Parameter Sweep
    # We sweep the shape exponent 'n' and the radius 'r'.
    # 'n' controls the "squareness" or "diamondness".
    # Higher n -> more square-like.
    # Lower n -> more diamond-like.
    # Circles (n=2) are known to be suboptimal for the 4-5 gap due to symmetry constraints on Fourier components.
    # We want to enhance the (2,2) component.
    
    best_score = -1.0
    best_cell = None
    
    # Range of n: 1.0 to 10.0
    # Range of radius: 3.0 to 6.0
    
    n_values = np.linspace(1.0, 10.0, 15)
    r_values = np.linspace(3.5, 6.0, 15)
    
    # Coarse sweep
    for n in n_values:
        if time.time() > deadline - 20:
            break
        for r in r_values:
            if time.time() > deadline - 5:
                break
            
            cell = generate_superellipse_cell(n, r)
            
            # Quick check: ensure it's not empty or full
            if np.sum(cell) == 0 or np.sum(cell) == size*size:
                continue
                
            score = compute_score(cell)
            
            if score > best_score:
                best_score = score
                best_cell = cell.copy()
                best_params = (n, r)
                
    # 3. Local Refinement around best coarse parameters
    if best_cell is not None and time.time() < deadline - 10:
        n_best, r_best = best_params
        
        # Fine sweep
        n_fine = np.linspace(max(0.5, n_best - 1.0), min(15.0, n_best + 1.0), 10)
        r_fine = np.linspace(max(3.0, r_best - 0.5), min(7.0, r_best + 0.5), 10)
        
        for n in n_fine:
            if time.time() > deadline - 5:
                break
            for r in r_fine:
                if time.time() > deadline - 2:
                    break
                
                cell = generate_superellipse_cell(n, r)
                if np.sum(cell) == 0 or np.sum(cell) == size*size:
                    continue
                    
                score = compute_score(cell)
                
                if score > best_score:
                    best_score = score
                    best_cell = cell.copy()
                    best_params = (n, r)

    # 4. Final Safety: If no valid cell was found (unlikely), return a default symmetric cell
    if best_cell is None:
        # Default to a simple 4-rod circle
        best_cell = generate_superellipse_cell(2.0, 4.5)

    # Ensure the returned cell is a list of lists of ints
    return best_cell.tolist()
