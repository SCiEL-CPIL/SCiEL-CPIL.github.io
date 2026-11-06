import numpy as np
import time
import random

def design(problem):
    size = problem.get("pixels", 32)
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)
    
    start_time = time.time()
    # Evaluator kills at 230s. We need to be safe.
    # Let's aim to finish in 150s to be safe, leaving 80s margin.
    deadline = start_time + 150.0

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
    # The evaluator uses 30 points. We use a dense grid to ensure we capture the min/max bands.
    # The irreducible BZ for square lattice is the triangle (0,0)-(pi,0)-(pi,pi).
    # Actually, the standard irreducible wedge is (0,0) to (pi,0) to (pi/2, pi/2).
    # Wait, the task says "30 wave vectors covering the boundary and interior".
    # Let's use a grid that covers the triangle (0,0)-(pi,0)-(pi,pi) which is half the BZ?
    # No, the irreducible BZ of a square lattice is 1/8 of the full BZ.
    # Vertices: (0,0), (pi,0), (pi/2, pi/2).
    # Let's generate points in this triangle.
    k_points = []
    n_k = 10 # Resolution
    for i in range(n_k + 1):
        for j in range(n_k + 1):
            if i + j <= n_k:
                # Map (i,j) to (kx, ky)
                # Let u = i/n_k, v = j/n_k. u+v <= 1.
                # kx = u * pi
                # ky = v * pi
                # This covers the triangle (0,0)-(pi,0)-(0,pi). 
                # Due to symmetry, this is equivalent to the irreducible wedge if we consider x>=y?
                # The irreducible wedge is x in [0, pi], y in [0, x].
                # So kx = u*pi, ky = v*pi with u+v<=1 maps to x in [0,pi], y in [0,pi-x].
                # This is exactly the irreducible wedge.
                k_points.append((u * np.pi, v * np.pi))
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

    def generate_superellipse_cell(size, center_x, center_y, radius, exponent):
        """
        Generate a 32x32 cell with 4 superelliptical rods.
        Centers at (8,8), (23,8), (8,23), (23,23) for size 32.
        Radius is the effective radius.
        Exponent controls shape: 2=circle, 4=square, >4=sharp square.
        """
        cell = np.zeros((size, size), dtype=int)
        # Centers for 4-rod design in 32x32
        # Symmetry center is (15.5, 15.5).
        # Rods should be centered in the 4 quadrants.
        # Quadrant 1: (0..15, 0..15). Center (7.5, 7.5) -> (8,8) or (7,7)?
        # Let's use (8,8), (23,8), (8,23), (23,23) which are symmetric about 15.5?
        # 8 + 23 = 31. 31/2 = 15.5. Yes.
        centers = [(8, 8), (23, 8), (8, 23), (23, 23)]
        
        # Precompute distances for speed
        # We only need to fill one quadrant and mirror?
        # No, superellipse is not separable.
        # But we can just loop. 32*32 = 1024 ops. Fast.
        
        for i in range(size):
            for j in range(size):
                # Check against all 4 centers
                # Optimization: Only check the closest center?
                # Due to symmetry, (i,j) is close to exactly one center unless it's near the middle.
                # But near the middle, it's far from all.
                # Let's just check the one in the correct quadrant.
                # Quadrant 0: i<16, j<16 -> (8,8)
                # Quadrant 1: i<16, j>=16 -> (8,23)
                # Quadrant 2: i>=16, j<16 -> (23,8)
                # Quadrant 3: i>=16, j>=16 -> (23,23)
                
                if i < 16 and j < 16:
                    cx, cy = 8, 8
                elif i < 16 and j >= 16:
                    cx, cy = 8, 23
                elif i >= 16 and j < 16:
                    cx, cy = 23, 8
                else:
                    cx, cy = 23, 23
                
                dx = j - cx
                dy = i - cy
                
                # Superellipse equation: |dx|/a + |dy|/b <= 1 is for p=1?
                # Standard: (|x/a|^p + |y/b|^p)^(1/p) <= 1
                # Here a=b=radius.
                # (|dx|/r)^p + (|dy|/r)^p <= 1
                # |dx/r|^p + |dy/r|^p <= 1
                
                val = (abs(dx) / radius) ** exponent + (abs(dy) / radius) ** exponent
                if val <= 1.0:
                    cell[i, j] = 1
        return cell

    best_score = -np.inf
    best_cell = None

    # 2. Deterministic Parameter Sweep
    # Parameters:
    # radius: 3.0 to 6.5 (pixels). 
    # exponent: 2.0 (circle) to 10.0 (sharp square).
    
    # Initial guess from literature/knowledge: 
    # 4-rod with square-like shapes often perform well for TM gaps.
    # Radius ~ 4.5-5.5. Exponent ~ 4-8.
    
    radii = np.arange(3.5, 6.6, 0.1)
    exponents = np.arange(2.0, 10.1, 0.5)
    
    # Coarse sweep first
    print(f"Starting coarse sweep at {time.time() - start_time:.1f}s")
    for r in radii:
        if time.time() > deadline - 30:
            break
        for e in exponents:
            if time.time() > deadline - 10:
                break
            cell = generate_superellipse_cell(size, 0, 0, r, e)
            score = compute_score(cell)
            if score > best_score:
                best_score = score
                best_cell = cell.copy()
                best_r = r
                best_e = e
                print(f"New best: r={r:.1f}, e={e:.1f}, score={score:.4f}")

    if best_cell is None:
        # Fallback to a simple circle
        best_cell = generate_superellipse_cell(size, 0, 0, 4.5, 2.0)
        best_score = compute_score(best_cell)
        best_r = 4.5
        best_e = 2.0

    # 3. Fine sweep around the best coarse point
    if time.time() < deadline - 20:
        print(f"Starting fine sweep around r={best_r}, e={best_e} at {time.time() - start_time:.1f}s")
        r_range = np.arange(best_r - 0.5, best_r + 0.51, 0.05)
        e_range = np.arange(best_e - 1.0, best_e + 1.01, 0.25)
        
        for r in r_range:
            if time.time() > deadline - 10:
                break
            for e in e_range:
                if time.time() > deadline - 5:
                    break
                cell = generate_superellipse_cell(size, 0, 0, r, e)
                score = compute_score(cell)
                if score > best_score:
                    best_score = score
                    best_cell = cell.copy()
                    print(f"Fine new best: r={r:.2f}, e={e:.2f}, score={score:.4f}")

    # 4. Final Symmetry Check and Return
    # The superellipse generation is inherently symmetric if centers are chosen correctly.
    # Let's verify.
    def is_symmetric(cell):
        if not np.array_equal(cell, cell[::-1, :]): return False
        if not np.array_equal(cell, cell[:, ::-1]): return False
        if not np.array_equal(cell, cell.T): return False
        return True

    if not is_symmetric(best_cell):
        # If somehow not symmetric (shouldn't happen), enforce it
        # by averaging? No, just revert to a known good symmetric one.
        # But our generator is symmetric.
        pass

    return best_cell.tolist()

