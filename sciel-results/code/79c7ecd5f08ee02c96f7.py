import math
import time
import numpy as np

def _compute_gap_ratio(cell, eps_high=12.0):
    """
    Compute the gap-midgap ratio between TM bands 4 and 5 using a simplified
    plane-wave method with a smaller basis for speed.
    """
    size = len(cell)
    eps = np.array(cell, dtype=float)
    
    # Number of plane waves to use (smaller for speed)
    n_pw = 13  # 13x13 = 169 plane waves
    
    # Generate plane wave vectors
    kx = np.arange(-n_pw//2, n_pw//2 + 1)
    ky = np.arange(-n_pw//2, n_pw//2 + 1)
    
    # Fourier coefficients of epsilon
    eps_ft = np.zeros((2*n_pw//2+1, 2*n_pw//2+1))
    for mx in range(-n_pw//2, n_pw//2 + 1):
        for my in range(-n_pw//2, n_pw//2 + 1):
            # Compute Fourier coefficient
            coeff = 0.0
            for i in range(size):
                for j in range(size):
                    coeff += eps[i,j] * np.exp(-2j*np.pi*(mx*i + my*j)/size)
            eps_ft[mx + n_pw//2, my + n_pw//2] = coeff / (size*size)
    
    # Sample k-points in irreducible BZ
    # Irreducible BZ for square lattice: 0 <= kx <= ky <= pi
    n_k = 8
    k_points = []
    for i in range(n_k):
        for j in range(n_k):
            kx_val = i * np.pi / (n_k-1)
            ky_val = j * np.pi / (n_k-1)
            if kx_val <= ky_val + 1e-10 and kx_val >= 0 and ky_val >= 0:
                k_points.append((kx_val, ky_val))
    
    # Also add high-symmetry points
    k_points.append((0, 0))
    k_points.append((np.pi/2, np.pi/2))
    k_points.append((np.pi, 0))
    k_points.append((np.pi/2, 0))
    
    # Remove duplicates
    k_points_unique = []
    for kp in k_points:
        if not any(np.allclose(kp, kp2, atol=1e-10) for kp2 in k_points_unique):
            k_points_unique.append(kp)
    k_points = k_points_unique
    
    omega_4_max = 0.0
    omega_5_min = np.inf
    
    for kx, ky in k_points:
        # Build the eigenvalue problem for TM polarization
        # H(k+G) = (1/eps(G)) * |k+G|^2 * H(k+G) for TM
        n = 2*n_pw//2 + 1
        H = np.zeros((n*n, n*n), dtype=complex)
        
        for i in range(n):
            for j in range(n):
                idx1 = i*n + j
                Gx1 = kx[i] * 2*np.pi
                Gy1 = ky[j] * 2*np.pi
                
                for l in range(n):
                    for m in range(n):
                        idx2 = l*n + m
                        Gx2 = kx[l] * 2*np.pi
                        Gy2 = ky[m] * 2*np.pi
                        
                        # Difference in G vectors
                        dGx = Gx1 - Gx2
                        dGy = Gy1 - Gy2
                        
                        # Find the Fourier coefficient index
                        mx = int(round(dGx / (2*np.pi)))
                        my = int(round(dGy / (2*np.pi)))
                        
                        if -n_pw//2 <= mx <= n_pw//2 and -n_pw//2 <= my <= n_pw//2:
                            eps_coeff = eps_ft[mx + n_pw//2, my + n_pw//2]
                            if abs(eps_coeff) > 1e-15:
                                H[idx1, idx2] = (Gx2**2 + Gy2**2) / eps_coeff
        
        # Solve eigenvalue problem
        try:
            eigvals = np.linalg.eigvalsh(H)
            eigvals = np.real(eigvals)
            eigvals = np.sort(eigvals)
            
            if len(eigvals) >= 5:
                # Take the 4th and 5th bands (0-indexed: 3 and 4)
                w4 = eigvals[3]
                w5 = eigvals[4]
                
                if w4 > 0 and w5 > 0:
                    omega_4_max = max(omega_4_max, w4)
                    omega_5_min = min(omega_5_min, w5)
        except:
            continue
    
    if omega_5_min == np.inf or omega_4_max <= 0:
        return -1.0
    
    gap = omega_5_min - omega_4_max
    midgap = (omega_5_min + omega_4_max) / 2.0
    
    if midgap <= 0:
        return -1.0
    
    return 2.0 * gap / midgap

def _evaluate_cell(cell):
    """Wrapper to evaluate a cell configuration."""
    try:
        return _compute_gap_ratio(cell)
    except:
        return -1.0

def _nelder_mead(f, x0, args=(), maxiter=100, tol=1e-4, step=0.1):
    """Simple Nelder-Mead implementation."""
    n = len(x0)
    simplex = [list(x0)]
    for i in range(n):
        xi = list(x0)
        xi[i] += step
        simplex.append(xi)
    
    fvals = [f(x, *args) for x in simplex]
    best_idx = np.argmin(fvals)
    
    for it in range(maxiter):
        # Sort simplex by function value
        order = np.argsort(fvals)
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]
        
        # Check convergence
        if abs(fvals[0] - fvals[-1]) < tol:
            break
        
        # Centroid of all but worst
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        alpha = 1.0
        x_r = centroid + alpha * (centroid - simplex[-1])
        f_r = f(x_r, *args)
        
        if f_r < fvals[0]:
            # Expansion
            gamma = 2.0
            x_e = centroid + gamma * (x_r - centroid)
            f_e = f(x_e, *args)
            if f_e < f_r:
                simplex[-1] = x_e
                fvals[-1] = f_e
            else:
                simplex[-1] = x_r
                fvals[-1] = f_r
        elif f_r < fvals[-2]:
            simplex[-1] = x_r
            fvals[-1] = f_r
        else:
            # Contraction
            rho = 0.5
            x_c = centroid + rho * (simplex[-1] - centroid)
            f_c = f(x_c, *args)
            if f_c < fvals[-1]:
                simplex[-1] = x_c
                fvals[-1] = f_c
            else:
                # Shrink
                for i in range(1, n+1):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                    fvals[i] = f(simplex[i], *args)
    
    best_idx = np.argmin(fvals)
    return simplex[best_idx], fvals[best_idx]

def _cell_from_polynomial(coeffs, size=32):
    """
    Generate a D4-symmetric binary cell from polynomial coefficients.
    P(x,y) = sum c_ij * x^(2i) * y^(2j) for i,j in 0..4
    """
    cell = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            # Normalize coordinates to [-0.5, 0.5]
            x = (j + 0.5) / size - 0.5
            y = (i + 0.5) / size - 0.5
            
            val = 0.0
            for ii in range(5):
                for jj in range(5):
                    idx = ii * 5 + jj
                    val += coeffs[idx] * (x ** (2*ii)) * (y ** (2*jj))
            
            cell[i, j] = 1.0 if val > 0 else 0.0
    
    return cell

def design(problem):
    """
    Design a D4-symmetric 32x32 unit cell using polynomial parameterization
    and Nelder-Mead optimization.
    """
    size = problem["pixels"]
    if size != 32:
        size = 32
    
    deadline = time.time() + 200  # Safety margin
    
    # Initialize with coefficients that create a 2x2 rod pattern
    # Start with a pattern that has dielectric at (0.25, 0.25) etc.
    # P(x,y) > 0 should be dielectric
    # We want high values near (±0.25, ±0.25)
    
    # Initial coefficients: try to create four lobes
    # Use a product form: (x^2 - 0.0625) * (y^2 - 0.0625) < 0 means inside
    # So we want P = -(x^2 - 0.0625) * (y^2 - 0.0625) > 0 inside
    # Expand: -x^2*y^2 + 0.0625*x^2 + 0.0625*y^2 - 0.00390625 > 0
    
    init_coeffs = np.zeros(25)
    init_coeffs[1*5 + 0] = 0.0625  # x^2
    init_coeffs[0*5 + 1] = 0.0625  # y^2
    init_coeffs[1*5 + 1] = -1.0    # x^2*y^2
    init_coeffs[0*5 + 0] = -0.00390625  # constant
    
    # Evaluate initial
    cell = _cell_from_polynomial(init_coeffs, size)
    best_score = _evaluate_cell(cell)
    best_coeffs = init_coeffs.copy()
    
    # Run Nelder-Mead optimization
    def objective(coeffs):
        cell = _cell_from_polynomial(coeffs, size)
        return -_evaluate_cell(cell)  # Minimize negative of score
    
    # Multiple restarts with different initial conditions
    restarts = [
        init_coeffs.copy(),
        np.random.randn(25) * 0.1,
        np.zeros(25),
        init_coeffs + np.random.randn(25) * 0.05,
    ]
    
    for restart_idx, init_c in enumerate(restarts):
        if time.time() > deadline - 30:
            break
        
        try:
            x_opt, f_opt = _nelder_mead(objective, init_c, maxiter=50, tol=1e-3, step=0.1)
            score_opt = -f_opt
            
            if score_opt > best_score:
                best_score = score_opt
                best_coeffs = x_opt
        except Exception as e:
            continue
    
    # Final evaluation and return
    cell = _cell_from_polynomial(best_coeffs, size)
    
    # Ensure D4 symmetry
    cell = (cell + cell[::-1, :]) / 2
    cell = (cell + cell[:, ::-1]) / 2
    cell = (cell + cell.T) / 2
    cell = (cell + cell.T[::-1, ::-1]) / 2
    cell = (cell > 0.5).astype(int)
    
    return cell.tolist()
