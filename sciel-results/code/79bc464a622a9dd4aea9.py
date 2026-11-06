"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import sys

# EVOLVE-BLOCK-START
def design(problem):
    """Design a 32x32 symmetric photonic crystal unit cell for max TM band gap."""
    start_time = time.time()
    deadline = start_time + 220  # Safety margin before 230s kill

    size = problem["pixels"]
    eps_low = problem["eps_low"]
    eps_high = problem["eps_high"]
    
    # Precompute pixel centers in [0,1)
    coords = np.linspace(0.5/size, 1 - 0.5/size, size)
    X, Y = np.meshgrid(coords, coords)
    # Center at 0.5
    Xc = X - 0.5
    Yc = Y - 0.5

    # Parameters: 
    # r: base radius
    # a: anisotropy (x^4 + y^4 term weight)
    # b: cross term (xy(x^2+y^2)) weight
    # t: threshold offset
    # We want to explore shapes like: r^2 < R^2 + A*(x^4+y^4) + B*x*y*(x^2+y^2)
    # Let's define a shape function f(x,y) = x^2+y^2 - r^2 - a*(x^4+y^4) - b*x*y*(x^2+y^2)
    # If f < 0, it's dielectric.
    
    def make_cell(params):
        r, a, b, t = params
        x2 = Xc**2
        y2 = Yc**2
        r2 = x2 + y2
        x4y4 = x2**2 + y2**2
        xy_term = Xc * Yc * r2
        # Shape equation: r2 - r^2 - a*x4y4 - b*xy_term - t < 0
        val = r2 - r**2 - a*x4y4 - b*xy_term - t
        cell = (val < 0).astype(int)
        # Ensure symmetry is exact by construction? 
        # The grid is symmetric around 0.5. 
        # cell[i,j] corresponds to x=coords[j], y=coords[i].
        # Symmetry check: cell[i,j] == cell[31-i, 31-j] and cell[i,j] == cell[j,i].
        # Due to floating point, we might break it. Let's enforce it.
        cell = np.maximum(cell, cell[::-1, ::-1])
        cell = np.maximum(cell, cell.T)
        # And ensure it's binary
        cell = (cell > 0).astype(int)
        return cell.tolist()

    def compute_gap(cell_list):
        # Fast plane-wave surrogate
        # We need the Fourier coefficients of the permittivity inverse (for TM) or permittivity (for TE).
        # TM polarization: H_z is the field. The equation involves 1/epsilon.
        # So we need the Fourier series of 1/epsilon.
        cell_np = np.array(cell_list)
        eps = np.where(cell_np == 1, eps_high, eps_low)
        inv_eps = 1.0 / eps
        
        # Compute Fourier coefficients up to order N
        # The evaluator uses 289 plane waves (17x17 grid of G vectors).
        # Let's use the same for fidelity.
        N = 17
        G = []
        for i in range(-N//2, N//2 + 1):
            for j in range(-N//2, N//2 + 1):
                G.append((i, j))
        
        # Compute 1/epsilon Fourier coefficients
        # c_G = (1/A) * sum_{pixel} (1/eps_{ij}) * exp(-i * G . r_{ij})
        # r_{ij} = (x_j, y_i) in [0,1)
        # Since the cell is periodic, we can use FFT if we pad, but direct sum is fine for 32x32.
        # Actually, FFT is faster.
        # Let's compute the 2D FFT of inv_eps.
        # The DFT gives coefficients for G from -N/2 to N/2-1.
        # We need to be careful with the scaling and the grid size.
        # The plane waves are exp(i * 2*pi * (m*x + n*y)).
        # The discrete sum approximates the integral.
        # c_mn = (1/size^2) * sum_{i,j} inv_eps[i,j] * exp(-i * 2*pi * (m*j/size + n*i/size))
        # This is exactly the 2D DFT normalized by size^2.
        
        fft_inv_eps = np.fft.fft2(inv_eps) / (size * size)
        
        # Map G indices to FFT indices
        # G = (m, n). FFT index for m is (m + size//2) % size.
        # We need the coefficients for the specific G vectors used in the basis.
        
        # Build the Hamiltonian matrix
        # For TM, the eigenvalue problem is:
        # sum_{G'} [ (k+G)^2 / epsilon_{G-G'} ] c_{G'} = omega^2 c_G
        # Wait, the standard form for TM is:
        # (k+G)^2 * c_G - sum_{G'} [ (k+G)^2 / epsilon_{G-G'} ] ... no.
        # Let's recall the PWE method for TM.
        # The wave equation is: -div(1/eps grad H_z) = omega^2 H_z
        # In Fourier: sum_{G'} (k+G)^2 (1/eps)_{G-G'} H_{G'} = omega^2 H_G
        # So the matrix element H_{G,G'} = (k+G)^2 * inv_eps_{G-G'}
        # where inv_eps_{G-G'} is the Fourier coefficient of 1/eps.
        
        # We need to extract the coefficients for G-G' where G, G' are in our basis.
        # Basis size is 289.
        # Let's create a lookup for inv_eps coefficients.
        # coeff[(m, n)] = fft_inv_eps[(m+size//2)%size, (n+size//2)%size]
        
        # Precompute the matrix of inv_eps coefficients for all differences
        # Actually, we can just index into the FFT array.
        
        # We need to solve the eigenvalue problem for 30 k-points.
        # To be fast, we'll only use a few k-points for the surrogate, or all 30 if fast enough.
        # 289x289 eig is ~ 289^3 ~ 24M ops. 30 points -> 720M ops.
        # In Python with NumPy, this might be slow. Let's check.
        # np.linalg.eigvalsh is BLAS-backed, should be fast.
        
        # Let's use a reduced set of k-points for the surrogate to save time,
        # or just use the critical ones: Gamma, X, M.
        # The gap is usually between bands that are degenerate at high-symmetry points or close to them.
        # For a square lattice, the 4th and 5th bands often interact near X or M.
        
        # Let's use 5 k-points: Gamma (0,0), X (0.5,0), M (0.5,0.5), and two intermediates.
        # k-points in units of 2*pi (so x, y in [0, 0.5] for irreducible BZ)
        k_points = [
            (0.0, 0.0),
            (0.5, 0.0),
            (0.5, 0.5),
            (0.25, 0.0),
            (0.25, 0.25),
            (0.0, 0.25),
            (0.4, 0.1),
            (0.1, 0.4),
        ]
        
        min_omega_5 = float('inf')
        max_omega_4 = -float('inf')
        
        for kx, ky in k_points:
            # Build Hamiltonian
            H = np.zeros((len(G), len(G)), dtype=np.complex128)
            for i, (gi, gj) in enumerate(G):
                # (k + G)^2
                kg_x = kx + gi
                kg_y = ky + gj
                kg_sq = kg_x**2 + kg_y**2
                
                for j, (gi2, gj2) in enumerate(G):
                    # Difference G - G'
                    dm = gi - gi2
                    dn = gj - gj2
                    # Get coefficient of 1/eps for this difference
                    # FFT index
                    idx_m = (dm + size//2) % size
                    idx_n = (dn + size//2) % size
                    coeff = fft_inv_eps[idx_m, idx_n]
                    H[i, j] = kg_sq * coeff
            
            # H is Hermitian? 
            # H_{G,G'} = (k+G)^2 * c_{G-G'}
            # H_{G',G} = (k+G')^2 * c_{G'-G}
            # c_{-G} = c_G^* for real eps.
            # So H_{G',G} = (k+G')^2 * c_{G-G'}^*
            # This is NOT Hermitian in this form unless (k+G)^2 = (k+G')^2, which is false.
            # Wait, the standard formulation is:
            # sum_{G'} (k+G)^2 / eps_{G-G'} H_{G'} = omega^2 H_G
            # This is a generalized eigenvalue problem? No, it's a standard one if we define the operator correctly.
            # Actually, the operator is T = (k+G)^2 * (1/eps)_G * ...
            # Let's check the Hermiticity.
            # The operator is L = (k+G)^2 * (1/eps)_{G-G'}
            # Is L Hermitian?
            # L_{G,G'} = (k+G)^2 c_{G-G'}
            # L_{G',G}^* = ((k+G')^2 c_{G'-G})^* = (k+G')^2 c_{G-G'}^* = (k+G')^2 c_{G-G'}
            # For L to be Hermitian, we need (k+G)^2 = (k+G')^2, which is not true.
            # Ah, the plane-wave method for TM is usually formulated as a generalized eigenvalue problem or by symmetrizing.
            # Actually, the matrix is NOT Hermitian. We must use np.linalg.eigvals, not eigvalsh.
            # And we take the real parts of the eigenvalues (they should be real).
            
            eigvals = np.linalg.eigvals(H)
            real_eigs = np.sort(np.real(eigvals))
            
            # Take the 5th and 4th bands
            # Note: The lowest band is the constant mode (G=0, k=0) which is zero.
            # We need to skip the zero mode? 
            # The task says "between the fourth and fifth TM bands".
            # Usually, the bands are indexed starting from 1.
            # Band 1 is the lowest non-zero? Or including zero?
            # In photonic crystals, the "gap" is between bands. The zero frequency is not a band.
            # Let's assume the evaluator skips the zero eigenvalue.
            # Let's find the 4th and 5th non-zero eigenvalues.
            non_zero = real_eigs[real_eigs > 1e-10]
            if len(non_zero) >= 5:
                w4 = non_zero[3] # 4th band
                w5 = non_zero[4] # 5th band
                max_omega_4 = max(max_omega_4, w4)
                min_omega_5 = min(min_omega_5, w5)
            else:
                return -1.0 # Invalid
        
        if min_omega_5 == float('inf'):
            return -1.0
            
        gap = min_omega_5 - max_omega_4
        midgap = (min_omega_5 + max_omega_4) / 2.0
        if midgap <= 0:
            return -1.0
        ratio = 2.0 * gap / midgap
        return ratio

    # Initial guess: a circle
    # r^2 < R^2 -> r=0.2, a=0, b=0, t=0
    x0 = [0.2, 0.0, 0.0, 0.0]
    
    # Nelder-Mead optimization
    # We need to be careful with the time budget.
    # Each evaluation of compute_gap takes some time.
    # Let's estimate: 8 k-points, 289x289 eig.
    # Let's run one evaluation to time it.
    
    t0 = time.time()
    score0 = compute_gap(make_cell(x0))
    t_eval = time.time() - t0
    sys.stderr.write(f"Initial score: {score0}, eval time: {t_eval:.2f}s\n")
    
    if time.time() > deadline - 10:
        return make_cell(x0)
    
    # Define objective function for minimization (negative score)
    def objective(params):
        if time.time() > deadline - 5:
            return 1e9 # Timeout penalty
        cell = make_cell(params)
        score = compute_gap(cell)
        if score < 0:
            return 1e9
        return -score
    
    # Run Nelder-Mead
    # Use scipy if available, otherwise a simple implementation.
    try:
        from scipy.optimize import minimize
        res = minimize(objective, x0, method='Nelder-Mead', 
                       options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-4})
        best_params = res.x
        best_score = -res.fun
    except ImportError:
        # Fallback: simple grid search or random restarts
        best_params = x0
        best_score = score0
        
        # Try some manual perturbations
        candidates = [
            [0.2, 0.1, 0.0, 0.0],
            [0.2, -0.1, 0.0, 0.0],
            [0.2, 0.0, 0.1, 0.0],
            [0.2, 0.0, -0.1, 0.0],
            [0.2, 0.2, 0.0, 0.0],
            [0.15, 0.0, 0.0, 0.0],
            [0.25, 0.0, 0.0, 0.0],
        ]
        
        for p in candidates:
            if time.time() > deadline - 10:
                break
            s = compute_gap(make_cell(p))
            if s > best_score:
                best_score = s
                best_params = p
                
        # Local refinement around best
        for _ in range(50):
            if time.time() > deadline - 10:
                break
            p = list(best_params)
            for i in range(len(p)):
                p[i] += np.random.uniform(-0.02, 0.02)
            s = compute_gap(make_cell(p))
            if s > best_score:
                best_score = s
                best_params = p

    # Final check
    if time.time() > deadline - 5:
        return make_cell(best_params)
    
    # Verify the best score
    final_cell = make_cell(best_params)
    final_score = compute_gap(final_cell)
    sys.stderr.write(f"Final score: {final_score}\n")
    
    return final_cell

# EVOLVE-BLOCK-END
