import numpy as np
import time

def design(problem):
    """
    Designs a C4-symmetric 32x32 photonic-crystal unit cell to maximize
    the TM band gap between bands 4 and 5 using a superellipse-based
    geometric parameterization with local binary search.
    """
    start_time = time.time()
    deadline = start_time + 215.0  # Safety margin before 230s kill

    size = problem["pixels"]
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)

    # Precompute grid coordinates
    x = np.linspace(0.5/size, 1 - 0.5/size, size)
    X, Y = np.meshgrid(x, x, indexing='ij')
    Xc = X - 0.5
    Yc = Y - 0.5

    # Precompute G vectors for plane-wave method
    N_pw = 8
    G_vals = np.arange(-N_pw, N_pw + 1)
    Gx, Gy = np.meshgrid(G_vals, G_vals, indexing='ij')
    G_vectors = np.column_stack((Gx.ravel(), Gy.ravel()))
    num_pw = len(G_vectors)

    # Precompute G-G' differences
    G_diff_x = Gx.ravel()[:, None] - Gx.ravel()[None, :]
    G_diff_y = Gy.ravel()[:, None] - Gy.ravel()[None, :]
    # Indices into the 32x32 FFT grid (mod 32)
    idx_x = (G_diff_x + 16) % 32
    idx_y = (G_diff_y + 16) % 32

    # K points: 30 points in irreducible BZ
    # The BZ of a square lattice with a=1 has boundaries at +/- pi (or +/- pi/a)
    # Irreducible BZ: kx >= ky >= 0, kx + ky <= pi (for the standard square BZ)
    # Actually, for a square lattice, the BZ is [-pi, pi] x [-pi, pi].
    # The 1/8 wedge is 0 <= ky <= kx, kx + ky <= pi? No.
    # The symmetry group of the square BZ is D4. The fundamental domain is a triangle
    # with vertices (0,0), (pi,0), (pi/2, pi/2).
    # Let's generate points on the edges and inside.
    k_points = []
    # Edge Gamma-M: (0,0) to (pi/2, pi/2)
    for t in np.linspace(0, np.pi/2, 10):
        k_points.append((t, t))
    # Edge M-X: (pi/2, pi/2) to (pi, 0)
    for t in np.linspace(np.pi/2, 0, 10):
        k_points.append((np.pi, t))
    # Edge Gamma-X: (0,0) to (pi, 0)
    for t in np.linspace(0, np.pi, 10):
        k_points.append((t, 0))

    # Remove duplicates
    seen = set()
    unique_k = []
    for k in k_points:
        key = (round(k[0], 10), round(k[1], 10))
        if key not in seen:
            seen.add(key)
            unique_k.append(k)

    def get_cell_from_params(params):
        """
        Generates a 32x32 binary cell from superellipse parameters.
        params: (n, r, offset)
        n: exponent (shape control, 1=diamond, 2=circle, inf=square)
        r: radius
        offset: shift from center (0.5, 0.5) - but symmetry requires offset=0 or symmetric
        For C4 symmetry, we can only have a single central feature or 4-fold symmetric features.
        Let's use a central superellipse hole (air) in a dielectric background.
        Or a central superellipse dielectric in an air background.
        Given eps_high=12, dielectric is 1.
        Usually, holes (air) in dielectric are better for band gaps.
        Let's try: Background = Dielectric (1). Hole = Air (0).
        The hole is a superellipse centered at (0.5, 0.5).
        Equation: |x-0.5|^n + |y-0.5|^n <= r^n
        """
        n = params[0]
        r = params[1]
        # n should be >= 1. If n is very large, it becomes a square.
        # If n < 1, it's concave. Let's allow n in [1, 10].
        
        # Clip n to avoid numerical issues with large powers
        n_clipped = np.clip(n, 1.0, 10.0)
        
        # |Xc| and |Yc| are precomputed? No, Xc, Yc are signed.
        abs_Xc = np.abs(Xc)
        abs_Yc = np.abs(Yc)
        
        # Superellipse test
        # To avoid overflow for large n, we can use log or just clip.
        # For n=10, 0.5^10 is small.
        # dist^n = (abs_Xc^n + abs_Yc^n)^(1/n) <= r
        # Equivalent to abs_Xc^n + abs_Yc^n <= r^n
        
        # Handle r=0
        if r <= 0:
            return np.ones((size, size), dtype=int)
            
        # Compute distance metric
        # For numerical stability, if n is large, abs_Xc^n underflows to 0 for abs_Xc < 1.
        # This is fine, it effectively creates a square of size r.
        
        dist_term = np.power(abs_Xc, n_clipped) + np.power(abs_Yc, n_clipped)
        hole_mask = dist_term <= np.power(r, n_clipped)
        
        # 1 = Dielectric, 0 = Air
        # Background is Dielectric (1), Hole is Air (0)
        cell = np.ones((size, size), dtype=int)
        cell[hole_mask] = 0
        
        return cell

    def evaluate_cell(cell):
        """
        Computes the band gap using a plane-wave method.
        cell: 32x32 array of 0/1.
        """
        # 1 = Dielectric (eps=12), 0 = Air (eps=1)
        eps = np.where(cell == 1, eps_high, eps_low)
        mu = 1.0 / eps

        # Compute Fourier coefficients of mu
        mu_fft = np.fft.fft2(mu)
        mu_fft /= (size * size)
        
        # Extract coefficients for G-G' differences
        # mu_diff[i,j] = mu_fft[idx_x[i,j], idx_y[i,j]]
        mu_diff = mu_fft[idx_x, idx_y]

        min_omega_5 = float('inf')
        max_omega_4 = float('-inf')

        for kx, ky in unique_k:
            k = np.array([kx, ky])
            # k is in radians per unit length. G is integer reciprocal lattice vectors.
            # The plane wave is exp(i (k + G) . r).
            # The eigenvalue equation for TM polarization is:
            # sum_{G'} mu(G - G') (k + G')^2 c_{G'} = omega^2 c_G
            # (k + G)^2 is the eigenvalue of the identity part?
            # Actually, the Hamiltonian H_{GG'} = mu(G-G') * (k+G')^2.
            # Wait, the standard PW expansion for TM is:
            # (k+G)^2 / eps(r) = (k+G)^2 * mu(r)
            # Projecting onto plane waves:
            # H_{GG'} = mu(G-G') * (k+G')^2
            # Note: (k+G')^2 depends only on G', so it's a diagonal matrix multiplied by mu matrix?
            # No, mu(G-G') is a matrix in G, G' space.
            # Let D be diagonal with D_{G'G'} = (k+G')^2.
            # Then H = M * D, where M_{GG'} = mu(G-G').
            # We need eigenvalues of H.
            
            kg_sq = np.sum((k[None, :] + G_vectors) ** 2, axis=1)
            # kg_sq is (k+G)^2 for each G.
            
            # H = mu_diff * kg_sq[None, :]  (broadcasting: mu_diff is (N,N), kg_sq is (N,))
            # H_{GG'} = mu_diff[G, G'] * kg_sq[G']
            
            H = mu_diff * kg_sq[None, :]
            
            # H is not Hermitian because mu(G-G') != mu*(G'-G) generally?
            # mu(r) is real, so mu(-G) = mu*(G).
            # mu_diff[G, G'] = mu(G-G').
            # mu_diff[G', G] = mu(G'-G) = mu(-(G-G')) = mu*(G-G') = mu_diff[G, G']*.
            # So M is Hermitian.
            # But H = M * D. If M and D don't commute, H is not Hermitian.
            # However, the problem is self-adjoint.
            # The correct formulation is:
            # sum_{G'} mu(G-G') (k+G')^2 c_{G'} = omega^2 c_G
            # This is a generalized eigenvalue problem? No.
            # It is a standard eigenvalue problem for the non-Hermitian matrix H.
            # But the eigenvalues should be real.
            # Alternatively, we can symmetrize.
            # Let's just use eigvals and take real parts.
            
            eigvals = np.linalg.eigvals(H)
            real_eigvals = np.sort(np.real(eigvals))
            
            if len(real_eigvals) >= 5:
                omega_4 = np.sqrt(real_eigvals[3])
                omega_5 = np.sqrt(real_eigvals[4])
                
                max_omega_4 = max(max_omega_4, omega_4)
                min_omega_5 = min(min_omega_5, omega_5)

        if min_omega_5 == float('inf') or max_omega_4 == float('-inf'):
            return -1.0

        gap = min_omega_5 - max_omega_4
        midgap = 0.5 * (min_omega_5 + max_omega_4)
        if midgap == 0:
            return -1.0
        ratio = 2 * gap / midgap
        return ratio

    # Optimization: Search over (n, r)
    best_score = -np.inf
    best_params = (2.0, 0.2)  # Circle, r=0.2

    # Grid search for initial good region
    n_vals = np.linspace(1.0, 5.0, 10)
    r_vals = np.linspace(0.1, 0.4, 10)
    
    for n in n_vals:
        if time.time() > deadline - 20:
            break
        for r in r_vals:
            params = (n, r)
            cell = get_cell_from_params(params)
            score = evaluate_cell(cell)
            if score > best_score:
                best_score = score
                best_params = params

    # Local refinement
    # Perturb n and r
    for _ in range(50):
        if time.time() > deadline - 10:
            break
        
        n_pert = best_params[0] + np.random.normal(0, 0.2)
        r_pert = best_params[1] + np.random.normal(0, 0.02)
        
        # Keep r positive
        r_pert = max(r_pert, 0.05)
        n_pert = max(n_pert, 1.0)
        
        params = (n_pert, r_pert)
        cell = get_cell_from_params(params)
        score = evaluate_cell(cell)
        
        if score > best_score:
            best_score = score
            best_params = params

    # Try the opposite: Dielectric island in Air background
    # This might be better for higher bands?
    best_score_opp = -np.inf
    best_params_opp = (2.0, 0.2)
    
    for n in n_vals:
        if time.time() > deadline - 20:
            break
        for r in r_vals:
            params = (n, r)
            # Create island
            cell = get_cell_from_params(params)
            # Invert: 1 becomes 0, 0 becomes 1
            cell = 1 - cell
            score = evaluate_cell(cell)
            if score > best_score_opp:
                best_score_opp = score
                best_params_opp = params

    if best_score_opp > best_score:
        best_score = best_score_opp
        best_params = best_params_opp
        # Note: We need to remember to invert in the final step
        final_invert = True
    else:
        final_invert = False

    # Final local refinement on the best type
    for _ in range(50):
        if time.time() > deadline - 5:
            break
        
        n_pert = best_params[0] + np.random.normal(0, 0.1)
        r_pert = best_params[1] + np.random.normal(0, 0.01)
        r_pert = max(r_pert, 0.05)
        n_pert = max(n_pert, 1.0)
        
        params = (n_pert, r_pert)
        cell = get_cell_from_params(params)
        if final_invert:
            cell = 1 - cell
        score = evaluate_cell(cell)
        
        if score > best_score:
            best_score = score
            best_params = params

    # Generate final cell
    n, r = best_params
    cell = get_cell_from_params((n, r))
    if final_invert:
        cell = 1 - cell

    # Convert to list of lists
    final_cell = [[int(cell[i, j]) for j in range(size)] for i in range(size)]
    
    return final_cell
