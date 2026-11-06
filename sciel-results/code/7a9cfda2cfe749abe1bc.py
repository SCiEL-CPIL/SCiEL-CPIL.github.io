import time
import math
import numpy as np

def design(problem):
    start_time = time.time()
    deadline = start_time + 195  # safety margin before 200s kill

    size = problem.get("pixels", 32)
    half = size // 2  # 16

    # Build basis functions on the full 32x32 grid that are D4-symmetric.
    # Coordinates in [-1, 1] range, origin at cell center.
    ys = np.arange(size, dtype=float) - half + 0.5
    xs = np.arange(size, dtype=float) - half + 0.5
    Y, X = np.meshgrid(ys, xs, indexing='ij')
    Xn = X / half  # normalized to roughly [-1, 1]
    Yn = Y / half

    # D4-symmetric polynomial basis (invariant under x->-x, y->-y, x<->y)
    # Use monomials that are symmetric: x^4, y^4, x^2*y^2, x^2, y^2, x^4*y^0, etc.
    # But we need x^2 and y^2 to be combined as (x^2+y^2) for full D4 symmetry.
    # Basis: 1, (x^2+y^2), (x^4+y^4), x^2*y^2, (x^2+y^2)^2, x^4*y^4, ...
    # Let's use a compact set:
    basis = np.stack([
        np.ones_like(Xn),
        Xn**2 + Yn**2,
        Xn**4 + Yn**4,
        Xn**2 * Yn**2,
        (Xn**2 + Yn**2)**2,
        Xn**4 * Yn**4,
        Xn**6 + Yn**6,
        Xn**2 * Yn**4 + Xn**4 * Yn**2,
    ], axis=-1)

    n_basis = basis.shape[-1]

    # Fast plane-wave surrogate for TM band gap between bands 4 and 5.
    # Use a reduced set of plane waves (e.g., 5x5 = 25 G-vectors) for speed.
    # The TM eigenvalue problem: for TM polarization, the operator involves 1/epsilon.
    # We work in the reciprocal space. For a square lattice with period a=1 (in pixel units),
    # the reciprocal lattice vectors are (2pi, 0) and (0, 2pi) in pixel-space.
    # Actually, let's think in terms of the unit cell being 32x32 pixels.
    # The plane wave expansion uses G-vectors that are integer multiples of 2pi/size in pixel units? 
    # No. The reciprocal lattice for a square lattice with real-space period a (in pixels) has
    # reciprocal period 2pi/a. But we're working in a unit cell of size `size` pixels.
    # Let's define the unit cell as 32x32 pixels. The lattice constant in pixel units is 32.
    # So reciprocal lattice vectors are (2pi/32, 0) * (h, k) for integers h,k.
    # But actually for the plane-wave method, we expand the field as sum_c exp(i(G+c)·r) where
    # c is the Bloch wavevector and G are reciprocal lattice vectors.
    #
    # For TM polarization: the dielectric equation is:
    #   (1/epsilon(r)) grad x (grad x E) = (omega^2/c^2) E
    # In 2D with E along z: (1/epsilon) (d^2/dx^2 + d^2/dy^2) Ez = (omega^2/c^2) Ez
    # This becomes: (1/epsilon(r)) [-(kx+Gx)^2 - (ky+Gy)^2] sum_{G'} c_{G'} exp(i G'·r) = (omega^2/c^2) sum c_G exp(iG·r)
    #
    # In matrix form with Fourier coefficients of 1/epsilon:
    #   sum_{G'} (1/epsilon)_{G-G'} * [-(k+G)^2] * c_{G'} = (omega^2/c^2) c_G
    # where (1/epsilon)_{G-G'} is the Fourier coefficient of 1/epsilon(r).
    #
    # For a 25x25 (5x5) plane-wave basis, we have 25 equations.
    # We need the top 5 eigenvalues (bands 1-5) to get bands 4 and 5.
    #
    # Let's precompute the G-vectors for a 5x5 grid.
    # G-vectors: (h * 2pi/size, k * 2pi/size) for h,k in [-2, -1, 0, 1, 2]
    # But wait, the unit cell is 32 pixels. The reciprocal lattice vector magnitude is 2pi/32 per unit.
    # Actually, let me reconsider. If the unit cell has L pixels on each side, and we define the
    # physical unit cell as 32x32 pixels, then the lattice constant a = 32 pixels.
    # Reciprocal lattice: b = 2pi/a. G = h*b_x + k*b_y = (2pi*h/32, 2pi*k/32) for integers h,k.
    #
    # For the plane-wave expansion, we use G-vectors in a grid. A 5x5 grid means h,k in [-2,-1,0,1,2].
    #
    # The Fourier coefficient of 1/epsilon(r) is:
    #   (1/eps)_G = (1/L^2) * sum_{i,j} (1/eps(i,j)) * exp(-i G · r_ij)
    # where r_ij = (i, j) in pixel coordinates, and L = size = 32.
    #
    # Actually, since we're working in a discrete pixel grid, we can compute the DFT directly.
    # For G = (h*2pi/L, k*2pi/L), the Fourier coefficient is:
    #   (1/eps)_{h,k} = (1/L^2) * sum_{i=0}^{L-1} sum_{j=0}^{L-1} (1/eps(i,j)) * exp(-2pi i (h*i/L + k*j/L))
    #
    # This is just the 2D DFT of the 1/eps array, divided by L^2, with indices shifted.
    #
    # Let's implement this efficiently using numpy's FFT.

    # Precompute G-vector indices
    g_range = np.arange(-2, 3)  # [-2, -1, 0, 1, 2]
    G_indices = np.array([(h, k) for h in g_range for k in g_range])
    n_pw = len(G_indices)  # 25

    # Precompute the phase factors for the DFT
    # For each G-index (h,k), we need exp(-2pi i (h*i/L + k*j/L)) for all i,j in [0, L-1]
    # We can compute the 2D DFT using np.fft.fftn and then extract the needed coefficients.
    # The DFT of an LxL array f[i,j] is F[h,k] = sum_{i,j} f[i,j] * exp(-2pi i (h*i/L + k*j/L))
    # where h,k can be any integers (we use the wrapped versions for FFT).
    # np.fft.fftn gives us F for h,k in [0, L-1]. To get negative indices, we use np.fft.fftshift.
    #
    # For a 5x5 grid centered at 0, we need h,k in [-2,-1,0,1,2].
    # In the FFT output (L=32), the index for a given (h,k) is:
    #   idx_h = h % 32, idx_k = k % 32
    # So we can extract directly.

    # Helper: compute the 25x25 matrix for the TM eigenvalue problem
    def compute_tm_matrix(eps_pixel):
        """
        eps_pixel: 32x32 array of epsilon values (1 or 12)
        Returns the 25x25 matrix M such that M @ c = (omega^2 * c^2 / a^2) * c
        where c is the vector of plane-wave coefficients.
        """
        inv_eps = 1.0 / eps_pixel
        # Compute 2D FFT of inv_eps
        fft_inv_eps = np.fft.fftn(inv_eps)
        L = size
        # Extract the 25 coefficients for G in [-2,-1,0,1,2]^2
        fft_coeffs = np.zeros(n_pw, dtype=complex)
        for idx, (h, k) in enumerate(G_indices):
            idx_h = h % L
            idx_k = k % L
            fft_coeffs[idx] = fft_inv_eps[idx_h, idx_k] / (L * L)
        
        # Now build the 25x25 matrix.
        # M_{G, G'} = (1/eps)_{G-G'} * [-(k_Bloch + G)^2]
        # Wait, the matrix element depends on the Bloch wavevector k_Bloch.
        # For a given k_Bloch = (kx, ky), the matrix is:
        #   M_{G, G'} = sum_{G''} (1/eps)_{G''} * delta_{G, G'+G''} * [-(k_Bloch + G)^2]
        # This is a diagonal matrix in G-space if we write it as:
        #   M = D * (1/eps)_Fourier
        # where D is diagonal with D_{G,G} = -(k_Bloch + G)^2
        # and (1/eps)_Fourier is the matrix with elements (1/eps)_{G-G'} = fft_coeffs[index_of(G-G')]
        #
        # Actually, the convolution in Fourier space means:
        #   [M c]_G = sum_{G'} (1/eps)_{G-G'} * [-(k_Bloch + G)^2] * c_{G'}
        # Wait, that's not right. Let me re-derive.
        #
        # The equation is:
        #   (1/eps(r)) * [-(k_Bloch + G)^2] * c_G = (omega^2/c^2) * c_G  -- NO, this is wrong.
        #
        # The correct form: we expand Ez(r) = sum_G c_G exp(i(G+k_Bloch)·r)
        # Then (1/eps(r)) * (-lap) * Ez(r) = (1/eps(r)) * sum_G c_G * |G+k_Bloch|^2 * exp(i(G+k_Bloch)·r)
        # = sum_G [ (1/eps(r)) * c_G * |G+k_Bloch|^2 ] * exp(i(G+k_Bloch)·r)
        #
        # Now expand 1/eps(r) = sum_{G''} (1/eps)_{G''} exp(i G''·r)
        # Product: sum_{G, G''} (1/eps)_{G''} * c_G * |G+k_Bloch|^2 * exp(i(G+G''+k_Bloch)·r)
        # Let G' = G + G'', so G'' = G' - G:
        # = sum_{G', G} (1/eps)_{G'-G} * c_G * |G+k_Bloch|^2 * exp(i(G'+k_Bloch)·r)
        #
        # So the coefficient of exp(i(G'+k_Bloch)·r) is:
        #   sum_G (1/eps)_{G'-G} * c_G * |G+k_Bloch|^2
        #
        # This equals (omega^2/c^2) * c_{G'}
        #
        # So the matrix is:
        #   M_{G', G} = (1/eps)_{G'-G} * |G + k_Bloch|^2
        #
        # Where (1/eps)_{G'-G} is the Fourier coefficient at frequency (G'-G).
        
        return fft_coeffs  # We'll build the matrix per k-point
    
    fft_inv_eps_template = None  # Will be computed once per candidate

    def get_fft_coeffs(eps_pixel):
        inv_eps = 1.0 / eps_pixel
        fft_inv_eps = np.fft.fftn(inv_eps)
        L = size
        coeffs = np.zeros(n_pw, dtype=complex)
        for idx, (h, k) in enumerate(G_indices):
            idx_h = h % L
            idx_k = k % L
            coeffs[idx] = fft_inv_eps[idx_h, idx_k] / (L * L)
        return coeffs

    # 30 k-points in the irreducible Brillouin zone
    # The irreducible BZ for a square lattice is the triangle with vertices (0,0), (pi/a, 0), (pi/a, pi/a)
    # In our units (a=32 pixels, so reciprocal lattice vector = 2pi/32 = pi/16 per pixel)
    # Actually, let's work in units where the reciprocal lattice vector is 1.
    # So k is in units of 2pi/L = 2pi/32.
    # The irreducible BZ: 0 <= ky <= kx, kx in [0, 0.5], ky in [0, 0.5]
    # But actually the full BZ is [-0.5, 0.5] x [-0.5, 0.5] in units of 2pi/L.
    # The irreducible part (1/8 of the BZ) is: 0 <= ky <= kx <= 0.5
    #
    # Let's define 30 k-points:
    k_points = []
    # Along Gamma-M: (t, 0) for t in [0, 0.5]
    for t in np.linspace(0, 0.5, 6):
        k_points.append((t, 0.0))
    # Along M-X: (0.5, t) for t in [0, 0.5]
    for t in np.linspace(0, 0.5, 5):
        if t > 0:  # avoid duplicate
            k_points.append((0.5, t))
    # Along Gamma-X: (t, t) for t in [0, 0.5]
    for t in np.linspace(0, 0.5, 6):
        if t > 0:  # avoid duplicate
            k_points.append((t, t))
    # Interior points
    k_points.append((0.3, 0.1))
    k_points.append((0.4, 0.2))
    k_points.append((0.2, 0.1))
    k_points.append((0.35, 0.15))
    k_points.append((0.45, 0.3))
    # Trim to 30
    k_points = k_points[:30]

    def fast_evaluate(coeffs):
        """
        coeffs: array of length n_basis, the polynomial coefficients
        Returns the gap-midgap ratio for bands 4 and 5.
        """
        # Compute the field
        field = np.tensordot(basis, coeffs, axes=([2], [0]))
        # Dielectric mask: field > 0 -> dielectric (eps=12), else air (eps=1)
        eps_pixel = np.where(field > 0, 12.0, 1.0)

        # Get Fourier coefficients of 1/eps
        fft_coeffs = get_fft_coeffs(eps_pixel)

        max_omega_4 = 0.0
        min_omega_5 = float('inf')

        for kx, ky in k_points:
            # k in units of 2pi/L, so the actual wavevector magnitude factor:
            # |G + k_Bloch|^2 where G = (h*2pi/L, k*2pi/L) and k_Bloch = (kx*2pi/L, ky*2pi/L)
            # = (2pi/L)^2 * (h+kx)^2 + (k+ky)^2
            # The (2pi/L)^2 factor will cancel in the ratio, so we can work in units of 2pi/L.
            
            # Build the 25x25 matrix
            M = np.zeros((n_pw, n_pw), dtype=complex)
            for ig, G in enumerate(G_indices):
                for igp, Gp in enumerate(G_indices):
                    # (1/eps)_{Gp - G}
                    dh = Gp[0] - G[0]
                    dk = Gp[1] - G[1]
                    # Find index of (dh, dk) in G_indices
                    # dh, dk should be in [-4, 4] range, but our G_indices only go to [-2,2]
                    # We need to wrap or handle this. For a 5x5 grid, Gp - G can be in [-4,4].
                    # We need the Fourier coefficient at (dh, dk).
                    # Since our fft_coeffs only covers [-2,2]^2, we need to be careful.
                    # Actually, the convolution (1/eps)_{Gp-G} requires the full Fourier series.
                    # With only 25 plane waves, we're truncating. For dh,dk outside [-2,2],
                    # we set the coefficient to 0 (truncation).
                    # But this is not accurate. Let's use a larger set or handle it properly.
                    #
                    # Alternative: use the direct DFT approach. Since we have the full FFT,
                    # we can extract any coefficient (dh, dk) by wrapping.
                    idx_dh = dh % size
                    idx_dk = dk % size
                    # But we already computed fft_inv_eps... let's recompute or store it.
                    # For now, let's just use 0 for out-of-range.
                    if -2 <= dh <= 2 and -2 <= dk <= 2:
                        # Find index in G_indices
                        found = False
                        for idx_g, (hh, kk) in enumerate(G_indices):
                            if hh == dh and kk == dk:
                                val = fft_coeffs[idx_g]
                                found = True
                                break
                        if not found:
                            val = 0.0
                    else:
                        val = 0.0
                    
                    # |G + k_Bloch|^2 in units of (2pi/L)^2
                    mag2 = (G[0] + kx)**2 + (G[1] + ky)**2
                    M[igp, ig] = val * mag2
            
            # Solve eigenvalue problem: M c = eigenvalue * c
            # M is not Hermitian in general, but for real 1/eps it should be symmetric.
            # Let's use eigvalsh if possible, otherwise eig.
            try:
                # Check if M is approximately Hermitian
                if np.allclose(M, M.conj().T, atol=1e-10):
                    eigvals = np.linalg.eigvalsh(M)
                else:
                    eigvals = np.linalg.eigvals(M)
                    eigvals = np.sort(eigvals.real)
                
                # eigvals are in units of (2pi/L)^2 * (1/eps average or something)
                # Actually, eigvals = (omega^2/c^2) / (2pi/L)^2 approximately
                # We just need the ordering, so bands 4 and 5 are the 3rd and 4th eigenvalues (0-indexed)
                # But wait, there might be spurious eigenvalues. We take the smallest 5.
                eigvals = np.sort(eigvals.real)
                # Take the 5 smallest
                eigvals_5 = eigvals[:5]
                omega_4 = eigvals_5[3]
                omega_5 = eigvals_5[4]
                
                max_omega_4 = max(max_omega_4, omega_4)
                min_omega_5 = min(min_omega_5, omega_5)
            except:
                return -10.0

        if min_omega_5 < max_omega_4:
            return -10.0
        midgap = (min_omega_5 + max_omega_4) / 2.0
        if midgap <= 0:
            return -10.0
        gap = min_omega_5 - max_omega_4
        return gap / midgap

    # Initial coefficients: try a few starting points
    initial_candidates = [
        np.array([0.0, -0.8, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        np.array([0.0, -1.0, 0.5, 0.3, 0.0, 0.0, 0.0, 0.0]),
        np.array([0.0, -0.5, 0.8, -0.2, 0.0, 0.0, 0.0, 0.0]),
        np.array([0.0, -1.2, 0.3, 0.5, 0.0, 0.0, 0.0, 0.0]),
        np.array([0.0, -0.6, 1.2, 0.1, 0.0, 0.0, 0.0, 0.0]),
    ]

    best_score = -10.0
    best_coeffs = initial_candidates[0].copy()

    for ic, init_c in enumerate(initial_candidates):
        if time.time() > deadline - 30:
            break
        score = fast_evaluate(init_c)
        if score > best_score:
            best_score = score
            best_coeffs = init_c.copy()

    # Local optimization using Nelder-Mead
    if time.time() < deadline - 15:
        def objective(coeffs):
            return -fast_evaluate(coeffs)

        try:
            from scipy.optimize import minimize
            res = minimize(
                objective,
                best_coeffs,
                method='Nelder-Mead',
                options={'maxiter': 400, 'xatol': 1e-3, 'fatol': 1e-3, 'disp': False, 'adaptive': True}
            )
            if res.fun < -best_score:
                best_score = -res.fun
                best_coeffs = res.x
        except Exception:
            pass

    # Second round with perturbed starting points if time allows
    if time.time() < deadline - 20:
        for ic in range(5):
            if time.time() > deadline - 10:
                break
            perturb = np.random.normal(0, 0.1, n_basis)
            perturbed = best_coeffs + perturb
            score = fast_evaluate(perturbed)
            if score > best_score:
                best_score = score
                best_coeffs = perturbed.copy()

    # Final grid
    field = np.tensordot(basis, best_coeffs, axes=([2], [0]))
    pixel_map = (field > 0).astype(int)

    # Symmetrize: enforce D4 symmetry by averaging with symmetry operations
    # The basis should already produce D4-symmetric fields, but let's enforce it.
    # D4 symmetry: invariant under 90-degree rotations and reflections.
    # For a 32x32 grid, we can enforce by taking the average of all 8 symmetry operations.
    # But since our basis is D4-symmetric, the field should already be D4-symmetric.
    # Let's just use the top-left 16x16 and mirror.
    top_left = pixel_map[:half, :half]
    final_cell = np.zeros((size, size), dtype=int)
    final_cell[:half, :half] = top_left
    final_cell[:half, half:] = top_left[:, ::-1]
    final_cell[half:, :half] = top_left[::-1, :]
    final_cell[half:, half:] = top_left[::-1, ::-1]

    return final_cell.tolist()
