"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import math

# EVOLVE-BLOCK-START
def design(problem):
    """
    Designs a 32x32 unit cell for a TM band gap between bands 4 and 5.
    Strategy:
    1. Use a continuous parameterization of the dielectric field using a low-order
       Chebyshev series constrained to C4v symmetry.
    2. Optimize the coefficients using stochastic hill-climbing with adaptive step sizes.
    3. The objective function is a fast, approximate plane-wave calculation (smaller basis)
       that correlates with the full 289-PW evaluator score.
    4. Time management ensures the function returns within the 200s limit.
    """
    size = problem["pixels"]  # 32
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)
    
    # Time management
    start_time = time.time()
    deadline = start_time + 190.0  # Safety margin for 200s limit

    # --- 1. Parameterization ---
    # We represent the dielectric field as:
    # eps(x, y) = 0.5 * (eps_low + eps_high) + 0.5 * (eps_high - eps_low) * tanh(f(x, y))
    # where f(x, y) is a sum of Chebyshev polynomials T_m(2x-1) * T_n(2y-1)
    # constrained by C4v symmetry.
    
    # Chebyshev polynomials T_k(t) are defined on [-1, 1].
    # Let u = 2*x - 1, v = 2*y - 1.
    # Symmetry requirements:
    # f(x, y) = f(1-x, y) => f(u, v) is even in u.
    # f(x, y) = f(x, 1-y) => f(u, v) is even in v.
    # f(x, y) = f(y, x) => f(u, v) = f(v, u).
    
    # So we sum over m, n >= 0 where m, n are even?
    # T_m(-u) = (-1)^m T_m(u). For evenness in u, m must be even.
    # For evenness in v, n must be even.
    # Symmetry f(u,v)=f(v,u) implies coefficients A[m,n] = A[n,m].
    
    # Let's use m, n in {0, 2, 4, 6, 8}.
    # Number of independent parameters:
    # Pairs (m, n) with m, n in {0, 2, 4, 6, 8} and m <= n.
    # (0,0), (0,2), (0,4), (0,6), (0,8)
    # (2,2), (2,4), (2,6), (2,8)
    # (4,4), (4,6), (4,8)
    # (6,6), (6,8)
    # (8,8)
    # Total: 5 + 4 + 3 + 2 + 1 = 15 parameters.
    
    cheb_orders = [0, 2, 4, 6, 8]
    param_indices = []
    for i, m in enumerate(cheb_orders):
        for j, n in enumerate(cheb_orders):
            if m <= n:
                param_indices.append((m, n))
    num_params = len(param_indices)

    # Precompute Chebyshev values on the grid
    # Grid points: pixel centers (i + 0.5) / size
    x_coords = np.array([(i + 0.5) / size for i in range(size)])
    y_coords = np.array([(i + 0.5) / size for i in range(size)])
    X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
    
    # Map to Chebyshev domain [-1, 1]
    U = 2 * X - 1
    V = 2 * Y - 1
    
    # Precompute T_m(U) and T_n(V) for m, n in cheb_orders
    # T_0 = 1
    # T_1 = t
    # T_k = 2*t*T_{k-1} - T_{k-2}
    def cheb_poly(t, k):
        if k == 0:
            return np.ones_like(t)
        if k == 1:
            return t.copy()
        T_prev2 = np.ones_like(t)
        T_prev1 = t.copy()
        for _ in range(2, k + 1):
            T_curr = 2 * t * T_prev1 - T_prev2
            T_prev2 = T_prev1
            T_prev1 = T_curr
        return T_prev1

    # Store basis functions for each parameter
    # basis[i] is a 32x32 array corresponding to param_indices[i]
    basis = []
    for m, n in param_indices:
        Tm_U = cheb_poly(U, m)
        Tn_V = cheb_poly(V, n)
        # The term is T_m(U) * T_n(V) + T_n(U) * T_m(V) if m != n, else T_m(U) * T_m(V)
        if m == n:
            basis.append(Tm_U * Tn_V)
        else:
            Tn_U = cheb_poly(U, n)
            Tm_V = cheb_poly(V, m)
            basis.append(Tm_U * Tn_V + Tn_U * Tm_V)
    
    def get_eps_grid(params):
        field = np.zeros((size, size))
        for i in range(num_params):
            field += params[i] * basis[i]
        eps_grid = 0.5 * (eps_low + eps_high) + 0.5 * (eps_high - eps_low) * np.tanh(field)
        return eps_grid

    # --- 2. Fast Evaluator ---
    # We need a fast proxy for the band gap.
    # The evaluator uses 289 plane waves (17x17 grid in reciprocal space).
    # We will use a smaller set, e.g., 25 (5x5) or 49 (7x7) plane waves.
    # Let's use 7x7 = 49 plane waves for a balance of speed and accuracy.
    
    pw_size = 7  # 7x7 = 49 PWs
    # Reciprocal lattice vectors: G = (h, k) where h, k are integers.
    # For a square lattice, G vectors are (h, k) * 2*pi / a.
    # We select the 49 G vectors closest to the origin.
    # h, k in range [-3, 3].
    
    G_vectors = []
    for h in range(-3, 4):
        for k in range(-3, 4):
            G_vectors.append((h, k))
    
    # Better approach: Compute FFT of 1/eps once per design iteration.
    # The FFT gives coefficients for h, k in [-16, 15] (or similar, depending on shift).
    # numpy.fft.fft2 returns indices 0..31.
    # Frequency h corresponds to index h mod 32.
    
    def get_fourier_coeff(fft_1eps, h, k):
        """
        Extracts the Fourier coefficient for (h, k) from the FFT of 1/eps.
        The FFT of a function f(x) on [0, 1) with N points gives:
        F[m] = sum_{n=0}^{N-1} f[n] exp(-i 2 pi m n / N)
        So F[m] corresponds to frequency m/N.
        For periodic extension, frequency h corresponds to index h % N.
        """
        idx_h = h % size
        idx_k = k % size
        return fft_1eps[idx_h, idx_k] / (size * size) # Normalization

    def evaluate_gap(eps_grid):
        """
        Computes the approximate band gap between bands 4 and 5.
        Returns the gap-midgap ratio.
        """
        # 1. Compute FFT of 1/eps
        inv_eps = 1.0 / eps_grid
        fft_1eps = np.fft.fft2(inv_eps)
        
        # 2. Build the Hamiltonian matrix for a set of k-points
        # We sample a few k-points in the irreducible BZ.
        # Irreducible BZ for square lattice: 0 <= kx <= ky, kx + ky <= 1 (in units of 2pi/a? No, 1/a)
        # Let's use units where the reciprocal lattice vector is 1.
        # k = (kx, ky).
        # Key points: Gamma (0,0), M (0.5, 0), X (0.5, 0.5).
        # The gap usually opens near M or X.
        # Let's sample points along the M-X line and Gamma-M line.
        
        k_points = []
        # Gamma to M: (0, t) for t in [0, 0.5]
        for t in np.linspace(0, 0.5, 5):
            k_points.append((0.0, t))
        # M to X: (t, 0.5-t) for t in [0, 0.5]
        for t in np.linspace(0, 0.5, 5):
            k_points.append((t, 0.5 - t))
        
        max_omega_4 = 0.0
        min_omega_5 = float('inf')
        
        for kx, ky in k_points:
            # Build H matrix
            # H_{ij} = (G_i + k) dot (G_j + k) * <1/eps>_{G_i - G_j}
            # For TM polarization, the operator is (G+k) x (G+k) * (1/eps) (scalar product in 2D)
            # Actually, for TM, the field is scalar z-component.
            # The equation is: sum_j (G_i + k) dot (G_j + k) / eps(G_i - G_j) * E_j = (omega^2 / c^2) E_i
            # Let's work with omega^2.
            
            N_pw = len(G_vectors)
            H = np.zeros((N_pw, N_pw), dtype=complex)
            
            for i, (hi, ki) in enumerate(G_vectors):
                # k-vector in fractional coordinates?
                # G vectors are integers. k vectors are fractional.
                # Total wavevector K_i = G_i + k.
                # K_i = (hi + kx, ki + ky)
                Kx_i = hi + kx
                Ky_i = ki + ky
                
                for j, (hj, kj) in enumerate(G_vectors):
                    Kx_j = hj + kx
                    Ky_j = kj + ky
                    
                    # Dot product
                    dot = Kx_i * Kx_j + Ky_i * Ky_j
                    
                    # Fourier coeff of 1/eps for G_i - G_j
                    dh = hi - hj
                    dk = ki - kj
                    coeff = get_fourier_coeff(fft_1eps, dh, dk)
                    
                    H[i, j] = dot * coeff
            
            # Solve eigenvalue problem
            # H is Hermitian (if coeff is real, which it is for symmetric eps).
            # 1/eps is real and symmetric, so its FFT is real (conjugate symmetric).
            # dot is real. So H is real symmetric?
            # coeff(dh, dk) is real if eps is symmetric.
            # Let's assume H is real symmetric for efficiency.
            H_real = H.real
            eigenvalues = np.linalg.eigvalsh(H_real)
            
            # eigenvalues are omega^2 (scaled).
            # We want bands 4 and 5 (1-indexed). So indices 3 and 4.
            # Sort eigenvalues
            # eigvalsh returns sorted.
            
            if len(eigenvalues) >= 5:
                w4_sq = eigenvalues[3]
                w5_sq = eigenvalues[4]
                
                # Convert to omega
                # Since we only care about the ratio, scaling doesn't matter.
                # But we need positive roots.
                w4 = np.sqrt(max(0, w4_sq))
                w5 = np.sqrt(max(0, w5_sq))
                
                if w4 > max_omega_4:
                    max_omega_4 = w4
                if w5 < min_omega_5:
                    min_omega_5 = w5
        
        if min_omega_5 == float('inf'):
            return 0.0
            
        gap = min_omega_5 - max_omega_4
        midgap = (min_omega_5 + max_omega_4) / 2.0
        
        if midgap == 0:
            return 0.0
            
        ratio = 2 * gap / midgap
        return ratio

    # --- 3. Optimization Loop ---
    
    # Initial guess: A central rod.
    # A central rod has high epsilon in the center.
    # f(x,y) should be positive in the center, negative at the edges.
    # T_0(0)=1, T_2(0)=-1, T_4(0)=1, T_6(0)=-1, T_8(0)=1.
    # At center (u=0, v=0):
    # f = A00*1 + A02*(-1) + A04*1 + A06*(-1) + A08*1 + A22*1 + A24*(-1) + A26*1 + A28*(-1) + A44*1 + A46*(-1) + A48*1 + A66*1 + A68*(-1) + A88*1
    # We want f > 0.
    # Let's start with A00 = 1.5 (uniform high eps) and add terms to create a hole?
    # No, we want a rod. So high eps in center.
    # Let's try A00 = 2.0, A22 = -2.0, A44 = 1.0, A66 = -1.0, A88 = 1.0?
    # This creates oscillations.
    # Simpler: Start with A00 = 1.5, others 0. This is a uniform slab (high eps).
    # Then optimize to find the shape.
    
    params = np.zeros(num_params)
    params[0] = 1.5  # A00
    
    # Step size for hill climbing
    step_size = 0.2
    best_score = evaluate_gap(get_eps_grid(params))
    best_params = params.copy()
    
    # Adaptive step size
    # If we improve, increase step size slightly or keep same.
    # If we fail, decrease step size.
    
    num_iterations = 0
    max_iterations = 5000
    
    while time.time() < deadline and num_iterations < max_iterations:
        num_iterations += 1
        
        # Pick a random parameter to perturb
        idx = np.random.randint(num_params)
        
        # Try a perturbation
        delta = np.random.choice([-1, 1]) * step_size
        
        new_params = params.copy()
        new_params[idx] += delta
        
        # Compute new score
        new_eps_grid = get_eps_grid(new_params)
        new_score = evaluate_gap(new_eps_grid)
        
        if new_score > best_score:
            params = new_params
            best_score = new_score
            best_params = params.copy()
            # Increase step size slightly
            step_size *= 1.05
            # Cap step size
            if step_size > 1.0:
                step_size = 1.0
        else:
            # Decrease step size
            step_size *= 0.9
            # Floor step size
            if step_size < 0.01:
                step_size = 0.01
        
        # Occasionally, do a larger jump (restarting) to escape local minima
        if num_iterations % 500 == 0:
            # Perturb all parameters slightly
            params = best_params + np.random.normal(0, 0.5, num_params)
            score = evaluate_gap(get_eps_grid(params))
            if score > best_score:
                best_score = score
                best_params = params.copy()
    
    # --- 4. Finalize Design ---
    
    # Use the best parameters found
    final_params = best_params
    final_eps_grid = get_eps_grid(final_params)
    
    # Threshold to binary 0/1
    # eps_low = 1, eps_high = 12.
    # Midpoint = 6.5.
    threshold = (eps_low + eps_high) / 2.0
    cell = (final_eps_grid > threshold).astype(int)
    
    # Enforce symmetry exactly (just in case of floating point errors in thresholding)
    # The construction ensures symmetry, but let's verify/force it.
    # Symmetry: cell[i][j] == cell[size-1-i][size-1-j] == cell[j][i]
    # Our basis functions are symmetric. The thresholding might break it if values are close to threshold.
    # We can average the 4 symmetric pixels and threshold the average.
    
    sym_cell = np.zeros_like(cell)
    for i in range(size):
        for j in range(size):
            # Symmetric equivalents:
            # (i, j)
            # (size-1-i, j)  -> reflection x
            # (i, size-1-j)  -> reflection y
            # (size-1-i, size-1-j) -> rotation 180
            # Also transpose: (j, i), etc.
            # C4v group has 8 elements.
            # Let's just average the 4 unique ones under the group?
            # Actually, the group is generated by reflection x, reflection y, and transpose.
            # If we enforce reflection x and y, and transpose, we get C4v.
            
            # For now, let's just take the majority vote or average of the 4 main symmetric points:
            # (i,j), (size-1-i, j), (i, size-1-j), (size-1-i, size-1-j)
            # And their transposes.
            # It's safer to just re-compute the average of the 8 symmetric points.
            
            points = [
                (i, j),
                (size - 1 - i, j),
                (i, size - 1 - j),
                (size - 1 - i, size - 1 - j),
                (j, i),
                (size - 1 - j, j),
                (j, size - 1 - i),
                (size - 1 - j, size - 1 - i)
            ]
            # Unique points?
            # For a 32x32 grid, these might not all be unique.
            # Let's just sum them and divide by 8.
            s = 0
            for pi, pj in points:
                s += cell[pi, pj]
            avg = s / 8.0
            sym_cell[i, j] = 1 if avg >= 0.5 else 0
    
    # Convert to list of lists
    return sym_cell.tolist()

# EVOLVE-BLOCK-END
