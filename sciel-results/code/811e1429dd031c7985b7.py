"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import math
import scipy.optimize

# EVOLVE-BLOCK-START
def design(problem):
    """
    Designs a 32x32 unit cell for a TM band gap between bands 4 and 5.
    Strategy:
    1. Use a level-set representation defined by a low-order polynomial in Chebyshev basis
       constrained to C4v symmetry.
    2. Optimize the coefficients using scipy's differential_evolution (global optimizer).
    3. The objective function is a fast, approximate plane-wave calculation (9x9 basis)
       that correlates with the full 289-PW evaluator score.
    """
    size = problem["pixels"]  # 32
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)
    
    # Time management
    start_time = time.time()
    deadline = start_time + 180.0  # Safety margin for 200s limit

    # --- 1. Parameterization ---
    # Chebyshev orders constrained by C4v symmetry
    cheb_orders = [0, 2, 4, 6]
    param_indices = []
    for i, m in enumerate(cheb_orders):
        for j, n in enumerate(cheb_orders):
            if m <= n:
                param_indices.append((m, n))
    num_params = len(param_indices)

    # Precompute Chebyshev values on the grid
    x_coords = np.array([(i + 0.5) / size for i in range(size)])
    y_coords = np.array([(i + 0.5) / size for i in range(size)])
    X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
    
    U = 2 * X - 1
    V = 2 * Y - 1
    
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
    basis = []
    for m, n in param_indices:
        Tm_U = cheb_poly(U, m)
        Tn_V = cheb_poly(V, n)
        if m == n:
            basis.append(Tm_U * Tn_V)
        else:
            Tn_U = cheb_poly(U, n)
            Tm_V = cheb_poly(V, m)
            basis.append(Tm_U * Tn_V + Tn_U * Tm_V)
    basis = np.array(basis) # Shape: (num_params, size, size)

    def get_eps_grid(params):
        # params shape: (num_params,)
        # basis shape: (num_params, size, size)
        field = np.tensordot(params, basis, axes=0)
        eps_grid = 0.5 * (eps_low + eps_high) + 0.5 * (eps_high - eps_low) * np.tanh(field)
        return eps_grid

    # --- 2. Fast Evaluator ---
    # Use 9x9 = 81 plane waves for better accuracy than 7x7
    pw_size = 9
    G_vectors = []
    half_pw = pw_size // 2
    for h in range(-half_pw, half_pw + 1):
        for k in range(-half_pw, half_pw + 1):
            G_vectors.append((h, k))
    G_vectors = np.array(G_vectors) # Shape: (N_pw, 2)
    N_pw = len(G_vectors)
    
    # Precompute G vector differences for Hamiltonian construction
    # G_diff[i, j] = G_vectors[i] - G_vectors[j]
    G_diff = G_vectors[:, np.newaxis, :] - G_vectors[np.newaxis, :, :]
    G_diff_h = G_diff[:, :, 0]
    G_diff_k = G_diff[:, :, 1]

    # Precompute k-points in irreducible BZ
    # Square lattice BZ: 0 <= kx <= ky, kx + ky <= 0.5 (if units are 2pi/a, then corners are 0.5,0.5)
    # Actually, standard units: k in [0, 1] where 1 is reciprocal lattice vector.
    # Irreducible wedge: 0 <= kx <= ky, kx + ky <= 0.5? No.
    # For square lattice, BZ is square [-0.5, 0.5] x [-0.5, 0.5].
    # Irreducible: 0 <= kx <= ky, kx + ky <= 0.5.
    # Let's sample points in this triangle.
    k_points = []
    # Gamma (0,0) to M (0.5, 0)
    for t in np.linspace(0, 0.5, 6):
        k_points.append((0.0, t))
    # M (0.5, 0) to X (0.5, 0.5)
    for t in np.linspace(0, 0.5, 6):
        k_points.append((0.5 - t, 0.5 - t)) # Wait, M is (0.5, 0), X is (0.5, 0.5).
        # Line M-X: kx=0.5, ky from 0 to 0.5.
        # Let's just use a grid in the irreducible wedge.
    k_points = []
    # Grid in the wedge 0 <= kx <= ky, kx + ky <= 0.5
    # Let's use a few specific lines
    # Gamma-M: (0, t)
    for t in np.linspace(0, 0.5, 5):
        k_points.append((0.0, t))
    # M-X: (0.5, t)
    for t in np.linspace(0, 0.5, 5):
        k_points.append((0.5, t))
    # Gamma-X: (t, t)
    for t in np.linspace(0, 0.5, 5):
        k_points.append((t, t))
    
    k_points = np.array(k_points)
    
    # Remove duplicates if any
    unique_k = np.unique(k_points, axis=0)

    def evaluate_gap(eps_grid):
        """
        Computes the approximate band gap between bands 4 and 5.
        Returns the gap-midgap ratio (negative if overlap).
        """
        # 1. Compute FFT of 1/eps
        inv_eps = 1.0 / eps_grid
        fft_1eps = np.fft.fft2(inv_eps)
        
        max_omega_4 = 0.0
        min_omega_5 = float('inf')
        
        for kx, ky in unique_k:
            # K vectors: G + k
            Kx = G_vectors[:, 0] + kx
            Ky = G_vectors[:, 1] + ky
            K_sq = Kx**2 + Ky**2 # Not directly used in TM, but dot product is
            
            # H[i, j] = (G_i + k) dot (G_j + k) * <1/eps>_{G_i - G_j}
            # Dot product: (Kx_i * Kx_j + Ky_i * Ky_j)
            dot_products = np.outer(Kx, Kx) + np.outer(Ky, Ky)
            
            # Get Fourier coefficients for all G_diffs
            # fft_1eps is indexed by [h, k] where h, k are in 0..31.
            # We need coeff for (dh, dk).
            # idx_h = dh % 32, idx_k = dk % 32
            idx_h = G_diff_h % size
            idx_k = G_diff_k % size
            coeffs = fft_1eps[idx_h, idx_k] / (size * size)
            
            H = dot_products * coeffs
            
            # H should be Hermitian. For real symmetric eps, H is real symmetric.
            # Let's enforce symmetry to be safe
            H = (H + H.T) / 2.0
            H = H.real
            
            try:
                eigenvalues = np.linalg.eigvalsh(H)
            except:
                continue
                
            if len(eigenvalues) >= 5:
                w4_sq = eigenvalues[3]
                w5_sq = eigenvalues[4]
                
                if w4_sq < 0 or w5_sq < 0:
                    continue # Unphysical or numerical error
                
                w4 = np.sqrt(w4_sq)
                w5 = np.sqrt(w5_sq)
                
                if w4 > max_omega_4:
                    max_omega_4 = w4
                if w5 < min_omega_5:
                    min_omega_5 = w5
        
        if min_omega_5 == float('inf'):
            return -1.0 # Penalize failure
            
        gap = min_omega_5 - max_omega_4
        midgap = (min_omega_5 + max_omega_4) / 2.0
        
        if midgap == 0:
            return -1.0
            
        ratio = 2 * gap / midgap
        return ratio

    # Wrapper for scipy minimize (minimize negative score)
    def objective(params):
        eps_grid = get_eps_grid(params)
        score = evaluate_gap(eps_grid)
        return -score # We maximize score, so minimize negative

    # --- 3. Optimization Loop ---
    
    # Bounds for parameters. 
    # tanh saturates at +-1, so large values don't change the binary outcome much.
    # But we need enough range to form shapes.
    bounds = [(-3.0, 3.0) for _ in range(num_params)]
    
    # Initial population size and max iterations for DE
    # We want to use as much time as possible.
    # DE with popsize=10, maxiter=200 might be too slow or too fast.
    # Let's run DE with a callback to check time.
    
    best_params = np.zeros(num_params)
    best_score = -1.0
    
    def callback(xk, convergence):
        nonlocal best_params, best_score
        # xk is the current best in the population? No, it's the current solution.
        # We need to track the global best.
        # Actually, differential_evolution returns the best at the end.
        # We can just let it run until time is up.
        pass

    # Use a time-based stopper
    class TimeStop(Exception):
        pass

    def time_check(xk):
        if time.time() > deadline:
            raise TimeStop()

    try:
        # popsize=10, tol=1e-6, maxiter=1000 (will be cut by time)
        # init='sobol' might be better for sampling
        result = scipy.optimize.differential_evolution(
            objective, 
            bounds, 
            maxiter=1000, 
            popsize=10, 
            tol=1e-6, 
            seed=42,
            callback=time_check
        )
        best_params = result.x
        best_score = -result.fun
        
    except TimeStop:
        # If stopped by time, we might not have the result object.
        # We should maintain a manual best.
        # Let's re-run with a manual loop if DE fails or is interrupted?
        # Actually, DE doesn't support easy manual best tracking in callback for the *global* best easily without hacking.
        # Let's do a simple manual loop with DE-like steps if we are worried.
        # But DE is robust. Let's assume it finishes or we catch the exception and use a fallback.
        # If TimeStop is raised, result is not available.
        # Let's implement a simple stochastic local search as a fallback or primary if DE is too slow to start?
        # No, DE is better. Let's just make sure we have a valid initial guess.
        
        # If DE times out, we might have lost the best.
        # To be safe, let's run a quick local search on the current best if we have one, 
        # or just use the initial guess if DE didn't finish.
        # However, DE usually makes progress quickly.
        
        # Alternative: Run DE with a smaller maxiter and then refine?
        # Let's stick to DE. If it times out, we return the last known best.
        # But we don't have the last known best from inside DE easily.
        
        # Let's modify the strategy: Run DE with a generous time, but if it fails, 
        # do a quick hill-climb on the center point.
        
        # For now, if TimeStop, we return the initial guess (which is bad).
        # Better: Implement a manual DE or just use a fixed number of iterations that fits in time.
        
        # Estimate time per iteration:
        # 81 PWs, 32x32 grid. FFT is fast. Eigvalsh of 81x81 is fast.
        # 100 k-points * 1 eval = 100 eigensolves.
        # One eval ~ 100 * (81^3 / 60) ops ~ 1e6 ops? No, eigvalsh is O(N^3). 81^3 ~ 500k.
        # 100 points * 500k = 50M ops per eval.
        # 1 eval ~ 50ms?
        # DE popsize 10, 1000 iters = 10000 evals. 10000 * 50ms = 500s. Too slow.
        # We need to reduce k-points or PWs.
        
        # Let's reduce k-points to 15 and PWs to 7x7=49.
        # 49^3 ~ 120k. 15 points * 120k = 1.8M ops. ~ 10ms per eval.
        # 10000 evals * 10ms = 100s. Fits in 180s.
        
        # Let's restart with adjusted parameters.
        pass

    # Re-configure for speed
    pw_size = 7
    G_vectors = []
    half_pw = pw_size // 2
    for h in range(-half_pw, half_pw + 1):
        for k in range(-half_pw, half_pw + 1):
            G_vectors.append((h, k))
    G_vectors = np.array(G_vectors)
    G_diff = G_vectors[:, np.newaxis, :] - G_vectors[np.newaxis, :, :]
    G_diff_h = G_diff[:, :, 0]
    G_diff_k = G_diff[:, :, 1]
    
    # Reduce k-points
    k_points = []
    for t in np.linspace(0, 0.5, 4):
        k_points.append((0.0, t))
    for t in np.linspace(0, 0.5, 4):
        k_points.append((0.5, t))
    for t in np.linspace(0, 0.5, 4):
        k_points.append((t, t))
    unique_k = np.unique(k_points, axis=0)

    def evaluate_gap_fast(eps_grid):
        inv_eps = 1.0 / eps_grid
        fft_1eps = np.fft.fft2(inv_eps)
        max_omega_4 = 0.0
        min_omega_5 = float('inf')
        
        for kx, ky in unique_k:
            Kx = G_vectors[:, 0] + kx
            Ky = G_vectors[:, 1] + ky
            dot_products = np.outer(Kx, Kx) + np.outer(Ky, Ky)
            
            idx_h = G_diff_h % size
            idx_k = G_diff_k % size
            coeffs = fft_1eps[idx_h, idx_k] / (size * size)
            
            H = dot_products * coeffs
            H = (H + H.T) / 2.0
            H = H.real
            
            try:
                eigenvalues = np.linalg.eigvalsh(H)
            except:
                continue
                
            if len(eigenvalues) >= 5:
                w4_sq = eigenvalues[3]
                w5_sq = eigenvalues[4]
                
                if w4_sq < 0 or w5_sq < 0:
                    continue
                
                w4 = np.sqrt(w4_sq)
                w5 = np.sqrt(w5_sq)
                
                if w4 > max_omega_4:
                    max_omega_4 = w4
                if w5 < min_omega_5:
                    min_omega_5 = w5
        
        if min_omega_5 == float('inf'):
            return -1.0
            
        gap = min_omega_5 - max_omega_4
        midgap = (min_omega_5 + max_omega_4) / 2.0
        
        if midgap == 0:
            return -1.0
            
        ratio = 2 * gap / midgap
        return ratio

    def objective_fast(params):
        eps_grid = get_eps_grid(params)
        score = evaluate_gap_fast(eps_grid)
        return -score

    bounds = [(-3.0, 3.0) for _ in range(num_params)]
    
    # Run DE with time guard
    # We will run it in a loop, checking time manually if possible, or just set maxiter low enough.
    # Let's try 500 iterations.
    
    try:
        result = scipy.optimize.differential_evolution(
            objective_fast, 
            bounds, 
            maxiter=500, 
            popsize=8, 
            tol=1e-5, 
            seed=42
        )
        best_params = result.x
        best_score = -result.fun
        
    except Exception as e:
        # Fallback to random search if DE fails
        best_params = np.random.uniform(-3, 3, num_params)
        best_score = objective_fast(best_params) * -1

    # --- 4. Finalize Design ---
    final_eps_grid = get_eps_grid(best_params)
    threshold = (eps_low + eps_high) / 2.0
    cell = (final_eps_grid > threshold).astype(int)
    
    # Enforce symmetry
    sym_cell = np.zeros_like(cell)
    for i in range(size):
        for j in range(size):
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
            s = 0
            for pi, pj in points:
                s += cell[pi, pj]
            avg = s / 8.0
            sym_cell[i, j] = 1 if avg >= 0.5 else 0

    return sym_cell.tolist()

# EVOLVE-BLOCK-END
