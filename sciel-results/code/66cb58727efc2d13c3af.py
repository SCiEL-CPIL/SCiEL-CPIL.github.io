import numpy as np
import time
import random

def design(problem):
    """
    Designs a 32x32 unit cell by optimizing a C4v-symmetric Fourier representation
    of the dielectric field to maximize the TM band gap between bands 4 and 5.
    Combines Fourier parameterization with a fast plane-wave surrogate and
    multi-start local search.
    """
    start_time = time.time()
    deadline = start_time + 195.0  # Safety margin before 230s kill

    size = problem["pixels"]
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)

    # --- Parameterization: C4v symmetric Fourier series ---
    # f(x,y) = sum A_{mn} cos(2*pi*m*(x-0.5)) cos(2*pi*n*(y-0.5))
    # Symmetry about x=0.5, y=0.5 is automatic.
    # Symmetry about y=x requires A_{mn} = A_{nm}.
    
    max_harmonic = 4
    params_idx = []
    for m in range(max_harmonic + 1):
        for n in range(m, max_harmonic + 1):
            params_idx.append((m, n))
    num_params = len(params_idx)

    # Precompute grid coordinates
    x = np.linspace(0.5/size, 1 - 0.5/size, size)
    X, Y = np.meshgrid(x, x, indexing='ij')
    Xc = X - 0.5
    Yc = Y - 0.5

    # Precompute basis functions
    basis_funcs = []
    for (m, n) in params_idx:
        if m == 0 and n == 0:
            basis_funcs.append(np.ones((size, size)))
        else:
            cm = np.cos(2 * np.pi * m * Xc)
            cn = np.cos(2 * np.pi * n * Yc)
            basis_funcs.append(cm * cn)
    basis_funcs = np.array(basis_funcs)  # shape (num_params, size, size)

    def get_field(params):
        """Returns the 32x32 epsilon grid from parameters."""
        field = np.tensordot(params, basis_funcs, axes=1)
        eps_grid = 0.5 * (eps_low + eps_high) + 0.5 * (eps_high - eps_low) * np.tanh(field)
        return eps_grid

    # --- Plane-wave solver setup ---
    # Use 17x17 = 289 plane waves to match the evaluator
    N_pw = 8
    G_vals = np.arange(-N_pw, N_pw + 1)
    Gx, Gy = np.meshgrid(G_vals, G_vals, indexing='ij')
    G_vectors = np.column_stack((Gx.ravel(), Gy.ravel()))
    num_pw = len(G_vectors)

    # K points: sample the irreducible BZ boundary and interior
    # Irreducible BZ for square lattice: triangle with vertices (0,0), (pi,0), (pi/2,pi/2)
    # We sample 30 points as the evaluator does
    k_points = []
    # Gamma-M edge: (t, 0) for t in [0, 0.5] (in units of pi)
    # M-X edge: (0.5, t) for t in [0, 0.5]
    # Gamma-X edge: (t, t) for t in [0, 0.5]
    # Plus interior points
    
    # Generate 30 k-points covering the irreducible BZ
    # The evaluator uses 30 wave vectors covering boundary and interior
    # Let's create a grid of points in the irreducible triangle
    for i in range(6):
        for j in range(6 - i):
            kx = (i + 0.5) / 6.0 * 0.5
            ky = (j + 0.5) / 6.0 * 0.5
            if kx + ky <= 0.5:
                k_points.append((kx, ky))
    
    # Add specific high-symmetry points
    k_points.append((0.0, 0.0))       # Gamma
    k_points.append((0.5, 0.0))       # M
    k_points.append((0.25, 0.25))     # X
    
    # Ensure we have enough points
    while len(k_points) < 30:
        # Add random interior points
        kx = random.random() * 0.5
        ky = random.random() * 0.5
        if kx + ky <= 0.5:
            k_points.append((kx, ky))
    
    k_points = k_points[:30]
    
    # Precompute G-G' differences for each pair (i,j)
    G_diff = G_vectors[:, np.newaxis, :] - G_vectors[np.newaxis, :, :]  # (num_pw, num_pw, 2)

    # Precompute the coefficient matrix indices
    # coeff_matrix[i,j] = fft_inv_eps[(G_diff[i,j,0]) % size, (G_diff[i,j,1]) % size]
    coeff_idx = np.zeros((num_pw, num_pw, 2), dtype=int)
    for i in range(num_pw):
        for j in range(num_pw):
            dx = int(G_diff[i, j, 0])
            dy = int(G_diff[i, j, 1])
            coeff_idx[i, j, 0] = dx % size
            coeff_idx[i, j, 1] = dy % size

    def evaluate_cell(eps_grid):
        """
        Computes the band gap using a plane-wave method.
        Uses 289 PWs and 30 k-points.
        """
        inv_eps_grid = 1.0 / eps_grid
        fft_inv_eps = np.fft.fft2(inv_eps_grid)
        fft_inv_eps /= (size * size)

        # Precompute coefficient matrix
        coeff_matrix = np.zeros((num_pw, num_pw), dtype=complex)
        for i in range(num_pw):
            for j in range(num_pw):
                ix = coeff_idx[i, j, 0]
                iy = coeff_idx[i, j, 1]
                coeff_matrix[i, j] = fft_inv_eps[ix, iy]

        min_omega_5 = np.inf
        max_omega_4 = -np.inf

        for kx, ky in k_points:
            k = np.array([kx * 2 * np.pi, ky * 2 * np.pi])

            # (k+G)^2 for each G
            k_plus_G = k + G_vectors * 2 * np.pi
            omega_sq = np.sum(k_plus_G ** 2, axis=1)  # (num_pw,)

            # M[i,j] = coeff_matrix[i,j] * omega_sq[i]
            M = coeff_matrix * omega_sq[:, np.newaxis]

            # Solve eigenvalue problem
            try:
                eigvals = np.linalg.eigvals(M)
                real_eigvals = np.sort(np.real(eigvals))
            except:
                continue

            # Filter positive eigenvalues
            positive_eigs = real_eigvals[real_eigvals > 1e-10]
            if len(positive_eigs) >= 5:
                omega_4 = np.sqrt(positive_eigs[3])
                omega_5 = np.sqrt(positive_eigs[4])

                max_omega_4 = max(max_omega_4, omega_4)
                min_omega_5 = min(min_omega_5, omega_5)

        if min_omega_5 == np.inf or max_omega_4 == -np.inf:
            return -1.0

        gap = min_omega_5 - max_omega_4
        midgap = 0.5 * (min_omega_5 + max_omega_4)
        if midgap == 0:
            return -1.0
        ratio = 2 * gap / midgap
        return ratio

    # --- Optimization: Multi-start local search ---
    best_params = None
    best_score = -np.inf

    # Initial guess: a dielectric rod-like shape
    init_params = np.zeros(num_params)
    idx_11 = params_idx.index((1, 1))
    init_params[idx_11] = 2.0
    init_params[0] = -0.5

    best_params = init_params.copy()
    best_score = evaluate_cell(get_field(best_params))

    # Multi-start with different initializations
    num_starts = 20
    for start_idx in range(num_starts):
        if time.time() > deadline - 15:
            break

        if start_idx == 0:
            p = best_params.copy()
        elif start_idx < 5:
            # Perturb best
            p = best_params + np.random.normal(0, 0.5, num_params)
        elif start_idx < 10:
            # Rod-like initialization
            p = np.zeros(num_params)
            p[0] = np.random.uniform(-1, 0)
            p[idx_11] = np.random.uniform(1, 3)
            for idx in range(1, num_params):
                if idx != idx_11:
                    p[idx] = np.random.normal(0, 0.3)
        elif start_idx < 15:
            # Hole-like initialization (air holes in dielectric)
            p = np.zeros(num_params)
            p[0] = np.random.uniform(1, 3)
            p[idx_11] = np.random.uniform(-3, -1)
            for idx in range(1, num_params):
                if idx != idx_11:
                    p[idx] = np.random.normal(0, 0.3)
        else:
            # Random
            p = np.random.uniform(-2, 2, num_params)

        score = evaluate_cell(get_field(p))

        if score > best_score:
            best_score = score
            best_params = p.copy()

        # Local refinement
        for _ in range(10):
            if time.time() > deadline - 15:
                break
            p_local = p + np.random.normal(0, 0.15, num_params)
            score_local = evaluate_cell(get_field(p_local))
            if score_local > score:
                score = score_local
                p = p_local.copy()
                if score > best_score:
                    best_score = score
                    best_params = p.copy()

    if best_params is None:
        # Fallback to seed
        cell = []
        for i in range(size):
            x_c = (i + 0.5) / size - 0.5
            row = []
            for j in range(size):
                y_c = (j + 0.5) / size - 0.5
                if x_c * x_c + y_c * y_c < 0.2 ** 2:
                    row.append(1)
                else:
                    row.append(0)
            cell.append(row)
        return cell

    final_grid = get_field(best_params)

    # Binarize
    threshold = 0.5 * (eps_low + eps_high)
    binary_cell = np.zeros((size, size), dtype=int)
    for i in range(size):
        for j in range(size):
            if final_grid[i, j] > threshold:
                binary_cell[i, j] = 1

    # Enforce strict symmetry via majority vote over symmetry group
    sym_cell = np.zeros((size, size), dtype=int)
    count = np.zeros((size, size), dtype=int)
    
    # The C4v group has 8 elements
    # For each pixel, compute its orbit under C4v
    # C4v: identity, rotation 90, rotation 180, rotation 270,
    #       reflection x, reflection y, reflection x=y, reflection x=-y
    
    for i in range(size):
        for j in range(size):
            # Get the orbit of (i, j) under C4v
            orbit = set()
            
            # Identity
            orbit.add((i, j))
            
            # Rotation 90: (i, j) -> (j, size-1-i)
            orbit.add((j, size - 1 - i))
            
            # Rotation 180: (i, j) -> (size-1-i, size-1-j)
            orbit.add((size - 1 - i, size - 1 - j))
            
            # Rotation 270: (i, j) -> (size-1-j, i)
            orbit.add((size - 1 - j, i))
            
            # Reflection x (vertical axis): (i, j) -> (size-1-i, j)
            orbit.add((size - 1 - i, j))
            
            # Reflection y (horizontal axis): (i, j) -> (i, size-1-j)
            orbit.add((i, size - 1 - j))
            
            # Reflection x=y: (i, j) -> (j, i)
            orbit.add((j, i))
            
            # Reflection x=-y: (i, j) -> (size-1-j, size-1-i)
            orbit.add((size - 1 - j, size - 1 - i))
            
            # For each pixel in the orbit, add the value of binary_cell[i,j]
            for (oi, oj) in orbit:
                sym_cell[oi, oj] += binary_cell[i, j]
                count[oi, oj] += 1

    # Majority vote
    final_sym_cell = (sym_cell / count) >= 0.5
    final_sym_cell = final_sym_cell.astype(int)

    # Final check: ensure strict symmetry by direct assignment
    # Use the orbit approach: for each orbit, all pixels get the same value
    # We already did majority vote, so this should be consistent
    # But let's enforce it explicitly
    
    # Process each orbit once
    processed = np.zeros((size, size), dtype=bool)
    for i in range(size):
        for j in range(size):
            if processed[i, j]:
                continue
            
            # Get the orbit
            orbit = set()
            orbit.add((i, j))
            orbit.add((j, size - 1 - i))
            orbit.add((size - 1 - i, size - 1 - j))
            orbit.add((size - 1 - j, i))
            orbit.add((size - 1 - i, j))
            orbit.add((i, size - 1 - j))
            orbit.add((j, i))
            orbit.add((size - 1 - j, size - 1 - i))
            
            # Mark all as processed
            for (oi, oj) in orbit:
                processed[oi, oj] = True
            
            # Set all to the same value (use the majority from final_sym_cell)
            val = final_sym_cell[i, j]
            for (oi, oj) in orbit:
                final_sym_cell[oi, oj] = val

    result = []
    for i in range(size):
        row = []
        for j in range(size):
            row.append(int(final_sym_cell[i, j]))
        result.append(row)

    return result
