"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

def design(problem):
    """
    Design a 32x32 photonic crystal unit cell to maximize the TM 4-5 band gap.
    Strategy:
    1. Generate candidate cells using Fourier synthesis (directly optimizing harmonic content).
    2. Evaluate candidates using a fast, vectorized plane-wave proxy.
    3. Refine the best candidate using binary gradient ascent (pixel flipping).
    """
    size = problem["pixels"]
    eps_high = problem["eps_high"]
    eps_low = problem["eps_low"]
    
    start_time = time.time()
    time_limit = 220.0  # Safety margin before 230s kill

    # --- Fast Plane Wave Proxy ---
    # Use a reduced basis for speed. N=9 (81 PWs) is a good balance.
    N_pw = 9
    indices = np.arange(-N_pw//2, N_pw//2 + 1)
    
    # Precompute G vectors and |G|^2
    Gx_grid, Gy_grid = np.meshgrid(indices, indices, indexing='ij')
    G2_grid = Gx_grid**2 + Gy_grid**2
    
    # Flatten G vectors
    Gx_flat = Gx_grid.flatten()
    Gy_flat = Gy_grid.flatten()
    G2_flat = G2_grid.flatten()
    N_basis = len(Gx_flat)
    
    # Key k-points in the irreducible Brillouin zone (in units of 2pi/a)
    # Irreducible BZ: 0 <= ky <= kx <= 0.5
    # We sample a grid of points to capture the gap robustly
    k_points = [
        (0.0, 0.0),    # Gamma
        (0.5, 0.0),    # X
        (0.5, 0.5),    # M
        (0.25, 0.0),   # Mid Gamma-X
        (0.25, 0.25),  # Mid Gamma-M
        (0.5, 0.25),   # Mid M-X
        (0.125, 0.0),
        (0.375, 0.0),
        (0.125, 0.125),
        (0.375, 0.125),
        (0.125, 0.375),
        (0.375, 0.375),
        (0.5, 0.125),
        (0.5, 0.375),
    ]

    # Precompute (G+k)^2 for all k-points and G vectors
    # kg2[k_idx, g_idx] = |G_g + k_k|^2
    kg2 = np.zeros((len(k_points), N_basis))
    for ki, (kx, ky) in enumerate(k_points):
        kgx = Gx_flat + kx
        kgy = Gy_flat + ky
        kg2[ki, :] = kgx**2 + kgy**2
        # Avoid division by zero in D_inv_sqrt later
        kg2[ki, kg2[ki, :] < 1e-10] = 1e-10

    D_inv_sqrt = 1.0 / np.sqrt(kg2)  # shape (n_k, n_basis)

    # Precompute phase factors for Fourier coefficients
    # Pixel coordinates
    px = (np.arange(size) + 0.5) / size - 0.5
    py = (np.arange(size) + 0.5) / size - 0.5
    PX, PY = np.meshgrid(px, py, indexing='ij')
    
    # G vectors in 2pi units for phase calculation
    # eps_G = (1/A) sum eps(r) exp(-i G . r)
    # G = 2pi * (gx, gy)
    # phase = exp(-i 2pi (gx px + gy py))
    phases = np.exp(-2j * np.pi * (np.outer(Gx_flat, px) + np.outer(Gy_flat, py)))
    # phases shape: (n_basis, size*size)
    
    def compute_eps_fourier(cell):
        """
        Compute Fourier coefficients of the dielectric permittivity.
        cell: 2D array of 0s and 1s.
        Returns: eps_f, shape (N_basis,)
        """
        eps = eps_low + (eps_high - eps_low) * cell
        eps_flat = eps.flatten()
        # eps_f[g] = sum(eps_flat * phases[g]) / (size*size)
        eps_f = (phases @ eps_flat) / (size * size)
        return eps_f

    def evaluate_gap(cell):
        """
        Fast evaluation of the TM 4-5 band gap ratio.
        cell: 2D array of 0s and 1s.
        Returns: gap_ratio (higher is better)
        """
        eps_f = compute_eps_fourier(cell)
        
        max_omega4 = 0.0
        min_omega5 = np.inf
        
        for ki in range(len(k_points)):
            # Construct the matrix A for this k-point
            # A_ij = eps_{G_i - G_j} * |G_j + k|^2
            # We need eps_{G_i - G_j}.
            # G_i - G_j = (gx_i - gx_j, gy_i - gy_j)
            # We can precompute the index mapping for differences.
            
            # For speed in the inner loop, we compute the matrix M = D^{-1/2} A D^{-1/2}
            # A_ij = eps_{ij} * kg2[j]
            # M_ij = (1/sqrt(kg2[i])) * eps_{ij} * kg2[j] * (1/sqrt(kg2[j]))
            # M_ij = eps_{ij} * sqrt(kg2[j] / kg2[i])
            
            # We need eps_{G_i - G_j} for all i, j.
            # Let's precompute the difference indices once outside the k-loop if possible,
            # but eps_f depends on the cell, so we just need the index map.
            
            # Difference grid
            dgx = Gx_flat[:, None] - Gx_flat[None, :]
            dgy = Gy_flat[:, None] - Gy_flat[None, :]
            
            # Map (dgx, dgy) to index in the N_pw x N_pw grid
            # Index = (dgx + N_pw//2) + (dgy + N_pw//2) * N_pw
            idx_map = (dgx + N_pw//2).astype(int) + (dgy + N_pw//2).astype(int) * N_pw
            
            # Get eps values for all differences
            # eps_f is a flat array of length N_basis
            eps_diff = eps_f[idx_map] # shape (N_basis, N_basis)
            
            # Construct M
            # M_ij = eps_diff[i, j] * sqrt(kg2[ki, j] / kg2[ki, i])
            ratio = np.sqrt(kg2[ki, None, :] / kg2[ki, :, None])
            M = eps_diff * ratio
            
            # Ensure Hermitian (should be, but numerical errors might creep in)
            M = (M + M.conj().T) / 2.0
            
            # Solve eigenvalue problem
            # We only need the 4th and 5th eigenvalues.
            # np.linalg.eigvalsh is fast for small matrices.
            try:
                evals = np.linalg.eigvalsh(M)
                # evals are omega^2
                if len(evals) >= 5:
                    omega4 = np.sqrt(evals[3])
                    omega5 = np.sqrt(evals[4])
                    max_omega4 = max(max_omega4, omega4)
                    min_omega5 = min(min_omega5, omega5)
            except:
                pass
        
        if min_omega5 == np.inf or max_omega4 == 0:
            return -1.0
            
        gap = 2 * (min_omega5 - max_omega4)
        midgap = min_omega5 + max_omega4
        if midgap == 0:
            return -1.0
        return gap / midgap

    # --- Cell Generation: Fourier Synthesis ---
    
    def generate_fourier_cell(modes, amplitudes, phases):
        """
        Generate a cell by summing Fourier modes.
        modes: list of (gx, gy)
        amplitudes: list of floats
        phases: list of floats
        """
        # Start with a mean value
        eps_cont = np.ones((size, size)) * 0.5
        
        for (gx, gy), amp, ph in zip(modes, amplitudes, phases):
            # Add mode: amp * cos(2pi(gx x + gy y) + ph)
            # This creates a real, symmetric field if we pair modes carefully,
            # but we will enforce symmetry later.
            term = amp * np.cos(2 * np.pi * (gx * PX + gy * PY) + ph)
            eps_cont += term
            
        # Binarize
        # Threshold at 0.5 to maintain roughly 50% fill factor
        cell = (eps_cont > 0.5).astype(np.uint8)
        
        # Enforce 4-fold symmetry and inversion symmetry
        # 1. Inversion: (i, j) -> (size-1-i, size-1-j)
        cell = (cell + cell[::-1, ::-1]) // 2
        # 2. Rotation 90: (i, j) -> (j, size-1-i)
        cell_rot = np.rot90(cell, 1)
        cell = (cell + cell_rot) // 2
        # 3. Transpose (Reflection): (i, j) -> (j, i)
        cell = (cell + cell.T) // 2
        # Re-apply inversion after transpose just in case
        cell = (cell + cell[::-1, ::-1]) // 2
        
        return cell

    # Define a set of modes to explore
    # Modes that contribute to the gap: (1,0), (0,1), (1,1), (2,0), (0,2), (2,1), etc.
    # We want to maximize the gap, so we want strong (2,0) and (0,2) components.
    
    base_modes = [
        (1, 0), (0, 1),
        (1, 1),
        (2, 0), (0, 2),
        (2, 1), (1, 2),
        (2, 2),
        (3, 0), (0, 3),
    ]
    
    # Randomize amplitudes and phases
    np.random.seed(42)
    
    best_cell = None
    best_score = -np.inf
    
    # Phase 1: Random Fourier Synthesis
    n_candidates = 50
    for _ in range(n_candidates):
        if time.time() - start_time > time_limit - 30:
            break
            
        # Random amplitudes
        amps = np.random.uniform(0.1, 0.5, size=len(base_modes))
        # Boost higher order modes
        amps[3:5] *= 1.5  # (2,0) and (0,2)
        
        # Random phases
        phs = np.random.uniform(0, 2*np.pi, size=len(base_modes))
        
        cell = generate_fourier_cell(base_modes, amps, phs)
        
        score = evaluate_gap(cell)
        
        if score > best_score:
            best_score = score
            best_cell = cell
            best_params = (amps.copy(), phs.copy())

    if best_cell is None:
        # Fallback
        best_cell = np.zeros((size, size), dtype=np.uint8)
        # Simple 4-rod
        cx, cy, r = 0.25, 0.25, 0.15
        X, Y = np.meshgrid(px, py, indexing='ij')
        d1 = np.sqrt((X - cx)**2 + (Y - cy)**2)
        d2 = np.sqrt((X + cx)**2 + (Y - cy)**2)
        d3 = np.sqrt((X - cx)**2 + (Y + cy)**2)
        d4 = np.sqrt((X + cx)**2 + (Y + cy)**2)
        mask = (d1 < r) | (d2 < r) | (d3 < r) | (d4 < r)
        best_cell = mask.astype(np.uint8)
        best_cell = (best_cell + best_cell[::-1, ::-1]) // 2
        best_cell = (best_cell + np.rot90(best_cell, 1)) // 2
        best_cell = (best_cell + best_cell.T) // 2
        best_score = evaluate_gap(best_cell)

    # Phase 2: Binary Gradient Ascent (Local Search)
    # Try flipping pixels that break symmetry? No, we must maintain symmetry.
    # We can only flip orbits of pixels under the symmetry group.
    
    # Identify unique orbits
    # An orbit is a set of pixels that are mapped to each other by symmetry.
    # If we flip one, we must flip all in the orbit.
    
    # Let's create a list of orbits.
    orbits = []
    visited = np.zeros((size, size), dtype=bool)
    
    def get_orbit(i, j):
        # Apply all 8 symmetries
        points = set()
        # Identity
        points.add((i, j))
        # Inversion
        points.add((size-1-i, size-1-j))
        # Rot 90
        points.add((j, size-1-i))
        # Rot 180
        points.add((size-1-i, size-1-j))
        # Rot 270
        points.add((size-1-j, i))
        # Transpose
        points.add((j, i))
        # Anti-transpose (Reflection across anti-diagonal)
        points.add((size-1-j, size-1-i))
        # Reflection across main diagonal is Transpose
        # Reflection across vertical axis?
        # The group is D4.
        # Let's just generate all 8.
        # 1. (i, j)
        # 2. (j, i)
        # 3. (size-1-i, j)
        # 4. (j, size-1-i)
        # 5. (size-1-i, size-1-j)
        # 6. (size-1-j, size-1-i)
        # 7. (size-1-j, i)
        # 8. (i, size-1-j)
        
        pts = [
            (i, j),
            (j, i),
            (size-1-i, j),
            (j, size-1-i),
            (size-1-i, size-1-j),
            (size-1-j, size-1-i),
            (size-1-j, i),
            (i, size-1-j)
        ]
        return pts

    # Find unique orbits
    for i in range(size):
        for j in range(size):
            if not visited[i, j]:
                orbit = get_orbit(i, j)
                orbits.append(orbit)
                for (oi, oj) in orbit:
                    visited[oi, oj] = True

    # Shuffle orbits to randomize search
    np.random.shuffle(orbits)
    
    current_cell = best_cell
    current_score = best_score
    
    # Local search loop
    for _ in range(200): # Max iterations
        if time.time() - start_time > time_limit - 10:
            break
            
        # Pick a random orbit
        # To make it faster, we can just iterate through a subset of orbits
        # Or pick a random orbit each time.
        
        improved = False
        # Try a few random orbits
        for _ in range(10):
            if time.time() - start_time > time_limit - 10:
                break
                
            orbit = np.random.choice(orbits)
            
            # Check current state of the orbit
            # If all are 0, try setting to 1. If all are 1, try setting to 0.
            # If mixed, it's not a valid symmetric state (shouldn't happen if we start symmetric and flip whole orbits).
            
            val = current_cell[orbit[0][0], orbit[0][1]]
            new_val = 1 - val
            
            # Apply flip
            for (oi, oj) in orbit:
                current_cell[oi, oj] = new_val
                
            new_score = evaluate_gap(current_cell)
            
            if new_score > current_score:
                current_score = new_score
                best_cell = current_cell.copy()
                best_score = current_score
                improved = True
            else:
                # Revert
                for (oi, oj) in orbit:
                    current_cell[oi, oj] = val
            
            if improved:
                break # Restart orbit selection if improved? Or continue?
                # Usually, continue to see if more improvements are possible,
                # but since we changed the landscape, let's restart the inner loop logic
                # or just break to pick a new random orbit.
                break
        
        if not improved:
            # Try a few more random orbits before giving up on this neighborhood
            # If no improvement in 10 tries, maybe try a different strategy or stop
            # For simplicity, if no improvement in a batch, we might be stuck in a local min.
            # Let's try a few more batches.
            pass

    return best_cell.tolist()
# EVOLVE-BLOCK-END
