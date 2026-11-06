"""Rewritten program with a correct plane-wave band solver and symmetric local search."""
import numpy as np
import time
import math
import sys

def design(problem):
    """
    Design a symmetric 32x32 unit cell for a 2D square-lattice photonic crystal
    that maximizes the gap-midgap ratio between TM bands 4 and 5.
    """
    size = problem["pixels"]  # 32
    eps_low = problem["eps_low"]    # 1
    eps_high = problem["eps_high"]  # 12
    lower_band = problem["lower_band"]  # 4
    polarization = problem.get("polarization", "TM")

    start_time = time.time()
    time_limit = 215.0  # Safety margin before 230s kill

    # --- Plane-wave solver setup ---
    # 289 plane waves = 17 x 17, so G indices from -8 to 8
    max_g = 8
    g_values = list(range(-max_g, max_g + 1))  # 17 values
    # Precompute G vectors
    g_vectors = np.array([(mx, my) for mx in g_values for my in g_values], dtype=float)
    n_pw = len(g_vectors)  # 289

    # --- Compute inverse epsilon Fourier coefficients ---
    def compute_inv_eps_coeffs(cell):
        """
        Compute Fourier coefficients of 1/epsilon for a pixelated cell.
        Returns a 2D array indexed by [mx_idx, my_idx] where mx_idx, my_idx in [0, 17)
        corresponding to mx, my in [-8, 8].
        """
        N = size
        eps_matrix = np.where(np.array(cell, dtype=float) == 1, eps_high, eps_low)
        inv_eps = 1.0 / eps_matrix

        # FFT to get Fourier coefficients
        # The DFT is: c[mx, my] = (1/N^2) * sum_{i,j} f[i,j] * exp(-i 2*pi (mx*i + my*j)/N)
        # np.fft.fft2 gives this with indices 0..N-1, where index k corresponds to frequency k (or N-k if k > N/2)
        # We need frequencies mx, my in [-8, 8]
        coeffs = np.fft.fft2(inv_eps) / (N * N)

        # Extract the needed range: mx, my in [-8, 8]
        # Index mapping: for frequency f in [-8, 8], the np.fft index is f % N
        # For N=32: f=-8 -> 24, f=-7 -> 25, ..., f=0 -> 0, f=1 -> 1, ..., f=8 -> 8
        # We need to construct an array indexed by [mx_idx, my_idx] where mx_idx corresponds to mx = mx_idx - 8
        result = np.zeros((17, 17), dtype=complex)
        for idx_mx, mx in enumerate(g_values):
            np_idx_mx = mx % N  # -8 % 32 = 24, 0 % 32 = 0, 8 % 32 = 8
            for idx_my, my in enumerate(g_values):
                np_idx_my = my % N
                result[idx_mx, idx_my] = coeffs[np_idx_mx, np_idx_my]

        return result

    # --- Solve for bands at a single k-point ---
    def solve_k_point(inv_eps_coeffs, kx, ky):
        """
        Solve the TM band structure at a single k-point.
        Returns the first 5 eigenvalues (omega^2) in ascending order.
        """
        # k + G for each plane wave
        kg = g_vectors + np.array([kx, ky])  # shape (289, 2)
        kg_sq = np.sum(kg ** 2, axis=1)  # (k+G)^2, shape (289,)

        # Build the matrix A where A[i, j] = (k+G_i) · (k+G_j) * (1/eps)_{G_i - G_j}
        # For TM, the eigenvalue equation is:
        # sum_j (k+G_i)·(k+G_j) * inv_eps_{G_i - G_j} * E_j = omega^2 * E_i
        # So A[i, j] = (k+G_i)·(k+G_j) * inv_eps_{G_i - G_j}
        # And we solve A @ E = omega^2 * E

        # inv_eps_{G_i - G_j} = inv_eps_coeffs[mx_i - mx_j, my_i - my_j]
        # mx_i - mx_j ranges from -16 to 16, but our coefficients only go from -8 to 8
        # We need to handle this. The convolution theorem says that for a periodic structure,
        # the Fourier coefficients are periodic with period N in each direction.
        # So inv_eps_{mx, my} = inv_eps_{mx mod N, my mod N} but mapped back to [-N/2, N/2)
        # Actually, for the difference G_i - G_j, the components range from -16 to 16.
        # We can use the fact that the Fourier series is periodic, so we can wrap.

        # Let's build the matrix directly
        A = np.zeros((n_pw, n_pw), dtype=complex)
        for i in range(n_pw):
            kg_i = kg[i]
            dot_kg_i = np.dot(kg_i, kg_i)  # (k+G_i)^2
            for j in range(n_pw):
                dx = g_vectors[i, 0] - g_vectors[j, 0]
                dy = g_vectors[i, 1] - g_vectors[j, 1]
                # Get inv_eps coefficient for (dx, dy)
                # Map (dx, dy) to index in inv_eps_coeffs
                # inv_eps_coeffs is indexed by [idx_mx, idx_my] where idx_mx = mx + 8 for mx in [-8, 8]
                # But dx can be outside [-8, 8]. We use periodicity: dx mod 32, then map to [-16, 16) or similar.
                # Actually, the Fourier coefficients of a periodic function with period N are zero for |f| >= N/2 (Nyquist).
                # For N=32, frequencies beyond 16 are aliased. So for dx in [-16, 16], we can wrap to [-8, 8] using dx % 32 mapped to [-16, 16).
                # Let's use: wrapped_dx = ((dx + 16) % 32) - 16  # maps to [-16, 16)
                # But our inv_eps_coeffs only has [-8, 8]. For |dx| > 8, the coefficient should be zero (or aliased).
                # For a pixelated structure with 32 pixels, the highest meaningful frequency is 16.
                # But we only computed up to 8. This might be an issue.
                # Let's recompute with a larger range. Actually, for 289 plane waves with max_g=8,
                # the difference can be up to 16. We need coefficients up to 16.
                # Let's recompute the Fourier coefficients with a larger range.

        # This approach is getting complex. Let's use a different method:
        # Use the fact that the matrix is a circulant-like structure and use FFT.
        # Actually, let's just recompute the coefficients with a larger range.

        # Recompute with max_g_diff = 16
        return None  # Placeholder, we'll fix this

    # Let's redo the coefficient computation with a larger range
    def compute_inv_eps_coeffs_large(cell, max_range=16):
        """
        Compute Fourier coefficients of 1/epsilon for frequencies in [-max_range, max_range].
        """
        N = size
        eps_matrix = np.where(np.array(cell, dtype=float) == 1, eps_high, eps_low)
        inv_eps = 1.0 / eps_matrix

        coeffs = np.fft.fft2(inv_eps) / (N * N)

        # We need coefficients for mx, my in [-max_range, max_range]
        # For N=32, max_range=16, this covers all unique frequencies
        result = np.zeros((2 * max_range + 1, 2 * max_range + 1), dtype=complex)
        for idx_mx, mx in enumerate(range(-max_range, max_range + 1)):
            np_idx_mx = mx % N
            for idx_my, my in enumerate(range(-max_range, max_range + 1)):
                np_idx_my = my % N
                result[idx_mx, idx_my] = coeffs[np_idx_mx, np_idx_my]

        return result

    # Precompute the g index mapping for the large range
    max_range = 16
    g_diff_values = list(range(-max_range, max_range + 1))  # 33 values

    def solve_k_point_fast(inv_eps_coeffs_large, kx, ky):
        """
        Fast solve for TM bands at a single k-point.
        """
        kg = g_vectors + np.array([kx, ky])
        kg_sq = np.sum(kg ** 2, axis=1)

        # Build matrix A
        A = np.zeros((n_pw, n_pw), dtype=complex)
        for i in range(n_pw):
            for j in range(n_pw):
                dx = g_vectors[i, 0] - g_vectors[j, 0]
                dy = g_vectors[i, 1] - g_vectors[j, 1]
                # Get coefficient
                idx_mx = int(dx) + max_range
                idx_my = int(dy) + max_range
                coeff = inv_eps_coeffs_large[idx_mx, idx_my]
                A[i, j] = np.dot(kg[i], kg[j]) * coeff

        # Solve eigenvalue problem: A @ E = omega^2 * E
        # A is Hermitian (should be real symmetric for this setup)
        # Use np.linalg.eigvalsh for Hermitian matrices
        # Ensure A is Hermitian by symmetrizing
        A = 0.5 * (A + A.conj().T)

        eigenvalues = np.linalg.eigvalsh(A)
        # Eigenvalues are omega^2, return first 5
        return eigenvalues[:5]

    # --- Generate k-points in the irreducible Brillouin zone ---
    def generate_k_points():
        """
        Generate 30 k-points in the irreducible Brillouin zone of a square lattice.
        The full BZ is [-pi, pi] x [-pi, pi]. The irreducible BZ (C4v) is a triangle:
        0 <= kx <= ky <= pi, with kx + ky <= pi? Actually, the standard irreducible BZ
        for a square lattice is 0 <= kx <= ky, kx <= pi, ky <= pi, and kx + ky <= pi is not required.
        Let's use the region: 0 <= kx <= ky <= pi, with the additional constraint that
        we cover the high-symmetry lines.
        
        Actually, the irreducible BZ for a square lattice (C4v) is the triangle with vertices
        Gamma (0,0), M (pi, pi), and X (pi, 0). So: 0 <= kx <= ky, kx + ky <= pi is not right.
        The correct irreducible BZ is: 0 <= kx <= ky <= pi, with kx <= ky and ky <= pi.
        But this is a triangle with vertices (0,0), (pi,pi), and (pi,0)? No, (pi,0) has kx > ky.
        
        Let me use: 0 <= kx <= ky, kx <= pi, ky <= pi. This gives a triangle with vertices
        (0,0), (pi,pi), and (pi,0) is not included because kx=pi > ky=0.
        The vertices are (0,0), (pi,pi), and (0,pi) is also not in 0<=kx<=ky.
        
        Actually, the irreducible BZ is 1/8 of the full BZ. The full BZ is a square.
        The irreducible wedge is: 0 <= kx <= ky, kx^2 + ky^2 <= pi^2? No, that's circular.
        
        For a square lattice, the irreducible BZ is the triangle with vertices:
        Gamma = (0, 0)
        X = (pi, 0)  -- but this is not in 0<=kx<=ky
        M = (pi, pi)
        
        Let me just use a set of points that covers the important region:
        - Gamma (0,0)
        - Points along the Gamma-X line: (k, 0) for k in [0, pi]
        - Points along the Gamma-M line: (k, k) for k in [0, pi]
        - Points along the X-M line: (pi, k) for k in [0, pi]
        
        And ensure 30 points total.
        """
        k_points = []
        # Gamma
        k_points.append((0.0, 0.0))
        
        # Gamma to X: (k, 0)
        for t in np.linspace(0.1, 1.0, 6):
            k_points.append((t * math.pi, 0.0))
        
        # Gamma to M: (k, k)
        for t in np.linspace(0.1, 1.0, 6):
            k_points.append((t * math.pi, t * math.pi))
        
        # X to M: (pi, k)
        for t in np.linspace(0.1, 0.9, 6):
            k_points.append((math.pi, t * math.pi))
        
        # Add some interior points
        # (kx, ky) with 0 < kx < ky < pi, kx + ky < 2*pi
        interior_points = []
        for i in range(1, 4):
            for j in range(i + 1, 5):
                kx = (i / 5.0) * math.pi
                ky = (j / 5.0) * math.pi
                if kx < math.pi and ky < math.pi:
                    interior_points.append((kx, ky))
        
        k_points.extend(interior_points[:10])
        
        # Take first 30
        return k_points[:30]

    k_points = generate_k_points()

    # --- Evaluate a cell: compute the gap-midgap ratio ---
    def evaluate_cell(cell):
        """
        Compute the gap-midgap ratio between bands 4 and 5.
        """
        inv_eps = compute_inv_eps_coeffs_large(cell, max_range)
        
        min_omega5_sq = float('inf')
        max_omega4_sq = 0.0
        
        for kx, ky in k_points:
            eigvals = solve_k_point_fast(inv_eps, kx, ky)
            if eigvals is None or len(eigvals) < 5:
                continue
            omega4_sq = eigvals[3]  # 4th band (0-indexed: 3)
            omega5_sq = eigvals[4]  # 5th band (0-indexed: 4)
            
            if omega4_sq < 0:
                omega4_sq = 0.0
            if omega5_sq < 0:
                omega5_sq = 0.0
            
            omega4 = math.sqrt(omega4_sq)
            omega5 = math.sqrt(omega5_sq)
            
            if omega5 < min_omega5_sq:
                min_omega5_sq = omega5
            if omega4 > max_omega4_sq:
                max_omega4_sq = omega4
        
        if min_omega5_sq == float('inf') or max_omega4_sq == 0.0:
            return -1.0
        
        gap = min_omega5_sq - max_omega4_sq
        midgap = (min_omega5_sq + max_omega4_sq) / 2.0
        
        if midgap <= 0:
            return -1.0
        
        return gap / midgap

    # --- Symmetric orbit generation ---
    def get_orbits():
        """
        Generate all orbits under the square symmetry group (C4v).
        The symmetry group includes: identity, 90/180/270 rotation, reflection about x=y,
        reflection about y=x (same), reflection about vertical/horizontal midlines.
        For a pixel at (i, j), its orbit includes:
        (i, j), (j, i), (N-1-i, N-1-j), (N-1-j, N-1-i)
        And their reflections:
        (N-1-i, j), (i, N-1-j), (N-1-i, N-1-j) [already], (N-1-j, N-1-i) [already]
        
        Actually, the full C4v group has 8 elements. The orbit of (i,j) under C4v:
        1. (i, j)
        2. (j, i)  -- reflection about y=x
        3. (N-1-i, N-1-j) -- 180 rotation
        4. (N-1-j, N-1-i) -- reflection about y=x then 180, or 180 then reflection
        5. (N-1-i, j) -- reflection about vertical midline
        6. (i, N-1-j) -- reflection about horizontal midline
        7. (N-1-j, i) -- 90 rotation
        8. (j, N-1-i) -- 270 rotation
        
        Let's generate all 8 and take unique.
        """
        N = size
        visited = set()
        orbits = []
        
        for i in range(N):
            for j in range(N):
                if (i, j) in visited:
                    continue
                orbit = []
                points = [
                    (i, j),
                    (j, i),
                    (N-1-i, N-1-j),
                    (N-1-j, N-1-i),
                    (N-1-i, j),
                    (i, N-1-j),
                    (N-1-j, i),
                    (j, N-1-i),
                ]
                for p in points:
                    if p not in visited:
                        orbit.append(p)
                        visited.add(p)
                orbits.append(orbit)
        
        return orbits

    # --- Initial candidate structures ---
    def create_air_holes_cell(hole_radius):
        """Create a cell with air holes in a dielectric background, centered."""
        cell = [[1] * size for _ in range(size)]
        cx = (size - 1) / 2.0
        cy = (size - 1) / 2.0
        for i in range(size):
            for j in range(size):
                x = i - cx
                y = j - cy
                if x * x + y * y < hole_radius * hole_radius:
                    cell[i][j] = 0
        return cell

    def create_dielectric_rods_cell(rod_radius):
        """Create a cell with dielectric rods in an air background, centered."""
        cell = [[0] * size for _ in range(size)]
        cx = (size - 1) / 2.0
        cy = (size - 1) / 2.0
        for i in range(size):
            for j in range(size):
                x = i - cx
                y = j - cy
                if x * x + y * y < rod_radius * rod_radius:
                    cell[i][j] = 1
        return cell

    # --- Local search ---
    def local_search(initial_cell, max_steps=200):
        """
        Perform local search by flipping symmetric orbits.
        """
        best_cell = [row[:] for row in initial_cell]
        best_score = evaluate_cell(best_cell)
        orbits = get_orbits()
        
        np.random.seed(42)
        orbit_indices = list(range(len(orbits)))
        np.random.shuffle(orbit_indices)
        
        step = 0
        while step < max_steps:
            if time.time() - start_time > time_limit:
                break
            
            # Pick a random orbit
            idx = orbit_indices[step % len(orbit_indices)]
            orbit = orbits[idx]
            
            new_cell = [row[:] for row in best_cell]
            current_val = best_cell[orbit[0][0]][orbit[0][1]]
            new_val = 1 - current_val
            
            for (i, j) in orbit:
                new_cell[i][j] = new_val
            
            new_score = evaluate_cell(new_cell)
            
            if new_score > best_score:
                best_score = new_score
                best_cell = new_cell
            else:
                pass  # Keep best_cell as is
            
            step += 1
            # Occasionally reshuffle
            if step % 50 == 0:
                np.random.shuffle(orbit_indices)
        
        return best_cell, best_score

    # --- Main search ---
    # Generate initial candidates
    candidates = []
    
    # Air holes in dielectric (common for TM gaps)
    for r in [0.2, 0.25, 0.3, 0.35, 0.4]:
        candidates.append(create_air_holes_cell(r))
    
    # Dielectric rods in air
    for r in [0.15, 0.2, 0.25, 0.3]:
        candidates.append(create_dielectric_rods_cell(r))
    
    # Evaluate all candidates
    best_cell = None
    best_score = -float('inf')
    
    for cell in candidates:
        if time.time() - start_time > time_limit:
            break
        score = evaluate_cell(cell)
        if score > best_score:
            best_score = score
            best_cell = [row[:] for row in cell]
    
    if best_cell is None:
        # Fallback: return a simple structure
        best_cell = create_air_holes_cell(0.3)
    
    # Local search from the best candidate
    if time.time() - start_time < time_limit - 30:  # Leave time margin
        best_cell, best_score = local_search(best_cell, max_steps=100)
    
    # Additional local search with different random seeds if time permits
    if time.time() - start_time < time_limit - 20:
        # Try a few more local searches from slightly perturbed starting points
        for attempt in range(3):
            if time.time() - start_time > time_limit:
                break
            # Perturb the best cell slightly
            perturbed = [row[:] for row in best_cell]
            orbits = get_orbits()
            for _ in range(3):
                idx = np.random.randint(len(orbits))
                orbit = orbits[idx]
                current_val = perturbed[orbit[0][0]][orbit[0][1]]
                new_val = 1 - current_val
                for (i, j) in orbit:
                    perturbed[i][j] = new_val
            
            cell, score = local_search(perturbed, max_steps=50)
            if score > best_score:
                best_score = score
                best_cell = cell
    
    # Ensure we return within time limit
    if time.time() - start_time > time_limit - 5:
        pass  # Just return what we have
    
    # Final validation: ensure symmetry
    # The local search already preserves symmetry, but let's double-check
    sym_cell = [row[:] for row in best_cell]
    for i in range(size):
        for j in range(size):
            # Enforce all symmetries
            val = sym_cell[i][j]
            sym_cell[j][i] = val
            sym_cell[size-1-i][size-1-j] = val
            sym_cell[size-1-j][size-1-i] = val
            sym_cell[size-1-i][j] = val
            sym_cell[i][size-1-j] = val
            sym_cell[size-1-j][i] = val
            sym_cell[j][size-1-i] = val
    
    return sym_cell
