"""Rewritten design program: direct pixel-level optimization with C4v symmetry
and a fast plane-wave band-gap proxy, using simulated annealing / hill climbing."""
import numpy as np
import time
import math

# EVOLVE-BLOCK-START
def design(problem):
    """
    Designs a 32x32 unit cell for a TM band gap between bands 4 and 5.
    Strategy:
    1. Represent the dielectric field as a binary 32x32 pixel map with C4v symmetry.
    2. Use a fast 9x9 plane-wave proxy to estimate the gap-midgap ratio.
    3. Optimize via simulated annealing with pixel-flip moves that respect C4v symmetry.
    """
    size = problem["pixels"]  # 32
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)

    start_time = time.time()
    deadline = start_time + 195.0  # Safety margin for 230s limit

    # --- C4v symmetry orbit computation ---
    # The C4v group has 8 elements. For each pixel (i,j), compute its orbit.
    # We'll work with unique orbits (fundamental domain) and flip all pixels in an orbit together.

    # Precompute orbits
    orbits = []
    orbit_set = set()
    for i in range(size):
        for j in range(size):
            if (i, j) in orbit_set:
                continue
            # Generate orbit under C4v
            pts = set()
            x, y = i, j
            # Identity
            pts.add((x, y))
            # Reflection x: (size-1-x, y)
            pts.add((size - 1 - x, y))
            # Reflection y: (x, size-1-y)
            pts.add((x, size - 1 - y))
            # 180 rotation: (size-1-x, size-1-y)
            pts.add((size - 1 - x, size - 1 - y))
            # Transpose: (y, x)
            pts.add((y, x))
            # Transpose + ref x: (size-1-y, x)
            pts.add((size - 1 - y, x))
            # Transpose + ref y: (y, size-1-x)
            pts.add((y, size - 1 - x))
            # Transpose + 180: (size-1-y, size-1-x)
            pts.add((size - 1 - y, size - 1 - x))

            pts = sorted(pts)
            orbits.append(pts)
            for p in pts:
                orbit_set.add(p)

    num_orbits = len(orbits)

    # --- Fast plane-wave evaluator ---
    # Use 9x9 = 81 plane waves for a good balance of speed and accuracy.
    pw_half = 4  # h, k in [-4, 4]
    G_vectors = []
    for h in range(-pw_half, pw_half + 1):
        for k in range(-pw_half, pw_half + 1):
            G_vectors.append((h, k))
    N_pw = len(G_vectors)

    # Precompute index mapping for G differences
    # dh, dk in [-8, 8]
    fft_size = size  # 32
    coeff_cache = {}

    def get_fourier_coeff(fft_1eps, dh, dk):
        """Extract Fourier coefficient for frequency (dh, dk) from FFT of 1/eps."""
        idx_h = dh % fft_size
        idx_k = dk % fft_size
        return fft_1eps[idx_h, idx_k] / (fft_size * fft_size)

    # K-points in the irreducible Brillouin zone
    # Irreducible BZ for square lattice: 0 <= kx <= ky, kx + ky <= 0.5 (in units of 1/a)
    # Key high-symmetry points: Gamma (0,0), M (0, 0.5), X (0.25, 0.25)
    # Let's sample along M-X and Gamma-M lines.
    k_points = []
    # Gamma to M: (0, t) for t in [0, 0.5]
    for t in np.linspace(0, 0.5, 6):
        k_points.append((0.0, t))
    # M to X: (t, 0.5 - t) for t in [0, 0.25]
    for t in np.linspace(0, 0.25, 6):
        k_points.append((t, 0.5 - t))
    # Also add a few interior points
    k_points.append((0.1, 0.2))
    k_points.append((0.15, 0.15))

    def evaluate_gap(cell):
        """
        Computes the approximate band gap between bands 4 and 5.
        Returns the gap-midgap ratio.
        """
        # Convert binary cell to epsilon grid
        eps_grid = np.where(cell == 1, eps_high, eps_low)
        inv_eps = 1.0 / eps_grid
        fft_1eps = np.fft.fft2(inv_eps)

        max_omega_4 = 0.0
        min_omega_5 = float('inf')

        for kx, ky in k_points:
            # Build H matrix
            H = np.zeros((N_pw, N_pw), dtype=float)

            for i in range(N_pw):
                hi, ki = G_vectors[i]
                Kx_i = hi + kx
                Ky_i = ki + ky

                for j in range(N_pw):
                    hj, kj = G_vectors[j]
                    Kx_j = hj + kx
                    Ky_j = kj + ky

                    dot = Kx_i * Kx_j + Ky_i * Ky_j
                    dh = hi - hj
                    dk = ki - kj
                    coeff = get_fourier_coeff(fft_1eps, dh, dk).real
                    H[i, j] = dot * coeff

            # Solve eigenvalue problem
            eigenvalues = np.linalg.eigvalsh(H)

            if len(eigenvalues) >= 5:
                w4_sq = eigenvalues[3]
                w5_sq = eigenvalues[4]

                w4 = math.sqrt(max(0.0, w4_sq))
                w5 = math.sqrt(max(0.0, w5_sq))

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

    # --- Initial design: central dielectric square ---
    # Start with a dielectric square in the center
    cell = np.zeros((size, size), dtype=int)
    # Central 12x12 block
    half = size // 2
    start = half - 6
    end = half + 6
    cell[start:end, start:end] = 1

    # Enforce C4v symmetry
    for orbit in orbits:
        # All pixels in the orbit should have the same value
        # Use the value of the first pixel
        val = cell[orbit[0][0], orbit[0][1]]
        for (i, j) in orbit:
            cell[i, j] = val

    best_score = evaluate_gap(cell)
    best_cell = cell.copy()

    # --- Simulated annealing ---
    # Temperature schedule
    T_start = 1.0
    T_end = 0.01
    cooling_rate = 0.9995

    T = T_start
    num_iterations = 0
    max_iterations = 100000
    no_improve_count = 0
    plateau_limit = 2000

    # Precompute orbit indices for fast access
    # orbit_to_pixel: for each orbit, store the list of (i,j) positions
    orbit_pixels = []
    for orbit in orbits:
        orbit_pixels.append(orbit)

    while time.time() < deadline and num_iterations < max_iterations:
        num_iterations += 1

        # Pick a random orbit to flip
        idx = np.random.randint(num_orbits)
        pixels = orbit_pixels[idx]

        # Flip all pixels in the orbit
        for (i, j) in pixels:
            cell[i, j] = 1 - cell[i, j]

        new_score = evaluate_gap(cell)

        if new_score > best_score:
            best_score = new_score
            best_cell = cell.copy()
            no_improve_count = 0
        else:
            # Accept with probability exp((new_score - best_score) / T)
            delta = new_score - best_score
            if delta < 0:
                prob = math.exp(delta / T)
                if np.random.random() > prob:
                    # Reject: flip back
                    for (i, j) in pixels:
                        cell[i, j] = 1 - cell[i, j]
            no_improve_count += 1

        # Cool down
        T = max(T_end, T * cooling_rate)

        # If plateau for too long, restart with a new random design
        if no_improve_count > plateau_limit:
            # Random restart
            cell = np.random.randint(0, 2, (size, size))
            for orbit in orbits:
                val = cell[orbit[0][0], orbit[0][1]]
                for (i, j) in orbit:
                    cell[i, j] = val
            score = evaluate_gap(cell)
            if score > best_score:
                best_score = score
                best_cell = cell.copy()
            no_improve_count = 0
            T = T_start * 0.5

    # Use the best cell found
    final_cell = best_cell

    # Final symmetry enforcement (should already be symmetric, but verify)
    sym_cell = np.zeros_like(final_cell)
    for orbit in orbits:
        val = final_cell[orbit[0][0], orbit[0][1]]
        for (i, j) in orbit:
            sym_cell[i, j] = val

    return sym_cell.tolist()

# EVOLVE-BLOCK-END
