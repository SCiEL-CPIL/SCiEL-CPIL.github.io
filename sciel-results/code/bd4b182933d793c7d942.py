"""
Combined program:
- Structural idea (inspiration 1): four circular rods at quarter-points (0.25, 0.25) etc.
- Refinement machinery (current program): parameterized shape (radius, x-offset, y-offset)
  with a fast 7x7 plane-wave evaluator to optimize the geometry.
- Symmetry enforcement via quadrant mirroring.
"""
import numpy as np
import time
import math

# EVOLVE-BLOCK-START

def design(problem):
    size = problem["pixels"]  # 32
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)

    start_time = time.time()
    deadline = start_time + 195.0  # safety margin before 200s/230s kill

    # --- Precompute fast plane-wave evaluator constants ---
    # Use 7x7 = 49 plane waves (h, k in [-3, 3])
    pw_range = 3
    G_vectors = []
    for h in range(-pw_range, pw_range + 1):
        for k in range(-pw_range, pw_range + 1):
            G_vectors.append((h, k))
    N_pw = len(G_vectors)

    # Precompute G differences for Fourier coefficients
    # diff_idx[i, j] = (G_i - G_j) mod size -> index into FFT
    G_arr = np.array(G_vectors, dtype=int)
    diff_h = G_arr[:, 0][:, None] - G_arr[:, 0][None, :]
    diff_k = G_arr[:, 1][:, None] - G_arr[:, 1][None, :]
    idx_h = diff_h % size
    idx_k = diff_k % size

    # K-points: sample along Gamma-M, M-X, and Gamma-X for the irreducible BZ
    # Use dimensionless coordinates where the reciprocal lattice vector is 1
    # (i.e., k is in units of 1/a, G vectors are integers)
    # Irreducible BZ: 0 <= kx <= ky, kx + ky <= 1
    k_points = []
    # Gamma to M: (0, t) for t in [0, 0.5]
    for t in np.linspace(0, 0.5, 6):
        k_points.append((0.0, t))
    # M to X: (t, 0.5 - t) for t in [0, 0.5]
    for t in np.linspace(0, 0.5, 6):
        k_points.append((t, 0.5 - t))
    # Gamma to X: (t, t) for t in [0, 0.5]
    for t in np.linspace(0, 0.5, 4):
        k_points.append((t, t))
    k_points = np.array(k_points)

    def build_H(fft_1eps, kx, ky):
        """Build the 49x49 Hamiltonian matrix for a given k-point."""
        # K_i = G_i + k, where G_i are integers and k is fractional
        Kx = G_arr[:, 0] + kx
        Ky = G_arr[:, 1] + ky

        # dot[i, j] = K_i dot K_j
        dot = Kx[:, None] * Kx[None, :] + Ky[:, None] * Ky[None, :]

        # Fourier coefficients of 1/eps
        coeffs = fft_1eps[idx_h, idx_k] / (size * size)

        return dot * coeffs

    def evaluate_gap(cell_binary):
        """
        Fast evaluation of the gap-midgap ratio between bands 4 and 5.
        cell_binary: 32x32 numpy array of 0/1 (1 = dielectric)
        """
        eps_grid = np.where(cell_binary == 1, eps_high, eps_low)
        inv_eps = 1.0 / eps_grid
        fft_1eps = np.fft.fft2(inv_eps)

        max_w4 = 0.0
        min_w5 = float('inf')

        for kx, ky in k_points:
            H = build_H(fft_1eps, kx, ky)
            # H should be real symmetric for real eps
            H_real = H.real
            evals = np.linalg.eigvalsh(H_real)
            if len(evals) >= 5:
                w4 = np.sqrt(max(0.0, evals[3]))
                w5 = np.sqrt(max(0.0, evals[4]))
                if w4 > max_w4:
                    max_w4 = w4
                if w5 < min_w5:
                    min_w5 = w5

        if min_w5 == float('inf') or max_w4 == 0.0:
            return 0.0
        gap = min_w5 - max_w4
        midgap = (min_w5 + max_w4) / 2.0
        if midgap == 0:
            return 0.0
        return 2.0 * gap / midgap

    # --- Parameterization: 4 circles at quarter-points ---
    # Parameters: radius, cx_offset, cy_offset
    # Centers are at (0.25 + cx_offset, 0.25 + cy_offset) and symmetric equivalents
    # Start with the known good design from inspiration 1: r=0.15, offsets=0
    params = np.array([0.15, 0.0, 0.0])
    param_names = ['radius', 'cx_offset', 'cy_offset']
    param_bounds = [(0.05, 0.35), (-0.1, 0.1), (-0.1, 0.1)]

    def make_cell(p):
        """Generate a symmetric 32x32 binary cell from parameters."""
        r, cx_off, cy_off = p
        cell = np.zeros((size, size), dtype=int)
        # Only need to fill the top-left quadrant (i < 16, j < 16)
        # due to symmetry, then mirror
        # Centers in the top-left quadrant region: (0.25 + cx_off, 0.25 + cy_off)
        # In pixel coordinates (0-indexed), center is at ((0.25+cx_off)*32, (0.25+cy_off)*32)
        cx_pix = (0.25 + cx_off) * size
        cy_pix = (0.25 + cy_off) * size
        r_pix = r * size
        r2 = r_pix * r_pix

        for i in range(size):
            for j in range(size):
                x = (i + 0.5) / size - 0.5  # [-0.5, 0.5)
                y = (j + 0.5) / size - 0.5
                # Check against all 4 centers
                # Centers at (+/-0.25+cx_off, +/-0.25+cy_off)
                # By symmetry, we can just check the one in the current quadrant
                # but it's simpler to check all 4
                found = False
                for sx in [1, -1]:
                    for sy in [1, -1]:
                        cx = sx * (0.25 + cx_off)
                        cy = sy * (0.25 + cy_off)
                        dx = x - cx
                        dy = y - cy
                        if dx * dx + dy * dy < r2:
                            found = True
                            break
                    if found:
                        break
                if found:
                    cell[i, j] = 1
        return cell

    # --- Optimization: stochastic hill-climbing with adaptive step ---
    best_cell = make_cell(params)
    best_score = evaluate_gap(best_cell)
    current_params = params.copy()
    step_sizes = np.array([0.01, 0.005, 0.005])  # initial step sizes per param
    min_step = np.array([0.001, 0.0005, 0.0005])
    max_step = np.array([0.05, 0.02, 0.02])

    iteration = 0
    while time.time() < deadline:
        iteration += 1

        # Pick a random parameter to perturb
        idx = np.random.randint(3)
        direction = np.random.choice([-1, 1])
        delta = direction * step_sizes[idx]

        new_params = current_params.copy()
        new_params[idx] += delta

        # Clamp to bounds
        for i in range(3):
            new_params[i] = np.clip(new_params[i], param_bounds[i][0], param_bounds[i][1])

        new_cell = make_cell(new_params)
        new_score = evaluate_gap(new_cell)

        if new_score > best_score:
            best_score = new_score
            best_cell = new_cell
            current_params = new_params.copy()
            # Increase step size for this parameter
            step_sizes[idx] = min(step_sizes[idx] * 1.2, max_step[idx])
            # Slightly decrease others
            for j in range(3):
                if j != idx:
                    step_sizes[j] = max(step_sizes[j] * 0.95, min_step[j])
        else:
            # Decrease step size for this parameter
            step_sizes[idx] = max(step_sizes[idx] * 0.8, min_step[idx])

        # Periodic restart: perturb all params slightly to escape local minima
        if iteration % 200 == 0:
            current_params = best_params + np.random.normal(0, 0.01, 3)
            for i in range(3):
                current_params[i] = np.clip(current_params[i], param_bounds[i][0], param_bounds[i][1])
            test_cell = make_cell(current_params)
            test_score = evaluate_gap(test_cell)
            if test_score > best_score:
                best_score = test_score
                best_cell = test_cell
            else:
                current_params = best_params.copy()

    # --- Finalize: enforce exact symmetry ---
    # The make_cell function already produces a symmetric cell, but let's verify
    # and enforce by mirroring the top-left 16x16 quadrant
    half = size // 2
    q = best_cell[:half, :half].copy()
    final_cell = np.zeros((size, size), dtype=int)

    for i in range(half):
        for j in range(half):
            val = q[i, j]
            final_cell[i, j] = val
            final_cell[i, size - 1 - j] = val
            final_cell[size - 1 - i, j] = val
            final_cell[size - 1 - i, size - 1 - j] = val

    # Enforce transpose symmetry: final_cell should be symmetric
    # The above construction ensures 180-degree and reflection symmetry.
    # For transpose: final_cell[i][j] == final_cell[j][i]
    # Since we mirrored from the quadrant, and the quadrant itself may not be
    # transpose-symmetric, we need to enforce it.
    # Take the average of cell and its transpose, then threshold.
    sym_cell = (final_cell + final_cell.T) / 2.0
    final_cell = (sym_cell >= 0.5).astype(int)

    # Final check: enforce all symmetries
    # cell == cell[::-1] row-wise
    # cell == cell[::-1] column-wise
    # cell == cell.T
    # The above should satisfy these. Let's do a final enforcement.
    for i in range(size):
        for j in range(size):
            # Get the 8 symmetric equivalents
            pts = [
                (i, j),
                (size - 1 - i, j),
                (i, size - 1 - j),
                (size - 1 - i, size - 1 - j),
                (j, i),
                (size - 1 - j, j),
                (j, size - 1 - i),
                (size - 1 - j, size - 1 - i)
            ]
            # Unique points
            unique_pts = list(set(pts))
            avg = np.mean([final_cell[pi, pj] for pi, pj in unique_pts])
            final_cell[i, j] = 1 if avg >= 0.5 else 0

    return final_cell.tolist()

# EVOLVE-BLOCK-END
