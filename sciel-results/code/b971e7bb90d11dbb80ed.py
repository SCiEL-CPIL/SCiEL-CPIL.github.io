"""Photonic crystal design: parametric ring/cross geometry + fast PW eval + targeted refinement."""
import numpy as np
import time
import sys

def design(problem):
    start_time = time.time()
    DEADLINE = start_time + 215.0  # safety margin before 230s kill

    size = problem["pixels"]       # 32
    eps_low = problem["eps_low"]   # 1
    eps_high = problem["eps_high"] # 12

    # ---- Symmetry orbits (square group about center) ----
    def get_orbits(sz):
        visited = set()
        orbits = []
        for i in range(sz):
            for j in range(sz):
                if (i, j) not in visited:
                    orbit = {
                        (i, j),
                        (sz-1-i, sz-1-j),
                        (i, sz-1-j),
                        (sz-1-i, j),
                        (j, i),
                        (sz-1-j, sz-1-i),
                    }
                    visited.update(orbit)
                    orbits.append(list(orbit))
        return orbits

    orbits = get_orbits(size)
    num_orbits = len(orbits)
    pixel_to_orbit = np.zeros((size, size), dtype=int)
    for o_idx, orbit in enumerate(orbits):
        for (i, j) in orbit:
            pixel_to_orbit[i, j] = o_idx

    def orbit_to_cell(orbit_vals):
        cell = np.zeros((size, size), dtype=np.uint8)
        for o_idx in range(num_orbits):
            val = orbit_vals[o_idx]
            for (i, j) in orbits[o_idx]:
                cell[i, j] = val
        return cell

    # ---- Plane-wave setup: 289 PWs => N=8 ----
    N_pw = 8
    G_range = np.arange(-N_pw, N_pw + 1)
    Gx, Gy = np.meshgrid(G_range, G_range, indexing='ij')
    G_vectors = np.column_stack((Gx.ravel(), Gy.ravel()))  # (289, 2)
    num_pw = len(G_vectors)

    # Lookup for G_diff
    G_lookup = {}
    for idx in range(num_pw):
        G_lookup[(int(G_vectors[idx, 0]), int(G_vectors[idx, 1]))] = idx

    # Precompute G_diff index matrix (289 x 289) for fast H construction
    # H_diff[a, b] = index into inv_eps_G for G_vectors[a] - G_vectors[b], or -1 if out of range
    G_diff_idx = np.full((num_pw, num_pw), -1, dtype=int)
    for a in range(num_pw):
        for b in range(num_pw):
            key = (int(G_vectors[a, 0] - G_vectors[b, 0]),
                   int(G_vectors[a, 1] - G_vectors[b, 1]))
            if key in G_lookup:
                G_diff_idx[a, b] = G_lookup[key]

    # ---- Fast Fourier of 1/eps using 2D DFT via matrix products ----
    # Precompute phase matrices for each Gx, Gy in [-8, 8]
    # phase_x[gx, i] = exp(-2j*pi*gx*i/size)
    phase_x = np.exp(-2j * np.pi * G_range[:, np.newaxis] * np.arange(size)[np.newaxis, :] / size)  # (17, 32)
    phase_y = np.exp(-2j * np.pi * G_range[:, np.newaxis] * np.arange(size)[np.newaxis, :] / size)  # (17, 32)

    def compute_fourier_eps_inv(cell):
        """Compute Fourier coefficients of 1/eps for the 289 plane waves."""
        inv_eps = np.where(cell == 0, 1.0, 1.0 / eps_high)  # (32, 32)
        # 2D DFT: inv_eps_G[gx, gy] = (1/size^2) * sum_i sum_j inv_eps[i,j] * exp(-2pi i (gx*i + gy*j)/size)
        # = (1/size^2) * phase_x[:, :] @ inv_eps @ phase_y[:, :].T
        # phase_x: (17, 32), inv_eps: (32, 32), phase_y.T: (32, 17)
        temp = phase_x @ inv_eps  # (17, 32)
        temp = temp @ phase_y.T   # (17, 17)
        temp = temp / (size * size)
        # Now temp[gx_idx, gy_idx] where gx_idx = gx + 8, gy_idx = gy + 8
        inv_eps_G = np.zeros(num_pw, dtype=complex)
        for idx in range(num_pw):
            gx = int(G_vectors[idx, 0])
            gy = int(G_vectors[idx, 1])
            inv_eps_G[idx] = temp[gx + N_pw, gy + N_pw]
        return inv_eps_G

    # ---- Fast band evaluation using Lanczos / power iteration for top few eigenvalues ----
    # We only need bands 4 and 5 (indices 3, 4 in 0-based sorted omega^2)
    # Use np.linalg.eigvalsh on the 289x289 matrix, but build H efficiently
    
    def build_H(inv_eps_G, kx, ky):
        """Build the 289x289 Hamiltonian matrix."""
        k_plus_G = G_vectors + np.array([kx, ky])
        kG2 = np.sum(k_plus_G ** 2, axis=1)  # (289,)
        # H[a, b] = inv_eps_G[G_diff_idx[a,b]] * kG2[b]
        # Use broadcasting
        # G_diff_idx: (289, 289), -1 means zero
        valid = G_diff_idx >= 0
        # Create a masked array
        G_diff_safe = np.where(valid, G_diff_idx, 0)
        H = np.zeros((num_pw, num_pw), dtype=complex)
        H[valid] = inv_eps_G[G_diff_safe[valid]] * kG2[np.newaxis, :]
        return H

    # K-points for evaluation (irreducible BZ, 30 points)
    # Use a denser set for better gap estimation
    k_points = []
    # High-symmetry and along high-symmetry lines
    for t in np.linspace(0, 0.5, 11):
        k_points.append((t, 0.0))       # Gamma to X
        k_points.append((0.5, t))       # X to M
        k_points.append((t, t))         # Gamma to M
    # Remove duplicates
    k_points = list(set(k_points))
    # Add some interior points
    for t1 in [0.1, 0.2, 0.3, 0.4]:
        for t2 in [0.05, 0.15, 0.25, 0.35]:
            if t1 >= t2:
                k_points.append((t1, t2))
    k_points = list(set(k_points))[:30]  # cap at 30

    def evaluate_cell_fast(cell):
        """Evaluate gap-midgap ratio for a given cell. Returns ratio (can be negative)."""
        inv_eps_G = compute_fourier_eps_inv(cell)
        
        min_w5_sq = float('inf')
        max_w4_sq = 0.0
        
        for kx, ky in k_points:
            H = build_H(inv_eps_G, kx, ky)
            # Solve eigenvalue problem - only need a few eigenvalues
            # Use eigvalsh for the full matrix (289x289 is manageable)
            eigvals = np.linalg.eigvalsh(H)
            eigvals = np.real(eigvals)
            # Bands 4 and 5 (1-indexed) => indices 3 and 4 (0-indexed)
            w4_sq = eigvals[3]
            w5_sq = eigvals[4]
            
            if w5_sq < min_w5_sq:
                min_w5_sq = w5_sq
            if w4_sq > max_w4_sq:
                max_w4_sq = w4_sq
        
        if max_w4_sq <= 0 or min_w5_sq <= 0:
            return -1.0
        
        min_w5 = np.sqrt(min_w5_sq)
        max_w4 = np.sqrt(max_w4_sq)
        
        if min_w5 <= max_w4:
            return -1.0
        
        midgap = (min_w5 + max_w4) / 2.0
        gap = min_w5 - max_w4
        ratio = 2.0 * gap / midgap
        return ratio

    # ---- Parametric design family: concentric rings / holes ----
    # Design parameters:
    # - r1: radius of central dielectric disk (or air hole)
    # - r2: radius of second ring
    # - r3: radius of third ring
    # - mode: 'solid' (center is dielectric) or 'hole' (center is air)
    # - fill: overall fill factor bias
    
    def parametric_cell(r1, r2, r3, center_dielectric=True, ring_width=0.02):
        """Generate a 32x32 cell from parametric ring geometry."""
        cell = np.zeros((size, size), dtype=np.uint8)
        cx, cy = (size - 1) / 2.0, (size - 1) / 2.0
        for i in range(size):
            for j in range(size):
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                r = np.sqrt(x*x + y*y)
                # Determine if this pixel is dielectric
                # Central region
                if r < r1:
                    cell[i, j] = 1 if center_dielectric else 0
                elif r1 <= r < r2:
                    cell[i, j] = 1 - (1 if center_dielectric else 0)  # opposite
                elif r2 <= r < r3:
                    cell[i, j] = 1 if center_dielectric else 0
                else:
                    cell[i, j] = 0 if center_dielectric else 1
        # Enforce symmetry (should already be symmetric, but ensure)
        cell = np.round((cell + cell[::-1, ::-1] + cell.T + cell[:, ::-1].T) / 4.0).astype(np.uint8)
        return cell

    def cell_to_orbit_vals(cell):
        """Convert a 32x32 cell to orbit values (majority vote per orbit)."""
        orbit_vals = np.zeros(num_orbits, dtype=int)
        for o_idx, orbit in enumerate(orbits):
            vals = [cell[i, j] for (i, j) in orbit]
            orbit_vals[o_idx] = int(np.mean(vals) > 0.5)
        return orbit_vals

    # ---- Phase 1: Coarse parametric sweep ----
    best_score = -float('inf')
    best_cell = None
    best_orbit_vals = None

    # Sweep over ring radii and center type
    r1_vals = np.linspace(0.02, 0.18, 8)
    r2_vals = np.linspace(0.18, 0.30, 6)
    r3_vals = np.linspace(0.30, 0.45, 5)
    center_types = [True, False]

    total_sweeps = len(r1_vals) * len(r2_vals) * len(r3_vals) * len(center_types)
    sweep_count = 0

    for r1 in r1_vals:
        for r2 in r2_vals:
            for r3 in r3_vals:
                for center_diel in center_types:
                    if time.time() > DEADLINE - 30:
                        break
                    sweep_count += 1
                    cell = parametric_cell(r1, r2, r3, center_diel)
                    score = evaluate_cell_fast(cell)
                    if score > best_score:
                        best_score = score
                        best_cell = cell.copy()
                        best_orbit_vals = cell_to_orbit_vals(cell)

    if time.time() > DEADLINE - 30:
        # Out of time, return best found
        if best_cell is not None:
            return best_cell.tolist()
        else:
            return np.zeros((size, size), dtype=int).tolist()

    # ---- Phase 2: Fine parametric sweep around best ----
    if best_cell is not None:
        # Refine around the best parameters
        # We don't track which params gave the best, so do a local orbit-level search instead
        pass

    # ---- Phase 3: Local search on orbit values (fast, targeted) ----
    if best_orbit_vals is not None:
        current_vals = best_orbit_vals.copy()
        current_score = best_score

        # Single-flip and double-flip local search
        flip_order = np.random.RandomState(42).permutation(num_orbits)

        improved = True
        max_rounds = 20
        round_num = 0

        while improved and round_num < max_rounds and time.time() < DEADLINE - 15:
            improved = False
            round_num += 1
            for o_idx in flip_order:
                if time.time() > DEADLINE - 15:
                    break
                new_vals = current_vals.copy()
                new_vals[o_idx] = 1 - new_vals[o_idx]
                cell = orbit_to_cell(new_vals)
                score = evaluate_cell_fast(cell)
                if score > current_score:
                    current_vals = new_vals
                    current_score = score
                    improved = True
                    if score > best_score:
                        best_score = score
                        best_cell = cell.copy()
                        best_orbit_vals = new_vals.copy()

        # Double-flip refinement
        if time.time() < DEADLINE - 10:
            for _ in range(50):
                if time.time() > DEADLINE - 10:
                    break
                i1, i2 = np.random.choice(num_orbits, 2, replace=False)
                new_vals = current_vals.copy()
                new_vals[i1] = 1 - new_vals[i1]
                new_vals[i2] = 1 - new_vals[i2]
                cell = orbit_to_cell(new_vals)
                score = evaluate_cell_fast(cell)
                if score > current_score:
                    current_vals = new_vals
                    current_score = score
                    if score > best_score:
                        best_score = score
                        best_cell = cell.copy()
                        best_orbit_vals = new_vals.copy()

    # ---- Final: ensure symmetry and return ----
    if best_cell is not None:
        # Final symmetry enforcement
        best_cell = np.round((best_cell + best_cell[::-1, ::-1] + best_cell.T + best_cell[:, ::-1].T) / 4.0).astype(np.uint8)
        return best_cell.tolist()
    else:
        # Fallback: simple centered disk
        cell = np.zeros((size, size), dtype=np.uint8)
        cx, cy = (size - 1) / 2.0, (size - 1) / 2.0
        for i in range(size):
            for j in range(size):
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                if np.sqrt(x*x + y*y) < 0.15:
                    cell[i, j] = 1
        cell = np.round((cell + cell[::-1, ::-1] + cell.T + cell[:, ::-1].T) / 4.0).astype(np.uint8)
        return cell.tolist()
