"""Photonic crystal design with vectorized PW solver and pixel-level local search."""
import numpy as np
import time
import math

# EVOLVE-BLOCK-START

def _build_pw_solver(size, eps_low, eps_high, n_pw, k_points):
    """
    Build a fast plane-wave band solver for a 2D square lattice.
    Returns a function that given a permittivity field (size x size array)
    computes the gap-midgap ratio for bands 4 and 5.
    """
    half = n_pw // 2
    idx = np.arange(-half, half + 1)
    Gx, Gy = np.meshgrid(idx, idx, indexing='ij')
    G2 = Gx**2 + Gy**2
    nG = len(Gx)
    
    Gx_flat = Gx.ravel()
    Gy_flat = Gy.ravel()
    
    # Precompute G difference indices for the circulant matrix
    # diff_idx[i, j] = index of (Gx_i - Gx_j, Gy_i - Gy_j) in the G array, or -1 if out of range
    g_map = {}
    for gi in range(nG):
        g_map[(int(Gx_flat[gi]), int(Gy_flat[gi]))] = gi
    
    diff_idx = np.zeros((nG, nG), dtype=np.int32)
    for i in range(nG):
        for j in range(nG):
            dx = int(Gx_flat[i]) - int(Gx_flat[j])
            dy = int(Gy_flat[i]) - int(Gy_flat[j])
            key = (dx, dy)
            if key in g_map:
                diff_idx[i, j] = g_map[key]
            else:
                diff_idx[i, j] = -1
    
    # k-points in irreducible BZ
    if k_points is None:
        k_points = []
        for i in range(7):
            k = i / 6.0 * 0.5
            k_points.append((k, 0.0))
        for i in range(1, 7):
            k = i / 6.0 * 0.5
            k_points.append((k, k))
        for i in range(1, 7):
            k = 0.5 - i / 6.0 * 0.5
            k_points.append((0.5, k))
        for i in range(1, 7):
            for j in range(1, 7):
                kx = i / 6.0 * 0.5
                ky = j / 6.0 * 0.5
                if ky <= kx and (kx, ky) not in k_points:
                    k_points.append((kx, ky))
        k_points = k_points[:30]
    
    # Precompute k vectors
    k_vecs = []
    for kx_frac, ky_frac in k_points:
        kx = 2.0 * math.pi * kx_frac
        ky = 2.0 * math.pi * ky_frac
        kvec2 = (Gx_flat + kx)**2 + (Gy_flat + ky)**2
        k_vecs.append(kvec2)
    
    # FFT setup
    N_fft = 2 * size
    fft_indices = np.zeros((nG, 2), dtype=np.int32)
    for gi in range(nG):
        gx = int(Gx_flat[gi])
        gy = int(Gy_flat[gi])
        fft_indices[gi, 0] = (gx + N_fft // 2) % N_fft
        fft_indices[gi, 1] = (gy + N_fft // 2) % N_fft
    
    def compute_bands(eps_field):
        """
        Compute TM band structure. eps_field is size x size array of 0/1.
        Returns the gap-midgap ratio for bands 4 and 5.
        """
        eps_full = eps_low + (eps_high - eps_low) * eps_field
        inv_eps = 1.0 / eps_full
        
        inv_eps_padded = np.zeros((N_fft, N_fft), dtype=np.float64)
        inv_eps_padded[:size, :size] = inv_eps
        inv_eps_fft = np.fft.fft2(inv_eps_padded) / (N_fft * N_fft)
        
        eps_fourier = np.zeros(nG, dtype=np.float64)
        for gi in range(nG):
            eps_fourier[gi] = inv_eps_fft[fft_indices[gi, 0], fft_indices[gi, 1]]
        
        max_omega_4 = 0.0
        min_omega_5 = np.inf
        
        for kvec2 in k_vecs:
            # M[i,j] = eps_fourier[diff_idx[i,j]] * kvec2[j]
            # Vectorized: M = eps_fourier[diff_idx] * kvec2[None, :]
            M = eps_fourier[diff_idx] * kvec2[None, :]
            
            # For TM with real eps, M should be symmetric
            M_sym = (M + M.T) / 2.0
            
            try:
                eigvals = np.linalg.eigvalsh(M_sym)
                eigvals = np.sort(eigvals)
            except:
                eigvals = np.linalg.eigvals(M)
                eigvals = np.sort(np.real(eigvals))
            
            if len(eigvals) >= 5:
                omega_4 = math.sqrt(max(0, eigvals[3]))
                omega_5 = math.sqrt(max(0, eigvals[4]))
                max_omega_4 = max(max_omega_4, omega_4)
                min_omega_5 = min(min_omega_5, omega_5)
        
        gap = min_omega_5 - max_omega_4
        midgap = min_omega_5 + max_omega_4
        if midgap <= 0:
            return 0.0
        return 2.0 * gap / midgap
    
    return compute_bands


def _apply_symmetry(cell, size):
    """
    Apply square symmetry: cell must be invariant under D4 group.
    Enforce by averaging over all 8 symmetry operations.
    """
    sym_cell = np.zeros((size, size), dtype=np.float64)
    count = 0
    for fr in [False, True]:
        for fc in [False, True]:
            for tr in [False, True]:
                c = cell.copy()
                if tr:
                    c = c.T
                if fr:
                    c = c[::-1, :]
                if fc:
                    c = c[:, ::-1]
                sym_cell += c
                count += 1
    sym_cell /= count
    return (sym_cell > 0.5).astype(int)


def _generate_initial_cells(size):
    """Generate diverse initial symmetric cells."""
    cells = []
    
    # 1. Central rod (circular)
    cell = np.zeros((size, size), dtype=int)
    center = size / 2.0
    radius = size * 0.25
    for i in range(size):
        for j in range(size):
            dx = i + 0.5 - center
            dy = j + 0.5 - center
            if dx*dx + dy*dy <= radius*radius:
                cell[i, j] = 1
    cell = _apply_symmetry(cell, size)
    cells.append(cell)
    
    # 2. Larger central rod
    cell = np.zeros((size, size), dtype=int)
    radius = size * 0.35
    for i in range(size):
        for j in range(size):
            dx = i + 0.5 - center
            dy = j + 0.5 - center
            if dx*dx + dy*dy <= radius*radius:
                cell[i, j] = 1
    cell = _apply_symmetry(cell, size)
    cells.append(cell)
    
    # 3. Square rod
    cell = np.zeros((size, size), dtype=int)
    r = size * 0.2
    for i in range(size):
        for j in range(size):
            dx = abs(i + 0.5 - center)
            dy = abs(j + 0.5 - center)
            if dx <= r and dy <= r:
                cell[i, j] = 1
    cell = _apply_symmetry(cell, size)
    cells.append(cell)
    
    # 4. Cross pattern
    cell = np.zeros((size, size), dtype=int)
    w = size * 0.15
    for i in range(size):
        for j in range(size):
            dx = abs(i + 0.5 - center)
            dy = abs(j + 0.5 - center)
            if dx <= w or dy <= w:
                cell[i, j] = 1
    cell = _apply_symmetry(cell, size)
    cells.append(cell)
    
    # 5. Ring
    cell = np.zeros((size, size), dtype=int)
    r_inner = size * 0.15
    r_outer = size * 0.35
    for i in range(size):
        for j in range(size):
            dx = i + 0.5 - center
            dy = j + 0.5 - center
            dist = math.sqrt(dx*dx + dy*dy)
            if r_inner <= dist <= r_outer:
                cell[i, j] = 1
    cell = _apply_symmetry(cell, size)
    cells.append(cell)
    
    # 6. Diagonal cross
    cell = np.zeros((size, size), dtype=int)
    w = size * 0.12
    for i in range(size):
        for j in range(size):
            dx = abs(i + 0.5 - center)
            dy = abs(j + 0.5 - center)
            if abs(dx - dy) <= w:
                cell[i, j] = 1
    cell = _apply_symmetry(cell, size)
    cells.append(cell)
    
    # 7. Four small rods at corners
    cell = np.zeros((size, size), dtype=int)
    r = size * 0.12
    offset = size * 0.3
    for cx, cy in [(center - offset, center - offset), (center + offset, center - offset),
                    (center - offset, center + offset), (center + offset, center + offset)]:
        for i in range(size):
            for j in range(size):
                dx = i + 0.5 - cx
                dy = j + 0.5 - cy
                if dx*dx + dy*dy <= r*r:
                    cell[i, j] = 1
    cell = _apply_symmetry(cell, size)
    cells.append(cell)
    
    # 8. Random with symmetry
    rng = np.random.RandomState(42)
    cell = rng.randint(0, 2, (size, size))
    cell = _apply_symmetry(cell, size)
    cells.append(cell)
    
    # 9. Higher filling fraction random
    rng = np.random.RandomState(123)
    cell = (rng.rand(size, size) < 0.5).astype(int)
    cell = _apply_symmetry(cell, size)
    cells.append(cell)
    
    # 10. Checkerboard-like
    cell = np.zeros((size, size), dtype=int)
    for i in range(size):
        for j in range(size):
            if (i // 4 + j // 4) % 2 == 0:
                cell[i, j] = 1
    cell = _apply_symmetry(cell, size)
    cells.append(cell)
    
    return cells


def _local_search(cell, size, compute_bands, time_limit, n_iter):
    """
    Pixel-level local search with symmetry enforcement.
    Tries flipping individual pixels (and their symmetric counterparts).
    """
    t_start = time.time()
    best_cell = cell.copy()
    best_score = compute_bands(cell.astype(np.float64))
    
    # Precompute symmetry groups: for each pixel, list all pixels that transform to it
    # Under D4, each pixel (i,j) maps to up to 8 pixels
    sym_groups = {}
    for i in range(size):
        for j in range(size):
            group = set()
            for fr in [False, True]:
                for fc in [False, True]:
                    for tr in [False, True]:
                        ii = i
                        jj = j
                        if tr:
                            ii, jj = jj, ii
                        if fr:
                            ii = size - 1 - ii
                        if fc:
                            jj = size - 1 - jj
                        group.add((ii, jj))
            sym_groups[(i, j)] = group
    
    # Get unique symmetry groups (to avoid redundant work)
    seen = set()
    unique_groups = []
    for key, group in sym_groups.items():
            frozen = frozenset(group)
            if frozen not in seen:
                seen.add(frozen)
                unique_groups.append(list(group))
    
    # Shuffle groups for exploration
    rng = np.random.RandomState(0)
    rng.shuffle(unique_groups)
    
    current_cell = best_cell.copy()
    current_score = best_score
    no_improve_count = 0
    max_no_improve = 50
    
    for iteration in range(n_iter):
        if time.time() - t_start > time_limit:
            break
        
        # Pick a random symmetry group to try flipping
        group = unique_groups[iteration % len(unique_groups)]
        
        # Try flipping all pixels in this group
        new_cell = current_cell.copy()
        old_val = new_cell[group[0][0], group[0][1]]
        new_val = 1 - old_val
        for (i, j) in group:
            new_cell[i, j] = new_val
        
        new_score = compute_bands(new_cell.astype(np.float64))
        
        if new_score > current_score:
            current_cell = new_cell
            current_score = new_score
            no_improve_count = 0
            if new_score > best_score:
                best_score = new_score
                best_cell = new_cell.copy()
        else:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                # Reset to best and try different strategy
                current_cell = best_cell.copy()
                current_score = best_score
                no_improve_count = 0
                # Shuffle groups
                rng.shuffle(unique_groups)
    
    return best_cell, best_score


def _hill_climb(cell, size, compute_bands, time_limit, n_iter):
    """
    More aggressive local search: try flipping single pixels (with symmetry).
    Uses a queue of promising moves.
    """
    t_start = time.time()
    best_cell = cell.copy()
    best_score = compute_bands(cell.astype(np.float64))
    
    # Precompute symmetry groups
    sym_groups = {}
    for i in range(size):
        for j in range(size):
            group = set()
            for fr in [False, True]:
                for fc in [False, True]:
                    for tr in [False, True]:
                        ii = i
                        jj = j
                        if tr:
                            ii, jj = jj, ii
                        if fr:
                            ii = size - 1 - ii
                        if fc:
                            jj = size - 1 - jj
                        group.add((ii, jj))
            sym_groups[(i, j)] = group
    
    seen = set()
    unique_groups = []
    for key, group in sym_groups.items():
        frozen = frozenset(group)
        if frozen not in seen:
            seen.add(frozen)
            unique_groups.append(list(group))
    
    rng = np.random.RandomState(1)
    rng.shuffle(unique_groups)
    
    current_cell = best_cell.copy()
    current_score = best_score
    stall_count = 0
    max_stall = 100
    
    for iteration in range(n_iter):
        if time.time() - t_start > time_limit:
            break
        
        # Try a random group
        group = unique_groups[iteration % len(unique_groups)]
        
        new_cell = current_cell.copy()
        old_val = new_cell[group[0][0], group[0][1]]
        new_val = 1 - old_val
        for (i, j) in group:
            new_cell[i, j] = new_val
        
        new_score = compute_bands(new_cell.astype(np.float64))
        
        if new_score > current_score + 1e-10:
            current_cell = new_cell
            current_score = new_score
            stall_count = 0
            if new_score > best_score:
                best_score = new_score
                best_cell = new_cell.copy()
        else:
            stall_count += 1
            if stall_count > max_stall:
                # Perturbation: flip a random group
                pert_group = unique_groups[rng.randint(len(unique_groups))]
                current_cell = current_cell.copy()
                old_val = current_cell[pert_group[0][0], pert_group[0][1]]
                for (i, j) in pert_group:
                    current_cell[i, j] = 1 - old_val
                current_score = compute_bands(current_cell.astype(np.float64))
                stall_count = 0
                max_stall = max(50, max_stall // 2)
    
    return best_cell, best_score


def design(problem):
    """
    Design a symmetric 32x32 photonic crystal unit cell maximizing the TM band gap
    between bands 4 and 5.
    """
    t_start = time.time()
    DEADLINE = 225.0  # Safety margin before 230s limit
    
    size = problem["pixels"]  # 32
    eps_low = problem["eps_low"]  # 1
    eps_high = problem["eps_high"]  # 12
    lower_band = problem.get("lower_band", 4)
    polarization = problem.get("polarization", "TM")
    
    # Set up PW solver
    n_pw = 17  # 17x17 = 289 plane waves
    k_points = None  # Will be generated inside solver
    
    compute_bands = _build_pw_solver(size, eps_low, eps_high, n_pw, k_points)
    
    # Generate initial cells
    initial_cells = _generate_initial_cells(size)
    
    # Evaluate all initial cells
    best_score = -np.inf
    best_cell = None
    for cell in initial_cells:
        if time.time() - t_start > DEADLINE - 10:
            break
        score = compute_bands(cell.astype(np.float64))
        if score > best_score:
            best_score = score
            best_cell = cell.copy()
    
    # Local search from best initial cell
    remaining_time = DEADLINE - (time.time() - t_start) - 5.0
    if remaining_time > 5.0 and best_cell is not None:
        best_cell, best_score = _local_search(
            best_cell, size, compute_bands, time_limit=remaining_time * 0.4, n_iter=5000
        )
    
    # If we have more time, try local search from other good initial cells
    remaining_time = DEADLINE - (time.time() - t_start) - 5.0
    if remaining_time > 10.0:
        for cell in initial_cells:
            if time.time() - t_start > DEADLINE - 5:
                break
            if np.array_equal(cell, best_cell):
                continue
            t_sub = time.time()
            cell2, score2 = _local_search(
                cell, size, compute_bands, time_limit=min(remaining_time / 3, 25.0), n_iter=2000
            )
            if score2 > best_score:
                best_score = score2
                best_cell = cell2
    
    # Final hill climb from best cell
    remaining_time = DEADLINE - (time.time() - t_start) - 3.0
    if remaining_time > 5.0 and best_cell is not None:
        best_cell, best_score = _hill_climb(
            best_cell, size, compute_bands, time_limit=remaining_time, n_iter=10000
        )
    
    # Final check
    if best_cell is None:
        # Fallback: return a simple rod
        best_cell = np.zeros((size, size), dtype=int)
        center = size / 2.0
        radius = size * 0.25
        for i in range(size):
            for j in range(size):
                dx = i + 0.5 - center
                dy = j + 0.5 - center
                if dx*dx + dy*dy <= radius*radius:
                    best_cell[i, j] = 1
        best_cell = _apply_symmetry(best_cell, size)
    
    return best_cell.tolist()

# EVOLVE-BLOCK-END
