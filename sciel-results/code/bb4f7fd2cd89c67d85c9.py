"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math

_SIZE = 32
_EPS_HIGH = 12.0
_EPS_LOW = 1.0

# Precompute pixel coordinates
_PIXEL_X = np.array([(i + 0.5) / _SIZE - 0.5 for i in range(_SIZE)])
_XY = np.meshgrid(_PIXEL_X, _PIXEL_X, indexing='ij')
_X_FLAT = _XY[0].flatten()
_Y_FLAT = _XY[1].flatten()

# Plane-wave basis for fast approximate solver (9x9 = 81 plane waves)
_G_RANGE_FAST = 4
_GV_FAST = [(m, n) for m in range(-_G_RANGE_FAST, _G_RANGE_FAST + 1)
            for n in range(-_G_RANGE_FAST, _G_RANGE_FAST + 1)]
_NPW_FAST = len(_GV_FAST)  # 81

# Precompute phase factors for fast solver
_PHASE_FAST = np.zeros((_NPW_FAST, _SIZE * _SIZE), dtype=np.complex128)
for ig, (gm, gn) in enumerate(_GV_FAST):
    phase = 2.0 * np.pi * (gm * _X_FLAT + gn * _Y_FLAT)
    _PHASE_FAST[ig, :] = np.exp(1j * phase)

# G index lookup for fast solver
_G_INDEX_FAST = {}
for idx, (gm, gn) in enumerate(_GV_FAST):
    _G_INDEX_FAST[(gm, gn)] = idx

# K-points in irreducible BZ (fractional coordinates, 1/8 triangle)
# Triangle: 0 <= kx <= ky, kx + ky <= 0.5 (in units of 2*pi/a)
_KPOINTS = []
# High symmetry and grid points
_KPOINTS.append((0.0, 0.0))       # Gamma
_KPOINTS.append((0.25, 0.25))     # X
_KPOINTS.append((0.0, 0.5))       # M
_KPOINTS.append((0.125, 0.125))
_KPOINTS.append((0.0, 0.25))
_KPOINTS.append((0.25, 0.0))
_KPOINTS.append((0.125, 0.25))
_KPOINTS.append((0.25, 0.125))
_KPOINTS.append((0.0, 0.375))
_KPOINTS.append((0.125, 0.375))
_KPOINTS.append((0.25, 0.375))
_KPOINTS.append((0.0, 0.125))
_KPOINTS.append((0.125, 0.0))
_KPOINTS.append((0.0625, 0.0625))
_KPOINTS.append((0.0, 0.0625))
_KPOINTS.append((0.0625, 0.0))
_KPOINTS.append((0.1875, 0.1875))
_KPOINTS.append((0.0, 0.4375))
_KPOINTS.append((0.4375, 0.0))
_KPOINTS.append((0.0625, 0.4375))
_KPOINTS.append((0.4375, 0.0625))
_KPOINTS.append((0.0, 0.3125))
_KPOINTS.append((0.3125, 0.0))
_KPOINTS.append((0.1875, 0.0))
_KPOINTS.append((0.0, 0.1875))
_KPOINTS.append((0.3125, 0.1875))
_KPOINTS.append((0.1875, 0.3125))
_KPOINTS.append((0.0, 0.21875))
_KPOINTS.append((0.21875, 0.0))
_KPOINTS.append((0.21875, 0.21875))

# Deduplicate
_seen_k = set()
_unique_kpts = []
for p in _KPOINTS:
    key = (round(p[0], 6), round(p[1], 6))
    if key not in _seen_k:
        _seen_k.add(key)
        _unique_kpts.append(p)
_KPOINTS = _unique_kpts[:30]


def _compute_fourier_coeffs_fast(cell_flat):
    """Compute Fourier coefficients of 1/eps for fast solver."""
    # 1/eps: 1 for air, 1/12 for dielectric
    # cell_flat: 0=air, 1=dielectric
    inv_eps = np.where(cell_flat > 0.5, 1.0 / _EPS_HIGH, 1.0)
    
    coeffs = np.zeros(_NPW_FAST, dtype=np.complex128)
    for ig in range(_NPW_FAST):
        coeffs[ig] = np.vdot(_PHASE_FAST[ig, :], inv_eps) / (_SIZE * _SIZE)
    return coeffs


def _fast_score(cell):
    """Fast approximate score using 81 plane waves.
    
    Returns gap-midgap ratio (2*(w5-w4)/(w5+w4)) or negative if overlap.
    """
    cell_flat = np.array(cell, dtype=np.float64).flatten()
    coeffs = _compute_fourier_coeffs_fast(cell_flat)
    
    # Build S matrix (1/eps in Fourier space)
    S = np.zeros((_NPW_FAST, _NPW_FAST), dtype=np.complex128)
    for i, (gi, gj) in enumerate(_GV_FAST):
        for j, (gj2, gk) in enumerate(_GV_FAST):
            dm = gi - gj2
            dn = gj - gk
            if -_G_RANGE_FAST <= dm <= _G_RANGE_FAST and -_G_RANGE_FAST <= dn <= _G_RANGE_FAST:
                S[i, j] = coeffs[_G_INDEX_FAST[(dm, dn)]]
            else:
                S[i, j] = 0.0
    
    # Ensure Hermitian
    S = 0.5 * (S + S.conj().T)
    
    band_4_max = -np.inf
    band_5_min = np.inf
    
    for kx_frac, ky_frac in _KPOINTS:
        # Convert to angular frequency units (k = 2*pi * k_frac)
        kx = 2.0 * np.pi * kx_frac
        ky = 2.0 * np.pi * ky_frac
        
        # Build D diagonal: |k+G|^2
        D_diag = np.zeros(_NPW_FAST, dtype=np.float64)
        for i, (gm, gn) in enumerate(_GV_FAST):
            kx_tot = kx + 2.0 * np.pi * gm
            ky_tot = ky + 2.0 * np.pi * gn
            D_diag[i] = kx_tot * kx_tot + ky_tot * ky_tot
        
        # Solve D E = omega^2 S E
        # Convert: S^{-1} D E = omega^2 E
        try:
            S_reg = S + 1e-10 * np.eye(_NPW_FAST)
            # Use eig for generalized problem
            # S E = (1/omega^2) D E
            # Or: D E = omega^2 S E
            # Standard approach: solve S^{-1} D E = omega^2 E
            A = np.linalg.solve(S_reg, np.diag(D_diag))
            A = 0.5 * (A + A.conj().T)
            eigvals = np.linalg.eigvalsh(A)
            eigvals = eigvals[eigvals > 0]
            if len(eigvals) >= 5:
                w4_sq = eigvals[3]
                w5_sq = eigvals[4]
                if w4_sq > 0 and w5_sq > 0:
                    w4 = math.sqrt(w4_sq)
                    w5 = math.sqrt(w5_sq)
                    band_4_max = max(band_4_max, w4)
                    band_5_min = min(band_5_min, w5)
        except Exception:
            continue
    
    if band_4_max <= 0 or band_5_min <= 0:
        return -1.0
    
    gap = band_5_min - band_4_max
    midgap = (band_5_min + band_4_max) / 2.0
    if midgap <= 0:
        return -1.0
    return 2.0 * gap / (band_5_min + band_4_max)


def _enforce_symmetry(cell):
    """Enforce 4-fold symmetry about cell center.
    
    Symmetries: 180-degree rotation, horizontal flip, vertical flip, transpose.
    For a 32x32 cell, the center is between pixels 15 and 16.
    cell[i][j] == cell[31-i][j] (vertical flip)
    cell[i][j] == cell[i][31-j] (horizontal flip)
    cell[i][j] == cell[j][i] (transpose)
    """
    cell = np.array(cell, dtype=np.int8).copy()
    size = _SIZE
    
    # The symmetry group generated by vertical flip, horizontal flip, and transpose
    # For 32x32, pixel (i,j) maps to:
    #   (31-i, j) - vertical flip
    #   (i, 31-j) - horizontal flip  
    #   (j, i) - transpose
    #   (31-j, 31-i) - 180 rotation
    #   (31-i, 31-j) - anti-transpose
    #   (j, 31-i) - ...
    #   (31-j, i) - ...
    
    # Simplest: iterate until fixed point using majority vote or max
    # Use the orbit approach: for each pixel, find all symmetric equivalents
    # and set them all to the same value
    
    # Group pixels into orbits under the symmetry group
    # The group has 8 elements: {id, vf, hf, t, 180, at, vf*t, hf*t}
    # For pixel (i,j), orbit is:
    #   (i,j), (31-i,j), (i,31-j), (31-i,31-j),
    #   (j,i), (j,31-i), (31-j,i), (31-j,31-i)
    
    visited = np.zeros((size, size), dtype=bool)
    for i in range(size):
        for j in range(size):
            if visited[i, j]:
                continue
            # Find orbit
            orbit = set()
            orbit.add((i, j))
            orbit.add((size-1-i, j))
            orbit.add((i, size-1-j))
            orbit.add((size-1-i, size-1-j))
            orbit.add((j, i))
            orbit.add((j, size-1-i))
            orbit.add((size-1-j, i))
            orbit.add((size-1-j, size-1-i))
            
            # Set all to majority value
            vals = [cell[p, q] for p, q in orbit]
            majority = 1 if sum(vals) >= len(vals) // 2 else 0
            for p, q in orbit:
                cell[p, q] = majority
                visited[p, q] = True
    
    return cell


def _make_cell_from_params(params, size=_SIZE):
    """Generate a symmetric cell from continuous parameters.
    
    params: list of (cx, cy, r, sign) for circles in UR quadrant.
    """
    cell = np.zeros((size, size), dtype=np.float64)
    
    for (cx, cy, r, sign) in params:
        dist2 = (_X_FLAT - cx)**2 + (_Y_FLAT - cy)**2
        mask = (dist2 < r**2).astype(np.float64).reshape(size, size)
        if sign > 0:
            cell += mask
        else:
            cell -= mask
    
    cell = (cell > 0).astype(np.float64)
    
    # Enforce exact symmetry
    cell = _enforce_symmetry(cell.astype(np.int8)).astype(np.float64)
    
    return cell


def _ga_score_population(pop, deadline):
    """Score a population of cells using the fast solver."""
    scores = []
    for cell in pop:
        if time.time() > deadline:
            scores.append(-1.0)
            continue
        try:
            s = _fast_score(cell)
            scores.append(s)
        except Exception:
            scores.append(-1.0)
    return scores


def _ga_crossover(parent1, parent2):
    """Uniform crossover between two 32x32 cells."""
    mask = np.random.rand(_SIZE, _SIZE) < 0.5
    child = np.where(mask, parent1, parent2)
    child = _enforce_symmetry(child.astype(np.int8))
    return child


def _ga_mutation(cell, rate=0.05):
    """Bit-flip mutation with symmetry enforcement."""
    child = cell.copy()
    mask = np.random.rand(_SIZE, _SIZE) < rate
    child[mask] = 1 - child[mask]
    child = _enforce_symmetry(child.astype(np.int8))
    return child


def design(problem):
    """Search for optimal photonic crystal unit cell using GA with fast PW solver."""
    size = problem["pixels"]
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    np.random.seed(42)
    
    # Initial population
    pop_size = 60
    population = []
    
    # Seed with known good designs
    # 1. Simple rod
    cell_rod = np.zeros((size, size), dtype=np.int8)
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            if x*x + y*y < 0.2**2:
                cell_rod[i, j] = 1
    cell_rod = _enforce_symmetry(cell_rod)
    population.append(cell_rod)
    
    # 2. Rod with hole
    cell_rod_hole = cell_rod.copy()
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            if x*x + y*y < 0.08**2:
                cell_rod_hole[i, j] = 0
    cell_rod_hole = _enforce_symmetry(cell_rod_hole)
    population.append(cell_rod_hole)
    
    # 3. Larger rod
    cell_big_rod = np.zeros((size, size), dtype=np.int8)
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            if x*x + y*y < 0.28**2:
                cell_big_rod[i, j] = 1
    cell_big_rod = _enforce_symmetry(cell_big_rod)
    population.append(cell_big_rod)
    
    # 4. Square rod
    cell_sq = np.zeros((size, size), dtype=np.int8)
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            if abs(x) < 0.15 and abs(y) < 0.15:
                cell_sq[i, j] = 1
    cell_sq = _enforce_symmetry(cell_sq)
    population.append(cell_sq)
    
    # 5. Diamond shape
    cell_diamond = np.zeros((size, size), dtype=np.int8)
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            if abs(x) + abs(y) < 0.25:
                cell_diamond[i, j] = 1
    cell_diamond = _enforce_symmetry(cell_diamond)
    population.append(cell_diamond)
    
    # 6. Cross shape
    cell_cross = np.zeros((size, size), dtype=np.int8)
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            if abs(x) < 0.12 or abs(y) < 0.12:
                cell_cross[i, j] = 1
    cell_cross = _enforce_symmetry(cell_cross)
    population.append(cell_cross)
    
    # 7. Rod with square hole
    cell_rod_sqhole = cell_rod.copy()
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            if abs(x) < 0.06 and abs(y) < 0.06:
                cell_rod_sqhole[i, j] = 0
    cell_rod_sqhole = _enforce_symmetry(cell_rod_sqhole)
    population.append(cell_rod_sqhole)
    
    # 8. Annulus
    cell_annulus = np.zeros((size, size), dtype=np.int8)
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            r2 = x*x + y*y
            if 0.1**2 < r2 < 0.25**2:
                cell_annulus[i, j] = 1
    cell_annulus = _enforce_symmetry(cell_annulus)
    population.append(cell_annulus)
    
    # Fill rest with random symmetric cells
    while len(population) < pop_size:
        cell = np.random.randint(0, 2, (size, size), dtype=np.int8)
        cell = _enforce_symmetry(cell)
        population.append(cell)
    
    # Score initial population
    scores = _ga_score_population(population, deadline)
    
    best_score = max(scores)
    best_cell = population[scores.index(best_score)].copy()
    
    # GA loop
    elite_frac = 0.1
    n_elite = max(2, int(pop_size * elite_frac))
    mutation_rate = 0.08
    crossover_rate = 0.7
    generations = 0
    
    while time.time() < deadline - 5.0:
        generations += 1
        
        # Sort by score
        idx_sorted = np.argsort(scores)[::-1]
        sorted_pop = [population[i] for i in idx_sorted]
        sorted_scores = [scores[i] for i in idx_sorted]
        
        # Elitism
        new_population = [sorted_pop[i].copy() for i in range(n_elite)]
        
        # Fill rest with offspring
        while len(new_population) < pop_size:
            if time.time() > deadline - 5.0:
                break
            
            # Select parents (tournament selection)
            p1_idx = np.random.randint(0, pop_size)
            p2_idx = np.random.randint(0, pop_size)
            # Tournament of 3
            t1 = np.random.randint(0, pop_size, 3)
            t2 = np.random.randint(0, pop_size, 3)
            p1_idx = t1[np.argmax([scores[i] for i in t1])]
            p2_idx = t2[np.argmax([scores[i] for i in t2])]
            
            parent1 = population[p1_idx]
            parent2 = population[p2_idx]
            
            if np.random.rand() < crossover_rate:
                child = _ga_crossover(parent1, parent2)
            else:
                child = parent1.copy()
            
            child = _ga_mutation(child, mutation_rate)
            new_population.append(child)
        
        # Score new population
        scores = _ga_score_population(new_population, deadline)
        population = new_population
        
        # Update best
        if len(scores) > 0:
            gen_best = max(scores)
            gen_best_idx = scores.index(gen_best)
            if gen_best > best_score:
                best_score = gen_best
                best_cell = population[gen_best_idx].copy()
        
        # Adaptive mutation rate
        if generations % 10 == 0:
            mutation_rate = max(0.02, mutation_rate * 0.9)
        
        # Occasionally add random immigrants
        if generations % 20 == 0 and time.time() < deadline - 10.0:
            for _ in range(3):
                cell = np.random.randint(0, 2, (size, size), dtype=np.int8)
                cell = _enforce_symmetry(cell)
                # Replace worst member
                worst_idx = np.argmin(scores)
                population[worst_idx] = cell
                scores[worst_idx] = _fast_score(cell)
    
    # Final: refine best cell with local search
    if time.time() < deadline - 3.0:
        # Try flipping individual pixels in the best cell
        current_best = best_cell.copy()
        current_score = best_score
        for _ in range(50):
            if time.time() > deadline - 3.0:
                break
            # Pick a random pixel
            i = np.random.randint(0, size)
            j = np.random.randint(0, size)
            trial = current_best.copy()
            trial[i, j] = 1 - trial[i, j]
            trial = _enforce_symmetry(trial)
            if not np.array_equal(trial, current_best):
                try:
                    trial_score = _fast_score(trial)
                    if trial_score > current_score:
                        current_best = trial
                        current_score = trial_score
                        if current_score > best_score:
                            best_score = current_score
                            best_cell = current_best.copy()
                except Exception:
                    pass
    
    # Convert to nested list
    result = []
    for i in range(size):
        row = []
        for j in range(size):
            row.append(int(best_cell[i, j]))
        result.append(row)
    
    return result
# EVOLVE-BLOCK-END
