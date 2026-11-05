"""Design a 32x32 symmetric photonic crystal unit cell for maximum TM band gap."""
import time
import math
import numpy as np
import random

_DEADLINE = 210.0
_start_time = time.time()


def _elapsed():
    return time.time() - _start_time


def _remaining():
    return _DEADLINE - _elapsed()


def _enforce_c4v(cell):
    """Enforce C4v symmetry (90-degree rotation + reflection about vertical axis)."""
    size = len(cell)
    result = [[0] * size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            # C4v orbit: (i,j), (size-1-j, i), (size-1-i, size-1-j), (j, size-1-i)
            # Plus reflections: (i, size-1-j), (size-1-j, size-1-i), (size-1-i, j), (j, i)
            positions = [
                (i, j),
                (size - 1 - j, i),
                (size - 1 - i, size - 1 - j),
                (j, size - 1 - i),
                (i, size - 1 - j),
                (size - 1 - j, size - 1 - i),
                (size - 1 - i, j),
                (j, i),
            ]
            vals = [cell[pi][pj] for pi, pj in positions]
            result[i][j] = 1 if sum(vals) >= 4 else 0
    return result


def _fast_fourier(cell, N=5):
    """Compute Fourier coefficients of 1/eps using FFT for speed."""
    size = len(cell)
    eps_high = 12.0
    eps_low = 1.0
    inv_eps = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            inv_eps[i][j] = 1.0 / (eps_high if cell[i][j] == 1 else eps_low)
    
    # Use FFT to compute all Fourier coefficients at once
    fft_2d = np.fft.fft2(inv_eps)
    coeffs = {}
    for m in range(-N + 1, N):
        for n in range(-N + 1, N):
            # FFT index: m maps to m mod size, but we need the DC component handling
            # For a periodic function sampled at size points, the FFT gives coefficients
            # c_k = (1/size^2) * sum f(x) exp(-2pi i k x/size)
            # Our desired coefficient: c_m = (1/A) integral f(x) exp(-2pi i m x) dA
            # With x = i/size, this is c_m = (1/size^2) sum f(i) exp(-2pi i m i/size)
            # Which is exactly (1/size^2) * FFT[f]_m (with appropriate indexing)
            idx_m = m % size
            idx_n = n % size
            coeffs[(m, n)] = fft_2d[idx_m, idx_n] / (size * size)
    return coeffs


def _fast_gap(cell, N=5, n_kpts=12):
    """Fast TM gap computation using low-order plane-wave expansion."""
    coeffs = _fast_fourier(cell, N=N)
    
    # k-points in irreducible BZ: 0 <= ky <= kx, kx + ky <= pi
    kpts = []
    # Along kx axis (ky=0)
    for i in range(6):
        kx = (i + 0.5) / 6.0 * np.pi
        kpts.append((kx, 0.0))
    # Along diagonal kx=ky
    for i in range(1, 6):
        kx = (i + 0.5) / 12.0 * np.pi
        kpts.append((kx, kx))
    # Along kx+ky=pi
    for i in range(1, 6):
        kx = (i + 0.5) / 6.0 * np.pi
        ky = np.pi - kx
        if 0 <= ky <= kx:
            kpts.append((kx, ky))
    # A few interior points
    kpts.append((np.pi/3, np.pi/6))
    kpts.append((np.pi/2.5, np.pi/5))
    
    max_omega4 = 0.0
    min_omega5 = float('inf')
    
    for kx, ky in kpts:
        g_vectors = []
        for m in range(-N + 1, N):
            for n in range(-N + 1, N):
                g_vectors.append((2 * np.pi * m, 2 * np.pi * n, m, n))
        
        npw = len(g_vectors)
        A = np.zeros((npw, npw), dtype=complex)
        
        for mi in range(npw):
            gx_m, gy_m, mm, mn = g_vectors[mi]
            kx_m = kx + gx_m
            ky_m = ky + gy_m
            km2 = kx_m * kx_m + ky_m * ky_m
            
            for ni in range(npw):
                gx_n, gy_n, nn, nnn = g_vectors[ni]
                dm = mm - nn
                dn = mn - nnn
                c = coeffs.get((dm, dn), 0.0)
                A[mi, ni] = c * km2
        
        try:
            eigenvalues = np.linalg.eigvalsh(A)
            eigenvalues = np.real(eigenvalues)
            eigenvalues = np.sort(eigenvalues)
            
            if len(eigenvalues) >= 5:
                omega4_sq = eigenvalues[3]
                omega5_sq = eigenvalues[4]
                
                if omega4_sq > 0 and omega5_sq > 0:
                    omega4 = np.sqrt(omega4_sq)
                    omega5 = np.sqrt(omega5_sq)
                    max_omega4 = max(max_omega4, omega4)
                    min_omega5 = min(min_omega5, omega5)
        except:
            continue
    
    if min_omega5 == float('inf') or max_omega4 == 0:
        return 0.0
    
    gap = min_omega5 - max_omega4
    midgap = (min_omega5 + max_omega4) / 2.0
    
    if midgap <= 0:
        return 0.0
    
    return 2.0 * gap / midgap


def _full_gap(cell, N=17, n_kpts=30):
    """Full TM gap computation using 17x17 plane-wave expansion."""
    coeffs = _fast_fourier(cell, N=N)
    
    kpts = []
    for i in range(8):
        kx = (i + 0.5) / 8.0 * np.pi
        kpts.append((kx, 0.0))
    for i in range(1, 8):
        kx = (i + 0.5) / 16.0 * np.pi
        kpts.append((kx, kx))
    for i in range(1, 8):
        kx = (i + 0.5) / 8.0 * np.pi
        ky = np.pi - kx
        if 0 <= ky <= kx:
            kpts.append((kx, ky))
    kpts.append((np.pi / 4, np.pi / 8))
    kpts.append((np.pi / 3, np.pi / 6))
    kpts.append((np.pi / 2.5, np.pi / 5))
    
    max_omega4 = 0.0
    min_omega5 = float('inf')
    
    for kx, ky in kpts:
        g_vectors = []
        for m in range(-N + 1, N):
            for n in range(-N + 1, N):
                g_vectors.append((2 * np.pi * m, 2 * np.pi * n, m, n))
        
        npw = len(g_vectors)
        A = np.zeros((npw, npw), dtype=complex)
        
        for mi in range(npw):
            gx_m, gy_m, mm, mn = g_vectors[mi]
            kx_m = kx + gx_m
            ky_m = ky + gy_m
            km2 = kx_m * kx_m + ky_m * ky_m
            
            for ni in range(npw):
                gx_n, gy_n, nn, nnn = g_vectors[ni]
                dm = mm - nn
                dn = mn - nnn
                c = coeffs.get((dm, dn), 0.0)
                A[mi, ni] = c * km2
        
        try:
            eigenvalues = np.linalg.eigvalsh(A)
            eigenvalues = np.real(eigenvalues)
            eigenvalues = np.sort(eigenvalues)
            
            if len(eigenvalues) >= 5:
                omega4_sq = eigenvalues[3]
                omega5_sq = eigenvalues[4]
                
                if omega4_sq > 0 and omega5_sq > 0:
                    omega4 = np.sqrt(omega4_sq)
                    omega5 = np.sqrt(omega5_sq)
                    max_omega4 = max(max_omega4, omega4)
                    min_omega5 = min(min_omega5, omega5)
        except:
            continue
    
    if min_omega5 == float('inf') or max_omega4 == 0:
        return 0.0
    
    gap = min_omega5 - max_omega4
    midgap = (min_omega5 + max_omega4) / 2.0
    
    if midgap <= 0:
        return 0.0
    
    return 2.0 * gap / midgap


def _random_cell(size=32, fill_prob=0.5):
    """Generate a random cell with C4v symmetry."""
    cell = [[0] * size for _ in range(size)]
    # Only fill the fundamental domain (1/8 of the cell)
    for i in range(size):
        for j in range(size):
            # Check if (i,j) is in the fundamental sector: i <= j and i+j <= size-1
            if i <= j and i + j <= size - 1:
                if random.random() < fill_prob:
                    cell[i][j] = 1
    # Enforce full C4v symmetry
    return _enforce_c4v(cell)


def _mutate_cell(cell, mutation_rate=0.1, size=32):
    """Mutate a cell by flipping random pixels while preserving C4v symmetry."""
    new_cell = [row[:] for row in cell]
    # Only mutate in the fundamental domain
    for i in range(size):
        for j in range(size):
            if i <= j and i + j <= size - 1:
                if random.random() < mutation_rate:
                    new_cell[i][j] = 1 - new_cell[i][j]
    return _enforce_c4v(new_cell)


def _crossover(cell1, cell2, size=32):
    """Crossover two cells by combining their fundamental domains."""
    child = [[0] * size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            if i <= j and i + j <= size - 1:
                if random.random() < 0.5:
                    child[i][j] = cell1[i][j]
                else:
                    child[i][j] = cell2[i][j]
    return _enforce_c4v(child)


def design(problem):
    global _start_time
    _start_time = time.time()
    
    size = problem.get("pixels", 32)
    random.seed(42)
    
    # Initialize population with diverse random cells
    pop_size = 20
    population = []
    for _ in range(pop_size):
        cell = _random_cell(size, fill_prob=random.uniform(0.3, 0.7))
        population.append(cell)
    
    # Evaluate initial population with fast method
    scores = [_fast_gap(cell) for cell in population]
    best_idx = np.argmax(scores)
    best_cell = [row[:] for row in population[best_idx]]
    best_score = scores[best_idx]
    
    # Evolutionary loop
    gen = 0
    max_gens = 500
    while _remaining() > 30 and gen < max_gens:
        gen += 1
        
        # Selection: keep top half
        sorted_indices = np.argsort(scores)[::-1]
        top_half = [population[i] for i in sorted_indices[:pop_size // 2]]
        
        # Generate offspring
        new_population = []
        for _ in range(pop_size // 2):
            # Crossover
            if random.random() < 0.5 and len(top_half) >= 2:
                idx1 = random.randint(0, len(top_half) - 1)
                idx2 = random.randint(0, len(top_half) - 1)
                child = _crossover(top_half[idx1], top_half[idx2], size)
            else:
                idx = random.randint(0, len(top_half) - 1)
                child = [row[:] for row in top_half[idx]]
            
            # Mutation
            mutation_rate = 0.05 + 0.05 * (gen / max_gens)  # Increase mutation over time
            child = _mutate_cell(child, mutation_rate=mutation_rate, size=size)
            new_population.append(child)
        
        # Add some fresh random individuals for diversity
        for _ in range(pop_size // 4):
            new_population.append(_random_cell(size, fill_prob=random.uniform(0.3, 0.7)))
        
        # Evaluate new population
        new_scores = [_fast_gap(cell) for cell in new_population]
        
        # Combine and select
        population = new_population
        scores = new_scores
        
        # Track best
        current_best_idx = np.argmax(scores)
        if scores[current_best_idx] > best_score:
            best_score = scores[current_best_idx]
            best_cell = [row[:] for row in population[current_best_idx]]
    
    # Final refinement with full plane-wave basis for top candidates
    if _remaining() > 10:
        # Get top 5 candidates from final population
        sorted_indices = np.argsort(scores)[::-1][:5]
        for idx in sorted_indices:
            if _remaining() < 5:
                break
            candidate = population[idx]
            full_score = _full_gap(candidate)
            if full_score > best_score:
                best_score = full_score
                best_cell = [row[:] for row in candidate]
    
    return best_cell
