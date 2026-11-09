"""New strategy: direct binary pixel search with fast FFT-based band solver and k-flip local search."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _enforce_symmetry(cell, size):
    """Enforce 4-fold rotational and mirror symmetry about center."""
    sym = np.zeros((size, size), dtype=int)
    for i in range(size // 2 + 1):
        for j in range(size // 2 + 1):
            val = cell[i][j]
            sym[i][j] = val
            sym[size - 1 - i][j] = val
            sym[i][size - 1 - j] = val
            sym[size - 1 - i][size - 1 - j] = val
    return sym

def _compute_eps_fourier_fast(eps_inv, size, G_vectors):
    """Compute eps^{-1} Fourier coefficients using FFT for speed."""
    N = len(G_vectors)
    eps_F = np.zeros((N, N), dtype=complex)
    
    # Use 2D FFT to compute all Fourier coefficients at once
    # eps_inv is size x size, we need F[(gx, gy)] = sum_{x,y} eps_inv[x,y] * exp(-2pi i (gx*x + gy*y)/size) / size^2
    # But we need F[(gx-gy)] for differences, so we can use the convolution theorem
    
    # Actually, let's just compute the 2D FFT of eps_inv
    eps_fft = np.fft.fft2(eps_inv) / (size * size)
    
    # Now eps_F[idx_g][idx_h] = sum_{x,y} eps_inv[x,y] * exp(2pi i ((hx-gx)*x + (hy-gy)*y) / size) / size^2
    # This is the Fourier coefficient at (hx-gx, hy-gy)
    # So eps_F[idx_g][idx_h] = eps_fft[(hx-gx) mod size, (hy-gy) mod size]
    
    for idx_g, (gx, gy) in enumerate(G_vectors):
        for idx_h, (hx, hy) in enumerate(G_vectors):
            dgx = (hx - gx) % size
            dgy = (hy - gy) % size
            eps_F[idx_g][idx_h] = eps_fft[dgy, dgx]  # Note: fft2 output is [y, x]
    
    return eps_F

def _score_cell_fast(cell, size, eps_high=12.0, Npw=21, n_kpoints=30):
    """Score a cell using fast plane-wave method with FFT-based Fourier coefficients."""
    half = (Npw - 1) // 2
    eps = np.array(cell, dtype=float)
    eps_inv = np.zeros_like(eps)
    eps_inv[eps > 0.5] = 1.0 / eps_high
    eps_inv[eps <= 0.5] = 1.0
    
    G_vectors = []
    for gx in range(-half, half + 1):
        for gy in range(-half, half + 1):
            G_vectors.append((gx, gy))
    N = len(G_vectors)
    
    # Compute epsilon^{-1} Fourier coefficients using FFT
    eps_F = _compute_eps_fourier_fast(eps_inv, size, G_vectors)
    
    # Define wave vectors in irreducible BZ
    k_points = []
    # Generate points along boundaries and interior of irreducible wedge
    # Square lattice: irreducible wedge is 0 <= ky <= kx, kx in [0, 0.5]
    # Use a grid in the wedge
    n_per_line = max(3, n_kpoints // 3)
    
    # Gamma-X line: (t, 0) for t in [0, 0.5]
    for i in range(n_per_line):
        t = i / max(1, n_per_line - 1) * 0.5
        k_points.append((t, 0.0))
    
    # Gamma-M line: (t, t) for t in [0, 0.5]
    for i in range(n_per_line):
        t = i / max(1, n_per_line - 1) * 0.5
        k_points.append((t, t))
    
    # X-M line: (0.5, t) for t in [0, 0.5]
    for i in range(n_per_line):
        t = i / max(1, n_per_line - 1) * 0.5
        k_points.append((0.5, t))
    
    # Add some interior points
    if len(k_points) < n_kpoints:
        for i in range(n_kpoints - len(k_points)):
            kx = random.uniform(0.1, 0.5)
            ky = random.uniform(0.0, kx)
            k_points.append((kx, ky))
    
    min_omega5 = float('inf')
    max_omega4 = 0.0
    
    for (kx, ky) in k_points:
        # Build TM eigenvalue matrix
        # For TM: A[G,G'] = (G+K)^2 * eps^{-1}_{G-G'}
        A = np.zeros((N, N), dtype=complex)
        for idx_g, (gx, gy) in enumerate(G_vectors):
            kgx = gx + kx
            kgy = gy + ky
            kg2 = kgx * kgx + kgy * kgy
            for idx_h, (hx, hy) in enumerate(G_vectors):
                A[idx_g][idx_h] = kg2 * eps_F[idx_h][idx_g]
        
        try:
            eigvals = np.linalg.eigvalsh(A)
            eigvals = np.sort(eigvals)
            if len(eigvals) >= 5:
                omega4 = np.sqrt(max(eigvals[3], 0))
                omega5 = np.sqrt(max(eigvals[4], 0))
                max_omega4 = max(max_omega4, omega4)
                min_omega5 = min(min_omega5, omega5)
        except:
            pass
    
    if max_omega4 > 0 and min_omega5 > 0:
        gap = 2.0 * (min_omega5 - max_omega4) / (min_omega5 + max_omega4)
        return gap
    return -1.0

def _generate_initial_candidates(size):
    """Generate physically motivated initial candidates."""
    candidates = []
    
    # 1. Central circle
    cell = np.zeros((size, size), dtype=int)
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            if x * x + y * y < 0.15 ** 2:
                cell[i][j] = 1
    cell = _enforce_symmetry(cell, size)
    candidates.append(cell)
    
    # 2. Ring
    cell = np.zeros((size, size), dtype=int)
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            r = np.sqrt(x * x + y * y)
            if 0.12 < r < 0.22:
                cell[i][j] = 1
    cell = _enforce_symmetry(cell, size)
    candidates.append(cell)
    
    # 3. Cross pattern
    cell = np.zeros((size, size), dtype=int)
    center = size // 2
    width = 3
    for i in range(size):
        for j in range(size):
            if abs(i - center) < width or abs(j - center) < width:
                cell[i][j] = 1
    cell = _enforce_symmetry(cell, size)
    candidates.append(cell)
    
    # 4. Four small circles at corners of a square
    cell = np.zeros((size, size), dtype=int)
    offset = 0.2
    radius = 0.08
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            for dx, dy in [(offset, offset), (-offset, offset), (offset, -offset), (-offset, -offset),
                          (offset, -offset), (-offset, -offset), (offset, offset), (-offset, offset)]:
                if (x - dx) ** 2 + (y - dy) ** 2 < radius ** 2:
                    cell[i][j] = 1
    cell = _enforce_symmetry(cell, size)
    candidates.append(cell)
    
    # 5. Checkerboard-like pattern
    cell = np.zeros((size, size), dtype=int)
    for i in range(size):
        for j in range(size):
            if (i // 4) % 2 == (j // 4) % 2:
                cell[i][j] = 1
    cell = _enforce_symmetry(cell, size)
    candidates.append(cell)
    
    # 6. Random
    cell = np.random.randint(0, 2, (size, size))
    cell = _enforce_symmetry(cell, size)
    candidates.append(cell)
    
    return candidates

def design(problem):
    """Optimize photonic crystal using direct binary pixel search with k-flip local search."""
    size = problem["pixels"]
    start_time = time.time()
    deadline = start_time + 220.0  # Leave margin before 230s limit
    
    random.seed(42)
    np.random.seed(42)
    
    best_cell = None
    best_score = -1.0
    
    # Try initial candidates
    candidates = _generate_initial_candidates(size)
    for cell in candidates:
        if time.time() > deadline - 30:
            break
        score = _score_cell_fast(cell, size, Npw=17, n_kpoints=20)
        if score > best_score:
            best_score = score
            best_cell = cell.copy()
    
    # If no good initial candidate, use random
    if best_cell is None:
        best_cell = np.random.randint(0, 2, (size, size))
        best_cell = _enforce_symmetry(best_cell, size)
        best_score = _score_cell_fast(best_cell, size, Npw=17, n_kpoints=20)
    
    # Local search with k-flip moves
    unique_positions = []
    for i in range(size // 2 + 1):
        for j in range(size // 2 + 1):
            unique_positions.append((i, j))
    
    # Adaptive k-flip search
    k_values = [1, 2, 3]
    max_iterations = 5000
    
    for it in range(max_iterations):
        if time.time() > deadline - 10:
            break
        
        # Choose k based on progress
        if it < max_iterations // 3:
            k = 1
        elif it < 2 * max_iterations // 3:
            k = 2
        else:
            k = 3
        
        # Try k-flip move
        flipped = best_cell.copy()
        positions = random.sample(unique_positions, min(k, len(unique_positions)))
        for i, j in positions:
            flipped[i][j] = 1 - flipped[i][j]
        
        flipped = _enforce_symmetry(flipped, size)
        
        # Quick score with smaller basis
        score = _score_cell_fast(flipped, size, Npw=13, n_kpoints=15)
        
        if score > best_score:
            best_score = score
            best_cell = flipped.copy()
    
    # Final refinement with single flips
    if time.time() < deadline - 5:
        for _ in range(200):
            if time.time() > deadline - 2:
                break
            idx = random.randint(0, len(unique_positions) - 1)
            i, j = unique_positions[idx]
            old_val = best_cell[i][j]
            best_cell[i][j] = 1 - old_val
            best_cell = _enforce_symmetry(best_cell, size)
            score = _score_cell_fast(best_cell, size, Npw=17, n_kpoints=20)
            if score > best_score:
                best_score = score
            else:
                best_cell[i][j] = old_val
                best_cell = _enforce_symmetry(best_cell, size)
    
    # Convert to nested list
    result = [[int(best_cell[i][j]) for j in range(size)] for i in range(size)]
    return result
# EVOLVE-BLOCK-END
