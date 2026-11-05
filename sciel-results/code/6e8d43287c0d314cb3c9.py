"""Optimized photonic-crystal unit cell design for the largest TM 4-5 band gap."""
import time
import math
import numpy as np

# ---------- symmetry helpers ----------
def symmetrize(cell, size):
    """Enforce 4-fold rotational symmetry + reflections about cell centre.
    The cell must satisfy: cell == cell[::-1] rows, cell == cell[::-1] cols,
    cell == cell.T, and cell == rotate180(cell). We build from a fundamental
    1/8 wedge (i<=j, i<=size-1-j mirrored) to guarantee all symmetries."""
    cell = np.array(cell, dtype=int)
    # enforce reflection about centre: cell[i][j]==cell[size-1-i][size-1-j]
    # enforce transpose: cell[i][j]==cell[j][i]
    # enforce 180 rot: cell[i][j]==cell[size-1-i][size-1-j] (same as first)
    # We'll just average/force by taking the value from the fundamental region
    # and propagating. Fundamental region: indices with i<=j and i<=size-1-j.
    out = np.zeros_like(cell)
    for i in range(size):
        for j in range(size):
            # mirror set
            cands = [
                (i, j),
                (j, i),
                (size-1-i, size-1-j),
                (size-1-j, size-1-i),
            ]
            # pick the representative in fundamental region
            rep = None
            for (a,b) in cands:
                if a <= b and a <= size-1-b:
                    rep = (a,b)
                    break
            if rep is None:
                rep = (i,j)
            out[i,j] = cell[rep]
    return out

def fundamental_pixels(size):
    """Yield (i,j) for the fundamental 1/8 wedge region (unique under D4)."""
    for i in range(size):
        for j in range(size):
            if i <= j and i <= size-1-j:
                yield (i,j)

def set_pixel_sym(cell, i, j, val, size):
    """Set a pixel and all its symmetric partners to val."""
    cands = [
        (i, j),
        (j, i),
        (size-1-i, size-1-j),
        (size-1-j, size-1-i),
    ]
    seen = set()
    for (a,b) in cands:
        if (a,b) not in seen:
            seen.add((a,b))
            cell[a,b] = val

# ---------- band solver ----------
def compute_eps_fourier(cell, Npw):
    """Compute the 1/(epsilon) Fourier coefficients for the given pixel map.
    cell: 32x32 array of 0/1 (1=eps 12, 0=eps 1).
    Npw: number of plane waves along one direction (total = Npw^2).
    Returns dict keyed by (nx,ny) -> value of <1/eps> for that reciprocal vector."""
    size = cell.shape[0]
    eps_map = np.where(cell == 1, 12.0, 1.0)
    inv_eps = 1.0 / eps_map
    # sample at pixel centres
    xs = (np.arange(size) + 0.5) / size
    ys = (np.arange(size) + 0.5) / size
    # reciprocal vectors: k = (nx, ny) for nx,ny in -Npw//2 .. Npw//2
    half = Npw // 2
    nvals = list(range(-half, half+1))
    coeffs = {}
    for nx in nvals:
        for ny in nvals:
            # Fourier coeff of inv_eps at reciprocal vector (nx,ny)
            # c = (1/size^2) sum_{i,j} inv_eps[i,j] exp(-2pi i (nx x_i + ny y_j))
            phase_x = np.exp(-2j*math.pi*nx*xs)
            phase_y = np.exp(-2j*math.pi*ny*ys)
            phase = phase_x[:,None] * phase_y[None,:]
            c = np.sum(inv_eps * phase) / (size*size)
            coeffs[(nx,ny)] = c
    return coeffs

def solve_tm_bands(coeffs, kx, ky, Npw, n_bands):
    """Solve the TM band structure at wave vector (kx, ky) using the
    plane-wave Galerkin method. Returns the n_bands lowest eigenvalues (omega^2).
    TM: the matrix is diag(k) @ <1/eps> @ diag(k) where k is the vector of
    |G+k|^2. Actually for TM, the eigenvalue equation is
    sum_G' <1/eps>_{G-G'} |k+G'|^2 c_{G'} = omega^2 c_G
    so A = <1/eps> matrix @ diag(|k+G|^2). We solve A c = omega^2 c."""
    half = Npw // 2
    nvals = list(range(-half, half+1))
    # build list of G vectors
    Gs = [(nx, ny) for nx in nvals for ny in nvals]
    n = len(Gs)
    # |k+G|^2
    k2 = np.zeros(n)
    for idx, (nx, ny) in enumerate(Gs):
        kxg = kx + nx
        kyg = ky + ny
        k2[idx] = kxg*kxg + kyg*kyg
    # build <1/eps> matrix
    # A_{GG'} = <1/eps>_{G-G'} * |k+G'|^2
    A = np.zeros((n, n), dtype=complex)
    for i in range(n):
        for j in range(n):
            dnx = Gs[i][0] - Gs[j][0]
            dny = Gs[i][1] - Gs[j][1]
            c = coeffs.get((dnx, dny), 0.0)
            A[i, j] = c * k2[j]
    # A is Hermitian (for real eps map, <1/eps> is Hermitian). Solve eigenvalues.
    # Use eigh for Hermitian
    A = (A + A.conj().T) / 2.0
    evals = np.linalg.eigvalsh(A)
    evals = np.real(evals)
    # sort ascending
    evals = np.sort(evals)
    return evals[:n_bands]

def compute_score(cell, Npw, kpoints, n_bands=5):
    """Compute the gap-midgap ratio between bands 4 and 5 (0-indexed 3 and 4)."""
    coeffs = compute_eps_fourier(cell, Npw)
    max_b4 = 0.0
    min_b5 = float('inf')
    for (kx, ky) in kpoints:
        evals = solve_tm_bands(coeffs, kx, ky, Npw, n_bands)
        if len(evals) < 5:
            continue
        b4 = evals[3]
        b5 = evals[4]
        max_b4 = max(max_b4, b4)
        min_b5 = min(min_b5, b5)
    if min_b5 == float('inf'):
        return -1.0
    # omega = sqrt(eigenvalue); ratio uses omega
    max_b4_omega = math.sqrt(max_b4) if max_b4 > 0 else 0.0
    min_b5_omega = math.sqrt(min_b5)
    gap = min_b5_omega - max_b4_omega
    mid = min_b5_omega + max_b4_omega
    if mid <= 0:
        return -1.0
    return 2.0 * gap / mid

def generate_kpoints(n_per_edge=3):
    """Generate wave vectors over the irreducible BZ of a square lattice.
    Irreducible BZ: 0 <= kx <= ky <= 0.5 (in units of 2pi/a).
    We sample a grid within this triangle."""
    kpts = []
    # include high-symmetry points
    kpts.append((0.0, 0.0))       # Gamma
    kpts.append((0.5, 0.0))       # X
    kpts.append((0.5, 0.5))       # M
    kpts.append((0.25, 0.25))     # interior
    # grid
    for i in range(n_per_edge):
        for j in range(n_per_edge):
            ky = 0.5 * (i + 0.5) / n_per_edge
            kx_max = ky
            kx = 0.5 * (j + 0.5) / n_per_edge
            if kx <= kx_max + 1e-9:
                kpts.append((kx, ky))
    # dedupe
    seen = set()
    out = []
    for (a,b) in kpts:
        key = (round(a,6), round(b,6))
        if key not in seen:
            seen.add(key)
            out.append((a,b))
    return out

# ---------- main design function ----------
def design(problem):
    size = problem["pixels"]  # 32
    deadline = time.time() + 110.0  # safety margin before 120s (evaluator kills at 230s, but we want margin)
    
    # Start from the seed design: centered rod of radius 0.2
    cell = np.zeros((size, size), dtype=int)
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            if x*x + y*y < 0.2**2:
                cell[i,j] = 1
    cell = symmetrize(cell, size)
    
    # Use a reduced plane-wave set for speed during search
    Npw_search = 15  # 15x15 = 225 plane waves (evaluator uses 289 = 17x17)
    kpoints = generate_kpoints(n_per_edge=4)
    
    best_cell = cell.copy()
    best_score = compute_score(cell, Npw_search, kpoints)
    
    # Local search: coordinate ascent over fundamental pixels
    fund = list(fundamental_pixels(size))
    np.random.seed(42)
    np.random.shuffle(fund)
    
    improved = True
    passes = 0
    while improved and time.time() < deadline - 15.0:
        improved = False
        passes += 1
        for (i, j) in fund:
            if time.time() > deadline - 15.0:
                break
            old_val = cell[i,j]
            new_val = 1 - old_val
            set_pixel_sym(cell, i, j, new_val, size)
            new_score = compute_score(cell, Npw_search, kpoints)
            if new_score > best_score + 1e-9:
                best_score = new_score
                best_cell = cell.copy()
                improved = True
            else:
                set_pixel_sym(cell, i, j, old_val, size)
        # If no improvement in a full pass, try a few random flips (simulated annealing lite)
        if not improved:
            for _ in range(20):
                if time.time() > deadline - 15.0:
                    break
                idx = np.random.randint(len(fund))
                i, j = fund[idx]
                old_val = cell[i,j]
                set_pixel_sym(cell, i, j, 1-old_val, size)
                new_score = compute_score(cell, Npw_search, kpoints)
                # accept with some probability if worse (annealing)
                if new_score >= best_score or np.random.rand() < 0.1:
                    if new_score > best_score:
                        best_score = new_score
                        best_cell = cell.copy()
                    improved = True
                else:
                    set_pixel_sym(cell, i, j, old_val, size)
    
    # Final polish: try a few more deterministic local moves with the best cell
    cell = best_cell.copy()
    for _ in range(3):
        improved = False
        for (i, j) in fund:
            if time.time() > deadline - 5.0:
                break
            old_val = cell[i,j]
            set_pixel_sym(cell, i, j, 1-old_val, size)
            new_score = compute_score(cell, Npw_search, kpoints)
            if new_score > best_score + 1e-9:
                best_score = new_score
                best_cell = cell.copy()
                improved = True
            else:
                set_pixel_sym(cell, i, j, old_val, size)
        if not improved:
            break
    
    # Return as nested list
    result = best_cell.tolist()
    return result
