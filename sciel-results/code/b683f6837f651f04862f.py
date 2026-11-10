"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np

_START_TIME = time.time()
_TIME_BUDGET = 150.0  # safety margin before 230s kill
_EPS_LOW = 1.0
_EPS_HIGH = 12.0


def _enforce_symmetry(cell):
    """Enforce 4-fold symmetry: 180-deg rotation + transpose. Returns 32x32 list."""
    n = len(cell)
    c = np.array(cell, dtype=float)
    # 180-degree rotation symmetry
    c = 0.5 * (c + c[::-1, ::-1])
    # transpose symmetry (diagonal mirror)
    c = 0.5 * (c + c.T)
    # re-enforce 180 after transpose
    c = 0.5 * (c + c[::-1, ::-1])
    c = 0.5 * (c + c.T)
    return (c > 0.5).astype(int)


def _fourier_eps(cell, N=289):
    """Compute Fourier coefficients of 1/eps for plane-wave method.
    cell: 2D array 0/1. Returns dict of (mx,my) -> coeff for |mx|,|my| <= M."""
    n = len(cell)
    eps = np.where(np.array(cell) == 1, _EPS_HIGH, _EPS_LOW)
    inv_eps = 1.0 / eps
    # 2D FFT to get Fourier coefficients
    F = np.fft.fft2(inv_eps)
    F = F / (n * n)
    # shift so zero freq at center
    F = np.fft.fftshift(F)
    M = (N - 1) // 2  # max |m|
    coeffs = {}
    for i in range(n):
        for j in range(n):
            mx = i - n // 2
            my = j - n // 2
            if abs(mx) <= M and abs(my) <= M:
                coeffs[(mx, my)] = F[i, j]
    return coeffs


def _build_pw_matrices(kx, ky, coeffs, M):
    """Build H and S matrices for plane-wave method at wavevector (kx,ky).
    Using normalized basis with S=I."""
    # Plane waves indexed by (mx,my) in [-M,M]
    pw = [(mx, my) for mx in range(-M, M + 1) for my in range(-M, M + 1)]
    N = len(pw)
    idx = {p: i for i, p in enumerate(pw)}
    H = np.zeros((N, N), dtype=complex)
    for i, (mx, my) in enumerate(pw):
        Gx = 2 * math.pi * mx
        Gy = 2 * math.pi * my
        kx2 = (kx - Gx) ** 2 + (ky - Gy) ** 2
        for j, (mx2, my2) in enumerate(pw):
            # coupling: k^2 term for i==j, else sum over G of inv_eps
            if i == j:
                H[i, j] = kx2
            else:
                Gdiffx = mx2 - mx
                Gdiffy = my2 - my
                c = coeffs.get((Gdiffx, Gdiffy), 0.0)
                H[i, j] = c * kx2
    return H


def _band_gaps(cell, kpoints, M=17):
    """Compute min omega_5 and max omega_4 over kpoints. Returns (max_w4, min_w5)."""
    coeffs = _fourier_eps(cell, N=2 * M + 1)
    max_w4 = 0.0
    min_w5 = float('inf')
    pw = [(mx, my) for mx in range(-M, M + 1) for my in range(-M, M + 1)]
    N = len(pw)
    for (kx, ky) in kpoints:
        H = _build_pw_matrices(kx, ky, coeffs, M)
        # Hermitian eigenvalue problem: H c = omega^2 c
        try:
            vals = np.linalg.eigvalsh(H)
        except np.linalg.LinAlgError:
            continue
        # TM: omega^2 = (k-G)^2 * ... need correct formulation.
        # For TM: omega^2 = sum over G' (k-G)^2 eps_G (k-G')
        # Let me use the standard form: eigenvalues of A where A_ij = (k-Gi)^2 * eps_(Gj-Gi)
        # This gives omega^2 directly.
        # Rebuild with correct TM formulation
        H2 = np.zeros((N, N), dtype=complex)
        for i, (mx, my) in enumerate(pw):
            Gx = 2 * math.pi * mx
            Gy = 2 * math.pi * my
            kx2 = (kx - Gx) ** 2 + (ky - Gy) ** 2
            for j, (mx2, my2) in enumerate(pw):
                Gdx = mx - mx2
                Gdy = my - my2
                c = coeffs.get((Gdx, Gdy), 0.0)
                H2[i, j] = kx2 * c
        try:
            vals = np.linalg.eigvalsh(H2)
        except np.linalg.LinAlgError:
            continue
        vals = np.sort(np.real(vals))
        # bands 4 and 5 (1-indexed) => indices 3 and 4
        if len(vals) >= 5:
            w4 = math.sqrt(max(0.0, vals[3]))
            w5 = math.sqrt(max(0.0, vals[4]))
            max_w4 = max(max_w4, w4)
            min_w5 = min(min_w5, w5)
    return max_w4, min_w5


def _irreducible_kpoints(nk=30):
    """Generate wavevectors in irreducible BZ of square lattice [0,pi]x[0,pi]
    with symmetry x>=y (first sextant)."""
    kpts = []
    # Boundary: from (0,0) to (pi,0) [Gamma-X], (pi,0) to (pi,pi) [X-M], (pi,pi) to (0,pi) [M-Gamma]
    # Irreducible: triangle Gamma(0,0)-X(pi,0)-M(pi,pi) with x>=y region... 
    # Standard square BZ irreducible wedge: 0<=y<=x<=pi, plus the line x=y.
    # Let's sample a grid in the triangle and boundary.
    n = 6
    for i in range(n + 1):
        x = math.pi * i / n
        for j in range(i + 1):
            y = math.pi * j / n
            kpts.append((x, y))
    # Add some interior points
    for i in range(3):
        for j in range(3):
            x = math.pi * (2 * i + 1) / 6
            y = math.pi * (2 * j + 1) / 6
            if y <= x:
                kpts.append((x, y))
    return kpts


def _make_cell(radius, n=32, shape='disk'):
    """Create a symmetric cell with a dielectric feature of given radius."""
    cell = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            x = (i + 0.5) / n - 0.5
            y = (j + 0.5) / n - 0.5
            if shape == 'disk':
                val = 1 if (x * x + y * y) < radius * radius else 0
            elif shape == 'square':
                val = 1 if (abs(x) < radius and abs(y) < radius) else 0
            elif shape == 'air_disk':
                # dielectric background with air hole
                val = 0 if (x * x + y * y) < radius * radius else 1
            else:
                val = 1 if (x * x + y * y) < radius * radius else 0
            cell[i, j] = val
    return _enforce_symmetry(cell)


def design(problem):
    size = problem.get("pixels", 32)
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)
    global _EPS_LOW, _EPS_HIGH
    _EPS_LOW = eps_low
    _EPS_HIGH = eps_high

    kpoints = _irreducible_kpoints(nk=30)
    M = 17  # plane-wave cutoff (289 planes)

    best_cell = None
    best_score = -1.0

    # Coarse search over radius and shape
    shapes = ['disk', 'square', 'air_disk']
    radii = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]

    for shape in shapes:
        for r in radii:
            if time.time() - _START_TIME > _TIME_BUDGET:
                break
            cell = _make_cell(r, size, shape)
            max_w4, min_w5 = _band_gaps(cell, kpoints, M)
            if min_w5 > max_w4:
                score = 2 * (min_w5 - max_w4) / (min_w5 + max_w4)
            else:
                score = -2 * (max_w4 - min_w5) / (min_w5 + max_w4) if (min_w5 + max_w4) > 0 else -1.0
            if score > best_score:
                best_score = score
                best_cell = cell
        if time.time() - _START_TIME > _TIME_BUDGET:
            break

    # Refine around best
    if best_cell is not None:
        # Determine best radius/shape by re-searching finer grid
        best_r = 0.2
        best_shape = 'disk'
        # Re-identify which (shape, r) gave best
        for shape in shapes:
            for r in radii:
                cell = _make_cell(r, size, shape)
                if np.array_equal(cell, best_cell):
                    best_r = r
                    best_shape = shape
                    break
            if np.array_equal(_make_cell(best_r, size, best_shape), best_cell):
                break

        # Fine search
        if time.time() - _START_TIME < _TIME_BUDGET:
            fine_r = np.linspace(max(0.05, best_r - 0.1), min(0.5, best_r + 0.1), 11)
            for r in fine_r:
                if time.time() - _START_TIME > _TIME_BUDGET:
                    break
                cell = _make_cell(r, size, best_shape)
                max_w4, min_w5 = _band_gaps(cell, kpoints, M)
                if min_w5 > max_w4:
                    score = 2 * (min_w5 - max_w4) / (min_w5 + max_w4)
                else:
                    score = -2 * (max_w4 - min_w5) / (min_w5 + max_w4) if (min_w5 + max_w4) > 0 else -1.0
                if score > best_score:
                    best_score = score
                    best_cell = cell

    # Fallback to seed if nothing found
    if best_cell is None:
        best_cell = _make_cell(0.2, size, 'disk')

    # Ensure exact symmetry and return as nested list
    best_cell = _enforce_symmetry(best_cell)
    return best_cell.tolist()
# EVOLVE-BLOCK-END

