"""Design a 32x32 C4v-symmetric photonic crystal unit cell maximizing TM band-4/5 gap.

Strategy: represent the unit cell as a level-set of a low-order Chebyshev polynomial
constrained to C4v symmetry. Optimize coefficients via adaptive stochastic hill-climbing.
"""
import time
import numpy as np
import scipy.special
import sys

# ---------- Deadline guard ----------
_DEADLINE = 180.0  # seconds; keep margin under 200s wall-clock / 230s kill
_start_time = None

def _elapsed():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time

# ---------- Chebyshev basis ----------
# Chebyshev polynomials T_n(x) for n=0..15
_N_CHEB = 16
_cheb_basis = None
_cheb_x = None

def _init_cheb():
    """Precompute Chebyshev basis values on a 32x32 grid (centered at -1..1)."""
    global _cheb_basis, _cheb_x
    if _cheb_basis is not None:
        return
    N = 32
    x = np.linspace(-1.0, 1.0, N)
    _cheb_x = x
    # T_0 = 1, T_1 = x, T_{n+1} = 2x*T_n - T_{n-1}
    basis = np.zeros((N_CHEB, N, N))
    basis[0, :, :] = 1.0
    if N_CHEB > 1:
        basis[1, :, :] = x[:, None] * np.ones((1, N))
    for n in range(2, N_CHEB):
        basis[n, :, :] = 2.0 * x[:, None] * basis[n-1, :, :] - basis[n-2, :, :]
    _cheb_basis = basis

def _evaluate_field(coeffs, N=32):
    """Evaluate the level-set field from Chebyshev coefficients."""
    _init_cheb()
    field = np.zeros((N, N), dtype=np.float64)
    for n in range(len(coeffs)):
        if abs(coeffs[n]) > 1e-12:
            field += coeffs[n] * _cheb_basis[n]
    return field

def _field_to_cell(field, N=32):
    """Convert continuous field to 0/1 cell via threshold at 0, then enforce C4v symmetry."""
    cell = (field > 0.0).astype(np.int8)
    return _enforce_c4v(cell, N)

def _enforce_c4v(cell, N=32):
    """Enforce C4v symmetry: cell[i][j] = cell[N-1-i][j] = cell[i][N-1-j] = cell[N-1-i][N-1-j] = cell[j][i] ..."""
    cell = np.array(cell, dtype=np.int8)
    result = cell.copy()
    # For each orbit under D4, take the value from the first element
    visited = np.zeros((N, N), dtype=bool)
    for i in range(N):
        for j in range(N):
            if visited[i, j]:
                continue
            # Generate the full orbit
            orbit = set()
            def add(ii, jj):
                ii = int(ii) % N
                jj = int(jj) % N
                orbit.add((ii, jj))
            # Identity and 90/180/270 rotations about center (N/2-0.5, N/2-0.5)
            cx = (N - 1) / 2.0
            # For C4 rotation about center: (i,j) -> (N-1-j, i)
            # For reflection: (i,j) -> (j, i)
            ii, jj = i, j
            for _ in range(4):
                add(ii, jj)
                # 90 deg rotation about center: new_i = N-1-jj, new_j = ii
                ii, jj = N - 1 - jj, ii
            # Reflections
            ii, jj = i, j
            for _ in range(4):
                add(ii, jj)
                ii, jj = N - 1 - jj, ii
            # Also the mirror orbit
            ii, jj = j, i
            for _ in range(4):
                add(ii, jj)
                ii, jj = N - 1 - jj, ii
            
            val = 1 if cell[i, j] == 1 else 0
            for (a, b) in orbit:
                result[a, b] = val
                visited[a, b] = True
    return result.tolist()

# ---------- Band structure calculation ----------
# Precompute plane-wave indices
_N_PW = 17  # +/- 17 reciprocal lattice vectors
_pw_indices = None
_pw_grid = None

def _init_pw():
    global _pw_indices, _pw_grid
    if _pw_indices is not None:
        return
    idx = list(range(-_N_PW, _N_PW + 1))
    _pw_grid = idx  # both m and n go from -N_PW to N_PW
    _pw_indices = [(m, n) for m in idx for n in idx]

# Precompute K+G squared magnitudes won't work since K varies;
# but we can precompute the Fourier coefficient lookup table structure.

def _compute_fourier_coeffs(cell, N=32):
    """Compute Fourier coefficients of 1/epsilon for a 32x32 pixel cell.
    
    1/eps = 1/1 + (1/12 - 1) * pix = 1 - (11/12) * pix
    So c_{1/eps}_{mn} = delta_mn * 1 + (1/12 - 1) * c_pix_{mn} = delta_mn - (11/12) c_pix_{mn}
    for (m,n) != (0,0), and c_{1/eps}_{00} = 1 - (11/12)*c_pix_00.
    
    c_pix_{mn} = (1/N^2) sum_{i,j} pix[i,j] exp(-i 2pi (m(i+0.5)/N + n(j+0.5)/N))
    """
    pix = np.array(cell, dtype=np.float64)
    
    # Use 2D FFT for efficiency
    # c_pix_{mn} = (1/N^2) sum_{i,j} pix[i,j] * exp(-i 2pi m i/N) * exp(-i 2pi m/2N) * exp(-i 2pi n j/N) * exp(-i 2pi n/2N)
    # = exp(-i pi m/N) * exp(-i pi n/N) * (1/N^2) sum_{i,j} pix[i,j] exp(-i 2pi mi/N) exp(-i 2pi nj/N)
    # The last term is the 2D DFT of pix.
    
    fft = np.fft.fft2(pix)
    # fft[m,n] = sum_{i,j} pix[i,j] exp(-i 2pi (m i + n j)/N) for m,n in 0..N-1
    # We need c_pix for (m,n) in range [-N_PW, N_PW]
    
    coeffs = np.zeros((2*_N_PW+1, 2*_N_PW+1), dtype=np.complex128)
    pix_area = N * N
    
    for m in range(-_N_PW, _N_PW + 1):
        for n in range(-_N_PW, _N_PW + 1):
            # Map to FFT indices: m_mod = m mod N, n_mod = n mod N
            m_mod = m % N
            n_mod = n % N
            phase = np.exp(-1j * np.pi * (m + n) / N)  # half-pixel shift
            c_pix = phase * fft[m_mod, n_mod] / pix_area
            if m == 0 and n == 0:
                coeffs[m + _N_PW, n + _N_PW] = 1.0 - (11.0/12.0) * c_pix
            else:
                coeffs[m + _N_PW, n + _N_PW] = -(11.0/12.0) * c_pix
    
    return coeffs

def _band_score(cell, n_k=30):
    """Compute the gap-midgap ratio for TM bands 4 and 5.
    
    Uses 289 plane waves (17x17) and 30 k-points in irreducible BZ.
    """
    _init_pw()
    N = len(cell)
    coeffs = _compute_fourier_coeffs(cell, N)
    
    # K-points in irreducible BZ (in units of 2pi/a, so K ranges 0..0.5)
    k_points = [
        (0.0, 0.0), (0.25, 0.0), (0.5, 0.0),
        (0.0, 0.25), (0.0, 0.5),
        (0.25, 0.25), (0.5, 0.5),
        (0.125, 0.0), (0.375, 0.0),
        (0.0, 0.125), (0.0, 0.375),
        (0.125, 0.125), (0.375, 0.375),
        (0.0625, 0.0), (0.4375, 0.0),
        (0.0, 0.0625), (0.0, 0.4375),
        (0.0625, 0.0625), (0.4375, 0.4375),
        (0.1875, 0.0), (0.3125, 0.0),
        (0.0, 0.1875), (0.0, 0.3125),
        (0.1875, 0.1875), (0.3125, 0.3125),
        (0.1, 0.05), (0.35, 0.15),
        (0.05, 0.1), (0.2, 0.1),
        (0.15, 0.2), (0.25, 0.1)
    ]
    k_points = k_points[:n_k]
    
    pw_list = _pw_indices
    M = len(pw_list)  # should be 289
    m_idx = np.array([p[0] for p in pw_list])
    n_idx = np.array([p[1] for p in pw_list])
    
    max_omega4 = 0.0
    min_omega5 = np.inf
    
    for (kx, ky) in k_points:
        Kx = kx * 2.0 * np.pi
        Ky = ky * 2.0 * np.pi
        
        # (K+G) components for each plane wave
        KGx = Kx + m_idx * 2.0 * np.pi
        KGy = Ky + n_idx * 2.0 * np.pi
        
        # Build Hamiltonian: H_{ab} = (K+G_a) . (K+G_b) * c_{1/eps}_{G_a - G_b}
        # For TM: H_{ab} = [(Kx+Gax)(Kx+Gbx) + (Ky+Gay)(Ky+Gby)] * c_{Ga-Gb}
        
        # Vectorized construction
        KGx_outer = np.outer(KGx, KGx)
        KGy_outer = np.outer(KGy, KGy)
        dot_prod = KGx_outer + KGy_outer  # M x M
        
        # c_{1/eps} for each pair (a,b): index by (m_a - m_b, n_a - n_b)
        # diff_m = m_a - m_b, diff_n = n_a - n_b
        diff_m = m_idx[:, None] - m_idx[None, :]  # M x M
        diff_n = n_idx[:, None] - n_idx[None, :]  # M x M
        
        # Convert to index into coeffs array
        cm_idx = diff_m + _N_PW
        cn_idx = diff_n + _N_PW
        
        # Only valid if |diff| <= N_PW
        valid = (np.abs(diff_m) <= _N_PW) & (np.abs(diff_n) <= _N_PW)
        
        C = np.zeros((M, M), dtype=np.float64)
        C[valid] = coeffs[cm_idx[valid], cn_idx[valid]].real
        
        H = dot_prod * C
        
        try:
            eigvals = np.linalg.eigvalsh(H)
        except np.linalg.LinAlgError:
            continue
        
        if len(eigvals) >= 5:
            omega4 = np.sqrt(max(eigvals[3], 0.0))
            omega5 = np.sqrt(max(eigvals[4], 0.0))
            if omega4 > max_omega4:
                max_omega4 = omega4
            if omega5 < min_omega5:
                min_omega5 = omega5
    
    if min_omega5 == np.inf or max_omega4 <= 0:
        return -1.0
    
    gap = 2.0 * (min_omega5 - max_omega4)
    midgap = min_omega5 + max_omega4
    return gap / midgap

# ---------- Search: adaptive stochastic hill-climbing on Chebyshev coefficients ----------
def _random_coeffs(scale=1.0):
    """Generate random Chebyshev coefficients."""
    c = np.random.randn(N_CHEB) * scale
    return c

def _perturb_coeffs(coeffs, step_sizes):
    """Perturb coefficients with per-dimension step sizes."""
    new_c = coeffs.copy()
    for i in range(len(coeffs)):
        new_c[i] += np.random.randn() * step_sizes[i]
    return new_c

def _initial_candidates(N=32):
    """Generate diverse initial candidates via Chebyshev coefficients."""
    _init_cheb()
    candidates = []
    
    # Candidate 1: uniform dielectric (all 1s)
    c = np.zeros(N_CHEB)
    c[0] = 5.0  # positive everywhere -> all dielectric
    candidates.append(c.copy())
    
    # Candidate 2: uniform air (all 0s)
    c = np.zeros(N_CHEB)
    c[0] = -5.0
    candidates.append(c.copy())
    
    # Candidate 3: central dielectric rod (positive in center, negative at edges)
    # T_0 + T_2: T_2 = 2x^2-1, so T_0 - T_2 = 2 - 2x^2 = 2(1-x^2), positive in center
    c = np.zeros(N_CHEB)
    c[0] = 1.0
    c[2] = -0.8
    candidates.append(c.copy())
    
    # Candidate 4: central air hole (negative in center, positive at edges)
    c = np.zeros(N_CHEB)
    c[0] = -1.0
    c[2] = 0.8
    candidates.append(c.copy())
    
    # Candidate 5: ring pattern
    # Use T_4 = 8x^4 - 8x^2 + 1
    c = np.zeros(N_CHEB)
    c[0] = 0.5
    c[2] = -1.2
    c[4] = 0.6
    candidates.append(c.copy())
    
    # Candidate 6: random
    candidates.append(_random_coeffs(0.5))
    
    # Candidate 7: random
    candidates.append(_random_coeffs(0.5))
    
    # Candidate 8: random
    candidates.append(_random_coeffs(0.5))
    
    return candidates

def design(problem):
    global _start_time
    _start_time = time.time()
    
    N = problem["pixels"]  # 32
    
    # Initialize Chebyshev basis
    _init_cheb()
    _init_pw()
    
    # Generate initial candidates
    candidates = _initial_candidates(N)
    
    best_score = -np.inf
    best_coeffs = None
    best_cell = None
    
    # Evaluate initial candidates
    for c in candidates:
        if _elapsed() > _DEADLINE - 60:
            break
        field = _evaluate_field(c, N)
        cell = _field_to_cell(field, N)
        score = _band_score(cell, n_k=30)
        if score > best_score:
            best_score = score
            best_coeffs = c.copy()
            best_cell = cell
    
    if best_cell is None:
        # Fallback: all dielectric
        best_cell = [[1]*N for _ in range(N)]
        best_score = 0.0
        best_coeffs = np.array([5.0] + [0.0]*(N_CHEB-1))
    
    # Adaptive stochastic hill-climbing
    step_sizes = np.ones(N_CHEB) * 0.15
    current_coeffs = best_coeffs.copy()
    current_score = best_score
    
    n_iter = 0
    no_improve = 0
    patience = 80
    
    while _elapsed() < _DEADLINE - 30:
        n_iter += 1
        
        # Adaptive step: shrink if no improvement, expand occasionally
        if no_improve > patience // 2:
            step_sizes *= 0.85
        elif no_improve > patience:
            # Reset: try a random restart with small perturbation of best
            if np.random.rand() < 0.3:
                current_coeffs = best_coeffs + np.random.randn(N_CHEB) * 0.2
                current_score = _band_score(_field_to_cell(_evaluate_field(current_coeffs, N), N), n_k=30)
                no_improve = 0
            else:
                step_sizes *= 0.85
        else:
            # Occasionally expand
            if np.random.rand() < 0.05:
                step_sizes *= 1.2
        
        # Perturb
        new_coeffs = _perturb_coeffs(current_coeffs, step_sizes)
        
        # Evaluate
        field = _evaluate_field(new_coeffs, N)
        cell = _field_to_cell(field, N)
        score = _band_score(cell, n_k=30)
        
        if score > current_score:
            current_coeffs = new_coeffs
            current_score = score
            no_improve = 0
            if score > best_score:
                best_score = score
                best_coeffs = new_coeffs.copy()
                best_cell = cell
        else:
            no_improve += 1
            # With some probability, accept worse solutions (simulated annealing-like)
            if np.random.rand() < 0.1 and score > -0.5:
                current_coeffs = new_coeffs
                current_score = score
    
    # Final: ensure we return the best cell found
    if best_cell is not None:
        return best_cell
    
    # Ultimate fallback
    return [[1 if (i < 16 or j < 16) else 0 for j in range(N)] for i in range(N)]
