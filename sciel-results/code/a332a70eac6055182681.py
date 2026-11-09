"""Photonic-crystal unit-cell design: maximize TM band-4/5 gap-midgap ratio."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _fourier_coeffs(eps_grid, gmax):
    """Compute exact Fourier coefficients of 1/eps for a pixelated 32x32 cell.
    
    Returns array c[gx, gy] for gx,gy in [-gmax, gmax].
    """
    N = eps_grid.shape[0]
    inv_eps = 1.0 / eps_grid.astype(float)
    
    # Precompute the pixel-integral factor F[i, g]
    # F[i, g] = integral_{i/N}^{(i+1)/N} exp(-i 2pi g x/N) dx
    F = np.zeros((N, 2*gmax+1), dtype=complex)
    for ig, g in enumerate(range(-gmax, gmax+1)):
        if g == 0:
            F[:, ig] = 1.0/N
        else:
            for i in range(N):
                F[i, ig] = np.exp(-1j * np.pi * g * (2*i+1)/N) * N/(2*np.pi*g) * np.sin(np.pi*g/N)
    
    # c[gx, gy] = (1/N^2) sum_{i,j} inv_eps[i,j] * F[i,gx] * F[j,gy]
    # = (1/N^2) * F[:,gx] @ inv_eps @ F[:,gy]  ... no, it's element-wise
    # Actually: c[gx,gy] = (1/N^2) * sum_i sum_j inv_eps[i,j] F[i,gx] F[j,gy]
    # = (1/N^2) * (F[:,gx] * inv_eps.sum(axis=1)[:,None]) ... no
    # Let me just do it as: for each (gx,gy), c = (1/N^2) * np.sum(inv_eps * np.outer(F[:,gx_idx], F[:,gy_idx]))
    # This is O((2gmax+1)^2 * N^2) which for gmax=8 is 289*1024 = 296k ops, fine.
    
    c = np.zeros((2*gmax+1, 2*gmax+1), dtype=complex)
    for igx in range(2*gmax+1):
        for igy in range(2*gmax+1):
            c[igx, igy] = (1.0/N**2) * np.sum(inv_eps * np.outer(F[:, igx], F[:, igy]))
    return c

def _solve_tm_bands(c, gmax, kx, ky, n_bands=6):
    """Solve TM eigenvalue problem at a given k-point.
    
    c: Fourier coefficients of 1/eps, indexed by [gx+gmax, gy+gmax]
    Returns sorted eigenvalues (omega^2) for the first n_bands.
    """
    # Number of plane waves in each direction
    n = 2*gmax + 1
    n2 = n * n
    
    # G vectors: (2pi*gx, 2pi*gy)
    g_vals = np.arange(-gmax, gmax+1)
    
    # Build the matrix M[i,j] = (k+Gi)·(k+Gj) * c[Gi-Gj]
    M = np.zeros((n2, n2), dtype=complex)
    
    # Precompute k+G for each plane wave
    kg = np.zeros((n2, 2))
    idx = 0
    for a in range(n):
        for b in range(n):
            kg[idx, 0] = kx + 2*np.pi*g_vals[a]
            kg[idx, 1] = ky + 2*np.pi*g_vals[b]
            idx += 1
    
    for i in range(n2):
        for j in range(i, n2):
            # Gi - Gj = (2pi*(gx_i - gx_j), 2pi*(gy_i - gy_j))
            # Need to check if (gx_i - gx_j, gy_i - gy_j) is within [-gmax, gmax]
            # i corresponds to (a_i, b_i), j to (a_j, b_j)
            a_i, b_i = divmod(i, n)
            a_j, b_j = divmod(j, n)
            dgx = g_vals[a_i] - g_vals[a_j]
            dgy = g_vals[b_i] - g_vals[b_j]
            
            if -gmax <= dgx <= gmax and -gmax <= dgy <= gmax:
                cval = c[dgx + gmax, dgy + gmax]
            else:
                cval = 0.0
            
            dot = np.dot(kg[i], kg[j])
            val = dot * cval
            M[i, j] = val
            M[j, i] = np.conj(val)
    
    # Solve eigenvalue problem
    try:
        eigvals = np.linalg.eigvalsh(M)
        eigvals = np.sort(eigvals.real)
        return eigvals[:n_bands]
    except:
        return None

def _score_cell(cell, gmax=8, n_kpts=30):
    """Compute the gap-midgap ratio for TM bands 4-5.
    
    cell: 32x32 array of 0/1
    Returns the gap-midgap ratio (higher is better).
    """
    N = 32
    eps_grid = np.array(cell, dtype=float)
    eps_grid[eps_grid == 1] = 12.0
    eps_grid[eps_grid == 0] = 1.0
    
    # Compute Fourier coefficients
    c = _fourier_coeffs(eps_grid, gmax)
    
    # K-points in the irreducible BZ
    # Path: (0,0) -> (pi,0) -> (pi,pi)
    kpts = []
    n1 = 15
    n2 = 15
    for t in np.linspace(0, 1, n1):
        kpts.append((np.pi*t, 0.0))
    for t in np.linspace(0, 1, n2):
        kpts.append((np.pi, np.pi*t))
    
    max_omega4 = 0.0
    min_omega5 = float('inf')
    
    for kx, ky in kpts:
        eigvals = _solve_tm_bands(c, gmax, kx, ky, n_bands=6)
        if eigvals is None or len(eigvals) < 5:
            continue
        omega4 = np.sqrt(max(eigvals[3], 0))
        omega5 = np.sqrt(max(eigvals[4], 0))
        max_omega4 = max(max_omega4, omega4)
        min_omega5 = min(min_omega5, omega5)
    
    if min_omega5 == float('inf') or max_omega4 <= 0:
        return -1.0
    
    gap_midgap = 2*(min_omega5 - max_omega4) / (min_omega5 + max_omega4)
    return gap_midgap

def _make_cell(params):
    """Create a D4-symmetric 32x32 cell from continuous parameters.
    
    params: array of 7 parameters
    [0] center_radius: radius of central dielectric blob
    [1] sat_radius: radius of 4 satellite blobs
    [2] sat_offset: offset of satellite blobs from center
    [3] ring_radius: radius of circular ring
    [4] ring_width: width of circular ring
    [5] axis_width: width of cross arms along axes
    [6] axis_length: length of cross arms along axes
    """
    N = 32
    cell = np.zeros((N, N), dtype=int)
    
    cr = params[0]
    sr = params[1]
    so = params[2]
    rr = params[3]
    rw = params[4]
    aw = params[5]
    al = params[6]
    
    for i in range(N):
        for j in range(N):
            x = (i + 0.5) / N - 0.5
            y = (j + 0.5) / N - 0.5
            r = np.sqrt(x*x + y*y)
            
            val = 0
            # Central blob
            if r < cr:
                val = 1
            # Satellite blobs at (±so, ±so)
            if sr > 0:
                for ox, oy in [(so, so), (so, -so), (-so, so), (-so, -so)]:
                    if (x-ox)**2 + (y-oy)**2 < sr*sr:
                        val = 1
                        break
            # Circular ring
            if rr > 0 and rw > 0:
                if abs(r - rr) < rw/2:
                    val = 1
            # Cross arms
            if aw > 0 and al > 0:
                if (abs(x) < aw/2 and abs(y) < al/2) or (abs(y) < aw/2 and abs(x) < al/2):
                    val = 1
            
            cell[i, j] = val
    
    return cell

def _de_optimize(score_fn, init_params, lo, hi, pop_size=12, max_gen=10, time_budget=60.0):
    """Differential evolution optimizer."""
    rng = np.random.RandomState(123)
    n_params = len(lo)
    
    # Initialize population
    pop = [init_params.copy()]
    for _ in range(pop_size - 1):
        p = lo + rng.random(n_params) * (hi - lo)
        pop.append(p)
    
    scores = [score_fn(p) for p in pop]
    best_idx = np.argmax(scores)
    best_p = pop[best_idx].copy()
    best_s = scores[best_idx]
    
    F = 0.8  # scaling factor
    CR = 0.9  # crossover rate
    
    start_time = time.time()
    for gen in range(max_gen):
        if time.time() - start_time > time_budget:
            break
        
        for i in range(pop_size):
            if time.time() - start_time > time_budget:
                break
            
            # Select 3 distinct individuals different from i
            a, b, c = [x for x in rng.choice(pop_size, 3, replace=False) if x != i]
            if len([a, b, c]) < 3:
                continue
            
            # Mutation
            v = pop[a] + F * (pop[b] - pop[c])
            v = np.clip(v, lo, hi)
            
            # Crossover
            trial = pop[i].copy()
            j_rand = rng.randint(n_params)
            for j in range(n_params):
                if rng.random() < CR or j == j_rand:
                    trial[j] = v[j]
            
            # Selection
            s_trial = score_fn(trial)
            if s_trial >= scores[i]:
                pop[i] = trial
                scores[i] = s_trial
                if s_trial > best_s:
                    best_s = s_trial
                    best_p = trial.copy()
    
    return best_p, best_s

def design(problem):
    """Design a 32x32 symmetric unit cell to maximize TM band-4/5 gap-midgap ratio."""
    start_time = time.time()
    deadline = start_time + 210.0  # safety margin before 230s kill
    
    size = problem.get("pixels", 32)
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)
    
    # Parameter bounds
    lo = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    hi = np.array([0.35, 0.20, 0.35, 0.35, 0.15, 0.30, 0.45])
    
    # Initial guess: a reasonable starting point
    init = np.array([0.15, 0.08, 0.22, 0.0, 0.0, 0.10, 0.30])
    
    # Quick evaluation with fewer k-points for speed during optimization
    def quick_score(params):
        cell = _make_cell(params)
        return _score_cell(cell, gmax=7, n_kpts=20)
    
    # Phase 1: DE optimization with a moderate budget
    best_p, best_s = _de_optimize(quick_score, init, lo, hi, pop_size=10, max_gen=8, time_budget=120.0)
    
    # Phase 2: refine with more k-points if time permits
    if time.time() < deadline - 30:
        def full_score(params):
            cell = _make_cell(params)
            return _score_cell(cell, gmax=8, n_kpts=30)
        
        # Local refinement around the best
        rng = np.random.RandomState(7)
        for _ in range(30):
            if time.time() > deadline - 30:
                break
            cand = best_p + rng.normal(0, 0.015, len(lo))
            cand = np.clip(cand, lo, hi)
            sc = full_score(cand)
            if sc > best_s:
                best_s = sc
                best_p = cand.copy()
        
        # Try a few different initial points with full scoring
        for init_pt in [
            np.array([0.20, 0.10, 0.25, 0.0, 0.0, 0.12, 0.35]),
            np.array([0.10, 0.05, 0.20, 0.25, 0.08, 0.0, 0.0]),
            np.array([0.25, 0.0, 0.0, 0.0, 0.0, 0.15, 0.40]),
        ]:
            if time.time() > deadline - 25:
                break
            cell = _make_cell(init_pt)
            sc = _score_cell(cell, gmax=8, n_kpts=30)
            if sc > best_s:
                best_s = sc
                best_p = init_pt.copy()
    
    # Final cell
    cell = _make_cell(best_p)
    
    # Verify symmetry (defensive check)
    cell = np.asarray(cell, dtype=int)
    # Ensure D4 symmetry by construction (already done in _make_cell)
    # But double-check: cell should be symmetric under transpose and reversal
    # For D4 symmetry about center: cell[i,j] == cell[j,i] == cell[N-1-i,j] == cell[i,N-1-j] == ...
    # Our construction ensures this by using (x,y) which is symmetric under (x,y)->(-x,-y), (x,y)->(y,x), etc.
    
    return cell.tolist()
# EVOLVE-BLOCK-END
