"""Photonic crystal unit cell design for TM band gap maximization."""
import time
import numpy as np

def _enforce_symmetry(cell_arr, size):
    """Enforce square symmetry: 180-rotation and transpose invariance."""
    b = np.array(cell_arr, dtype=np.int8)
    r180 = b[::-1, ::-1]
    t = b.T
    at = t[::-1, ::-1]
    votes = (b + r180 + t + at)
    cell = (votes >= 2).astype(np.int8)
    return cell

def _compute_gap(cell, size, eps_high=12.0, eps_low=1.0, n_pw=10, k_points=None):
    """Fast PWE calculation for TM bands 4 and 5 gap ratio."""
    if k_points is None:
        # Key k-points: Gamma, X, M, and interior points
        k_points = [
            (0.0, 0.0),
            (0.25*np.pi, 0.0), (0.5*np.pi, 0.0), (0.75*np.pi, 0.0), (np.pi, 0.0),
            (0.25*np.pi, 0.25*np.pi), (0.5*np.pi, 0.5*np.pi), (0.75*np.pi, 0.75*np.pi), (np.pi, np.pi),
            (0.5*np.pi, 0.0), (0.75*np.pi, 0.0),
            (0.5*np.pi, 0.25*np.pi), (0.5*np.pi, 0.5*np.pi),
            (np.pi, 0.0), (np.pi, 0.5*np.pi), (np.pi, np.pi),
            (0.0, 0.5*np.pi), (0.5*np.pi, 0.5*np.pi),
        ]
    
    cell_arr = np.array(cell, dtype=float)
    eps_arr = np.where(cell_arr > 0.5, eps_high, eps_low)
    inv_eps_arr = 1.0 / eps_arr
    
    # Compute Fourier coefficients of 1/eps
    G_range = range(-n_pw, n_pw + 1)
    Gs = np.array([(mx, my) for mx in G_range for my in G_range], dtype=float)
    nG = len(Gs)
    
    # Fast 2D FFT approach for Fourier coefficients
    inv_eps_2d = inv_eps_arr
    # Pad to 2*n_pw+1 size for proper DFT
    N_fft = 2 * n_pw + 1
    inv_eps_padded = np.zeros((N_fft, N_fft), dtype=complex)
    # Map size x size to N_fft x N_fft
    scale = N_fft / size
    for i in range(size):
        for j in range(size):
            fi = int(i * scale)
            fj = int(j * scale)
            inv_eps_padded[fi, fj] += inv_eps_arr[i, j]
    # Normalize
    inv_eps_padded /= (size * size)
    
    # 2D FFT to get Fourier coefficients
    inv_eps_F = np.fft.fft2(inv_eps_padded)
    # Reorder to match Gs ordering
    inv_eps_G = np.zeros(nG, dtype=complex)
    for idx, (mx, my) in enumerate(Gs):
        # FFT gives frequencies 0..N-1, map to -n_pw..n_pw
        fx = mx if mx >= 0 else mx + N_fft
        fy = my if my >= 0 else my + N_fft
        inv_eps_G[idx] = inv_eps_F[fx, fy]
    
    # Precompute Gs differences lookup
    Gs_dict = {}
    for idx, (mx, my) in enumerate(Gs):
        Gs_dict[(int(mx), int(my))] = idx
    
    max_band4 = 0.0
    min_band5 = np.inf
    
    for kx, ky in k_points:
        k = np.array([kx, ky])
        kGs = k + Gs  # (nG, 2)
        
        # Dot products: (k+Gi)·(k+Gj)
        dot = np.outer(kGs[:, 0], kGs[:, 0]) + np.outer(kGs[:, 1], kGs[:, 1])
        
        # Build inv_eps matrix
        inv_eps_mat = np.zeros((nG, nG), dtype=complex)
        for i in range(nG):
            for j in range(nG):
                dmx = int(Gs[i, 0] - Gs[j, 0])
                dmy = int(Gs[i, 1] - Gs[j, 1])
                key = (dmx, dmy)
                if key in Gs_dict:
                    inv_eps_mat[i, j] = inv_eps_G[Gs_dict[key]]
        
        A = dot * inv_eps_mat
        A = (A + A.conj().T) / 2.0
        
        try:
            eigvals = np.linalg.eigvalsh(A)
            eigvals = np.real(eigvals)
            eigvals = eigvals[eigvals > 1e-10]
        except:
            continue
        
        if len(eigvals) >= 5:
            b4 = eigvals[3]
            b5 = eigvals[4]
            max_band4 = max(max_band4, b4)
            min_band5 = min(min_band5, b5)
    
    if min_band5 == np.inf or max_band4 <= 0:
        return -1.0, None
    
    gap = min_band5 - max_band4
    midgap = 0.5 * (min_band5 + max_band4)
    if midgap <= 0:
        return -1.0, None
    ratio = 2.0 * gap / midgap
    return ratio, (max_band4, min_band5)

def _make_initial_cell(size, shape_type='circle', params=None):
    """Generate initial symmetric cell."""
    if params is None:
        params = {}
    
    base = np.zeros((size, size), dtype=np.int8)
    center = size / 2.0
    
    if shape_type == 'circle':
        r = params.get('radius', 0.25)
        for i in range(size):
            for j in range(size):
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                if x*x + y*y < r*r:
                    base[i, j] = 1
    elif shape_type == 'square':
        hw = params.get('half_width', 0.3)
        cr = params.get('corner_radius', 0.05)
        for i in range(size):
            for j in range(size):
                x = abs((i + 0.5) / size - 0.5)
                y = abs((j + 0.5) / size - 0.5)
                if x < hw and y < hw:
                    if x > hw - cr and y > hw - cr:
                        dx = x - (hw - cr)
                        dy = y - (hw - cr)
                        if dx*dx + dy*dy <= cr*cr:
                            base[i, j] = 1
                    else:
                        base[i, j] = 1
    elif shape_type == 'cross':
        aw = params.get('arm_width', 6)
        ln = params.get('length', 12)
        for i in range(size):
            for j in range(size):
                dx = abs(i + 0.5 - center)
                dy = abs(j + 0.5 - center)
                if dx < ln and dy < aw/2:
                    base[i, j] = 1
                elif dy < ln and dx < aw/2:
                    base[i, j] = 1
    elif shape_type == 'ring':
        r_inner = params.get('r_inner', 0.15)
        r_outer = params.get('r_outer', 0.35)
        for i in range(size):
            for j in range(size):
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                r2 = x*x + y*y
                if r_inner*r_inner < r2 < r_outer*r_outer:
                    base[i, j] = 1
    
    return _enforce_symmetry(base, size).tolist()

def design(problem):
    """Design photonic crystal unit cell to maximize TM band 4-5 gap."""
    size = problem["pixels"]
    eps_high = problem.get("eps_high", 12.0)
    eps_low = problem.get("eps_low", 1.0)
    
    start_time = time.time()
    deadline = start_time + 200.0  # Safety margin for 230s limit
    
    best_cell = None
    best_score = -1.0
    
    # Generate diverse initial candidates
    candidates = []
    for r in [0.15, 0.2, 0.25, 0.3, 0.35]:
        candidates.append(_make_initial_cell(size, 'circle', {'radius': r}))
    for hw in [0.2, 0.25, 0.3, 0.35]:
        for cr in [0.02, 0.05, 0.1]:
            candidates.append(_make_initial_cell(size, 'square', {'half_width': hw, 'corner_radius': cr}))
    for aw in [4, 6, 8, 10]:
        for ln in [8, 12, 16]:
            candidates.append(_make_initial_cell(size, 'cross', {'arm_width': aw, 'length': ln}))
    for ri in [0.1, 0.15, 0.2]:
        for ro in [0.3, 0.35, 0.4]:
            if ro > ri:
                candidates.append(_make_initial_cell(size, 'ring', {'r_inner': ri, 'r_outer': ro}))
    
    # Evaluate initial candidates
    for cell in candidates:
        if time.time() > deadline - 30:
            break
        try:
            score, _ = _compute_gap(cell, size, eps_high, eps_low, n_pw=8)
        except:
            score = -1.0
        if score > best_score:
            best_score = score
            best_cell = cell
    
    # Local search with random mutations
    if best_cell is not None:
        current_cell = best_cell
        current_score = best_score
        rng = np.random.default_rng(12345)
        
        max_steps = 2000
        for step in range(max_steps):
            if time.time() > deadline - 10:
                break
            
            # Adaptive mutation rate
            n_flips = rng.integers(1, 6)
            mut_cell = np.array(current_cell, dtype=np.int8)
            for _ in range(n_flips):
                i = rng.integers(0, size)
                j = rng.integers(0, size)
                mut_cell[i, j] = 1 - mut_cell[i, j]
            
            sym_cell = _enforce_symmetry(mut_cell, size)
            sym_list = sym_cell.tolist()
            
            try:
                score, _ = _compute_gap(sym_list, size, eps_high, eps_low, n_pw=8)
            except:
                score = -1.0
            
            if score > current_score:
                current_cell = sym_list
                current_score = score
                if score > best_score:
                    best_score = score
                    best_cell = sym_list
    
    # Final refinement: try small perturbations around best
    if best_cell is not None:
        current_cell = best_cell
        current_score = best_score
        rng2 = np.random.default_rng(99999)
        
        for step in range(500):
            if time.time() > deadline - 5:
                break
            
            # Single pixel flip
            i = rng2.integers(0, size)
            j = rng2.integers(0, size)
            mut_cell = np.array(current_cell, dtype=np.int8)
            mut_cell[i, j] = 1 - mut_cell[i, j]
            
            sym_cell = _enforce_symmetry(mut_cell, size)
            sym_list = sym_cell.tolist()
            
            try:
                score, _ = _compute_gap(sym_list, size, eps_high, eps_low, n_pw=8)
            except:
                score = -1.0
            
            if score > current_score:
                current_cell = sym_list
                current_score = score
                if score > best_score:
                    best_score = score
                    best_cell = sym_list
    
    if best_cell is None:
        best_cell = _make_initial_cell(size, 'circle', {'radius': 0.25})
    
    return best_cell
