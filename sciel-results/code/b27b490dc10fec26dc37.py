"""Improved QWOT-based antireflection coating design with per-layer thickness optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _reflectance_spectrum_vec(n_indices, thicknesses, n0, ns, wavelengths):
    """
    Vectorized normal-incidence reflectance using characteristic matrix method.
    n_indices: array of refractive indices (length N)
    thicknesses: array of physical thicknesses in nm (length N)
    n0, ns: ambient and substrate indices
    wavelengths: array of wavelengths in nm (length W)
    Returns: array of reflectance fractions (length W)
    """
    N = len(n_indices)
    W = len(wavelengths)
    
    if N == 0:
        r = (n0 - ns) / (n0 + ns)
        return np.full(W, r**2)
    
    # Precompute 2*pi/lambda for all wavelengths
    inv_lambda = 2.0 * np.pi / wavelengths  # shape (W,)
    
    # For each layer, precompute cos and sin of delta for all wavelengths
    # delta_j = n_j * d_j * 2*pi/lambda
    n_arr = np.asarray(n_indices, dtype=np.float64)  # (N,)
    d_arr = np.asarray(thicknesses, dtype=np.float64)  # (N,)
    
    # delta shape: (N, W)
    delta = np.outer(n_arr * d_arr, inv_lambda)  # (N, W)
    cos_d = np.cos(delta)  # (N, W)
    sin_d = np.sin(delta)  # (N, W)
    
    # Initialize M matrices: M00, M01, M10, M11 each shape (W,)
    M00 = np.ones(W, dtype=np.complex128)
    M01 = np.zeros(W, dtype=np.complex128)
    M10 = np.zeros(W, dtype=np.complex128)
    M11 = np.ones(W, dtype=np.complex128)
    
    for j in range(N):
        n_j = n_arr[j]
        c = cos_d[j]  # (W,)
        s = sin_d[j]  # (W,)
        
        # Layer matrix: [[c, i*s/n_j], [i*n_j*s, c]]
        # M_new = M @ M_j
        i_s_over_n = 1j * s / n_j  # (W,)
        i_n_s = 1j * n_j * s  # (W,)
        
        new_M00 = M00 * c + M01 * i_s_over_n
        new_M01 = M00 * i_s_over_n + M01 * c
        new_M10 = M10 * c + M11 * i_n_s
        new_M11 = M10 * i_n_s + M11 * c
        
        M00, M01, M10, M11 = new_M00, new_M01, new_M10, new_M11
    
    B = M00 + M01 / ns
    C = M10 + M11 / ns
    Y = C / B
    r = (n0 - Y) / (n0 + Y)
    R = r.real**2 + r.imag**2
    return R

def _eval_thick(seq, thicknesses, problem):
    """
    Evaluate mean reflectance given layer labels and explicit thicknesses.
    Returns score in percent, or 100.0 if constraints violated.
    """
    if len(seq) == 0:
        r = (problem["ambient_index"] - problem["substrate_index"]) / (problem["ambient_index"] + problem["substrate_index"])
        return r**2 * 100.0
    
    mat_dict = problem["materials"]
    n0 = problem["ambient_index"]
    ns = problem["substrate_index"]
    wavelengths = problem["_wavelengths"]
    max_layer_nm = problem["max_layer_nm"]
    max_total_nm = problem["max_total_nm"]
    
    n_indices = np.array([mat_dict[label] for label in seq], dtype=np.float64)
    d_arr = np.asarray(thicknesses, dtype=np.float64)
    
    if np.any(d_arr < 0) or np.any(d_arr > max_layer_nm):
        return 100.0
    if np.sum(d_arr) > max_total_nm:
        return 100.0
    
    R = _reflectance_spectrum_vec(n_indices, d_arr, n0, ns, wavelengths)
    return float(np.mean(R) * 100.0)

def _eval_scale(seq, scale, problem):
    """
    Evaluate with single global scale factor.
    """
    if len(seq) == 0:
        r = (problem["ambient_index"] - problem["substrate_index"]) / (problem["ambient_index"] + problem["substrate_index"])
        return r**2 * 100.0
    
    mat_dict = problem["materials"]
    lambda_0 = problem["_lambda0"]
    n_indices = np.array([mat_dict[label] for label in seq], dtype=np.float64)
    thicknesses = scale * lambda_0 / (4.0 * n_indices)
    return _eval_thick(seq, thicknesses, problem)

def _optimize_per_layer(seq, problem, max_iter=30):
    """
    Optimize each layer thickness independently using golden-section search.
    Returns (best_thicknesses_array, best_score).
    """
    if len(seq) == 0:
        return np.array([]), 100.0
    
    mat_dict = problem["materials"]
    lambda_0 = problem["_lambda0"]
    max_layer_nm = problem["max_layer_nm"]
    max_total_nm = problem["max_total_nm"]
    
    n_indices = np.array([mat_dict[label] for label in seq], dtype=np.float64)
    N = len(seq)
    
    # Start from QWOT baseline
    thicknesses = lambda_0 / (4.0 * n_indices)
    
    # Ensure total constraint
    total = np.sum(thicknesses)
    if total > max_total_nm:
        thicknesses *= max_total_nm / total
    
    best_score = _eval_thick(seq, thicknesses, problem)
    
    # Iterate: optimize each layer in turn
    for iteration in range(max_iter):
        improved = False
        for j in range(N):
            # Get current thickness of layer j
            d_j = thicknesses[j]
            
            # Determine bounds for layer j
            d_min = 0.0
            d_max = max_layer_nm
            
            # Total constraint: sum of others + d_j <= max_total_nm
            sum_others = np.sum(thicknesses) - d_j
            d_max = min(d_max, max_total_nm - sum_others)
            
            if d_max <= d_min:
                continue
            
            # Golden section search for layer j
            phi = (np.sqrt(5.0) - 1.0) / 2.0
            a, b = d_min, d_max
            
            # Start near current value if valid
            if d_min < d_j < d_max:
                # Narrow search around current value
                span = (b - a) * 0.3
                a = max(d_min, d_j - span)
                b = min(d_max, d_j + span)
            
            c = b - phi * (b - a)
            d = a + phi * (b - a)
            
            def eval_j(d_val):
                thicknesses[j] = d_val
                return _eval_thick(seq, thicknesses, problem)
            
            fc = eval_j(c)
            fd = eval_j(d)
            
            for _ in range(25):
                if fc < fd:
                    b = d
                    d = c
                    fd = fc
                    c = b - phi * (b - a)
                    fc = eval_j(c)
                else:
                    a = c
                    c = d
                    fc = fd
                    d = a + phi * (b - a)
                    fd = eval_j(d)
            
            best_d = (a + b) / 2.0
            thicknesses[j] = best_d
            sc = eval_j(best_d)
            
            if sc < best_score - 1e-6:
                best_score = sc
                improved = True
        
        if not improved:
            break
    
    return thicknesses.copy(), best_score

def _optimize_scale(seq, problem):
    """
    Golden-section + fine grid optimization of scale for a fixed sequence.
    Returns (best_scale, best_score).
    """
    if len(seq) == 0:
        return 1.0, 100.0
    
    mat_dict = problem["materials"]
    lambda_0 = problem["_lambda0"]
    max_layer_nm = problem["max_layer_nm"]
    max_total_nm = problem["max_total_nm"]
    
    # Determine valid scale range
    s_min = 0.05
    s_max = 10.0
    
    for label in seq:
        n = mat_dict[label]
        s_max = min(s_max, max_layer_nm * 4.0 * n / lambda_0)
    
    sum_inv_n = sum(1.0 / mat_dict[label] for label in seq)
    if sum_inv_n > 0:
        s_max = min(s_max, max_total_nm * 4.0 / (lambda_0 * sum_inv_n))
    
    s_max = max(s_max, s_min)
    
    # Golden section search
    phi = (np.sqrt(5.0) - 1.0) / 2.0
    a, b = s_min, s_max
    c = b - phi * (b - a)
    d = a + phi * (b - a)
    fc = _eval_scale(seq, c, problem)
    fd = _eval_scale(seq, d, problem)
    
    for _ in range(40):
        if fc < fd:
            b = d
            d = c
            fd = fc
            c = b - phi * (b - a)
            fc = _eval_scale(seq, c, problem)
        else:
            a = c
            c = d
            fc = fd
            d = a + phi * (b - a)
            fd = _eval_scale(seq, d, problem)
    
    best_s = (a + b) / 2.0
    best_score = _eval_scale(seq, best_s, problem)
    
    # Fine grid around golden section result
    grid = np.linspace(max(s_min, best_s - 0.15), min(s_max, best_s + 0.15), 60)
    for s in grid:
        sc = _eval_scale(seq, s, problem)
        if sc < best_score:
            best_score = sc
            best_s = s
    
    # Even finer
    grid2 = np.linspace(max(s_min, best_s - 0.03), min(s_max, best_s + 0.03), 60)
    for s in grid2:
        sc = _eval_scale(seq, s, problem)
        if sc < best_score:
            best_score = sc
            best_s = s
    
    return best_s, best_score

def design(problem):
    """
    Design an antireflection coating using QWOT-based combinatorial search
    with per-layer thickness optimization.
    """
    start_time = time.time()
    time_budget = 55.0
    
    band = problem["band_nm"]
    lambda_0 = 0.5 * (band[0] + band[1])
    mat_dict = problem["materials"]
    max_layers = problem["max_layers"]
    
    # Precompute
    problem["_lambda0"] = lambda_0
    problem["_wavelengths"] = np.linspace(band[0], band[1], 401)
    
    np.random.seed(12345)
    
    # Build initial candidate sequences: alternating patterns with various lengths
    # plus some structured patterns known to be effective
    initial_seqs = []
    
    # Alternating patterns starting with L or H, lengths 1-20
    for length in range(1, max_layers + 1):
        initial_seqs.append(['L' if i % 2 == 0 else 'H' for i in range(length)])
        initial_seqs.append(['H' if i % 2 == 0 else 'L' for i in range(length)])
    
    # Some structured patterns: blocks of same type
    for length in range(3, max_layers + 1, 2):
        # LHHLH pattern
        pat = []
        for i in range(length):
            if i % 4 in [0, 1]:
                pat.append('L')
            else:
                pat.append('H')
        initial_seqs.append(pat)
        
        # HLLHH pattern
        pat = []
        for i in range(length):
            if i % 4 in [0, 1]:
                pat.append('H')
            else:
                pat.append('L')
        initial_seqs.append(pat)
    
    # Evaluate all initial sequences with per-layer optimization
    best_seq = None
    best_thick = None
    best_score = 100.0
    
    for seq in initial_seqs:
        if len(seq) > max_layers:
            continue
        thick, score = _optimize_per_layer(seq, problem, max_iter=15)
        if score < best_score:
            best_score = score
            best_seq = list(seq)
            best_thick = thick.copy()
    
    if best_seq is None:
        best_seq = ['L']
        best_thick, best_score = _optimize_per_layer(best_seq, problem, max_iter=20)
    
    # Simulated annealing on the discrete sequence
    # with periodic per-layer thickness re-optimization
    T_init = 0.5
    T_final = 0.001
    total_iters = 25000
    cooling = (T_final / T_init) ** (1.0 / total_iters)
    
    current_seq = list(best_seq)
    current_thick = best_thick.copy()
    current_score = best_score
    
    reopt_interval = 30  # re-optimize thicknesses every N iterations
    
    for it in range(total_iters):
        if time.time() - start_time > time_budget - 5:
            break
        
        T = T_init * (cooling ** it)
        
        # Generate neighbor
        r = np.random.rand()
        new_seq = list(current_seq)
        
        if r < 0.30:
            # Flip a random layer
            if len(new_seq) > 0:
                idx = np.random.randint(len(new_seq))
                new_seq[idx] = 'H' if new_seq[idx] == 'L' else 'L'
        elif r < 0.45:
            # Insert a layer
            if len(new_seq) < max_layers:
                pos = np.random.randint(len(new_seq) + 1)
                new_seq.insert(pos, 'L' if np.random.rand() < 0.5 else 'H')
        elif r < 0.60:
            # Remove a layer
            if len(new_seq) > 1:
                idx = np.random.randint(len(new_seq))
                new_seq.pop(idx)
        elif r < 0.75:
            # Swap two adjacent layers
            if len(new_seq) >= 2:
                idx = np.random.randint(len(new_seq) - 1)
                new_seq[idx], new_seq[idx+1] = new_seq[idx+1], new_seq[idx]
        elif r < 0.85:
            # Replace a random contiguous block with random layers
            if len(new_seq) >= 2:
                block_len = np.random.randint(2, min(4, len(new_seq) + 1))
                start = np.random.randint(0, len(new_seq) - block_len + 1)
                for k in range(block_len):
                    new_seq[start + k] = 'L' if np.random.rand() < 0.5 else 'H'
        else:
            # Move a layer to a different position
            if len(new_seq) >= 3:
                idx = np.random.randint(len(new_seq))
                layer = new_seq.pop(idx)
                new_idx = np.random.randint(len(new_seq) + 1)
                new_seq.insert(new_idx, layer)
        
        # Quick evaluation with current thicknesses (scaled if length changed)
        if len(new_seq) == len(current_seq):
            new_thick = current_thick.copy()
        else:
            # Re-initialize thicknesses for new length
            n_indices = np.array([mat_dict[label] for label in new_seq], dtype=np.float64)
            new_thick = lambda_0 / (4.0 * n_indices)
            total = np.sum(new_thick)
            if total > problem["max_total_nm"]:
                new_thick *= problem["max_total_nm"] / total
        
        new_score_fast = _eval_thick(new_seq, new_thick, problem)
        
        # If significantly worse, skip detailed evaluation
        if new_score_fast > current_score + 5.0:
            continue
        
        # Re-optimize thicknesses periodically or when sequence improves
        if it % reopt_interval == 0 or (new_score_fast < current_score):
            new_thick, new_score = _optimize_per_layer(new_seq, problem, max_iter=20)
        else:
            new_score = new_score_fast
        
        # Accept/reject
        if new_score < current_score:
            current_seq = new_seq
            current_thick = new_thick
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_seq = list(current_seq)
                best_thick = current_thick.copy()
        else:
            delta = new_score - current_score
            if delta > 0 and T > 0.001:
                accept_prob = np.exp(-delta / T)
                if np.random.rand() < accept_prob:
                    current_seq = new_seq
                    current_thick = new_thick
                    current_score = new_score
    
    # Final refinement with more iterations
    best_thick, best_score = _optimize_per_layer(best_seq, problem, max_iter=50)
    
    # Local polish: try all single-layer modifications
    if time.time() - start_time < time_budget - 3:
        for idx in range(len(best_seq)):
            trial = list(best_seq)
            trial[idx] = 'H' if trial[idx] == 'L' else 'L'
            thick, sc = _optimize_per_layer(trial, problem, max_iter=25)
            if sc < best_score:
                best_score = sc
                best_seq = trial
                best_thick = thick
        
        # Try insertions
        for pos in range(len(best_seq) + 1):
            for label in ['L', 'H']:
                trial = list(best_seq)
                trial.insert(pos, label)
                if len(trial) <= max_layers:
                    thick, sc = _optimize_per_layer(trial, problem, max_iter=25)
                    if sc < best_score:
                        best_score = sc
                        best_seq = trial
                        best_thick = thick
        
        # Try deletions
        for idx in range(len(best_seq)):
            if len(best_seq) > 1:
                trial = list(best_seq)
                trial.pop(idx)
                thick, sc = _optimize_per_layer(trial, problem, max_iter=25)
                if sc < best_score:
                    best_score = sc
                    best_seq = trial
                    best_thick = thick
    
    # Convert to output
    materials_labels = list(best_seq)
    thicknesses = [float(d) for d in best_thick]
    
    return materials_labels, thicknesses
# EVOLVE-BLOCK-END
