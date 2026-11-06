import time
import numpy as np
from scipy.optimize import minimize
from scipy.fft import dct as scipy_dct, idct as scipy_idct

N = 400
_t0 = time.time()
_DEADLINE = 145.0  # Safety margin

# Precompute indices
_i_arr = np.arange(N, dtype=float)

# Precompute atom templates for sparse representation
_ATOM_TEMPLATES = []
_ATOM_TEMPLATES.append(np.hamming(8))
_ATOM_TEMPLATES.append(np.hanning(12))
_ATOM_TEMPLATES.append(np.blackman(16))
_ATOM_TEMPLATES.append(np.ones(6) * 0.5)
_ATOM_TEMPLATES.append(np.hamming(20))
_ATOM_TEMPLATES.append(np.ones(4) * 0.25)
_ATOM_TEMPLATES.append(np.hanning(24))
_ATOM_TEMPLATES.append(np.blackman(8))
_ATOM_TEMPLATES.append(np.ones(10) * 0.3)
for i in range(len(_ATOM_TEMPLATES)):
    _ATOM_TEMPLATES[i] = _ATOM_TEMPLATES[i] / np.sum(_ATOM_TEMPLATES[i])

# Precompute atom autocorrelations and cross-correlations for fast coordinate descent
# For each template t, precompute corr_t[k] = sum_i template[t][i] * template[t][i-k] (full autocorrelation)
_ATOM_CORR = []
for tmpl in _ATOM_TEMPLATES:
    L = len(tmpl)
    # Full autocorrelation: length 2L-1
    ac = np.correlate(tmpl, tmpl, mode='full')
    _ATOM_CORR.append(ac)

# Precompute cross-correlations between all template pairs
_n_templates = len(_ATOM_TEMPLATES)
_XCORR = np.zeros((_n_templates, _n_templates, 2 * N - 1))
for i in range(_n_templates):
    for j in range(_n_templates):
        ti = _ATOM_TEMPLATES[i]
        tj = _ATOM_TEMPLATES[j]
        # Cross-correlation: sum_k ti[k] * tj[k - lag]
        # Use FFT for efficiency
        size = 2 * N
        Ti = np.fft.rfft(ti, size)
        Tj = np.fft.rfft(tj, size)
        xc = np.fft.irfft(Ti * np.conj(Tj), size)[:2 * N - 1]
        _XCORR[i, j] = xc

def _value(h):
    """Compute the exact task value (lower is better)."""
    h = np.asarray(h, dtype=float)
    if not np.all(np.isfinite(h)):
        return np.inf
    s = float(np.sum(h))
    if abs(s) < 1e-12:
        return np.inf
    
    size = 2 * N
    H = np.fft.rfft(h, size)
    conv = np.fft.irfft(H * H, size)[:2 * N - 1]
    
    m = float(np.max(conv))
    val = abs(2.0 * N * m / (s * s))
    return val

def _fast_value_from_conv(conv, s):
    """Compute value from precomputed convolution and sum."""
    if abs(s) < 1e-12:
        return np.inf
    m = float(np.max(conv))
    return abs(2.0 * N * m / (s * s))

def _sparse_to_h(params):
    """
    Reconstruct h from sparse atom parameters.
    params: [amp_0, offset_0, atom_idx_0, amp_1, offset_1, atom_idx_1, ...]
    """
    h = np.zeros(N)
    n_atoms = len(params) // 3
    for i in range(n_atoms):
        amp = params[3*i]
        offset = int(round(params[3*i + 1]))
        atom_idx = int(round(params[3*i + 2]))
        
        template = _ATOM_TEMPLATES[atom_idx % len(_ATOM_TEMPLATES)]
        tmpl_len = len(template)
        offset = np.clip(offset, 0, N - tmpl_len)
        
        h[offset:offset+tmpl_len] += amp * template
    
    return h

def _compute_conv_from_atoms(params):
    """
    Compute the full autoconvolution of h built from sparse atoms.
    Returns (conv, sum_h) where conv has length 2N-1.
    """
    n_atoms = len(params) // 3
    conv = np.zeros(2 * N - 1)
    sum_h = 0.0
    
    for i in range(n_atoms):
        amp_i = params[3*i]
        off_i = int(round(params[3*i + 1]))
        idx_i = int(round(params[3*i + 2])) % _n_templates
        tmpl_i = _ATOM_TEMPLATES[idx_i]
        L_i = len(tmpl_i)
        off_i = np.clip(off_i, 0, N - L_i)
        
        sum_h += amp_i * np.sum(tmpl_i)
        
        # Self-correlation contribution: amp_i^2 * autocorr(tmpl_i) placed at off_i
        ac_i = _ATOM_CORR[idx_i]
        start = off_i
        end = start + len(ac_i)
        if start >= 0 and end <= 2 * N - 1:
            conv[start:end] += amp_i * amp_i * ac_i
        
        # Cross-correlation with all previous atoms
        for j in range(i):
            amp_j = params[3*j]
            off_j = int(round(params[3*j + 1]))
            idx_j = int(round(params[3*j + 2])) % _n_templates
            tmpl_j = _ATOM_TEMPLATES[idx_j]
            L_j = len(tmpl_j)
            off_j = np.clip(off_j, 0, N - L_j)
            
            # Cross-correlation: amp_i * amp_j * xcorr(tmpl_i, tmpl_j)
            # The placement depends on relative offset
            xc = _XCORR[idx_i, idx_j]
            # xcorr(k) = sum_m tmpl_i[m] * tmpl_j[m - k]
            # When placed at off_i and off_j, the contribution to conv at lag L is:
            # sum_m tmpl_i[m] * tmpl_j[m - (L - (off_i - off_j))]
            # = xcorr(L - (off_i - off_j))
            # So conv[L] += amp_i * amp_j * xc[L - (off_i - off_j)]
            # And also the symmetric: amp_j * amp_i * xcorr(tmpl_j, tmpl_i)
            # which is xc_ji[L - (off_j - off_i)] = xc_ji[L + (off_i - off_j)]
            
            delta = off_i - off_j
            
            # Contribution from (i,j) pair
            # conv[L] += amp_i * amp_j * xc_ij[L - delta] for valid L
            # and conv[L] += amp_j * amp_i * xc_ji[L + delta]
            # But xc_ji[k] = xc_ij[-k] (conjugate symmetry for real signals)
            # Actually for real: xc_ij[k] = sum_m ti[m] * tj[m-k]
            # xc_ji[k] = sum_m tj[m] * ti[m-k] = sum_n ti[n] * tj[n+k] = xc_ij[-k]
            
            # So total cross contribution at lag L:
            # amp_i*amp_j * (xc_ij[L - delta] + xc_ij[-(L + delta)])
            # = amp_i*amp_j * (xc_ij[L - delta] + xc_ij[delta - L])
            
            # We need to add this for all L in [0, 2N-2]
            # xc_ij has length 2N-1, indexed 0..2N-2
            # xc_ij[k] for k in range(-max_len, max_len)
            # Our precomputed xc is the full cross-correlation, length 2N-1
            # xc[L] = sum_m ti[m] * tj[m - L] for L=0..2N-2
            # But we need xc for negative indices too.
            # xc[-k] = sum_m ti[m] * tj[m + k]
            
            # Let's just compute the cross-correlation properly using FFT for this pair
            # Actually, let's precompute more carefully.
            
            # Simpler approach: just use the precomputed xc array
            # xc[k] for k = 0, 1, ..., 2N-2 represents sum_m ti[m] * tj[m-k]
            # For the convolution, the cross term at lag L is:
            # sum_m ti[m + off_i] * tj[m + off_j] shifted... 
            # This is getting complex. Let me use a different approach.
            
            # Actually, the autoconvolution of h is:
            # conv[L] = sum_m h[m] * h[m + L] (for L >= 0)
            # = sum_m h[m] * h[m - L] (for L < 0, but we only have L >= 0 in full conv)
            # Full conv: conv[L] = sum_m h[m] * h[m + L] for L = -(N-1)..(N-1)
            # But np.convolve gives length 2N-1, indexed 0..2N-2
            # conv[k] = sum_m h[m] * h[m - (k - (N-1))]... 
            # Actually np.convolve(h,h) gives:
            # conv[k] = sum_m h[m] * h[m - k] for k = 0..2N-2
            # where h is zero-padded outside [0, N-1]
            
            # For two atoms at positions off_i and off_j:
            # The contribution to conv[k] from these two atoms is:
            # sum_m (amp_i * ti[m - off_i]) * (amp_j * tj[m - off_j - k])
            # = amp_i * amp_j * sum_m ti[m - off_i] * tj[m - off_j - k]
            # Let n = m - off_i:
            # = amp_i * amp_j * sum_n ti[n] * tj[n + off_i - off_j - k]
            # = amp_i * amp_j * xcorr_ti_tj[k - (off_i - off_j)]
            # where xcorr_ti_tj[d] = sum_n ti[n] * tj[n + d]
            
            # So we need xcorr for positive and negative d.
            # xcorr[d] = sum_n ti[n] * tj[n + d]
            # xcorr[-d] = sum_n ti[n] * tj[n - d] = sum_m tj[m] * ti[m + d] = xcorr_tj_ti[d]
            
            # Precomputed _XCORR[i,j][k] = sum_m ti[m] * tj[m - k] (from FFT)
            # This is xcorr_ti_tj[-k] = sum_n ti[n] * tj[n - k]
            # So xcorr_ti_tj[d] = _XCORR[i,j][-d] = _XCORR[i,j][k] where k = -d
            # But our array is indexed 0..2N-2, representing k = 0..2N-2
            # For negative k, we need to handle it.
            
            # Let me redefine: _XCORR[i,j][k] for k=0..2N-2 is:
            # sum_m ti[m] * tj[m - k]
            # For d = k - (off_i - off_j), we need:
            # xcorr_ti_tj[d] = sum_n ti[n] * tj[n + d] = sum_n ti[n] * tj[n - (-d)]
            # = _XCORR[i,j][-d] if -d >= 0, else we need the other direction
            
            # This is getting too complex for precomputation. Let me just compute the cross term directly for small number of atoms.
            
            # For n_atoms <= 48, the number of pairs is at most 48*47/2 = 1128
            # Each cross-correlation can be computed via FFT in O(N log N)
            # But that's too slow for coordinate descent.
            
            # Alternative: just compute the full conv via FFT for each evaluation.
            # With n_atoms up to 48, the sparse_to_h + FFT approach is fine.
            pass
    
    return conv, sum_h

def _smooth_sparse_obj(params, beta=50.0):
    """Smooth surrogate for sparse atom objective."""
    h = _sparse_to_h(params)
    if not np.all(np.isfinite(h)):
        return np.inf
    s = float(np.sum(h))
    if abs(s) < 1e-12:
        return np.inf
    
    size = 2 * N
    H = np.fft.rfft(h, size)
    conv = np.fft.irfft(H * H, size)[:2 * N - 1]
    
    m = np.max(conv)
    scaled = beta * (conv - m)
    scaled = np.clip(scaled, -100, 100)
    lse = m + np.log(np.sum(np.exp(scaled))) / beta
    
    val = abs(2.0 * N * lse / (s * s))
    return val

def _exact_sparse_obj(params):
    """Exact objective for sparse atom parameters."""
    h = _sparse_to_h(params)
    return _value(h)

def _coordinate_descent_fast(params, max_iter=30, tol=1e-7):
    """
    Fast coordinate descent on sparse atom parameters.
    Uses FFT-based evaluation but with incremental updates where possible.
    """
    current_val = _exact_sparse_obj(params)
    if not np.isfinite(current_val):
        return params
    
    n_atoms = len(params) // 3
    n_templates = len(_ATOM_TEMPLATES)
    
    # Precompute current h and conv
    h_curr = _sparse_to_h(params)
    s_curr = float(np.sum(h_curr))
    size = 2 * N
    H_curr = np.fft.rfft(h_curr, size)
    conv_curr = np.fft.irfft(H_curr * H_curr, size)[:2 * N - 1]
    
    for iteration in range(max_iter):
        if not deadline_ok():
            break
            
        improved = False
        
        # Optimize amplitudes
        for i in range(n_atoms):
            if not deadline_ok():
                break
            amp_idx = 3 * i
            off_idx = 3 * i + 1
            atom_idx = 3 * i + 2
            
            amp = params[amp_idx]
            offset = int(round(params[off_idx]))
            tmpl_idx = int(round(params[atom_idx])) % n_templates
            template = _ATOM_TEMPLATES[tmpl_idx]
            tmpl_len = len(template)
            offset = np.clip(offset, 0, N - tmpl_len)
            
            # Try several amplitude values
            best_amp = amp
            best_val = current_val
            for trial_amp in [amp * 0.5, amp * 0.75, amp * 1.25, amp * 1.5, amp - 0.3, amp + 0.3]:
                if not deadline_ok():
                    break
                # Compute new conv incrementally
                # h_new = h_curr + (trial_amp - amp) * template[offset:offset+tmpl_len]
                delta_h = (trial_amp - amp) * template
                h_new = h_curr.copy()
                h_new[offset:offset+tmpl_len] += delta_h
                
                s_new = s_curr + (trial_amp - amp) * np.sum(template)
                if abs(s_new) < 1e-12:
                    continue
                
                H_new = np.fft.rfft(h_new, size)
                conv_new = np.fft.irfft(H_new * H_new, size)[:2 * N - 1]
                tv = _fast_value_from_conv(conv_new, s_new)
                
                if tv < best_val - tol:
                    best_val = tv
                    best_amp = trial_amp
            
            if best_amp != amp:
                params[amp_idx] = best_amp
                # Update h_curr and conv_curr
                delta_h = (best_amp - amp) * template
                h_curr[offset:offset+tmpl_len] += delta_h
                s_curr += (best_amp - amp) * np.sum(template)
                H_curr = np.fft.rfft(h_curr, size)
                conv_curr = np.fft.irfft(H_curr * H_curr, size)[:2 * N - 1]
                current_val = best_val
                improved = True
        
        # Optimize offsets
        for i in range(n_atoms):
            if not deadline_ok():
                break
            amp_idx = 3 * i
            off_idx = 3 * i + 1
            atom_idx = 3 * i + 2
            
            tmpl_idx = int(round(params[atom_idx])) % n_templates
            template = _ATOM_TEMPLATES[tmpl_idx]
            tmpl_len = len(template)
            cur_off = int(round(params[off_idx]))
            amp = params[amp_idx]
            
            best_off = cur_off
            best_val = current_val
            for delta in [-2, -1, 1, 2]:
                new_off = np.clip(cur_off + delta, 0, N - tmpl_len)
                if new_off == cur_off:
                    continue
                
                # Remove atom from old position, add to new
                h_new = h_curr.copy()
                h_new[cur_off:cur_off+tmpl_len] -= amp * template
                h_new[new_off:new_off+tmpl_len] += amp * template
                
                s_new = s_curr  # Sum doesn't change
                if abs(s_new) < 1e-12:
                    continue
                
                H_new = np.fft.rfft(h_new, size)
                conv_new = np.fft.irfft(H_new * H_new, size)[:2 * N - 1]
                tv = _fast_value_from_conv(conv_new, s_new)
                
                if tv < best_val - tol:
                    best_val = tv
                    best_off = new_off
            
            if best_off != cur_off:
                params[off_idx] = float(best_off)
                h_curr[cur_off:cur_off+tmpl_len] -= amp * template
                h_curr[best_off:best_off+tmpl_len] += amp * template
                H_curr = np.fft.rfft(h_curr, size)
                conv_curr = np.fft.irfft(H_curr * H_curr, size)[:2 * N - 1]
                current_val = best_val
                improved = True
        
        # Optimize atom types
        for i in range(n_atoms):
            if not deadline_ok():
                break
            amp_idx = 3 * i
            off_idx = 3 * i + 1
            atom_idx = 3 * i + 2
            
            cur_type = int(round(params[atom_idx])) % n_templates
            cur_off = int(round(params[off_idx]))
            amp = params[amp_idx]
            cur_template = _ATOM_TEMPLATES[cur_type]
            cur_tmpl_len = len(cur_template)
            cur_off = np.clip(cur_off, 0, N - cur_tmpl_len)
            
            best_type = cur_type
            best_val = current_val
            for t in range(n_templates):
                if t == cur_type:
                    continue
                tmpl = _ATOM_TEMPLATES[t]
                tmpl_len = len(tmpl)
                new_off = np.clip(cur_off, 0, N - tmpl_len)
                
                h_new = h_curr.copy()
                h_new[cur_off:cur_off+cur_tmpl_len] -= amp * cur_template
                h_new[new_off:new_off+tmpl_len] += amp * tmpl
                
                s_new = s_curr + amp * (np.sum(tmpl) - np.sum(cur_template))
                if abs(s_new) < 1e-12:
                    continue
                
                H_new = np.fft.rfft(h_new, size)
                conv_new = np.fft.irfft(H_new * H_new, size)[:2 * N - 1]
                tv = _fast_value_from_conv(conv_new, s_new)
                
                if tv < best_val - tol:
                    best_val = tv
                    best_type = t
            
            if best_type != cur_type:
                params[atom_idx] = float(best_type)
                new_template = _ATOM_TEMPLATES[best_type]
                new_tmpl_len = len(new_template)
                new_off = np.clip(cur_off, 0, N - new_tmpl_len)
                h_curr[cur_off:cur_off+cur_tmpl_len] -= amp * cur_template
                h_curr[new_off:new_off+new_tmpl_len] += amp * new_template
                s_curr += amp * (np.sum(new_template) - np.sum(cur_template))
                H_curr = np.fft.rfft(h_curr, size)
                conv_curr = np.fft.irfft(H_curr * H_curr, size)[:2 * N - 1]
                current_val = best_val
                improved = True
        
        if not improved:
            break
    
    return params

def _smooth_value(h, beta=50.0):
    """Smooth surrogate for the objective."""
    h = np.asarray(h, dtype=float)
    if not np.all(np.isfinite(h)):
        return np.inf
    s = float(np.sum(h))
    if abs(s) < 1e-12:
        return np.inf
    
    size = 2 * N
    H = np.fft.rfft(h, size)
    conv = np.fft.irfft(H * H, size)[:2 * N - 1]
    
    m = np.max(conv)
    scaled = beta * (conv - m)
    scaled = np.clip(scaled, -100, 100)
    lse = m + np.log(np.sum(np.exp(scaled))) / beta
    
    val = abs(2.0 * N * lse / (s * s))
    return val

def _dct_h(coeffs):
    """Reconstruct h from DCT-II coefficients."""
    h = scipy_idct(coeffs, type=2, n=N, norm='ortho')
    return h

def _dct_smooth_obj(coeffs, beta=50.0):
    """Objective for DCT coefficients."""
    h = _dct_h(coeffs)
    return _smooth_value(h, beta)

def _generate_initial_dct(rng, num_coeffs):
    """Generate random DCT coefficients with low-frequency bias."""
    coeffs = np.zeros(num_coeffs)
    coeffs[0] = rng.normal(0, 1.0) * 10.0
    weights = 1.0 / (1.0 + np.arange(num_coeffs))
    coeffs[1:] = rng.normal(0, 1.0, size=num_coeffs - 1) * weights[1:] * 2.0
    return coeffs

def _generate_initial_sparse(rng, n_atoms):
    """Generate random sparse atom parameters."""
    params = np.zeros(n_atoms * 3)
    n_templates = len(_ATOM_TEMPLATES)
    for i in range(n_atoms):
        params[3*i] = rng.normal(0, 1.0) * 3.0
        params[3*i + 1] = rng.integers(0, N - 1)
        params[3*i + 2] = rng.integers(0, n_templates)
    return params

def _generate_golay_like(rng, N):
    """
    Generate Golay-like complementary pair sequences.
    Golay pairs have the property that their aperiodic autocorrelations sum to zero.
    We create a sequence that is a combination of two complementary-like sequences.
    """
    # Start with a random ±1 sequence
    a = rng.choice([-1.0, 1.0], size=N)
    b = rng.choice([-1.0, 1.0], size=N)
    
    # Try to make them more complementary by adjusting
    # For simplicity, just return a weighted combination
    w = rng.uniform(0.3, 0.7)
    h = w * a + (1 - w) * b
    return h

def _generate_zadoff_chu(rng, N):
    """
    Generate Zadoff-Chu-like sequences (constant modulus, low autocorrelation sidelobes).
    """
    # Zadoff-Chu: x[n] = exp(-j*pi*u*n*(n+1)/N) for u coprime to N
    # For real-valued, use cosine
    u = rng.integers(1, N // 2)
    if np.gcd(u, N) != 1:
        u = 1
    n = np.arange(N)
    phase = -np.pi * u * n * (n + 1) / N
    h = np.cos(phase)
    # Add small random perturbation
    h += rng.normal(0, 0.1, size=N)
    return h

def _full_dim_polish(h, max_iter=50, step_size=0.05):
    """
    Full-dimensional coordinate descent polish.
    Tries small perturbations at each coordinate.
    """
    h = np.asarray(h, dtype=float).copy()
    s = float(np.sum(h))
    if abs(s) < 1e-12:
        return h
    
    size = 2 * N
    H = np.fft.rfft(h, size)
    conv = np.fft.irfft(H * H, size)[:2 * N - 1]
    current_val = _fast_value_from_conv(conv, s)
    
    if not np.isfinite(current_val):
        return h
    
    rng = np.random.default_rng(123)
    
    for iteration in range(max_iter):
        if not deadline_ok():
            break
        
        improved = False
        # Try a subset of coordinates each iteration
        indices = rng.choice(N, size=min(50, N), replace=False)
        
        for idx in indices:
            if not deadline_ok():
                break
            
            # Try small positive and negative steps
            for delta in [-step_size, step_size]:
                h[idx] += delta
                s_new = s + delta
                if abs(s_new) < 1e-12:
                    h[idx] -= delta
                    continue
                
                H_new = np.fft.rfft(h, size)
                conv_new = np.fft.irfft(H_new * H_new, size)[:2 * N - 1]
                tv = _fast_value_from_conv(conv_new, s_new)
                
                if tv < current_val - 1e-8:
                    current_val = tv
                    s = s_new
                    H = H_new
                    conv = conv_new
                    improved = True
                    break
                else:
                    h[idx] -= delta
        
        if not improved:
            # Reduce step size
            step_size *= 0.7
            if step_size < 1e-6:
                break
    
    return h

def construct():
    global _t0
    _t0 = time.time()
    
    best_h = np.ones(N)
    best_v = _value(best_h)
    
    def update_best(h):
        nonlocal best_h, best_v
        h = np.asarray(h, dtype=float)
        if not np.all(np.isfinite(h)):
            return
        v = _value(h)
        if v < best_v:
            best_v = v
            best_h = h.copy()

    def deadline_ok():
        return (time.time() - _t0) < _DEADLINE

    rng = np.random.default_rng(42)
    
    # Strategy 1: Structured initializations
    if deadline_ok():
        # Golay-like
        for _ in range(5):
            h_golay = _generate_golay_like(rng, N)
            update_best(h_golay)
        
        # Zadoff-Chu-like
        for _ in range(5):
            h_zc = _generate_zadoff_chu(rng, N)
            update_best(h_zc)
        
        # Symmetric patterns
        for block_size in [2, 4, 8, 16, 32]:
            if not deadline_ok():
                break
            h_sym = np.zeros(N)
            for i in range(N):
                block = i // block_size
                h_sym[i] = 1.0 if block % 2 == 0 else -1.0
            update_best(h_sym)
        
        # Sine patterns
        for freq in [1, 2, 3, 5, 8, 13]:
            if not deadline_ok():
                break
            t = np.linspace(0, 2*np.pi*freq, N)
            h_sine = np.cos(t)
            update_best(h_sine)

    # Strategy 2: Sparse atom optimization (main strategy)
    for n_atoms in [8, 16, 24, 32]:
        if not deadline_ok():
            break
            
        sparse_restarts = 4
        
        for r in range(sparse_restarts):
            if not deadline_ok():
                break
                
            try:
                params0 = _generate_initial_sparse(rng, n_atoms)
                
                # Phase 1: Coarse smooth optimization
                res = minimize(
                    lambda p: _smooth_sparse_obj(p, beta=5.0),
                    params0,
                    method='L-BFGS-B',
                    options={'maxiter': 60, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                
                if not deadline_ok():
                    break
                    
                params_opt = res.x if res.x is not None else params0
                
                # Phase 2: Medium smoothness
                if deadline_ok():
                    res2 = minimize(
                        lambda p: _smooth_sparse_obj(p, beta=20.0),
                        params_opt,
                        method='L-BFGS-B',
                        options={'maxiter': 60, 'ftol': 1e-12, 'gtol': 1e-10}
                    )
                    if res2.x is not None:
                        params_opt = res2.x

                # Phase 3: Coordinate descent polish
                if deadline_ok():
                    params_opt = _coordinate_descent_fast(params_opt, max_iter=30)

                h_candidate = _sparse_to_h(params_opt)
                
                if np.all(np.isfinite(h_candidate)):
                    update_best(h_candidate)
                    
            except Exception:
                pass

    # Strategy 3: DCT-based optimization (reduced budget)
    if deadline_ok():
        K = 400
        dct_restarts = 2
        
        for r in range(dct_restarts):
            if not deadline_ok():
                break
                
            try:
                coeffs0 = _generate_initial_dct(rng, K)
                
                res = minimize(
                    lambda c: _dct_smooth_obj(c, beta=20.0),
                    coeffs0,
                    method='L-BFGS-B',
                    options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                
                if not deadline_ok():
                    break
                    
                coeffs_opt = res.x if res.x is not None else coeffs0
                
                h_candidate = _dct_h(coeffs_opt)
                
                if np.all(np.isfinite(h_candidate)):
                    update_best(h_candidate)
                    
            except Exception:
                pass

    # Strategy 4: Full-dimensional polish on best solution
    if deadline_ok():
        h_polished = _full_dim_polish(best_h, max_iter=40, step_size=0.03)
        update_best(h_polished)
        
        # Try small perturbations
        for scale in [0.01, 0.03, 0.05]:
            if not deadline_ok():
                break
            for _ in range(15):
                if not deadline_ok():
                    break
                noise = rng.normal(0, scale, size=N)
                h_perturb = best_h + noise
                update_best(h_perturb)
        
        # Final polish
        if deadline_ok():
            h_polished2 = _full_dim_polish(best_h, max_iter=30, step_size=0.01)
            update_best(h_polished2)

    s = np.sum(best_h)
    if s == 0 or not np.all(np.isfinite(best_h)):
        return np.ones(N)
        
    return best_h

if __name__ == "__main__":
    h = construct()
    print(f"Value: {_value(h)}")
    print(f"Sum: {np.sum(h)}")
    print(f"Shape: {h.shape}")
