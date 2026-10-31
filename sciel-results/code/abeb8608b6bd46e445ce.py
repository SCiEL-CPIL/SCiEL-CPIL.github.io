import numpy as np
import time
import scipy.optimize
import scipy.signal

N = 400

def _compute_bound(f):
    """
    Computes the peak sidelobe magnitude of the linear autocorrelation.
    """
    n_fft = 2 * N
    F = np.fft.rfft(f, n=n_fft)
    PSD = F * np.conj(F)
    R = np.fft.irfft(PSD, n=n_fft)
    # R[0] is the peak (energy)
    # R[1:N] are the sidelobes
    return np.max(np.abs(R[1:N]))

def _objective_l2(f):
    """
    Objective function for optimization: Sum of squared sidelobes.
    This is differentiable and smooth, suitable for L-BFGS-B.
    """
    n_fft = 2 * N
    F = np.fft.rfft(f, n=n_fft)
    PSD = F * np.conj(F)
    R = np.fft.irfft(PSD, n=n_fft)
    sidelobes = R[1:N]
    return np.sum(sidelobes**2)

def _project_sign(f):
    """
    Project a real-valued vector to signed step function values (+1/-1).
    """
    return np.sign(f)

def _local_search(f, max_iter=100, time_limit=5):
    """
    Local search: try flipping individual samples to reduce peak sidelobe.
    """
    start_time = time.time()
    current_f = f.copy()
    current_bound = _compute_bound(current_f)
    
    for iteration in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        improved = False
        # Try flipping each sample
        for i in range(N):
            old_val = current_f[i]
            current_f[i] = -old_val
            new_bound = _compute_bound(current_f)
            if new_bound < current_bound:
                current_bound = new_bound
                improved = True
            else:
                current_f[i] = old_val
        
        if not improved:
            break
    
    return current_f, current_bound

def _gen_m_sequence(k=9):
    """
    Generate an m-sequence of length 2^k - 1, then pad/truncate to N samples.
    """
    length = (1 << k) - 1
    state = 1
    seq = []
    # For k=9, use primitive polynomial x^9 + x^5 + 1
    # Taps at 5 and 0 (0-indexed from LSB)
    for _ in range(length):
        seq.append(state & 1)
        fb = ((state >> 4) ^ state) & 1
        state = (state >> 1) | (fb << (k - 1))
    
    # Map 0->-1, 1->1
    mseq = 2 * np.array(seq, dtype=float) - 1
    
    # Truncate or pad to N
    if len(mseq) >= N:
        return mseq[:N]
    else:
        padded = np.zeros(N)
        padded[:len(mseq)] = mseq
        return padded

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 150s kill
    
    best_f = None
    best_score = float('inf')
    
    # List of initial guesses
    initial_guesses = []
    
    # 1. Kaiser windows with different beta values
    for beta in [4, 6, 8, 10, 12]:
        f_kaiser = scipy.signal.windows.kaiser(N, beta=beta)
        initial_guesses.append(f_kaiser)
        
    # 2. Gaussian window
    x = np.linspace(-1, 1, N)
    f_gauss = np.exp(-3 * x**2)
    initial_guesses.append(f_gauss)
    
    # 3. M-sequence
    f_mseq = _gen_m_sequence(9)
    initial_guesses.append(f_mseq)
    
    # 4. Random signed sequences
    rng = np.random.default_rng(42)
    for _ in range(3):
        f_rand = rng.choice([-1, 1], size=N).astype(float)
        initial_guesses.append(f_rand)
    
    # Optimize each initial guess
    for i, f_init in enumerate(initial_guesses):
        if time.time() > deadline - 10:
            break
            
        # Normalize initial guess
        norm = np.linalg.norm(f_init)
        if norm > 0:
            f_init = f_init / norm
        
        # Run L-BFGS-B optimization on L2 objective
        try:
            res = scipy.optimize.minimize(
                _objective_l2, 
                f_init, 
                method='L-BFGS-B',
                options={'maxiter': 30, 'ftol': 1e-10, 'gtol': 1e-6}
            )
            f_opt = res.x
        except Exception:
            f_opt = f_init
        
        # Project to signed step function
        f_proj = _project_sign(f_opt)
        
        # Evaluate true bound
        score = _compute_bound(f_proj)
        
        if score < best_score:
            best_score = score
            best_f = f_proj.copy()
            
        # Local search refinement on the projected solution
        if time.time() < deadline - 5:
            f_ls, score_ls = _local_search(f_proj, max_iter=50, time_limit=3)
            if score_ls < best_score:
                best_score = score_ls
                best_f = f_ls.copy()
                
    # Final local search on the best found
    if best_f is not None and time.time() < deadline - 2:
        f_final, score_final = _local_search(best_f, max_iter=100, time_limit=5)
        if score_final < best_score:
            best_score = score_final
            best_f = f_final.copy()
    
    # Fallback
    if best_f is None:
        best_f = np.ones(N)
    
    return best_f
