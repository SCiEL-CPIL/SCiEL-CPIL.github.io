"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _generate_de_bruijn_sequence(n):
    """
    Generate a de Bruijn sequence of length n over alphabet {0,1}.
    A de Bruijn sequence B(k,n) contains every possible k-length sequence
    exactly once in a cyclic sequence of length n.
    
    For our purpose, we want a binary sequence of length N with good
    autocorrelation properties. We use a de Bruijn torus approach:
    generate a de Bruijn sequence of order k where 2^k >= N, then take
    the first N bits.
    
    The de Bruijn sequence has the property that all k-bit subsequences
    appear exactly once, which distributes autocorrelation energy evenly.
    """
    # Find smallest k such that 2^k >= N
    k = 1
    while (2 ** k) < N:
        k += 1
    
    # Generate de Bruijn sequence using the standard algorithm
    # B(k, 2) - de Bruijn sequence for binary alphabet, order k
    a = [0] * (2 * (2 ** k))
    sequence = []
    
    def db(t, p):
        if t > k:
            if k % p == 0:
                sequence.extend(a[1:p+1])
        else:
            a[t] = a[t - p]
            db(t + 1, p)
            for j in range(a[t - p] + 1, 2):
                a[t] = j
                db(t + 1, t)
    
    db(1, 1)
    
    # The de Bruijn sequence is cyclic with period 2^k
    # Take the first N bits
    de_bruijn = sequence[:N]
    
    # Convert to signed: 0 -> -1, 1 -> +1
    signed = np.where(np.array(de_bruijn) == 1, 1.0, -1.0)
    return signed


def _compute_acf_fft(s):
    """Compute autocorrelation using FFT."""
    n = len(s)
    n_fft = 2 * n
    f = np.fft.rfft(s, n=n_fft)
    acf = np.fft.irfft(f * np.conj(f), n=n_fft)[:n]
    return acf


def _peak_sidelobe(acf):
    """Peak absolute sidelobe (excluding main lobe at lag 0)."""
    if len(acf) <= 1:
        return 0.0
    return np.max(np.abs(acf[1:]))


def _c3_upper_bound(h):
    """
    Compute the C3 upper bound for a signed step function h.
    Metric: max_k |(h*h)_k| / (sum_i |h_i|)^2
    """
    n = len(h)
    n_fft = 2 * n
    H = np.fft.rfft(h, n=n_fft)
    conv = np.fft.irfft(H * H, n=n_fft)[:n]
    
    abs_conv = np.abs(conv)
    sum_abs_h = np.sum(np.abs(h))
    
    if sum_abs_h < 1e-10:
        return np.inf
    
    return np.max(abs_conv) / (sum_abs_h ** 2)


def _discrete_refine(s, deadline, max_iterations=5000):
    """
    Discrete refinement: iteratively flip bits to reduce the peak sidelobe.
    Uses FFT-based autocorrelation for efficiency.
    """
    acf = _compute_acf_fft(s)
    current_val = _peak_sidelobe(acf)
    best_val = current_val
    best_s = s.copy()
    
    no_improve_count = 0
    
    for iteration in range(max_iterations):
        if time.time() > deadline - 3:
            break
        
        improved = False
        best_flip_val = current_val
        best_flip_idx = -1
        
        # Try flipping each position
        for i in range(N):
            # Compute new autocorrelation after flipping position i
            si = s[i]
            new_acf = acf.copy()
            
            # For lag k: new_acf[k] = acf[k] - 2*si*s[i+k] (for i+k < N)
            # and new_acf[k] = acf[k] - 2*s[i-k]*si (for i-k >= 0)
            k_max = N - i
            if k_max > 0:
                new_acf[:k_max] -= 2 * si * s[i:i+k_max]
            if i > 0:
                new_acf[:i+1] -= 2 * s[:i+1] * si
            
            new_val = _peak_sidelobe(new_acf)
            
            if new_val < best_flip_val - 1e-12:
                best_flip_val = new_val
                best_flip_idx = i
        
        if best_flip_idx >= 0:
            s[best_flip_idx] = -s[best_flip_idx]
            acf = _compute_acf_fft(s)
            current_val = _peak_sidelobe(acf)
            improved = True
            
            if current_val < best_val:
                best_val = current_val
                best_s = s.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            no_improve_count += 1
        
        if no_improve_count > 100:
            break
    
    return best_s, best_val


def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin: 150s limit, keep 5s buffer
    
    best_seq = None
    best_val = np.inf
    
    # Step 1: Generate de Bruijn sequence as deterministic starting point
    de_bruijn_seq = _generate_de_bruijn_sequence(N)
    
    # Evaluate the de Bruijn sequence
    db_val = _c3_upper_bound(de_bruijn_seq)
    if db_val < best_val:
        best_val = db_val
        best_seq = de_bruijn_seq.copy()
    
    # Step 2: Continuous relaxation using L-BFGS-B
    # Parameterize h = tanh(x) so h is in (-1, 1)
    # Start from the de Bruijn sequence mapped to x-space
    
    def objective(x):
        h = np.tanh(x)
        n = len(h)
        n_fft = 2 * n
        H = np.fft.rfft(h, n=n_fft)
        conv = np.fft.irfft(H * H, n=n_fft)[:n]
        
        abs_conv = np.sqrt(conv**2 + 1e-10)
        
        # Soft max for differentiability
        T = 3.0
        max_abs = np.max(abs_conv)
        if max_abs < 1e-10:
            return 1e10
        
        shifted = (abs_conv - max_abs) / T
        log_sum_exp = np.log(np.sum(np.exp(shifted))) + max_abs / T
        soft_max = T * log_sum_exp
        
        sum_abs_h = np.sum(np.sqrt(h**2 + 1e-10))
        
        if sum_abs_h < 1e-10:
            return 1e10
        
        return soft_max / (sum_abs_h ** 2)
    
    # Initialize x from de Bruijn sequence
    # tanh(x) = h => x = atanh(h)
    # For h = ±1, atanh is infinite, so clip
    h_init = de_bruijn_seq.copy()
    h_clipped = np.clip(h_init, -0.999, 0.999)
    x0 = np.arctanh(h_clipped)
    
    # Run L-BFGS-B optimization
    try:
        res = minimize(objective, x0, method='L-BFGS-B',
                       options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8})
        
        h_opt = np.tanh(res.x)
        
        # Evaluate continuous solution
        cont_val = _c3_upper_bound(h_opt)
        if cont_val < best_val:
            best_val = cont_val
            best_seq = h_opt.copy()
        
        # Step 3: Discretize the continuous solution
        # Round to nearest ±1
        h_discrete = np.where(h_opt >= 0, 1.0, -1.0)
        
        disc_val = _c3_upper_bound(h_discrete)
        if disc_val < best_val:
            best_val = disc_val
            best_seq = h_discrete.copy()
        
        # Step 4: Discrete refinement from the discretized continuous solution
        if time.time() < deadline - 5:
            refined_seq, refined_val = _discrete_refine(h_discrete.copy(), deadline)
            if refined_val < best_val:
                best_val = refined_val
                best_seq = refined_seq.copy()
        
    except Exception:
        pass
    
    # Step 5: Also try discrete refinement from the original de Bruijn sequence
    if time.time() < deadline - 5:
        refined_db, refined_db_val = _discrete_refine(de_bruijn_seq.copy(), deadline)
        if refined_db_val < best_val:
            best_val = refined_db_val
            best_seq = refined_db.copy()
    
    # Step 6: Try a few random restarts with discrete refinement if time permits
    np.random.seed(42)
    for restart in range(10):
        if time.time() > deadline - 5:
            break
        
        s_random = np.random.choice([-1.0, 1.0], size=N)
        rand_val = _c3_upper_bound(s_random)
        
        if rand_val < best_val:
            best_val = rand_val
            best_seq = s_random.copy()
        
        if time.time() < deadline - 5:
            refined_r, refined_r_val = _discrete_refine(s_random.copy(), deadline)
            if refined_r_val < best_val:
                best_val = refined_r_val
                best_seq = refined_r.copy()
    
    if best_seq is None:
        return np.ones(N)
    
    return best_seq
# EVOLVE-BLOCK-END
