import numpy as np
import time
from scipy.optimize import minimize

N = 400
TIME_LIMIT = 120.0
SAFETY_MARGIN = 5.0
DEADLINE = time.time() + TIME_LIMIT - SAFETY_MARGIN

# Precompute FFT size for autoconvolution
FFT_SIZE = 1
while FFT_SIZE < 2 * N - 1:
    FFT_SIZE *= 2


def _autoconv(f):
    """Compute the linear (full) autoconvolution of f using FFT."""
    F = np.fft.rfft(f, n=FFT_SIZE)
    C = np.fft.irfft(F * F, n=FFT_SIZE)
    return C[:2 * N - 1]


def _objective_vectorized(f):
    """
    Compute max(|autoconv(f)|) / (sum(f))^2.
    If sum(f) == 0, return a large penalty.
    """
    S = np.sum(f)
    if abs(S) < 1e-12:
        return 1e10
    C = _autoconv(f)
    return np.max(np.abs(C)) / (S * S)


def _objective_smooth(f, t=0.01):
    """
    Smooth surrogate using log-sum-exp for the max.
    """
    S = np.sum(f)
    if abs(S) < 1e-12:
        return 1e10
    C = _autoconv(f)
    absC = np.abs(C)
    max_val = np.max(absC)
    lse = max_val + np.log(np.sum(np.exp(t * (absC - max_val))))
    return (lse / t) / (S * S)


def _gradient_smooth(f, t=0.01):
    """
    Gradient of the smooth objective using FFT-based approach.
    d/d f_i [max_k |C(k)|] where C = f * f
    C(k) = sum_j f(j) f(j+k-i) ... using correlation form
    """
    S = np.sum(f)
    if abs(S) < 1e-12:
        return np.zeros_like(f)
    
    C = _autoconv(f)
    absC = np.abs(C)
    signC = np.sign(C)
    max_val = np.max(absC)
    
    exp_term = np.exp(t * (absC - max_val))
    sum_exp = np.sum(exp_term)
    weights = exp_term / sum_exp
    
    A = weights * signC  # length 2N-1
    
    # Gradient of C(k) w.r.t. f[i]: C(k) = sum_j f(j)*f(j+k)
    # dC(k)/df[i] = sum over pairs where i is involved
    # Using FFT: if A is the weight vector for C, then grad[i] = 2 * (A * f)[i] in correlation sense
    # Actually: C(k) = sum_j f(j) f(j+k), so dC(k)/df(i) = f(i+k) + f(i-k) (with bounds)
    # So grad[i] = sum_k A[k] * (f(i+k) + f(i-k))
    # This is equivalent to 2 * (A_convolve_f) at appropriate indices
    
    # Compute using FFT for efficiency
    # A has length 2N-1, f has length N
    # We need: for each i in [0,N), grad[i] = sum_k A[k] * f[i+k] + A[k] * f[i-k]
    # = 2 * sum_k A[k] * f[|i-k|] ... let me think more carefully
    
    # C(k) = sum_{j=0}^{N-1} f(j) f(j+k) for k in [-(N-1), N-1]
    # dC(k)/df(i) = f(i+k) if 0<=i+k<N, plus f(i-k) if 0<=i-k<N
    # So grad[i] = sum_k A[k] * (f(i+k) * 1[0<=i+k<N] + f(i-k) * 1[0<=i-k<N])
    
    # This is a convolution-like operation. Let's compute it directly but efficiently.
    # For each i, we sum over k from -(N-1) to (N-1):
    #   A[k+N-1] * f[i+k] (if valid) + A[k+N-1] * f[i-k] (if valid)
    
    # Use FFT: define B as the "correlation kernel"
    # grad[i] = sum_k A[k+N-1] * f[(i+k) mod N] ... no, it's linear not circular
    
    # Let's just do it with numpy operations efficiently
    # Create padded A for the two terms
    # Term 1: sum_k A[k] * f[i+k] for valid i+k
    # Term 2: sum_k A[k] * f[i-k] for valid i-k
    
    # Using FFT for linear convolution:
    # term1[i] = sum_k A[k] * f[i+k] = (A_reversed * f)[i + N - 1] ... 
    # Let me use a simpler approach: np.convolve
    
    # A has indices 0 to 2N-2, corresponding to k = -(N-1) to N-1
    # For term1: sum over k where i+k in [0,N-1], A[k+N-1] * f[i+k]
    #   Let j = i+k, so k = j-i, A[j-i+N-1] * f[j], sum over j where j-i+N-1 in [0,2N-2]
    #   i.e., j in [i-(N-1), i+N-1] intersect [0,N-1]
    #   This is a sliding dot product
    
    # Simpler: just compute directly with vectorized ops
    grad = np.zeros(N)
    # For each offset k, A[k+N-1] contributes to all i where both i and i+k are in [0,N)
    for k in range(2 * N - 1):
        a_val = A[k]
        if abs(a_val) < 1e-15:
            continue
        k_offset = k - (N - 1)  # actual lag
        if k_offset >= 0:
            # C(k) = sum_j f(j) f(j+k), dC/df(i) gets f(i+k) contribution
            # i ranges [0, N-1-k], i+k ranges [k, N-1]
            n_valid = N - k_offset
            if n_valid > 0:
                grad[:n_valid] += a_val * f[k_offset:k_offset + n_valid]
                # Also f(i-k) term: i-k = i - k_offset, valid when i >= k_offset
                # i ranges [k_offset, N-1]
                grad[k_offset:] += a_val * f[:n_valid]
        else:
            k_pos = -k_offset
            n_valid = N - k_pos
            if n_valid > 0:
                # k is negative lag: C(-k_pos) = sum_j f(j+k_pos) f(j)
                # dC/df(i) gets f(i-k_pos) from first term and f(i+k_pos) from second
                grad[:n_valid] += a_val * f[k_pos:k_pos + n_valid]
                grad[k_pos:] += a_val * f[:n_valid]
    
    LSE_val = max_val + np.log(np.sum(np.exp(t * (absC - max_val))))
    G_val = LSE_val / t
    
    return (1.0 / (S * S)) * grad - (G_val / (S * S * S)) * 2.0 * S


def _local_search_binary(f, max_iter=2000, time_budget=8.0):
    """
    Local search on binary {-1, 1} sequences.
    Tries single flips and pair flips, accepting if it improves the objective.
    """
    f = f.copy()
    S = np.sum(f)
    C = _autoconv(f)
    score = np.max(np.abs(C)) / (S * S)
    
    start_time = time.time()
    no_improve_count = 0
    
    for it in range(max_iter):
        if time.time() - start_time > time_budget:
            break
        if time.time() > DEADLINE:
            break
        
        k_max = np.argmax(np.abs(C))
        
        # Collect candidate positions for single flips
        candidates_single = set()
        for _ in range(15):
            candidates_single.add(np.random.randint(N))
        
        for j in range(N):
            if 0 <= k_max - j < N:
                candidates_single.add(j)
        
        # Collect candidate pairs for pair flips
        candidates_pair = set()
        for _ in range(8):
            i1 = np.random.randint(N)
            i2 = np.random.randint(N)
            if i1 != i2:
                candidates_pair.add((min(i1, i2), max(i1, i2)))
        
        # Add pairs near the peak
        for j in range(min(20, N)):
            if 0 <= k_max - j < N and j != k_max - j:
                candidates_pair.add((min(j, k_max - j), max(j, k_max - j)))
        
        best_improvement = 0
        best_move = None  # (type, indices)
        
        # Try single flips
        for i in candidates_single:
            v = f[i]
            k_start = i
            k_end = min(i + N, 2 * N - 1)
            k_end_clamped = min(k_end, len(C))
            f_slice = f[:k_end_clamped - k_start]
            
            delta_C = np.zeros_like(C)
            delta_C[k_start:k_end_clamped] = -2.0 * v * f_slice
            
            new_C = C + delta_C
            new_S = S - 2.0 * v
            
            if abs(new_S) < 1e-6:
                continue
            
            new_score = np.max(np.abs(new_C)) / (new_S * new_S)
            
            if new_score < score - best_improvement:
                best_improvement = score - new_score
                best_move = ('single', (i,))
        
        # Try pair flips
        for (i, j) in candidates_pair:
            vi, vj = f[i], f[j]
            
            delta_C = np.zeros_like(C)
            # Flip i: delta at positions i..i+N-1
            k_start_i = i
            k_end_i = min(i + N, 2 * N - 1)
            delta_C[k_start_i:k_end_i] += -2.0 * vi * f[:k_end_i - k_start_i]
            # Flip j: delta at positions j..j+N-1
            k_start_j = j
            k_end_j = min(j + N, 2 * N - 1)
            delta_C[k_start_j:k_end_j] += -2.0 * vj * f[:k_end_j - k_start_j]
            
            new_C = C + delta_C
            new_S = S - 2.0 * vi - 2.0 * vj
            
            if abs(new_S) < 1e-6:
                continue
            
            new_score = np.max(np.abs(new_C)) / (new_S * new_S)
            
            if new_score < score - best_improvement:
                best_improvement = score - new_score
                best_move = ('pair', (i, j))
        
        if best_move is not None:
            if best_move[0] == 'single':
                i = best_move[1][0]
                f[i] = -f[i]
                S -= 2.0 * f[i] * -1  # S -= 2*v, v was old value
                # Actually: old v = f[i], new f[i] = -v, so S changes by -2*v
                # But we already flipped, so let's recompute
                S = np.sum(f)
            else:
                i, j = best_move[1]
                f[i] = -f[i]
                f[j] = -f[j]
                S = np.sum(f)
            
            C = _autoconv(f)
            score = np.max(np.abs(C)) / (S * S)
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count > 50:
                break
    
    return f, score


def _local_search_real(f, max_iter=2000, time_budget=8.0, step=0.5):
    """
    Local search on real-valued sequences.
    Adjusts individual values by ±step, accepting if it improves the objective.
    """
    f = f.copy()
    S = np.sum(f)
    C = _autoconv(f)
    score = np.max(np.abs(C)) / (S * S)
    
    start_time = time.time()
    no_improve_count = 0
    
    for it in range(max_iter):
        if time.time() - start_time > time_budget:
            break
        if time.time() > DEADLINE:
            break
        
        best_improvement = 0
        best_change = None
        
        # Try a random subset of positions
        trial_positions = np.random.choice(N, size=min(50, N), replace=False)
        
        for i in trial_positions:
            for delta in [-step, step]:
                f[i] += delta
                new_S = S + delta
                if abs(new_S) < 1e-6:
                    f[i] -= delta
                    continue
                
                C = _autoconv(f)
                new_score = np.max(np.abs(C)) / (new_S * new_S)
                
                if new_score < score - best_improvement:
                    best_improvement = score - new_score
                    best_change = (i, delta)
                
                f[i] -= delta
                S = np.sum(f)
        
        if best_change is not None:
            i, delta = best_change
            f[i] += delta
            S += delta
            C = _autoconv(f)
            score = np.max(np.abs(C)) / (S * S)
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count > 30:
                step *= 0.7
                if step < 0.01:
                    break
    
    return f, score


def _spectral_init(rng, N):
    """
    Initialize using windowed sinc in frequency domain to structurally
    suppress peak autoconvolution sidelobes.
    """
    # Design a frequency-domain window that minimizes peak sidelobes
    # Use Blackman-Harris window
    n = N
    w = np.zeros(n)
    for i in range(n):
        x = i / (n - 1)
        # Blackman-Harris 4-term
        w[i] = (0.35875 - 0.48829 * np.cos(2 * np.pi * x) +
                0.14128 * np.cos(4 * np.pi * x) - 0.01168 * np.cos(6 * np.pi * x))
    
    # Create a random phase pattern
    phases = rng.uniform(0, 2 * np.pi, n)
    
    # Frequency domain signal
    freq_signal = w * np.exp(1j * phases)
    
    # Convert to time domain
    time_signal = np.fft.ifft(freq_signal).real
    
    # Sign it to get binary-like sequence
    f = np.sign(time_signal)
    f[f == 0] = 1.0
    
    # Ensure non-zero sum
    S = np.sum(f)
    if abs(S) < N * 0.05:
        # Adjust some signs to increase |S|
        for _ in range(N // 4):
            idx = rng.integers(0, N)
            if f[idx] * S < 0:
                f[idx] = -f[idx]
                S = np.sum(f)
            if abs(S) > N * 0.2:
                break
    
    return f


def construct():
    start_time = time.time()
    
    best_f = np.ones(N)
    best_score = _objective_vectorized(best_f)
    
    rng = np.random.default_rng(42)
    
    # Spectral initialization for diverse starting points
    spectral_starts = []
    for _ in range(6):
        if time.time() > DEADLINE:
            break
        f_spec = _spectral_init(rng, N)
        spectral_starts.append(f_spec)
    
    # Continuous optimization with smooth objective
    for restart in range(4):
        if time.time() > DEADLINE:
            break
        
        if restart < len(spectral_starts):
            f0 = spectral_starts[restart].copy()
        else:
            f0 = rng.uniform(-1, 1, N)
        
        if abs(np.sum(f0)) > 1e-6:
            f0 = f0 / np.sum(f0)
        else:
            f0 = np.ones(N)
        
        ts = [0.1, 0.05, 0.01, 0.005]
        current_f = f0
        
        for t in ts:
            if time.time() > DEADLINE:
                break
            
            try:
                res = minimize(
                    _objective_smooth,
                    current_f,
                    args=(t,),
                    method='L-BFGS-B',
                    jac=_gradient_smooth,
                    options={'maxiter': 40, 'maxls': 40, 'ftol': 1e-9, 'gtol': 1e-7}
                )
                
                if res.fun < best_score * 1.01:
                    true_score = _objective_vectorized(res.x)
                    if true_score < best_score:
                        best_score = true_score
                        best_f = res.x.copy()
                    current_f = res.x
            except Exception:
                continue
    
    # Binary local search with diverse initializations
    binary_starts = []
    for _ in range(10):
        if time.time() > DEADLINE:
            break
        f0 = rng.choice([-1.0, 1.0], size=N)
        S0 = np.sum(f0)
        
        if abs(S0) < 5:
            for _ in range(N // 2):
                idx = rng.integers(0, N)
                if f0[idx] * S0 < 0:
                    f0[idx] = -f0[idx]
                    S0 = np.sum(f0)
            if abs(S0) < 5:
                continue
        binary_starts.append(f0)
    
    # Also add spectral-based binary starts
    for f_spec in spectral_starts:
        if time.time() > DEADLINE:
            break
        f_bin = np.sign(f_spec)
        f_bin[f_bin == 0] = 1.0
        S_bin = np.sum(f_bin)
        if abs(S_bin) >= 5:
            binary_starts.append(f_bin)
    
    for f0 in binary_starts:
        if time.time() > DEADLINE:
            break
        
        remaining_time = DEADLINE - time.time()
        time_budget = min(6.0, remaining_time * 0.7)
        
        if time_budget < 0.5:
            break
        
        f_local, score_local = _local_search_binary(f0, max_iter=5000, time_budget=time_budget)
        
        if score_local < best_score:
            best_score = score_local
            best_f = f_local.copy()
    
    # Real-valued local search from best found
    if time.time() < DEADLINE:
        remaining_time = DEADLINE - time.time()
        if remaining_time > 8:
            time_budget = min(8.0, remaining_time * 0.8)
            f_start = best_f.copy()
            f_local, score_local = _local_search_real(f_start, max_iter=3000, time_budget=time_budget, step=0.5)
            if score_local < best_score:
                best_score = score_local
                best_f = f_local.copy()
    
    # Final polish: try small perturbations around best
    if time.time() < DEADLINE:
        remaining_time = DEADLINE - time.time()
        if remaining_time > 5:
            f_polish = best_f.copy()
            S = np.sum(f_polish)
            C = _autoconv(f_polish)
            score = np.max(np.abs(C)) / (S * S)
            
            for _ in range(200):
                if time.time() > DEADLINE:
                    break
                idx = rng.integers(0, N)
                delta = rng.choice([-0.1, 0.1, -0.2, 0.2])
                f_polish[idx] += delta
                new_S = np.sum(f_polish)
                if abs(new_S) < 1e-6:
                    f_polish[idx] -= delta
                    continue
                new_C = _autoconv(f_polish)
                new_score = np.max(np.abs(new_C)) / (new_S * new_S)
                if new_score < score:
                    score = new_score
                else:
                    f_polish[idx] -= delta
            
            if score < best_score:
                best_score = score
                best_f = f_polish.copy()
    
    return best_f
