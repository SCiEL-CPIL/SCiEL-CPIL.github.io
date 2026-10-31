import numpy as np
import time

N = 400
TIME_LIMIT = 150.0
SAFETY_MARGIN = 10.0
DEADLINE = time.time() + TIME_LIMIT - SAFETY_MARGIN

# Precompute FFT size
FFT_SIZE = 1
while FFT_SIZE < 2 * N - 1:
    FFT_SIZE *= 2

def _autoconv_fft(f):
    """Compute linear autoconvolution using FFT."""
    F = np.fft.rfft(f, n=FFT_SIZE)
    C = np.fft.irfft(F * F, n=FFT_SIZE)
    return C[:2 * N - 1]

def _score(f):
    S = np.sum(f)
    if abs(S) < 1e-12:
        return 1e10
    C = _autoconv_fft(f)
    return np.max(np.abs(C)) / (S * S)

def _local_search_binary_fast(f, max_iter=1000, time_budget=5.0):
    """
    Local search on binary {-1, 1} sequences using vectorized flip evaluation.
    """
    f = np.sign(f).astype(np.float64)
    S = np.sum(f)
    if abs(S) < 1e-6:
        return f, 1e10

    C = _autoconv_fft(f)
    score = np.max(np.abs(C)) / (S * S)

    start_time = time.time()
    no_improve_count = 0

    for it in range(max_iter):
        if time.time() - start_time > time_budget:
            break
        if time.time() > DEADLINE:
            break

        # Vectorized evaluation of all single-flip candidates
        # For flip at position i: delta = -2*f[i]
        # C_new[k] = C[k] + 2*delta_i*f[k-i] for k in [i, i+N-1]
        #            + delta_i^2 at k=2i (if 2i < 2N-1)
        # We build a matrix of shape (N, 2N-1) where row i is the convolution
        # contribution from flipping position i.

        deltas = -2.0 * f  # shape (N,)

        # Build the flip-contribution matrix M[i, k] = contribution to C[k] from flipping f[i]
        # M[i, k] = 2 * deltas[i] * f[k-i] for i <= k < i+N, plus deltas[i]^2 at k=2i
        # Use FFT-based approach: for each i, the cross-convolution term is
        # 2*delta_i * (f convolved with unit impulse at i) which is just a shifted f.
        # Vectorize by building a Toeplitz-like structure.

        # Actually, let's use a different vectorization:
        # The change in C from flipping position i is:
        # dC[k] = 2*delta_i * f[k-i] + delta_i^2 * delta(k-2i)
        # where delta(k-2i) is 1 if k==2i else 0.
        #
        # We can compute all dC vectors at once using matrix operations.
        # Let's build a matrix A of shape (N, 2N-1) where A[i, k] = f[k-i] if 0 <= k-i < N else 0
        # This is a Hankel matrix of f.

        # Build Hankel matrix H where H[i, k] = f[k-i] for valid k-i
        # H[i, k] is nonzero when i <= k < i+N
        idx = np.arange(2 * N - 1)
        # For row i, valid columns are i to i+N-1
        # H[i, k] = f[k - i] if i <= k < i + N else 0
        # We can build this using broadcasting
        rows = np.arange(N)[:, None]  # (N, 1)
        cols = idx[None, :]           # (1, 2N-1)
        valid = (cols >= rows) & (cols < rows + N)  # (N, 2N-1)
        # f[k-i] where valid
        # k-i ranges from 0 to N-1
        diff = cols - rows  # (N, 2N-1)
        # Clip to valid range for indexing
        diff_clipped = np.clip(diff, 0, N - 1)
        H = np.where(valid, f[diff_clipped], 0.0)  # (N, 2N-1)

        # dC matrix: dC[i, k] = 2 * deltas[i] * H[i, k]
        dC_cross = 2.0 * deltas[:, None] * H  # (N, 2N-1)

        # Add the delta_i^2 term at k = 2i
        # k = 2i is valid if 2i < 2N-1, i.e., i < N - 0.5, always true for i < N
        two_i = 2 * np.arange(N)
        # dC_cross[i, 2i] += deltas[i]^2
        dC_cross[np.arange(N), two_i] += deltas**2

        # Full candidate convolutions: C_candidates[i, :] = C + dC_cross[i, :]
        # Shape: (N, 2N-1)
        C_candidates = C[None, :] + dC_cross  # (N, 2N-1)

        # New sums: new_S[i] = S - 2*f[i]
        new_S = S - 2.0 * f  # (N,)

        # Compute scores for all candidates
        # score[i] = max(|C_candidates[i]|) / (new_S[i])^2
        max_abs_C = np.max(np.abs(C_candidates), axis=1)  # (N,)
        valid_mask = np.abs(new_S) > 1e-6
        scores = np.full(N, np.inf)
        scores[valid_mask] = max_abs_C[valid_mask] / (new_S[valid_mask] ** 2)

        # Find best flip
        best_idx = np.argmin(scores)
        if scores[best_idx] < score - 1e-12:
            f[best_idx] = -f[best_idx]
            S = new_S[best_idx]
            C = C_candidates[best_idx]
            score = scores[best_idx]
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count > 10:
                break

    return f, score

def _soft_max_polish(f, time_budget=10.0, beta=50.0):
    """
    Smooth surrogate polish using soft-max (log-sum-exp) approximation of max|C|.
    Minimizes log(sum(exp(beta*|C|))) / S^2 via gradient descent on the binary constraint.
    """
    f = np.sign(f).astype(np.float64)
    S = np.sum(f)
    if abs(S) < 1e-6:
        return f, 1e10

    start_time = time.time()
    lr = 0.5

    C = _autoconv_fft(f)
    score = np.max(np.abs(C)) / (S * S)

    for it in range(500):
        if time.time() - start_time > time_budget:
            break
        if time.time() > DEADLINE:
            break

        C = _autoconv_fft(f)
        S = np.sum(f)
        if abs(S) < 1e-6:
            break

        # Soft-max of |C|: log(sum(exp(beta * |C|)))
        abs_C = np.abs(C)
        beta_abs = beta * abs_C
        # Numerically stable log-sum-exp
        max_val = np.max(beta_abs)
        log_sum_exp = max_val + np.log(np.sum(np.exp(beta_abs - max_val)))

        # Gradient of log-sum-exp w.r.t. C: exp(beta*|C|) / sum(exp(beta*|C|)) * sign(C)
        # d/dC_k [log sum exp(beta|C_j|)] = beta * exp(beta|C_k|) / sum_j exp(beta|C_j|) * sign(C_k)
        exp_vals = np.exp(beta_abs - max_val)
        weights = exp_vals / np.sum(exp_vals)
        grad_C = beta * weights * np.sign(C)

        # Gradient of objective = max|C|/S^2 approximated by log-sum-exp/S^2
        # d/dC_k [log_sum_exp / S^2] = grad_C[k] / S^2
        # d/dS [log_sum_exp / S^2] = -2 * log_sum_exp / S^3
        # Chain rule: d/d f_i = sum_k dC_k/df_i * grad_C[k]/S^2 + dS/df_i * (-2*log_sum_exp/S^3)
        # dC_k/df_i = f[k-i] + f[k+i] (for valid indices), plus f[0] if k==0... 
        # Actually for autoconv: C[k] = sum_i f[i]*f[k-i], so dC_k/df_i = f[k-i] (if k-i in [0,N-1])
        # and also f[i-k] from the other term... wait, autoconv is symmetric.
        # C[k] = sum_{i} f[i] f[k-i] where both indices must be in [0,N-1]
        # dC_k/df_i = f[k-i] if 0 <= k-i <= N-1, else 0
        # (The product rule: derivative of f[i]*f[k-i] w.r.t. f[i] is f[k-i], 
        #  and derivative of f[k-i]*f[i] is the same since it's the same term)
        # Wait, C[k] = sum_i f[i]*f[k-i]. If we differentiate w.r.t. f[j]:
        # dC_k/df_j = f[k-j] (from the term where i=j) 
        # But also, if k-j is a valid index, the term f[k-j]*f[j] contributes f[k-j]... 
        # No, each pair (i, k-i) appears once in the sum. So dC_k/df_j = f[k-j] if 0<=k-j<N.

        # Build gradient w.r.t. f
        # grad_f[j] = sum_k dC_k/df_j * grad_C[k] / S^2 - 2*log_sum_exp/S^3
        # dC_k/df_j = f[k-j] if 0 <= k-j < N
        # So sum_k f[k-j] * grad_C[k] = (f * grad_C)[j] ... this is a cross-correlation!
        # Actually: sum_k f[k-j] * grad_C[k] for k in [0, 2N-2]
        # Let m = k-j, then k = m+j, sum_m f[m] * grad_C[m+j] for valid m+j
        # This is the cross-correlation of f and grad_C at lag j.

        # Compute cross-correlation: corr[j] = sum_k f[k-j] * grad_C[k]
        # Using FFT
        F_f = np.fft.rfft(f, n=FFT_SIZE)
        F_gc = np.fft.rfft(grad_C, n=FFT_SIZE)
        # Cross-correlation via FFT: ifft(conj(F_f) * F_gc)
        cross_corr = np.fft.irfft(np.conj(F_f) * F_gc, n=FFT_SIZE)
        # cross_corr[k] = sum_j f[j] * grad_C[j+k] ... we need the right convention
        # Let's just do it directly for N=400, it's fast enough
        # Actually for N=400, a direct O(N^2) computation is 160k ops, very fast.
        # grad_f[j] = sum_{k=0}^{2N-2} f[k-j] * grad_C[k] where 0<=k-j<N
        # = sum_{m=0}^{N-1} f[m] * grad_C[m+j] where m+j < 2N-1

        # Vectorized direct computation
        # For each j, we need sum_m f[m] * grad_C[m+j] for m in [0, N-1] and m+j in [0, 2N-2]
        # This is a sliding dot product
        # Use stride tricks or just loop (N=400 is small)
        grad_f = np.zeros(N)
        for j in range(N):
            # m ranges from 0 to N-1, need m+j < 2N-1 => m < 2N-1-j
            m_end = min(N, 2 * N - 1 - j)
            if m_end > 0:
                grad_f[j] = np.dot(f[:m_end], grad_C[j:j+m_end])

        # Full gradient of the objective w.r.t. f
        grad_obj = grad_f / (S * S) - 2.0 * log_sum_exp / (S ** 3)

        # Projected gradient step on the hypercube {-1, 1}^N
        # Use a soft update: move f in the direction of -grad, then re-project
        # For binary, we can use a sigmoid-like update or just flip the most negative gradient
        # Simple approach: flip positions where gradient is most negative (largest improvement)
        # Threshold-based update
        threshold = np.percentile(grad_obj, 5)  # flip bottom 5% of gradients
        flip_mask = grad_obj < threshold
        if np.any(flip_mask):
            f[flip_mask] *= -1.0

        # Check for improvement in the hard objective
        C_new = _autoconv_fft(f)
        S_new = np.sum(f)
        if abs(S_new) < 1e-6:
            break
        score_new = np.max(np.abs(C_new)) / (S_new * S_new)
        if score_new > score * 1.05:
            # Revert
            f[flip_mask] *= -1.0
            break
        else:
            score = score_new

    return f, score

def _legendre_symbol(a, p):
    """
    Compute the Legendre symbol (a/p).
    Returns 1 if a is a quadratic residue mod p, -1 if non-residue, 0 if a % p == 0.
    """
    if a % p == 0:
        return 0
    res = pow(a, (p - 1) // 2, p)
    if res == 1:
        return 1
    elif res == p - 1:
        return -1
    else:
        return 0

def _generate_legendre_sequence(p, N):
    """
    Generate a sequence of length N based on the Legendre symbol of a prime p.
    h[i] = Legendre(i+1, p) for i in range(N).
    If Legendre is 0, map to 1.
    """
    h = np.zeros(N)
    for i in range(N):
        sym = _legendre_symbol(i + 1, p)
        h[i] = 1.0 if sym >= 0 else -1.0
    return h

def construct():
    start_time = time.time()
    
    best_f = np.ones(N)
    best_score = _score(best_f)
    
    rng = np.random.default_rng(42)
    
    # Strategy 1: Algebraic initializations (Legendre sequences)
    # These have good autocorrelation properties theoretically.
    primes_to_try = [397, 401, 389, 409, 383, 419, 379, 421, 373, 431, 433, 439, 443, 449, 457]
    
    for p in primes_to_try:
        if time.time() > DEADLINE:
            break
        h = _generate_legendre_sequence(p, N)
        s = np.sum(h)
        if abs(s) < N * 0.1:
             # Ensure sum is not too small
             if s < 0:
                 h = -h
             if abs(np.sum(h)) < 1e-6:
                 h[0] = 1.0
        
        h_opt, val_opt = _local_search_binary_fast(h, max_iter=2000, time_budget=10.0)
        if val_opt < best_score:
            best_score = val_opt
            best_f = h_opt.copy()

    # Strategy 2: Structured initializations (Cosines, Chirps, Blocks)
    x = np.arange(N)
    initializations = []
    
    # Cosines
    for k in range(1, 11):
        h = np.sign(np.cos(2 * np.pi * k * x / N))
        h[h == 0] = 1.0
        initializations.append(h)
        
    # Chirps
    for freq_mult in [1.0, 2.0, 5.0, 10.0]:
        h = np.sign(np.cos(2 * np.pi * freq_mult * x * x / (2 * N)))
        h[h == 0] = 1.0
        initializations.append(h)
        
    # Blocks
    for _ in range(10):
        h = np.ones(N)
        pos = 0
        while pos < N:
            block_len = rng.integers(2, 20)
            sign = rng.choice([-1.0, 1.0])
            h[pos:pos+block_len] = sign
            pos += block_len
        initializations.append(h)
        
    # Mostly +1 with sparse -1
    for n_neg in [5, 10, 20, 50, 100]:
        h = np.ones(N)
        neg_idx = rng.choice(N, n_neg, replace=False)
        h[neg_idx] = -1.0
        initializations.append(h)

    for h in initializations:
        if time.time() > DEADLINE:
            break
        s = np.sum(h)
        if abs(s) < N * 0.1:
             if s < 0:
                 h = -h
             if abs(np.sum(h)) < 1e-6:
                 h[0] = 1.0
                 
        h_opt, val_opt = _local_search_binary_fast(h, max_iter=2000, time_budget=10.0)
        if val_opt < best_score:
            best_score = val_opt
            best_f = h_opt.copy()

    # Strategy 3: Random Restarts
    # Use different densities of 1s to ensure S is not too small
    # Increased count due to vectorized local search being much faster
    for _ in range(150):
        if time.time() > DEADLINE:
            break

        f_rand = rng.choice([-1.0, 1.0], size=N)
        S_rand = np.sum(f_rand)
        if abs(S_rand) < N * 0.2:
            continue

        f_rand_local, score_rand_local = _local_search_binary_fast(f_rand, max_iter=2000, time_budget=3.0)

        if score_rand_local < best_score:
            best_score = score_rand_local
            best_f = f_rand_local.copy()

    # Strategy 4: Soft-max surrogate polish
    # Use smooth log-sum-exp approximation to escape local minima of the hard max
    if time.time() < DEADLINE - 15.0:
        remaining = DEADLINE - time.time() - 15.0
        # Try multiple beta values
        for beta_val in [20.0, 50.0, 100.0]:
            if time.time() > DEADLINE - 5.0:
                break
            time_left = DEADLINE - time.time() - 5.0
            if time_left < 2.0:
                break
            f_smooth, score_smooth = _soft_max_polish(best_f, time_budget=min(time_left * 0.3, 15.0), beta=beta_val)
            if score_smooth < best_score:
                best_score = score_smooth
                best_f = f_smooth.copy()

    # Final Polish: Run local search on the best found so far with more time
    if time.time() < DEADLINE - 5.0:
        f_final, score_final = _local_search_binary_fast(best_f, max_iter=5000, time_budget=DEADLINE - time.time() - 2.0)
        if score_final < best_score:
            best_score = score_final
            best_f = f_final

    return best_f
