"""Structural rewrite: simulated annealing + FFT autoconv + targeted hot-lag perturbation for C3' minimization."""
import numpy as np
import time

N = 400

def _fft_autoconv(a):
    """Compute full autoconvolution via FFT (circular, then extract linear)."""
    n = len(a)
    size = 2 * n - 1
    f = np.fft.rfft(a, size=size)
    ac = np.fft.irfft(f * f, n=size)
    # ac[k] for k=0..n-1 is autoconv at lag k (positive), k=n..2n-2 is negative lags
    # Full linear autoconv: ac_full[k] = sum_i a[i]*a[i+k] for k=-(n-1)..(n-1)
    # irfft gives circular, but with size=2n-1 it's the linear convolution
    # ac[0] = sum a[i]^2, ac[k] for k=1..n-1 = sum_{i=0}^{n-1-k} a[i]*a[i+k]
    # ac[n] = sum_{i=0}^{n-1} a[i]*a[i-n+... ] -- actually for autoconv, ac is symmetric
    # The full autoconvolution of length 2n-1:
    # ac_full[k] for k=0..2n-2, where ac_full[k] = sum_i a[i]*a[i+k-n+1]... 
    # Actually np.fft.irfft of |F|^2 gives the circular autocorrelation of the zero-padded signal,
    # which equals the linear autocorrelation.
    # ac[0] = sum a[i]^2
    # ac[k], k=1..n-1: sum_{i=0}^{n-1-k} a[i]*a[i+k]  (positive lag)
    # ac[n], ac[n+1], ... : these are the negative lags, ac[n+j] = ac[n-1-j]... 
    # Wait, for real signals, the autocorrelation is symmetric: ac[k] = ac[-k]
    # The irfft output of length 2n-1: ac[0..n-1] are lags 0..n-1, ac[n..2n-2] are lags -(n-1)..-1
    # Actually: ac_full[j] for j=0..2n-2 where j=0 is lag 0, j=1 is lag 1, ..., j=n-1 is lag n-1,
    # j=n is lag -(n-1), j=n+1 is lag -(n-2), ..., j=2n-2 is lag -1
    return ac

def compute_objective(a):
    """Compute max abs autoconv / (sum a)^2. Return inf if sum is 0."""
    s = np.sum(a)
    if abs(s) < 1e-12:
        return np.inf
    ac = _fft_autoconv(a)
    return np.max(np.abs(ac)) / (s * s)

def _hot_lags(ac, top_k=20):
    """Return indices of the top_k largest |ac| values (excluding center if needed)."""
    idx = np.argsort(-np.abs(ac))[:top_k]
    return idx

def _positions_in_lag(lag, n):
    """Return pairs (i, j) such that j - i = lag mod n, but for linear autoconv, 
    lag k means pairs (i, i+k) for i=0..n-1-k. For negative lag -k, pairs (i, i-k) for i=k..n-1."""
    # We'll just return a set of positions that participate in this lag
    # For lag k (0 <= k < n): positions involved are 0..n-1 (all, since both i and i+k appear)
    # Actually, for a given lag k, the pairs are (i, i+k) for i=0..n-1-k
    # So positions involved: {0, 1, ..., n-1} (all positions appear in some pair for most lags)
    # For lag k: positions 0..n-1-k appear as first element, positions k..n-1 appear as second
    # Union: 0..n-1 (all)
    # This isn't very selective. Let me think differently.
    # The contribution to ac[k] is sum_i a[i]*a[i+k]. If we flip a[i], it affects ac[k] for all k.
    # But the "hot" lags are those with large |ac[k]|. Flipping a position that is in many hot lag pairs
    # is more impactful.
    # For simplicity, return all positions (since every position affects every lag).
    # Better: return positions weighted by how many hot lags they participate in.
    return None  # placeholder

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    best_a = None
    best_obj = np.inf
    
    rng = np.random.default_rng(12345)
    
    # Diverse initializations
    initializations = []
    
    # 1. All ones
    a1 = np.ones(N)
    initializations.append(a1)
    
    # 2. Random signs
    for trial in range(30):
        a = rng.choice([-1.0, 1.0], size=N)
        initializations.append(a)
    
    # 3. Chirp patterns with various periods
    for period in [N // 4, N // 3, N // 2, N // 5, N // 7]:
        t = np.arange(N)
        a = np.sign(np.cos(2 * np.pi * t / period))
        a[a == 0] = 1.0
        initializations.append(a)
    
    # 4. Block patterns
    for nblocks in [4, 8, 16, 32]:
        block_size = N // nblocks
        a = np.zeros(N)
        for b in range(nblocks):
            a[b*block_size:(b+1)*block_size] = 1.0 if b % 2 == 0 else -1.0
        initializations.append(a)
    
    # 5. Sparse random
    for density in [0.1, 0.2, 0.3, 0.5]:
        for trial in range(15):
            mask = rng.random(N) < density
            a = np.zeros(N)
            a[mask] = rng.choice([-1.0, 1.0], size=np.sum(mask))
            initializations.append(a)
    
    # 6. Walsh-Hadamard-like patterns (low autocorrelation structure)
    # Generate using Kronecker products of small low-autocorrelation blocks
    for base_len in [4, 8, 16]:
        # Use a known low-autocorrelation sequence as base
        # Barker-like: for length 7, [1,1,1,-1,1,-1,-1]
        if base_len == 4:
            base = np.array([1, 1, -1, 1])
        elif base_len == 8:
            base = np.array([1, 1, 1, -1, 1, -1, -1, 1])
        else:
            base = np.array([1, 1, 1, -1, 1, -1, -1, 1, 1, 1, -1, 1, -1, 1, 1, -1])
        # Tile with sign flips
        for sign_flip in [0, 1]:
            a = np.tile(base, N // base_len + 1)[:N]
            if sign_flip:
                a = a * np.sign(np.cos(2 * np.pi * np.arange(N) / (N // 2)))
                a[a == 0] = 1.0
            initializations.append(a)
    
    # Evaluate all initializations
    for a in initializations:
        if time.time() > deadline:
            break
        obj = compute_objective(a)
        if obj < best_obj:
            best_obj = obj
            best_a = a.copy()
    
    # Simulated annealing optimization
    def _simulated_annealing(a_init, max_time=30.0, T0=1.0, T_end=0.01, cooling=0.999):
        """Simulated annealing to minimize max abs autoconv / sum^2."""
        a = a_init.copy().astype(np.float64)
        s = np.sum(a)
        if abs(s) < 1e-12:
            # Add a small perturbation to avoid zero sum
            a[rng.integers(N)] += 2.0
            s = np.sum(a)
        
        current_obj = compute_objective(a)
        best_local_obj = current_obj
        best_local_a = a.copy()
        
        T = T0
        t_start = time.time()
        no_improve_count = 0
        
        while time.time() - t_start < max_time and T > T_end:
            # Choose a position to flip, biased toward positions that appear in hot lags
            # For now, uniform random, but we can bias later
            i = rng.integers(N)
            
            # Flip
            a[i] *= -1
            new_s = np.sum(a)
            if abs(new_s) < 1e-12:
                a[i] *= -1  # revert
                continue
            
            new_obj = compute_objective(a)
            delta = new_obj - current_obj
            
            if delta < 0 or rng.random() < np.exp(-delta / max(T, 1e-10)):
                # Accept
                current_obj = new_obj
                if new_obj < best_local_obj:
                    best_local_obj = new_obj
                    best_local_a = a.copy()
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                # Revert
                a[i] *= -1
                no_improve_count += 1
            
            T *= cooling
            
            # Adaptive cooling: cool faster if no improvement for a while
            if no_improve_count > 500:
                T *= 0.95
                no_improve_count = 0
        
        return best_local_a, best_local_obj
    
    # Run SA on best candidate and a few diverse ones
    sa_candidates = [best_a.copy()]
    # Add a few random ones
    for _ in range(5):
        sa_candidates.append(rng.choice([-1.0, 1.0], size=N))
    
    for i, cand in enumerate(sa_candidates):
        if time.time() > deadline - 5:
            break
        remaining = deadline - time.time()
        max_time = min(remaining * 0.6, 40.0)
        if max_time < 2:
            break
        opt_a, opt_obj = _simulated_annealing(cand, max_time=max_time)
        if opt_obj < best_obj:
            best_obj = opt_obj
            best_a = opt_a
    
    # Final polish: greedy sign-flip on the best
    if time.time() < deadline - 2:
        a = best_a.copy()
        s = np.sum(a)
        if abs(s) < 1e-12:
            a[rng.integers(N)] += 2.0
            s = np.sum(a)
        current_obj = compute_objective(a)
        improved = True
        max_iters = 200
        for it in range(max_iters):
            if time.time() > deadline - 1:
                break
            improved_this_round = False
            for i in range(N):
                if time.time() > deadline - 1:
                    break
                a[i] *= -1
                new_s = np.sum(a)
                if abs(new_s) < 1e-12:
                    a[i] *= -1
                    continue
                new_obj = compute_objective(a)
                if new_obj < current_obj:
                    current_obj = new_obj
                    improved_this_round = True
                else:
                    a[i] *= -1
            if not improved_this_round:
                break
        
        if current_obj < best_obj:
            best_obj = current_obj
            best_a = a
    
    return best_a
