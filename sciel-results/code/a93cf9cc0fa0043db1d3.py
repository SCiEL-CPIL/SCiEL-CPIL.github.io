import time
import numpy as np

N = 400
L = 1024  # FFT length (power of 2 > 2N)

def _pad(f):
    out = np.zeros(L)
    out[:N] = f
    return out

def _score(f):
    fp = _pad(f)
    F = np.fft.rfft(fp)
    r = np.fft.irfft(F * np.conj(F), n=L)
    c = r[:N]
    return 2.0 * np.sum(c[1:] ** 2)

def _best_single_flip_gain(f):
    fp = _pad(f)
    F = np.fft.rfft(fp)
    F2 = F * np.conj(F)
    r = np.fft.irfft(F2, n=L)
    c = r[:N].copy()
    conv = np.fft.irfft(F2 * F, n=L)
    S = conv[:N] - N * f
    # T[i] = sum_{k=1}^{N-1} (f[i+k]+f[i-k])^2  (with bounds, ignoring out-of-range)
    # We compute this directly O(N^2) but N=400 so it's 160k ops, fine.
    T = np.zeros(N)
    for k in range(1, N):
        # f[i+k] valid for i=0..N-1-k
        # f[i-k] valid for i=k..N-1
        # We need to handle bounds carefully.
        # For term (f[i+k]+f[i-k])^2, it's only defined when both indices are in [0,N-1].
        # Otherwise, one term is 0? No, the sum is over k where the index is in range.
        # Actually the definition is sum_{k=1}^{N-1} c_k d_k. d_k is the change in autocorrelation.
        # My derivation assumed the standard autocorrelation sum.
        # Let's just use the formula derived: Delta J = -8 f[i] S[i] + 8 * something.
        # Actually, let's just do a direct check for the best flip, it might be faster than deriving complex T.
        pass
    
    # Let's stick to a simpler but slightly slower per-sweep method if the complex one is buggy.
    # Or, let's verify the complex one.
    # S[i] = sum_{m != i} c_{|m-i|} f[m].
    # If we flip f[i] -> -f[i], new f' has f'[i] = -f[i], others same.
    # New autocorr c' = c + delta_c.
    # delta_c_k = f'[i+k]f'[i] + f'[i]f'[i+k] - (f[i+k]f[i] + f[i]f[i+k]) ... wait.
    # c_k = sum_{j=0}^{N-1-k} f[j+k]f[j].
    # c'_k = sum f'[j+k]f'[j].
    # Only terms involving index i change.
    # If j=i, term f'[i+k]f'[i] = (-f[i+k])(-f[i]) = f[i+k]f[i] (unchanged if both flip? No, only i flips).
    # f'[i] = -f[i].
    # Term j=i: f'[i+k]f'[i] = f[i+k](-f[i]) = -f[i+k]f[i]. Change: -2 f[i]f[i+k].
    # Term j+k=i => j=i-k: f'[i]f'[i-k] = (-f[i])f[i-k] = -f[i]f[i-k]. Change: -2 f[i]f[i-k].
    # So delta_c_k = -2 f[i] (f[i+k] + f[i-k]) (with boundary terms vanishing).
    # This matches my previous d_k derivation.
    
    # So J' = 2 sum_{k>=1} (c_k + delta_c_k)^2 = J + 4 sum c_k delta_c_k + 2 sum delta_c_k^2.
    # sum c_k delta_c_k = sum c_k (-2 f[i] (f[i+k]+f[i-k])) = -2 f[i] sum c_k (f[i+k]+f[i-k]) = -2 f[i] S[i].
    # So 4 sum c_k delta_c_k = -8 f[i] S[i].
    
    # sum delta_c_k^2 = sum_{k=1}^{N-1} 4 f[i]^2 (f[i+k]+f[i-k])^2 = 4 f[i]^2 * (sum_{k=1}^{N-1} (f[i+k]+f[i-k])^2).
    # Let Q[i] = sum_{k=1}^{N-1} (f[i+k]+f[i-k])^2.
    # Note f[i]^2 = 1.
    # So 2 sum delta_c_k^2 = 8 Q[i].
    
    # Delta J = -8 f[i] S[i] + 8 Q[i].
    # We want to minimize J, so we want the most negative Delta J.
    # We need Q[i].
    # Q[i] = sum_{k=1}^{N-1} f[i+k]^2 + f[i-k]^2 + 2 f[i+k]f[i-k] = sum (1+1) + 2 sum f[i+k]f[i-k] = 2(N-1) + 2 * (something).
    # This is getting complicated.
    
    # Given N=400, O(N^2) per sweep is 160,000 ops. 100 sweeps = 16M ops. Very fast in numpy if vectorized, or even in python if careful.
    # Let's just compute Q[i] directly with vectorization if possible, or just pick the index with smallest -8 f[i] S[i] and verify.
    # Actually, S[i] is the dominant term usually.
    # Let's just evaluate Delta J for all i using the S[i] approx and check top candidates.
    
    gains = -8.0 * f * S
    # This ignores the +8Q[i] term. Q[i] is positive and bounded (max ~2N).
    # S[i] can be large.
    # Let's just try flipping the argmin of gains, then check.
    
    best_i = np.argmin(gains)
    return best_i, gains

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    # Initialize with a good heuristic or random
    # Random is often good for this type of problem to avoid local minima of simple patterns
    rng = np.random.default_rng(42)
    f = rng.choice([-1, 1], size=N)
    
    best_score = _score(f)
    best_f = f.copy()
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Greedy: find best single flip
        # We can do multiple passes per outer loop or just one
        fp = _pad(f)
        F = np.fft.rfft(fp)
        F2 = F * np.conj(F)
        r = np.fft.irfft(F2, n=L)
        c = r[:N]
        conv = np.fft.irfft(F2 * F, n=L)
        S = conv[:N] - N * f
        
        # Compute approximate gain
        gains = -8.0 * f * S
        
        # Find best flip
        for _ in range(10):  # Do a few flips per sweep
            if time.time() > deadline: break
            i = np.argmin(gains)
            
            # Try flipping i
            f[i] = -f[i]
            new_score = _score(f)
            
            if new_score < best_score:
                best_score = new_score
                best_f = f.copy()
                improved = True
                # Recalculate S and gains after flip
                fp = _pad(f)
                F = np.fft.rfft(fp)
                F2 = F * np.conj(F)
                r = np.fft.irfft(F2, n=L)
                c = r[:N]
                conv = np.fft.irfft(F2 * F, n=L)
                S = conv[:N] - N * f
                gains = -8.0 * f * S
            else:
                # Revert
                f[i] = -f[i]
                break # No improvement found in this batch, stop inner loop
                
        # If we didn't improve in the inner loop, maybe try a random kick?
        if not improved:
            # Random restart or kick
            for _ in range(5):
                if time.time() > deadline: break
                idx = rng.integers(N)
                f[idx] = -f[idx]
                new_score = _score(f)
                if new_score < best_score:
                    best_score = new_score
                    best_f = f.copy()
                    improved = True
                else:
                    f[idx] = -f[idx]
                    
    return best_f
