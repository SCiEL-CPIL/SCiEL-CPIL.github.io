import numpy as np
import time
from scipy.optimize import minimize

N = 400
_FSIZE = 1
while _FSIZE < 2 * N - 1:
    _FSIZE *= 2

def _autoconv_full(h):
    """Full linear autoconvolution via FFT."""
    H = np.fft.rfft(h, _FSIZE)
    ac = np.fft.irfft(H * H, _FSIZE)
    return ac[:2*N-1]

def compute_raw_value(x):
    """
    Computes the raw value: max(|autoconvolution|) / (sum(x))^2.
    Assumes x is a 1D numpy array.
    """
    X = np.fft.rfft(x, _FSIZE)
    conv = np.fft.irfft(X * X, _FSIZE)
    conv = conv[:2*N-1]
    max_abs_conv = np.max(np.abs(conv))
    integral = np.sum(x)
    if abs(integral) < 1e-12:
        return float('inf')
    return max_abs_conv / (integral * integral)

def _soft_max_abs(ac, beta=50.0):
    """Smooth approximation of max(|ac|) using log-sum-exp."""
    abs_ac = np.abs(ac)
    m = np.max(abs_ac)
    if m < 1e-15:
        return 0.0
    scaled = beta * (abs_ac - m)
    scaled = np.clip(scaled, -50, 50)
    log_sum = np.log(np.sum(np.exp(scaled)))
    return m + log_sum / beta

def _smooth_obj(h, beta=50.0):
    """Smooth objective: soft_max_abs(autoconv(h)) / (sum(h))^2."""
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    ac = _autoconv_full(h)
    sm = _soft_max_abs(ac, beta)
    return sm / (s * s)

def _local_search_flip(h, max_passes=20, deadline=None):
    """
    Local optimization: try flipping each element's sign, keep if improves score.
    Works on binary {-1, 1} sequences.
    """
    h = h.copy()
    h = np.sign(h)
    h[h == 0] = 1
    
    ac = _autoconv_full(h)
    sum_h = np.sum(h)
    best_val = np.max(np.abs(ac)) / (sum_h * sum_h)
    
    for _ in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        for i in range(N):
            if deadline is not None and time.time() > deadline:
                break
            old_val = h[i]
            h[i] = -old_val
            delta = -2.0 * old_val
            sum_h += delta
            
            # Update ac
            # ac[k] = sum_j h[j]*h[j+k]
            # Terms involving h[i]:
            # 1. j=i: h[i]*h[i+k] for k=0..N-1-i
            # 2. j+k=i => j=i-k: h[i-k]*h[i] for k=0..i
            # So ac[:N-i] += delta * h[i:N]
            #    ac[:i+1] += delta * h[0:i+1][::-1]
            
            ac[:N-i] += delta * h[i:N]
            ac[:i+1] += delta * h[0:i+1][::-1]
            
            new_val = np.max(np.abs(ac)) / (sum_h * sum_h)
            if new_val < best_val:
                best_val = new_val
                improved = True
            else:
                # Revert
                h[i] = old_val
                sum_h -= delta
                ac[:N-i] -= delta * h[i:N]
                ac[:i+1] -= delta * h[0:i+1][::-1]
        if not improved:
            break
    
    return h, best_val

def _local_search_block(h, deadline=None, block_sizes=[3, 5, 7, 10]):
    """
    Block flip local search: flip contiguous blocks of signs.
    """
    h = h.copy()
    h = np.sign(h)
    h[h == 0] = 1
    best_val = compute_raw_value(h)
    improved = True
    while improved and (deadline is None or time.time() < deadline):
        improved = False
        for bs in block_sizes:
            for start in range(0, N - bs + 1, max(1, (N - bs) // 100)):
                if deadline is not None and time.time() > deadline:
                    break
                h[start:start+bs] = -h[start:start+bs]
                new_val = compute_raw_value(h)
                if new_val < best_val:
                    best_val = new_val
                    improved = True
                else:
                    h[start:start+bs] = -h[start:start+bs]  # revert
    return h, best_val

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin before 150s kill
    
    # 1. Baseline: All ones
    best_x = np.ones(N, dtype=float)
    best_val = compute_raw_value(best_x)
    
    # 2. Generate diverse candidates
    candidates = [best_x.copy()]
    
    t = np.arange(N)
    
    # Chirp sequences
    for freq in [1.0, 2.0, 0.5, 3.0, 4.0, 0.25]:
        phase = 2 * np.pi * freq * (t**2) / N
        x_chirp = np.sign(np.cos(phase))
        x_chirp[x_chirp == 0] = 1
        candidates.append(x_chirp)
    
    # Linear chirp (constant frequency)
    for freq in [1.0, 2.0, 0.5, 3.0, 4.0, 0.25, 5.0]:
        phase = 2 * np.pi * freq * t / N
        x_lin = np.sign(np.cos(phase))
        x_lin[x_lin == 0] = 1
        candidates.append(x_lin)
    
    # Barker-like structured patterns
    for period in [3, 4, 5, 6, 7, 8]:
        for offset in range(period):
            x_pat = np.array([1 if (i + offset) % period < period//2 else -1 for i in range(N)], dtype=float)
            candidates.append(x_pat)
    
    # Random sequences
    rng = np.random.default_rng(42)
    for _ in range(25):
        x_rand = rng.choice([1.0, -1.0], N)
        candidates.append(x_rand)
    
    # Golden ratio based
    phi = (1 + np.sqrt(5)) / 2
    x_gold = np.sign(np.cos(2 * np.pi * phi * t))
    x_gold[x_gold == 0] = 1
    candidates.append(x_gold)
    
    # Thue-Morse like (alternating blocks)
    def thue_morse(n):
        tm = [1]
        while len(tm) < n:
            tm.extend([-x for x in tm])
        return tm[:n]
    x_tm = np.array(thue_morse(N), dtype=float)
    candidates.append(x_tm)
    
    # Rabinowitz-Kantor style (balanced)
    for balance in [0.3, 0.4, 0.5, 0.6, 0.7]:
        x_bal = rng.choice([1.0, -1.0], N, p=[balance, 1-balance])
        candidates.append(x_bal)
    
    # 3. Evaluate all candidates
    for x in candidates:
        val = compute_raw_value(x)
        if val < best_val:
            best_val = val
            best_x = x.copy()
    
    # 4. Smooth surrogate optimization with adaptive beta annealing
    scored_candidates = []
    for x in candidates:
        val = compute_raw_value(x)
        scored_candidates.append((val, x))
    scored_candidates.sort(key=lambda t: t[0])
    
    # Optimize from top 5 diverse starts with annealed beta
    top_starts = scored_candidates[:5]
    
    for init_val, h_init in top_starts:
        if time.time() > deadline - 40:
            break
        
        # Normalize to sum=1 for optimization
        s0 = np.sum(h_init)
        if abs(s0) < 1e-6:
            continue
        h_norm = h_init / s0
        
        # Adaptive beta annealing: start low (smooth) then increase (sharp)
        for beta in [10, 30, 80, 150]:
            if time.time() > deadline - 20:
                break
            
            try:
                def obj(x, _beta=beta):
                    return _smooth_obj(x, beta=_beta)
                
                res = minimize(obj, h_norm, method='L-BFGS-B',
                               options={'maxiter': 150, 'disp': False, 'gtol': 1e-8})
                h_opt = res.x
                
                # Evaluate the actual objective
                val = compute_raw_value(h_opt)
                if val < best_val:
                    best_val = val
                    best_x = h_opt.copy()
                
                # Also try sign-binarized version
                h_bin = np.sign(h_opt)
                h_bin[h_bin == 0] = 1
                val_bin = compute_raw_value(h_bin)
                if val_bin < best_val:
                    best_val = val_bin
                    best_x = h_bin.copy()
                    
            except Exception:
                pass
    
    # 5. Local search refinement on best result (incremental, fast)
    if time.time() < deadline - 15:
        # First, binarize if not already binary
        h_work = np.sign(best_x)
        h_work[h_work == 0] = 1
        
        # Single-bit flip local search with incremental updates
        h_work, val_work = _local_search_flip(h_work, max_passes=20, deadline=deadline - 8)
        if val_work < best_val:
            best_val = val_work
            best_x = h_work.copy()
        
        # Block flip local search
        if time.time() < deadline - 5:
            h_work, val_work = _local_search_block(h_work, deadline=deadline - 3, 
                                                    block_sizes=[3, 5, 7, 10])
            if val_work < best_val:
                best_val = val_work
                best_x = h_work.copy()
        
        # Second round of single-bit flips after block moves
        if time.time() < deadline - 3:
            h_work = np.sign(best_x)
            h_work[h_work == 0] = 1
            h_work, val_work = _local_search_flip(h_work, max_passes=10, deadline=deadline - 2)
            if val_work < best_val:
                best_val = val_work
                best_x = h_work.copy()
    
    # 6. Final random restarts with local search if time remains
    if time.time() < deadline - 25:
        # More diverse restarts: random, biased, and structured perturbations
        restart_types = ['random', 'biased', 'perturb']
        for restart_idx in range(15):
            if time.time() > deadline - 12:
                break
            rtype = restart_types[restart_idx % len(restart_types)]
            if rtype == 'random':
                x_rand = rng.choice([1.0, -1.0], N)
            elif rtype == 'biased':
                # Biased toward +1 to increase sum
                p_plus = 0.55 + 0.1 * (restart_idx % 5)
                x_rand = rng.choice([1.0, -1.0], N, p=[p_plus, 1-p_plus])
            else:
                # Perturb current best
                x_rand = best_x.copy()
                flip_mask = rng.random(N) < 0.15
                x_rand[flip_mask] = -x_rand[flip_mask]
                x_rand = np.sign(x_rand)
                x_rand[x_rand == 0] = 1
            
            h_work, val_work = _local_search_flip(x_rand, max_passes=8, deadline=deadline - 8)
            if val_work < best_val:
                best_val = val_work
                best_x = h_work.copy()
    
    # 7. Final high-precision polish: more passes of incremental flip search
    if time.time() < deadline - 2:
        h_work = np.sign(best_x)
        h_work[h_work == 0] = 1
        h_work, val_work = _local_search_flip(h_work, max_passes=50, deadline=deadline - 1)
        if val_work < best_val:
            best_val = val_work
            best_x = h_work.copy()
        
        # One more block search round if time permits
        if time.time() < deadline - 1:
            h_work, val_work = _local_search_block(h_work, deadline=deadline - 0.5,
                                                    block_sizes=[2, 3, 5, 7, 11])
            if val_work < best_val:
                best_val = val_work
                best_x = h_work.copy()
    
    # Ensure we return a valid array
    return best_x

if __name__ == "__main__":
    t0 = time.time()
    x = construct()
    t1 = time.time()
    val = compute_raw_value(x)
    print(f"Best raw value: {val}")
    print(f"Sum: {np.sum(x)}")
    print(f"Time: {t1-t0:.2f}s")
