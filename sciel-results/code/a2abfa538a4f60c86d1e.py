import time
import numpy as np

N = 100
SCALE = 2.0 / N

def _compute_O(h):
    """
    Compute the overlap vector O for all shifts k=0..N-1.
    O_pos[k] = SCALE * sum_{j=k}^{N-1} h[j]*(1-h[j-k])
    O_neg[k] = SCALE * sum_{j=0}^{N-1-k} h[j]*(1-h[j+k])
    Uses FFT for O(N log N) speed.
    """
    # h * (1-h) autocorrelation via FFT
    # corr[k] = sum_j h[j] * (1-h[j-k]) for valid j
    # This is cross-correlation of h and (1-h)
    
    # For O_pos[k] = sum_{j=k}^{N-1} h[j]*(1-h[j-k])
    # Let's compute full cross-correlation
    # cross_corr[m] = sum_j h[j+m] * (1-h[j]) for m where valid
    
    # Use FFT-based correlation
    # corr = ifft(fft(h, 2*N) * conj(fft(1-h, 2*N)))
    # But we need specific shifts
    
    # O_pos[k]: sum_{j=k}^{N-1} h[j]*(1-h[j-k]) = sum_{i=0}^{N-1-k} h[i+k]*(1-h[i])
    # This is cross-correlation at lag k: sum_i h[i+k]*(1-h[i])
    
    # O_neg[k]: sum_{j=0}^{N-1-k} h[j]*(1-h[j+k]) = sum_{i=0}^{N-1-k} h[i]*(1-h[i+k])
    # This is cross-correlation at lag -k: sum_i h[i]*(1-h[i+k])
    
    n_fft = 2 * N
    H = np.fft.rfft(h, n_fft)
    H1 = np.fft.rfft(1.0 - h, n_fft)
    
    # Cross-correlation: corr[m] = sum_i h[i+m] * (1-h[i])
    # = ifft(conj(H) * H1)[m] for m >= 0, and appropriate for m < 0
    # Actually: ifft(H * conj(H1))[m] = sum_i h[i] * (1-h[i+m])
    # ifft(conj(H) * H1)[m] = sum_i h[i+m] * (1-h[i])
    
    # Let's compute both
    corr_pos = np.fft.irfft(np.conj(H) * H1, n_fft)[:N]  # sum_i h[i+k]*(1-h[i]) for k=0..N-1
    corr_neg = np.fft.irfft(H * np.conj(H1), n_fft)[:N]  # sum_i h[i]*(1-h[i+k]) for k=0..N-1
    
    # O_pos[k] = SCALE * corr_pos[k]
    # O_neg[k] = SCALE * corr_neg[k]
    
    O_pos = SCALE * corr_pos
    O_neg = SCALE * corr_neg
    
    return np.concatenate([O_pos, O_neg])

def _step(h, tau):
    """
    Compute the gradient of the softmax-smoothed max overlap.
    """
    O = _compute_O(h)
    Omax = float(O.max())
    z = tau * (O - Omax)
    ez = np.exp(z)
    w = ez / ez.sum()
    w_pos, w_neg = w[:N], w[N:]
    
    K = np.arange(N)[:, None]
    I = np.arange(N)[None, :]
    
    ge = (I >= K)
    lt = (I < (N - K))
    
    idx_p = np.clip(I - K, 0, N - 1)
    idx_m = np.clip(I + K, 0, N - 1)
    
    Gpos = SCALE * (ge.astype(float) - ge * h[idx_p] - lt * h[idx_m])
    Gneg = SCALE * (lt.astype(float) - lt * h[idx_m] - ge * h[idx_p])
    
    grad = w_pos @ Gpos + w_neg @ Gneg
    return Omax, grad

def _project(z):
    """
    Project z onto {h in [0,1], sum h = N/2}
    """
    lo = float(z.min()) - 1.0
    hi = float(z.max()) + 1.0
    target = N / 2.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        s = np.clip(z - mid, 0.0, 1.0).sum()
        if s > target:
            lo = mid
        else:
            hi = mid
    t = 0.5 * (lo + hi)
    return np.clip(z - t, 0.0, 1.0)

def _optimize(init, deadline):
    h = _project(np.asarray(init, dtype=float).copy())
    
    # Schedule of tau values (smoothness)
    tau_schedule = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0]
    
    best_h_local = h.copy()
    best_val_local = float(_compute_O(h).max())
    
    giter = 0
    for tau in tau_schedule:
        base_eta = 0.5 / (1.0 + 0.1 * tau)
        
        for _ in range(200):
            if time.time() > deadline - 2.0:
                return best_h_local, best_val_local
                
            Omax, grad = _step(h, tau)
            if not np.all(np.isfinite(grad)):
                break
            
            eta = base_eta
            h_new = _project(h - eta * grad)
            O_new = _compute_O(h_new)
            val_new = float(O_new.max())
            
            if val_new < best_val_local:
                best_h_local = h_new.copy()
                best_val_local = val_new
                
            h = h_new
            giter += 1

    return best_h_local, best_val_local

def construct():
    start = time.time()
    deadline = start + 145.0  # Safety margin

    seed = np.full(N, 0.5)
    best_h = seed.copy()
    best_val = float(_compute_O(seed).max())  # 0.5

    idx = np.arange(N)
    inits = []
    
    # 1. Bang-bang random inits
    rng = np.random.default_rng(42)
    for _ in range(10):
        bb = np.zeros(N)
        bb[rng.choice(N, N // 2, replace=False)] = 1.0
        inits.append(bb)
        
    # 2. Structured continuous inits
    for f in (1, 2, 3, 4, 5, 7, 11):
        inits.append(0.5 + 0.4 * np.cos(2.0 * np.pi * f * idx / N))
        
    # 3. Random continuous
    for _ in range(10):
        inits.append(rng.random(N))
        
    # 4. "Sawtooth" or periodic bang-bang
    for period in [3, 4, 5, 6, 7, 8, 10, 11, 13]:
        bb = np.zeros(N)
        ones = period // 2
        pattern = np.zeros(period)
        pattern[:ones] = 1.0
        bb = np.tile(pattern, N // period + 1)[:N]
        if bb.sum() > N/2:
             idxs = np.where(bb==1)[0]
             to_remove = rng.choice(idxs, int(bb.sum() - N/2), replace=False)
             bb[to_remove] = 0
        elif bb.sum() < N/2:
             idxs = np.where(bb==0)[0]
             to_add = rng.choice(idxs, int(N/2 - bb.sum()), replace=False)
             bb[to_add] = 1
        inits.append(bb)
        
    # 5. Block patterns
    for block_size in [2, 3, 4, 5]:
        for offset in range(block_size):
            bb = np.zeros(N)
            for i in range(N):
                if (i - offset) % (2 * block_size) < block_size:
                    bb[i] = 1.0
            if bb.sum() > N/2:
                 idxs = np.where(bb==1)[0]
                 to_remove = rng.choice(idxs, int(bb.sum() - N/2), replace=False)
                 bb[to_remove] = 0
            elif bb.sum() < N/2:
                 idxs = np.where(bb==0)[0]
                 to_add = rng.choice(idxs, int(N/2 - bb.sum()), replace=False)
                 bb[to_add] = 1
            inits.append(bb)

    for init in inits:
        if time.time() > deadline - 10.0:
            break
        try:
            h, val = _optimize(init, deadline)
            if np.all(np.isfinite(h)) and val < best_val:
                best_h = h
                best_val = val
        except Exception:
            continue

    # Final polish with adaptive tau and eta - two phases
    h = best_h.copy()
    
    # Phase 1: moderate tau
    tau = 200.0
    eta = 0.05
    for _ in range(2000):
        if time.time() > deadline - 5.0:
            break
        Omax, grad = _step(h, tau)
        if not np.all(np.isfinite(grad)):
            break
        h_new = _project(h - eta * grad)
        val_new = float(_compute_O(h_new).max())
        if val_new < best_val:
            best_h = h_new
            best_val = val_new
            h = h_new
            eta = min(eta * 1.02, 0.1)
        else:
            eta *= 0.9
    
    # Phase 2: high tau for fine-tuning
    tau = 1000.0
    eta = 0.01
    for _ in range(1000):
        if time.time() > deadline - 1.0:
            break
        Omax, grad = _step(h, tau)
        if not np.all(np.isfinite(grad)):
            break
        h_new = _project(h - eta * grad)
        val_new = float(_compute_O(h_new).max())
        if val_new < best_val:
            best_h = h_new
            best_val = val_new
            h = h_new
            eta = min(eta * 1.01, 0.02)
        else:
            eta *= 0.95

    # Ensure exact sum and bounds
    best_h = _project(best_h)
    for _ in range(20):
        d = N / 2.0 - best_h.sum()
        if abs(d) < 1e-14:
            break
        best_h = np.clip(best_h + d / N, 0.0, 1.0)
        best_h = _project(best_h)
        
    # Verify final result
    final_val = float(_compute_O(best_h).max())
    if final_val < best_val:
        best_val = final_val
    
    return best_h
