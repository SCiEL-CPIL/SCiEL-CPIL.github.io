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

def _max_abs_ac(h):
    return np.max(np.abs(_autoconv_full(h)))

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

def _compute_ac_from_fft(X):
    """Compute autoconvolution from pre-computed FFT."""
    conv = np.fft.irfft(X * X, _FSIZE)
    return conv[:2*N-1]

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

class _IncrementalAutoConv:
    """
    Incremental autoconvolution tracker for single-bit sign flips.
    Maintains the autoconvolution and updates it in O(N) time per flip.
    """
    def __init__(self, h):
        self.h = h.copy()
        self.ac = _autoconv_full(h)  # shape (2N-1,)
        self.max_abs = np.max(np.abs(self.ac))
        self.integral = np.sum(h)
        self.N = len(h)
    
    def flip(self, i):
        """Flip sign at index i and update autoconvolution in O(N)."""
        h_i = self.h[i]
        self.h[i] = -h_i
        # Update autoconvolution for k=1..N-1 (k=0 unchanged since h[i]^2 invariant)
        for k in range(1, self.N):
            delta = 0.0
            if i + k < self.N:
                delta -= 2.0 * h_i * self.h[i+k]
            if i - k >= 0:
                delta -= 2.0 * h_i * self.h[i-k]
            self.ac[k] += delta
        
        # Update max_abs
        self.max_abs = np.max(np.abs(self.ac))
        self.integral = np.sum(self.h)
    
    def revert(self, i):
        """Revert flip at index i (same as flip since it's involutive)."""
        self.flip(i)
    
    def get_value(self):
        """Get current objective value."""
        if abs(self.integral) < 1e-12:
            return float('inf')
        return self.max_abs / (self.integral * self.integral)

def _local_search_flip_fast(h, max_passes=20, deadline=None):
    """
    Fast local optimization using incremental autoconvolution.
    Each flip evaluation is O(N) instead of O(N log N).
    """
    h = h.copy()
    h = np.sign(h)
    h[h == 0] = 1
    tracker = _IncrementalAutoConv(h)
    best_val = tracker.get_value()
    
    for _ in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        for i in range(N):
            if deadline is not None and time.time() > deadline:
                break
            tracker.flip(i)
            new_val = tracker.get_value()
            if new_val < best_val:
                best_val = new_val
                improved = True
            else:
                tracker.revert(i)
        if not improved:
            break
    return h, best_val

def _local_search_block(h, deadline=None, block_sizes=[3,5,7,10]):
    """
    Block flip local search: flip contiguous blocks of signs.
    Uses FFT incrementally for speedup.
    """
    h = h.copy()
    h = np.sign(h)
    h[h == 0] = 1
    best_val = compute_raw_value(h)
    H = np.fft.rfft(h, _FSIZE)
    improved = True
    while improved and (deadline is None or time.time() < deadline):
        improved = False
        for bs in block_sizes:
            step = max(1, (N - bs) // 80)
            for start in range(0, N - bs + 1, step):
                if deadline is not None and time.time() > deadline:
                    break
                # Flip block
                block = h[start:start+bs].copy()
                h[start:start+bs] = -h[start:start+bs]
                # Update FFT
                delta = np.zeros(_FSIZE//2+1, dtype=complex)
                for j in range(bs):
                    idx = start + j
                    delta += -2 * block[j] * np.exp(-2j * np.pi * np.arange(_FSIZE//2+1) * idx / _FSIZE)
                H_new = H + delta
                conv = np.fft.irfft(H_new * H_new, _FSIZE)[:2*N-1]
                integral = np.sum(h)
                if abs(integral) < 1e-12:
                    new_val = float('inf')
                else:
                    new_val = np.max(np.abs(conv)) / (integral * integral)
                
                if new_val < best_val:
                    best_val = new_val
                    H = H_new
                    improved = True
                else:
                    h[start:start+bs] = block  # revert
    return h, best_val

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill
    
    # 1. Baseline: All ones
    best_x = np.ones(N, dtype=float)
    best_val = compute_raw_value(best_x)
    
    # 2. Generate diverse candidates
    candidates = [best_x.copy()]
    
    # Chirp sequences
    t = np.arange(N)
    for freq in [1.0, 2.0, 0.5, 3.0, 1.5, 2.5]:
        phase = 2 * np.pi * freq * (t**2) / N
        x_chirp = np.sign(np.cos(phase))
        x_chirp[x_chirp == 0] = 1
        candidates.append(x_chirp)
    
    # Linear chirp (constant frequency)
    for freq in [1.0, 2.0, 0.5, 1.5, 3.0]:
        phase = 2 * np.pi * freq * t / N
        x_lin = np.sign(np.cos(phase))
        x_lin[x_lin == 0] = 1
        candidates.append(x_lin)
    
    # Barker-like structured
    x_barker = np.array([1 if i % 4 in [0, 2] else -1 for i in range(N)], dtype=float)
    candidates.append(x_barker)
    x_barker2 = np.array([1 if i % 6 in [0, 1, 2] else -1 for i in range(N)], dtype=float)
    candidates.append(x_barker2)
    
    # Random sequences
    rng = np.random.default_rng(42)
    for _ in range(20):
        x_rand = rng.choice([1.0, -1.0], N)
        candidates.append(x_rand)
    
    # Golden ratio based
    phi = (1 + np.sqrt(5)) / 2
    x_gold = np.sign(np.cos(2 * np.pi * phi * t))
    x_gold[x_gold == 0] = 1
    candidates.append(x_gold)
    
    # Quadratic residues (pseudorandom-like structure)
    for p in [7, 11, 13, 17]:
        x_quad = np.array([1 if (i*i) % p < p//2 else -1 for i in range(N)], dtype=float)
        candidates.append(x_quad)
    
    # 3. Evaluate all candidates
    for x in candidates:
        val = compute_raw_value(x)
        if val < best_val:
            best_val = val
            best_x = x.copy()
    
    # 4. Smooth surrogate optimization from best candidates
    scored_candidates = []
    for x in candidates:
        val = compute_raw_value(x)
        scored_candidates.append((val, x))
    scored_candidates.sort(key=lambda t: t[0])
    
    # Optimize from top 10 diverse starts
    top_starts = scored_candidates[:10]
    
    for init_val, h_init in top_starts:
        if time.time() > deadline - 35:
            break
        
        # Normalize to sum=1 for optimization
        s0 = np.sum(h_init)
        if abs(s0) < 1e-6:
            continue
        h_norm = h_init / s0
        
        # Multi-stage beta optimization
        for beta in [10, 25, 50, 100]:
            if time.time() > deadline - 20:
                break
            
            try:
                def obj(x, b=beta):
                    return _smooth_obj(x, beta=b)
                
                res = minimize(obj, h_norm, method='L-BFGS-B',
                               options={'maxiter': 300, 'disp': False, 'gtol': 1e-9})
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
    
    # 5. Fast local search refinement using incremental autoconvolution
    if time.time() < deadline - 15:
        h_work = np.sign(best_x)
        h_work[h_work == 0] = 1
        h_work, val_work = _local_search_flip_fast(h_work, max_passes=15, deadline=deadline - 8)
        if val_work < best_val:
            best_val = val_work
            best_x = h_work.copy()
    
    # 6. Block flip local search for larger moves
    if time.time() < deadline - 10:
        h_work = np.sign(best_x)
        h_work[h_work == 0] = 1
        h_work, val_work = _local_search_block(h_work, deadline=deadline - 5, 
                                                block_sizes=[3, 5, 7, 11])
        if val_work < best_val:
            best_val = val_work
            best_x = h_work.copy()
    
    # 7. Random restarts with fast local search
    if time.time() < deadline - 20:
        for _ in range(15):
            if time.time() > deadline - 12:
                break
            x_rand = rng.choice([1.0, -1.0], N)
            h_work, val_work = _local_search_flip_fast(x_rand, max_passes=5, deadline=deadline - 6)
            if val_work < best_val:
                best_val = val_work
                best_x = h_work.copy()
    
    # 8. Final polish: multiple rounds of fast local search
    if time.time() < deadline - 3:
        h_work = np.sign(best_x)
        h_work[h_work == 0] = 1
        h_work, val_work = _local_search_flip_fast(h_work, max_passes=10, deadline=deadline - 1)
        if val_work < best_val:
            best_val = val_work
            best_x = h_work.copy()
    
    # Ensure we return a valid array
    return best_x

if __name__ == "__main__":
    x = construct()
    val = compute_raw_value(x)
    print(f"Best raw value: {val}")
    print(f"Sum: {np.sum(x)}")
