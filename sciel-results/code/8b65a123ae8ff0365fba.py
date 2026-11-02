"""Autoconvolution minimization via block-flip simulated annealing with incremental FFT updates.

We choose 400 signed step heights (+/-1) to minimize the maximum absolute
value of the (linear) autoconvolution, which is the C3 upper bound we want
to lower. 

Optimization:
- Uses incremental FFT updates for small block flips.
- Precomputes Fourier basis vectors to update the spectrum in O(kL) instead of O(L log L) for the forward transform.
- This allows for more search steps within the time budget.
"""
import time
import numpy as np

N = 400
L = 1024  # FFT length >= 2*N-1 = 799
_SEED = 12345

# Precompute Fourier basis for incremental updates
# W[i, k] = exp(-2j * pi * i * k / L) for i in 0..N-1, k in 0..L//2
_k_indices = np.arange(L // 2 + 1)
_i_indices = np.arange(N)
# Create meshgrid for exponent calculation
# Shape: (N, L//2 + 1)
_exponents = -2j * np.pi * np.outer(_i_indices, _k_indices) / L
_W_basis = np.exp(_exponents)


def _autocorr_fast(h):
    """Linear one-sided autocorrelation ac[k] = sum_i h[i] h[i+k], k = 0..N-1."""
    F = np.fft.rfft(h, n=L)
    ac = np.fft.irfft(F * np.conj(F), n=L)[:N]
    return ac


def _hard_score_from_ac(ac):
    """Compute score from autocorrelation array."""
    dc = ac[0]
    if dc <= 1e-30:
        return 1e9
    return float(np.max(np.abs(ac)) / dc)


def _smooth_score_from_ac(ac, beta):
    """Smooth (log-sum-exp) surrogate of the peak sidelobe energy, guides SA."""
    dc = ac[0]
    if dc <= 1e-30:
        return 1e9
    y = (ac[1:] / dc) ** 2
    m = float(np.max(y))
    z = beta * (y - m)
    return m + float(np.sum(np.exp(z))) / beta


def construct():
    try:
        t0 = time.time()
        deadline = t0 + 140.0  # safety margin before 150s limit
        rng = np.random.default_rng(_SEED)

        # ---------- diverse initializations ----------
        idx = np.arange(N)
        inits = [
            np.ones(N),
            rng.choice(np.array([-1.0, 1.0]), size=N),
            np.cos(2 * np.pi * idx ** 2 / (2 * N)),
            np.cos(2 * np.pi * idx ** 2 / (N + 1)),
            np.cos(2 * np.pi * idx ** 2 / (3 * N)),
            rng.choice(np.array([-1.0, 1.0]), size=N),
            np.cos(2 * np.pi * idx / N),
            np.sin(2 * np.pi * idx / N),
            np.cos(2 * np.pi * 2 * idx / N),
            rng.choice(np.array([-1.0, 1.0]), size=N),
        ]
        best_h = None
        best_hard = 1e9
        for h in inits:
            ac = _autocorr_fast(h)
            s = _hard_score_from_ac(ac)
            if s < best_hard:
                best_hard = s
                best_h = h.copy()

        h = best_h.copy()
        F = np.fft.rfft(h, n=L)
        ac = _autocorr_fast(h)
        cur_hard = _hard_score_from_ac(ac)
        cur_smooth = _smooth_score_from_ac(ac, 30.0)

        # ---------- simulated annealing with coordinated block sign-flips ----------
        T = 0.02
        T_min = 1e-4
        decay = 0.9997
        beta = 30.0
        sizes = np.array([1, 2, 3, 4, 8, 12, 16, 20, 24, 32, 48, 64])
        
        steps = 0
        while time.time() < deadline - 5.0:
            # Periodically recompute F to avoid drift
            if steps % 1000 == 0 and steps > 0:
                F = np.fft.rfft(h, n=L)
            
            if rng.random() < 0.70:
                size = int(rng.choice(sizes))
                start = int(rng.integers(0, N - size + 1))
                sl = slice(start, start + size)
                idx_flip = np.arange(start, start + size)
                
                # Incremental FFT update
                # delta_h = -2 * h[sl]
                # delta_F = FFT(delta_h) = sum_{i in sl} -2 h[i] W[i, :]
                h_block = h[idx_flip]
                W_block = _W_basis[idx_flip, :]
                delta_F = -2.0 * (W_block @ h_block)
                F_new = F + delta_F
                
                # Compute score from new F
                P = F_new * np.conj(F_new)
                ac_new = np.fft.irfft(P, n=L)[:N]
                new_hard = _hard_score_from_ac(ac_new)
                new_smooth = _smooth_score_from_ac(ac_new, beta)
                
                dS = new_smooth - cur_smooth
                if new_hard < cur_hard or (dS <= 0 or rng.random() < np.exp(-dS / T)):
                    # Accept
                    h[sl] *= -1.0
                    F = F_new
                    cur_hard = new_hard
                    cur_smooth = new_smooth
                    if new_hard < best_hard:
                        best_hard = new_hard
                        best_h = h.copy()
                else:
                    # Reject, keep old h and F
                    pass
            else:
                i = int(rng.integers(0, N))
                idx_flip = np.array([i])
                
                # Incremental FFT update
                h_block = h[idx_flip]
                W_block = _W_basis[idx_flip, :]
                delta_F = -2.0 * (W_block @ h_block)
                F_new = F + delta_F
                
                # Compute score from new F
                P = F_new * np.conj(F_new)
                ac_new = np.fft.irfft(P, n=L)[:N]
                new_hard = _hard_score_from_ac(ac_new)
                new_smooth = _smooth_score_from_ac(ac_new, beta)
                
                dS = new_smooth - cur_smooth
                if new_hard < cur_hard or (dS <= 0 or rng.random() < np.exp(-dS / T)):
                    # Accept
                    h[i] *= -1.0
                    F = F_new
                    cur_hard = new_hard
                    cur_smooth = new_smooth
                    if new_hard < best_hard:
                        best_hard = new_hard
                        best_h = h.copy()
                else:
                    # Reject
                    pass
            
            T = max(T_min, T * decay)
            steps += 1

        # ---------- greedy multi-scale local polish ----------
        h = best_h.copy()
        F = np.fft.rfft(h, n=L)
        ac = _autocorr_fast(h)
        best_hard = _hard_score_from_ac(ac)
        
        sizes_polish = [1, 2, 4, 8, 16, 32]
        for _pass in range(40):
            if time.time() > deadline - 1.0:
                break
            any_improve = False
            for size in sizes_polish:
                for start in range(0, N - size + 1):
                    if time.time() > deadline - 1.0:
                        return best_h
                    idx_flip = np.arange(start, start + size)
                    
                    # Incremental FFT update
                    h_block = h[idx_flip]
                    W_block = _W_basis[idx_flip, :]
                    delta_F = -2.0 * (W_block @ h_block)
                    F_new = F + delta_F
                    
                    # Compute score from new F
                    P = F_new * np.conj(F_new)
                    ac_new = np.fft.irfft(P, n=L)[:N]
                    new_hard = _hard_score_from_ac(ac_new)
                    
                    if new_hard < best_hard - 1e-12:
                        # Accept
                        h[idx_flip] *= -1.0
                        F = F_new
                        best_hard = new_hard
                        best_h = h.copy()
                        any_improve = True
                    else:
                        # Reject, F remains unchanged
                        pass
            if not any_improve:
                break

        return best_h
    except Exception:
        return np.ones(N)
