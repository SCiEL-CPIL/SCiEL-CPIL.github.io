import numpy as np
import time

N = 400
_NFFT = 1 << (2 * N - 1).bit_length()  # next power of 2 >= 2N for fast FFT


def _autocorr_fast(h):
    # linear one-sided autocorrelation ac[k] = sum_i h[i] h[i+k], k = 0..N-1
    # Use real FFT trick: autocorr of real signal = IFFT(|FFT|^2)
    F = np.fft.rfft(h, n=_NFFT)
    ac = np.fft.irfft(F * np.conj(F), n=_NFFT)[:N]
    return ac


def _autocorr(h):
    return _autocorr_fast(h)


def _hard_score(h):
    # the actual metric: peak autoconvolution concentration (lower is better)
    ac = _autocorr(h)
    dc = ac[0]
    if dc <= 1e-30:
        return 1e9
    return float(np.max(np.abs(ac)) / dc)


def _smooth_score(h, beta):
    # smooth (log-sum-exp) surrogate of the peak sidelobe energy, guides SA
    ac = _autocorr(h)
    dc = ac[0]
    if dc <= 1e-30:
        return 1e9
    y = (ac[1:] / dc) ** 2
    m = float(np.max(y))
    z = beta * (y - m)
    return m + float(np.log(np.sum(np.exp(z)))) / beta


def _hard_score_approx(h):
    """Faster approximate score: uses a subset of lags to estimate the peak."""
    ac = _autocorr_fast(h)
    dc = ac[0]
    if dc <= 1e-30:
        return 1e9
    # Check all lags but use a quick max
    return float(np.max(np.abs(ac)) / dc)

def _thue_morse(n):
    """Generate Thue-Morse sequence of length n."""
    t = [1]
    while len(t) < n:
        t += [-x for x in t]
    return np.array(t[:n])

def construct():
    try:
        t0 = time.time()
        deadline = t0 + 115.0          # safety margin inside the 120s limit
        rng = np.random.default_rng(20240517)

        # ---------- diverse initializations ----------
        idx = np.arange(N)
        inits = [
            _thue_morse(N),
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
            s = _hard_score(h)
            if s < best_hard:
                best_hard = s
                best_h = h.copy()

        h = best_h.copy()
        cur_hard = best_hard
        cur_smooth = _smooth_score(h, 30.0)

        # ---------- Targeted Greedy Search ----------
        # Strategy: Identify the lag k with the max sidelobe.
        # Try flipping bits that reduce the contribution to C[k].
        # C[k] = sum h[i]h[i+k]. Flipping h[i] changes C[k] by -2 h[i]h[i+k].
        # If C[k] > 0, we want to flip i where h[i]h[i+k] > 0.
        # If C[k] < 0, we want to flip i where h[i]h[i+k] < 0.
        
        max_iterations = 5000
        for _iter in range(max_iterations):
            if time.time() > deadline - 10.0:
                break
            
            ac = _autocorr(h)
            dc = ac[0]
            if dc <= 1e-30:
                break
            
            # Find the lag with the maximum absolute sidelobe
            # We exclude lag 0
            sidelobes = np.abs(ac[1:])
            k_star = int(np.argmax(sidelobes)) + 1
            c_star = ac[k_star]
            
            if np.abs(c_star) / dc < best_hard:
                # If the current peak is already below our best recorded (shouldn't happen if best_hard is updated correctly, 
                # but good for safety), update best
                if np.max(sidelobes) / dc < best_hard:
                    best_hard = np.max(sidelobes) / dc
                    best_h = h.copy()
                continue

            # Determine which flips would reduce |C[k_star]|
            # We want to change C[k_star] towards 0.
            # Delta C = -2 * h[i] * h[i+k]
            # We want sign(Delta C) == -sign(C[k_star])
            # So -h[i]h[i+k] has same sign as -C[k_star]
            # h[i]h[i+k] has same sign as C[k_star]
            
            # Get valid indices i such that i+k < N
            max_i = N - k_star
            indices = np.arange(max_i)
            products = h[indices] * h[indices + k_star]
            
            # We want products to have the same sign as c_star
            if c_star > 0:
                candidates = indices[products > 0]
            else:
                candidates = indices[products < 0]
            
            if len(candidates) == 0:
                # No single bit flip can reduce this specific peak? 
                # Try random moves or next iteration
                i = int(rng.integers(0, N))
                h[i] *= -1.0
                new_hard = _hard_score(h)
                if new_hard < cur_hard:
                    cur_hard = new_hard
                    if new_hard < best_hard:
                        best_hard = new_hard
                        best_h = h.copy()
                else:
                    h[i] *= -1.0 # revert
                continue

            # Evaluate the best candidate
            # We can't test all candidates with full FFT each time (too slow).
            # Approximate improvement:
            # New C[k_star] = C[k_star] - 2 * h[i] * h[i+k]
            # We want to minimize |New C[k_star]|.
            # Since we picked candidates where sign(product) == sign(c_star), 
            # the magnitude will decrease.
            # The best candidate is the one with largest |product|? 
            # Actually, product is +/- 1. So all candidates reduce |C| by 2.
            # However, flipping one bit affects ALL other lags.
            # So we must pick one and test.
            # Let's pick the candidate that minimizes the *global* peak estimate.
            # To be efficient, we'll just pick a random candidate from the top few or just one.
            # Let's pick the one that is most "central" or just the first one? 
            # Better: Pick 3-5 random candidates, test them, keep the best.
            
            num_tests = 5
            if len(candidates) > num_tests:
                test_indices = rng.choice(candidates, num_tests, replace=False)
            else:
                test_indices = candidates
            
            best_candidate = None
            best_candidate_score = cur_hard
            best_candidate_ac = None
            
            for i in test_indices:
                h[i] *= -1.0
                new_hard = _hard_score(h)
                if new_hard < best_candidate_score:
                    best_candidate_score = new_hard
                    best_candidate = i
                    best_candidate_ac = _autocorr(h) # reuse for next step? No, we revert.
                h[i] *= -1.0 # revert for next test
                
            if best_candidate is not None and best_candidate_score < cur_hard:
                h[best_candidate] *= -1.0
                cur_hard = best_candidate_score
                if cur_hard < best_hard:
                    best_hard = cur_hard
                    best_h = h.copy()
            else:
                # If no improvement found in targeted search, do a random move to escape local minima
                # with small probability, or just try a random block flip
                if rng.random() < 0.1:
                    size = int(rng.choice([1, 2, 4, 8]))
                    start = int(rng.integers(0, N - size + 1))
                    sl = slice(start, start + size)
                    old = h[sl].copy()
                    h[sl] *= -1.0
                    new_hard = _hard_score(h)
                    if new_hard < cur_hard:
                        cur_hard = new_hard
                        if new_hard < best_hard:
                            best_hard = new_hard
                            best_h = h.copy()
                    else:
                        h[sl] = old

        # ---------- final single-bit polish ----------
        h = best_h.copy()
        for _pass in range(200):
            if time.time() > deadline - 0.5:
                break
            any_improve = False
            for i in range(N):
                old = h[i]
                h[i] *= -1.0
                new_hard = _hard_score(h)
                if new_hard < best_hard - 1e-12:
                    best_hard = new_hard
                    best_h = h.copy()
                    any_improve = True
                else:
                    h[i] = old
            if not any_improve:
                break

        return best_h
    except Exception:
        return np.ones(N)
