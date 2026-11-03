import time
import numpy as np

N = 100
_DEADLINE = time.time() + 140  # Safety margin before 150s kill


def _linear_overlap_values(h):
    n = len(h)
    n_fft = 2 * n
    h_pad = np.zeros(n_fft)
    h_pad[:n] = h
    H = np.fft.rfft(h_pad, n=n_fft)
    corr = np.fft.irfft(H * np.conj(H), n=n_fft)
    r = corr[:n]
    cs = (2.0 / n) * (n / 2.0 - r)
    return cs


def _score(h):
    h = np.asarray(h, dtype=np.float64)
    cs = _linear_overlap_values(h)
    return float(np.max(cs))


def _fix_sum(h):
    h = np.array(h, dtype=np.float64, copy=True)
    target = N / 2.0

    if np.all(h >= -1e-15) and np.all(h <= 1.0 + 1e-15) and abs(np.sum(h) - target) < 1e-12:
        h = np.clip(h, 0.0, 1.0)
        diff = target - np.sum(h)
        if abs(diff) > 1e-15:
            mask = (h > 1e-10) & (h < 1.0 - 1e-10)
            if np.any(mask):
                h[mask] += diff / np.sum(mask)
                h = np.clip(h, 0.0, 1.0)
        return h

    alpha_low = -max(h) - 1.0
    alpha_high = 1.0 - min(h) + 1.0

    for _ in range(60):
        alpha_mid = (alpha_low + alpha_high) / 2.0
        h_new = np.clip(h + alpha_mid, 0.0, 1.0)
        s = np.sum(h_new)
        if s < target:
            alpha_low = alpha_mid
        else:
            alpha_high = alpha_mid

    h_final = np.clip(h + (alpha_low + alpha_high) / 2.0, 0.0, 1.0)

    diff = target - np.sum(h_final)
    if abs(diff) > 1e-14:
        mask = (h_final > 1e-10) & (h_final < 1.0 - 1e-10)
        if np.any(mask):
            h_final[mask] += diff / np.sum(mask)
            h_final = np.clip(h_final, 0.0, 1.0)

    diff = target - np.sum(h_final)
    if abs(diff) > 1e-14:
        if diff > 0:
            idx = np.argmin(h_final)
        else:
            idx = np.argmax(h_final)
        h_final[idx] += diff
        h_final = np.clip(h_final, 0.0, 1.0)

    return h_final


def _smooth_max_val_grad(h, beta):
    h = np.asarray(h, dtype=np.float64)
    n = len(h)
    cs = _linear_overlap_values(h)
    m = np.max(cs)
    exp_terms = np.exp(beta * (cs - m))
    sum_exp = np.sum(exp_terms)
    w = exp_terms / sum_exp
    val = m + np.log(sum_exp) / beta

    # FFT-based gradient
    n_fft = 2 * n
    w_pad = np.zeros(n_fft)
    w_pad[:n] = w
    W = np.fft.rfft(w_pad, n=n_fft)
    H = np.fft.rfft(h, n=n_fft)

    conv_wh = np.fft.irfft(W * H, n=n_fft)[:n]

    h_flip = np.zeros(n)
    h_flip[0] = h[0]
    h_flip[1:] = h[1:][::-1]
    h_flip_pad = np.zeros(n_fft)
    h_flip_pad[:n] = h_flip
    HF = np.fft.rfft(h_flip_pad, n=n_fft)
    conv_whf = np.fft.irfft(W * HF, n=n_fft)[:n]

    grad = -(2.0 / n) * (conv_wh + conv_whf)
    return val, grad


def _accelerated_smooth_max(h0, beta, max_iter=1000, lr_init=0.05):
    h = np.array(h0, dtype=np.float64, copy=True)
    h = _fix_sum(h)
    h_prev = h.copy()
    t_k = 1.0
    lr = lr_init

    best_val = _score(h)
    best_h = h.copy()

    for it in range(max_iter):
        if time.time() > _DEADLINE - 5:
            break

        val, grad = _smooth_max_val_grad(h, beta)
        h_new = h - lr * grad
        h_new = _fix_sum(h_new)

        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t_k * t_k)) / 2.0
        h_next = h_new + ((t_k - 1.0) / t_new) * (h_new - h_prev)
        h_next = _fix_sum(h_next)

        h_prev = h
        h = h_next
        t_k = t_new

        if it % 50 == 49:
            lr = max(0.001, min(0.1, lr * 0.97))

        if it % 30 == 29:
            current_score = _score(h)
            if current_score < best_val:
                best_val = current_score
                best_h = h.copy()

    return best_h, best_val


def _epigraph_feasibility(h0, t_target, beta, max_iter=300, lr_init=0.03):
    h = np.array(h0, dtype=np.float64, copy=True)
    h = _fix_sum(h)
    h_prev = h.copy()
    t_k = 1.0
    lr = lr_init

    best_val = _score(h)
    best_h = h.copy()

    for it in range(max_iter):
        if time.time() > _DEADLINE - 5:
            break

        val, grad = _smooth_max_val_grad(h, beta)
        if val > t_target:
            h_new = h - lr * grad
        else:
            h_new = h - lr * 0.1 * grad

        h_new = _fix_sum(h_new)

        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t_k * t_k)) / 2.0
        h_next = h_new + ((t_k - 1.0) / t_new) * (h_new - h_prev)
        h_next = _fix_sum(h_next)

        h_prev = h
        h = h_next
        t_k = t_new

        if it % 80 == 79:
            lr = max(0.001, min(0.05, lr * 0.95))

        if it % 40 == 39:
            current_score = _score(h)
            if current_score < best_val:
                best_val = current_score
                best_h = h.copy()

    return best_h, best_val

def _discrete_local_search(h_init, max_iter=5000):
    """
    Structural change: Discrete local search on binary-like sequences.
    We maintain sum = N/2 by flipping pairs (0->1, 1->0).
    """
    h = np.array(h_init, dtype=np.float64, copy=True)
    # Project to binary-like: Threshold at 0.5?
    # Better: Start with the best continuous h, round to binary, then fix sum.
    h_bin = (h > 0.5).astype(float)
    # Fix sum to be exactly N/2
    current_sum = np.sum(h_bin)
    target_sum = N / 2.0
    if current_sum < target_sum:
        # Need to turn some 0s to 1s
        zeros = np.where(h_bin == 0.0)[0]
        # Sort zeros by h value (higher h is "closer" to 1)
        order = np.argsort(-h[zeros])
        count_needed = int(round(target_sum - current_sum))
        if count_needed > 0:
            for idx in order[:count_needed]:
                h_bin[zeros[idx]] = 1.0
    elif current_sum > target_sum:
        # Need to turn some 1s to 0s
        ones = np.where(h_bin == 1.0)[0]
        # Sort ones by h value (lower h is "closer" to 0)
        order = np.argsort(h[ones])
        count_needed = int(round(current_sum - target_sum))
        if count_needed > 0:
            for idx in order[:count_needed]:
                h_bin[ones[idx]] = 0.0
        
    h = h_bin
    current_score = _score(h)
    best_h = h.copy()
    best_score = current_score
    
    # Precompute overlap for speed? No, FFT is fast enough for N=100.
    # But we can optimize the check.
    
    for it in range(max_iter):
        if time.time() > _DEADLINE - 10:
            break
            
        # Randomly select a 1 and a 0 to swap
        ones = np.where(h == 1.0)[0]
        zeros = np.where(h == 0.0)[0]
        if len(ones) == 0 or len(zeros) == 0:
            break
        
        i = np.random.choice(ones)
        j = np.random.choice(zeros)
        
        # Try the flip
        h[i] = 0.0
        h[j] = 1.0
        
        new_score = _score(h)
        
        if new_score < best_score:
            best_score = new_score
            best_h = h.copy()
            current_score = new_score
        else:
            # Revert
            h[i] = 1.0
            h[j] = 0.0
            # If we reverted, current_score is still the old best_score? 
            # Actually, if we revert, we are back to the previous state.
            # We should only keep the state if it improved.
            # So if it didn't improve, we revert and current_score remains best_score?
            # No, current_score should track the *current* h.
            # If we revert, current h is the previous best.
            # So current_score = best_score.
            current_score = best_score
            
    return best_h, best_score

def construct():
    t0 = time.time()
    np.random.seed(42)

    best_h = None
    best_score = float('inf')

    def _update_best(h):
        nonlocal best_h, best_score
        h = _fix_sum(h)
        s = _score(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()

    i = np.arange(N)

    # Phase 1: Generate diverse structured starts
    starts = []

    for _ in range(10):
        h_rand = np.random.rand(N)
        starts.append(_fix_sum(h_rand))

    for f in [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20]:
        for phase in [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4, 7*np.pi/4]:
            h_sine = 0.5 + 0.3 * np.sin(2 * np.pi * f * i / N + phase)
            starts.append(_fix_sum(h_sine))

    for f in [1, 2, 3, 4, 5, 6]:
        for amp in [0.15, 0.25, 0.35, 0.45]:
            h_cos = 0.5 + amp * np.cos(2 * np.pi * f * i / N)
            starts.append(_fix_sum(h_cos))

    # Multi-frequency combinations
    for f1, f2, a1, a2 in [(1, 2, 0.3, 0.2), (1, 3, 0.25, 0.25), (2, 3, 0.2, 0.3),
                            (1, 4, 0.3, 0.15), (2, 5, 0.2, 0.2), (3, 5, 0.2, 0.25)]:
        h_combo = 0.5 + a1 * np.cos(2 * np.pi * f1 * i / N) + a2 * np.cos(2 * np.pi * f2 * i / N)
        starts.append(_fix_sum(h_combo))
        h_combo2 = 0.5 + a1 * np.sin(2 * np.pi * f1 * i / N) + a2 * np.sin(2 * np.pi * f2 * i / N)
        starts.append(_fix_sum(h_combo2))

    # Sawtooth-like
    h_saw = 0.5 + 0.5 * np.sin(2.0 * np.pi * i / N) * np.sign(np.sin(4.0 * np.pi * i / N))
    starts.append(_fix_sum(h_saw))

    # Alternating block patterns
    for block_size in [2, 3, 4, 5]:
        h_block = np.zeros(N)
        for j in range(N):
            h_block[j] = 0.8 if (j // block_size) % 2 == 0 else 0.2
        starts.append(_fix_sum(h_block))

    for h0 in starts:
        _update_best(h0)

    # Phase 2: Bisection on t with epigraph feasibility
    if best_h is not None:
        t_low = 0.0
        t_high = best_score

        for bisect_iter in range(25):
            if time.time() > _DEADLINE - 25:
                break

            t_mid = (t_low + t_high) / 2.0
            beta = 30.0 + bisect_iter * 12

            h_candidate, s_candidate = _epigraph_feasibility(
                best_h, t_mid, beta, max_iter=200, lr_init=0.03
            )
            if s_candidate < best_score:
                _update_best(h_candidate)

            # Try from perturbed start
            h_pert = best_h + 0.03 * np.random.randn(N)
            h_pert = _fix_sum(h_pert)
            h_candidate2, s_candidate2 = _epigraph_feasibility(
                h_pert, t_mid, beta, max_iter=120, lr_init=0.03
            )
            if s_candidate2 < best_score:
                _update_best(h_candidate2)

            if best_score <= t_mid + 1e-7:
                t_high = t_mid
            else:
                t_low = t_mid

            if t_high - t_low < 1e-9:
                break

    # Phase 3: Full-space Nesterov accelerated gradient with multiple betas
    if best_h is not None:
        for beta in [50.0, 100.0, 200.0, 350.0]:
            if time.time() > _DEADLINE - 15:
                break
            h_candidate, s_candidate = _accelerated_smooth_max(best_h, beta, max_iter=500, lr_init=0.04)
            if s_candidate < best_score:
                _update_best(h_candidate)

    # Phase 4: Final high-beta polish
    if best_h is not None and time.time() < _DEADLINE - 8:
        h_candidate, s_candidate = _accelerated_smooth_max(best_h, beta=500.0, max_iter=300, lr_init=0.015)
        if s_candidate < best_score:
            _update_best(h_candidate)

    # Phase 5: Ultra-high-beta final polish with tiny steps
    if best_h is not None and time.time() < _DEADLINE - 4:
        h_candidate, s_candidate = _accelerated_smooth_max(best_h, beta=1000.0, max_iter=150, lr_init=0.004)
        if s_candidate < best_score:
            _update_best(h_candidate)

    # Phase 6: Final micro-polish
    if best_h is not None and time.time() < _DEADLINE - 2:
        h_candidate, s_candidate = _accelerated_smooth_max(best_h, beta=2000.0, max_iter=80, lr_init=0.001)
        if s_candidate < best_score:
            _update_best(h_candidate)

    # Phase 7: NEW STRUCTURAL CHANGE: Discrete Local Search
    if best_h is not None and time.time() < _DEADLINE - 2:
        # Try discrete search from the best continuous solution
        h_discrete, s_discrete = _discrete_local_search(best_h, max_iter=20000)
        if s_discrete < best_score:
            _update_best(h_discrete)
            
        # Also try a few random binary starts
        for _ in range(5):
            h_rand_bin = np.zeros(N)
            indices = np.random.choice(N, N//2, replace=False)
            h_rand_bin[indices] = 1.0
            h_discrete2, s_discrete2 = _discrete_local_search(h_rand_bin, max_iter=5000)
            if s_discrete2 < best_score:
                _update_best(h_discrete2)

    if best_h is None:
        best_h = np.full(N, 0.5)

    best_h = _fix_sum(best_h)
    return best_h
