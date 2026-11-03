"""
Erdos minimum overlap: minimize max overlap of h with shifted 1-h.
Combines FFT-based smooth-max optimization (Nesterov accelerated gradient descent
with epigraph formulation) from the inspiration with a chirp-structured
parameterization for initialization and refinement, plus multi-stage polish.
"""
import time
import numpy as np

N = 100
_T_START = time.time()
_DEADLINE = _T_START + 140.0  # Safety margin before 150s kill
TARGET_SUM = N / 2.0


def _linear_overlap_values(h):
    """
    Compute C_k = (2/N) * sum_i h_i * (1 - h_{(i-k)%N})
    = (2/N) * (sum_i h_i - sum_i h_i * h_{(i-k)%N})
    = (2/N) * (N/2 - r_k)  where r_k is the circular autocorrelation.
    """
    n = len(h)
    n_fft = 1
    while n_fft < 2 * n:
        n_fft *= 2
    h_pad = np.zeros(n_fft)
    h_pad[:n] = h
    H = np.fft.rfft(h_pad)
    corr = np.fft.irfft(H * np.conj(H), n=n_fft)
    r = corr[:n]
    cs = (2.0 / n) * (n / 2.0 - r)
    return cs


def _score(h):
    """Compute the maximum overlap value."""
    h = np.asarray(h, dtype=np.float64)
    cs = _linear_overlap_values(h)
    return float(np.max(cs))


def _project(h, target_sum):
    """
    Project h onto the set {x : 0 <= x_i <= 1, sum(x) = target_sum}.
    Uses bisection on the Lagrange multiplier for the sum constraint.
    """
    h = np.asarray(h, dtype=np.float64).copy()
    low = -np.max(h) - 1.0
    high = 1.0 - np.min(h) + 1.0

    current_sum = np.sum(h)
    if abs(current_sum - target_sum) < 1e-12 and np.all(h >= -1e-12) and np.all(h <= 1 + 1e-12):
        return np.clip(h, 0.0, 1.0)

    def sum_of_clipped(offset):
        return np.sum(np.clip(h + offset, 0.0, 1.0))

    for _ in range(60):
        mid = (low + high) / 2.0
        s_mid = sum_of_clipped(mid)
        if s_mid > target_sum:
            high = mid
        else:
            low = mid

    offset = (low + high) / 2.0
    h_proj = np.clip(h + offset, 0.0, 1.0)

    diff = target_sum - np.sum(h_proj)
    if abs(diff) > 1e-14:
        if diff > 0:
            idx = np.argmin(h_proj)
            h_proj[idx] = min(1.0, h_proj[idx] + diff)
        else:
            idx = np.argmax(h_proj)
            h_proj[idx] = max(0.0, h_proj[idx] + diff)
        h_proj = np.clip(h_proj, 0.0, 1.0)

    return h_proj


def _smooth_max_obj_grad(h, beta):
    """
    Compute the value and gradient of the smooth max of C_k.
    F(h) = (1/beta) * log(sum(exp(beta * C_k)))
    grad F = sum(w_k * grad C_k)
    where C_k = (2/N)(N/2 - r_k), r_k = sum_i h_i h_{(i-k)%N}
    dC_k/dh_j = (2/N)(-h_{(j-k)%N} - h_{(j+k)%N})
    """
    cs = _linear_overlap_values(h)

    cs_max = np.max(cs)
    exp_cs = np.exp(beta * (cs - cs_max))
    sum_exp = np.sum(exp_cs)
    w = exp_cs / sum_exp

    val = (1.0 / beta) * (np.log(sum_exp) + beta * cs_max)

    n_fft = 1
    while n_fft < 2 * N:
        n_fft *= 2

    w_pad = np.zeros(n_fft)
    w_pad[:N] = w
    h_pad = np.zeros(n_fft)
    h_pad[:N] = h

    W = np.fft.rfft(w_pad)
    H = np.fft.rfft(h_pad)

    # A_j = sum_k w_k * h_{(j-k)%N}  (circular convolution w * h)
    conv = np.fft.irfft(W * H, n=n_fft)
    A = conv[:N]

    # B_j = sum_k w_k * h_{(j+k)%N}  (circular cross-correlation)
    w_rev = w[::-1]
    w_rev_pad = np.zeros(n_fft)
    w_rev_pad[:N] = w_rev
    W_rev = np.fft.rfft(w_rev_pad)
    conv_rev = np.fft.irfft(W_rev * H, n=n_fft)
    B = conv_rev[N-1 : N-1+N]

    grad = -(2.0 / N) * (A + B)

    return val, grad


def _epigraph_obj_grad(h, t, beta):
    """
    Epigraph formulation: minimize t + (1/beta)*log(sum(exp(beta*(C_k - t))))
    Returns (val, grad_h, grad_t).
    """
    cs = _linear_overlap_values(h)
    diff = cs - t
    diff_max = np.max(diff)
    exp_diff = np.exp(beta * (diff - diff_max))
    sum_exp = np.sum(exp_diff)
    w = exp_diff / sum_exp

    val = t + (1.0 / beta) * (np.log(sum_exp) + beta * diff_max)

    n_fft = 1
    while n_fft < 2 * N:
        n_fft *= 2

    w_pad = np.zeros(n_fft)
    w_pad[:N] = w
    h_pad = np.zeros(n_fft)
    h_pad[:N] = h

    W = np.fft.rfft(w_pad)
    H = np.fft.rfft(h_pad)

    conv = np.fft.irfft(W * H, n=n_fft)
    A = conv[:N]

    w_rev = w[::-1]
    w_rev_pad = np.zeros(n_fft)
    w_rev_pad[:N] = w_rev
    W_rev = np.fft.rfft(w_rev_pad)
    conv_rev = np.fft.irfft(W_rev * H, n=n_fft)
    B = conv_rev[N-1 : N-1+N]

    grad_h = -(2.0 / N) * (A + B)
    grad_t = 1.0 - np.sum(w)

    return val, grad_h, grad_t


def _chirp_initialization(sweep_rate, phase_offset=0.0, amplitude_scale=0.5):
    """
    Generate step heights from a linear chirp with quadratic phase.
    h_i = amplitude_scale * (0.5 + 0.5 * cos(2*pi*sweep_rate*i/N + phase_offset))
    This produces low-autocorrelation structures.
    """
    i = np.arange(N)
    phase = 2.0 * np.pi * sweep_rate * i / N + phase_offset
    h = amplitude_scale * (0.5 + 0.5 * np.cos(phase))
    h = np.clip(h, 0.0, 1.0)
    # Ensure correct sum
    current_sum = np.sum(h)
    if current_sum > 0:
        h = h * (TARGET_SUM / current_sum)
        h = np.clip(h, 0.0, 1.0)
    h = _project(h, TARGET_SUM)
    return h


def _nesterov_accelerated_gradient(h0, beta, n_iter=300, lr=0.05):
    """
    Nesterov accelerated gradient descent on the smooth max objective
    with adaptive learning rate and backtracking line search.
    """
    h = np.array(h0, dtype=np.float64)
    h = _project(h, TARGET_SUM)

    h_prev = h.copy()
    t = 1.0

    best_h = h.copy()
    best_score = _score(h)

    lr_current = lr
    max_backtracks = 10

    for i in range(n_iter):
        if time.time() > _DEADLINE - 5:
            break

        t_new = (1 + np.sqrt(1 + 4 * t * t)) / 2
        momentum = (t - 1) / t_new

        y = h + momentum * (h - h_prev)

        val_y, grad = _smooth_max_obj_grad(y, beta)

        step_lr = lr_current
        h_new = None
        for bt in range(max_backtracks):
            h_candidate = y - step_lr * grad
            h_candidate = _project(h_candidate, TARGET_SUM)
            val_candidate = _score(h_candidate)
            if val_candidate <= val_y + 1e-8:
                h_new = h_candidate
                lr_current = step_lr
                break
            step_lr *= 0.5

        if h_new is None:
            h_new = y - lr_current * 0.5 * grad
            h_new = _project(h_new, TARGET_SUM)
            lr_current *= 0.5

        h_prev = h
        h = h_new
        t = t_new

        current_score = _score(h)
        if current_score < best_score:
            best_score = current_score
            best_h = h.copy()

    return best_h, best_score


def _nesterov_epigraph(h0, beta, n_iter=300, lr=0.03):
    """
    Nesterov accelerated gradient descent using epigraph formulation.
    """
    h = np.array(h0, dtype=np.float64)
    h = _project(h, TARGET_SUM)

    t_val = _score(h)

    h_prev = h.copy()
    t_prev = t_val
    t_mom = 1.0

    best_h = h.copy()
    best_score = _score(h)

    lr_current = lr
    max_backtracks = 10

    for i in range(n_iter):
        if time.time() > _DEADLINE - 5:
            break

        t_mom_new = (1 + np.sqrt(1 + 4 * t_mom * t_mom)) / 2
        momentum = (t_mom - 1) / t_mom_new

        y_h = h + momentum * (h - h_prev)
        y_t = t_val + momentum * (t_val - t_prev)

        val_y, grad_h, grad_t = _epigraph_obj_grad(y_h, y_t, beta)

        step_lr = lr_current
        h_new = None
        t_new_val = None
        for bt in range(max_backtracks):
            h_candidate = y_h - step_lr * grad_h
            h_candidate = _project(h_candidate, TARGET_SUM)
            t_candidate = y_t - step_lr * grad_t
            t_candidate = max(0.0, t_candidate)

            val_candidate = _score(h_candidate)
            if val_candidate <= val_y + 1e-8:
                h_new = h_candidate
                t_new_val = t_candidate
                lr_current = step_lr
                break
            step_lr *= 0.5

        if h_new is None:
            h_new = y_h - lr_current * 0.5 * grad_h
            h_new = _project(h_new, TARGET_SUM)
            t_new_val = max(0.0, y_t - lr_current * 0.5 * grad_t)
            lr_current *= 0.5

        h_prev = h
        t_prev = t_val
        h = h_new
        t_val = t_new_val
        t_mom = t_mom_new

        current_score = _score(h)
        if current_score < best_score:
            best_score = current_score
            best_h = h.copy()

    return best_h, best_score


def _high_beta_polish(h0, n_iter=80, beta=300.0, lr=0.003):
    """
    Final high-precision polish pass with very high beta and small learning rate.
    Uses plain projected gradient descent for stability at high beta.
    """
    h = np.array(h0, dtype=np.float64)
    h = _project(h, TARGET_SUM)

    best_h = h.copy()
    best_score = _score(h)

    for i in range(n_iter):
        if time.time() > _DEADLINE - 2:
            break

        val, grad = _smooth_max_obj_grad(h, beta)

        h_new = h - lr * grad
        h_new = _project(h_new, TARGET_SUM)

        current_score = _score(h_new)
        if current_score < best_score:
            best_score = current_score
            best_h = h_new.copy()
            h = h_new.copy()
        else:
            lr *= 0.95

    return best_h, best_score


def _coordinate_descent_polish(h0, n_iter=200):
    """
    Multi-stage coordinate descent: for each coordinate, find the best value
    in [0,1] that maintains the sum constraint and minimizes the score.
    """
    h = np.array(h0, dtype=np.float64)
    h = _project(h, TARGET_SUM)

    best_h = h.copy()
    best_score = _score(h)

    for iteration in range(n_iter):
        if time.time() > _DEADLINE - 2:
            break

        improved = False
        for j in range(N):
            if time.time() > _DEADLINE - 2:
                break

            old_val = h[j]
            # Try a few candidate values for h[j]
            candidates = [0.0, 1.0, old_val - 0.1, old_val + 0.1, old_val - 0.01, old_val + 0.01]
            candidates = [max(0.0, min(1.0, c)) for c in candidates]

            best_candidate = old_val
            best_candidate_score = best_score

            for c in candidates:
                if abs(c - old_val) < 1e-12:
                    continue
                h_trial = h.copy()
                h_trial[j] = c
                # Adjust sum constraint by distributing the difference
                diff = TARGET_SUM - np.sum(h_trial)
                if abs(diff) > 1e-10:
                    # Distribute to other coordinates
                    others = [i for i in range(N) if i != j]
                    if len(others) > 0:
                        # Add to coordinate with most room
                        if diff > 0:
                            idx = others[np.argmin(h_trial[others])]
                            add = min(diff, 1.0 - h_trial[idx])
                            h_trial[idx] += add
                            remaining = diff - add
                            if abs(remaining) > 1e-10:
                                h_trial += remaining / (N - 1)
                                h_trial[j] = c
                                h_trial = np.clip(h_trial, 0.0, 1.0)
                        else:
                            idx = others[np.argmax(h_trial[others])]
                            sub = min(-diff, h_trial[idx])
                            h_trial[idx] -= sub
                            remaining = diff + sub
                            if abs(remaining) > 1e-10:
                                h_trial += remaining / (N - 1)
                                h_trial[j] = c
                                h_trial = np.clip(h_trial, 0.0, 1.0)

                h_trial = _project(h_trial, TARGET_SUM)
                s = _score(h_trial)
                if s < best_candidate_score - 1e-12:
                    best_candidate = c
                    best_candidate_score = s

            if best_candidate != old_val:
                h[j] = best_candidate
                h = _project(h, TARGET_SUM)
                current_score = _score(h)
                if current_score < best_score:
                    best_score = current_score
                    best_h = h.copy()
                    improved = True

        if not improved:
            break

    return best_h, best_score


def _generate_initializations():
    """Generate diverse initializations including chirp-based ones."""
    inits = []

    # 1. Uniform
    inits.append(np.full(N, 0.5))

    # 2. Alternating
    alt = np.zeros(N)
    alt[::2] = 1.0
    alt[1::2] = 0.0
    inits.append(alt.copy())

    # 3. Block: first half 1, second half 0
    block = np.zeros(N)
    block[:N//2] = 1.0
    inits.append(block.copy())

    # 4. Chirp-based initializations (key structural idea)
    for sweep in [1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 13.0, 17.0, 23.0, 31.0]:
        for amp in [0.3, 0.5, 0.7]:
            h = _chirp_initialization(sweep, phase_offset=0.0, amplitude_scale=amp)
            inits.append(h)

    # 5. Random with correct sum
    np.random.seed(42)
    for trial in range(8):
        x_rand = np.random.uniform(0, 1, N)
        x_rand = _project(x_rand, TARGET_SUM)
        inits.append(x_rand)

    # 6. Frequency-based
    for freq in [1, 2, 3, 5, 7, 11]:
        if freq < N:
            t = np.linspace(0, 2 * np.pi * freq, N, endpoint=False)
            x_freq = 0.5 + 0.5 * np.sin(t)
            x_freq = _project(x_freq, TARGET_SUM)
            inits.append(x_freq)

    # 7. Chirp with random phase offsets
    for sweep in [3.0, 5.0, 7.0]:
        for phase in [0.0, np.pi/3, 2*np.pi/3, np.pi]:
            h = _chirp_initialization(sweep, phase_offset=phase, amplitude_scale=0.5)
            inits.append(h)

    return inits


def construct():
    start_time = time.time()

    def time_left():
        return _DEADLINE - time.time()

    # Generate initializations
    initializations = _generate_initializations()

    # Evaluate all initializations
    best_x = None
    best_score = float('inf')

    for x_init in initializations:
        if time_left() < 3.0:
            break
        x_init = _project(x_init, TARGET_SUM)
        score = _score(x_init)
        if score < best_score:
            best_score = score
            best_x = x_init.copy()

    # Main optimization: Nesterov accelerated gradient with smooth max
    # Try multiple initializations with different betas
    top_inits = initializations[:min(len(initializations), 15)]

    for x_init in top_inits:
        if time_left() < 15.0:
            break

        x_init = _project(x_init, TARGET_SUM)

        for beta in [50.0, 100.0, 200.0]:
            if time_left() < 10.0:
                break

            x_opt, score_opt = _nesterov_accelerated_gradient(x_init, beta, n_iter=200, lr=0.05)
            if score_opt < best_score:
                best_score = score_opt
                best_x = x_opt.copy()

    # Epigraph formulation
    if time_left() > 20.0:
        for x_init in top_inits[:8]:
            if time_left() < 10.0:
                break

            x_init = _project(x_init, TARGET_SUM)

            for beta in [100.0, 200.0]:
                if time_left() < 8.0:
                    break

                x_opt, score_opt = _nesterov_epigraph(x_init, beta, n_iter=150, lr=0.03)
                if score_opt < best_score:
                    best_score = score_opt
                    best_x = x_opt.copy()

    # High-beta polish
    if time_left() > 10.0:
        x_polished, score_polished = _high_beta_polish(best_x, n_iter=60, beta=300.0, lr=0.003)
        if score_polished < best_score:
            best_score = score_polished
            best_x = x_polished.copy()

    # Coordinate descent polish
    if time_left() > 5.0:
        x_cd, score_cd = _coordinate_descent_polish(best_x, n_iter=100)
        if score_cd < best_score:
            best_score = score_cd
            best_x = x_cd.copy()

    # Final high-beta polish
    if time_left() > 3.0:
        x_final, score_final = _high_beta_polish(best_x, n_iter=30, beta=500.0, lr=0.001)
        if score_final < best_score:
            best_score = score_final
            best_x = x_final.copy()

    # Final enforcement of constraints
    best_x = _project(best_x, TARGET_SUM)
    final_score = _score(best_x)

    return best_x
