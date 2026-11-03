import numpy as np
import time

N = 50

def _compute_ratio(h):
    """Compute ||h||_2^2 / (||h||_1 * sup|h*h|) for nonnegative h."""
    h = np.asarray(h, dtype=float)
    if np.all(h < 1e-15):
        return 0.0
    l2_sq = np.sum(h**2)
    l1 = np.sum(h)
    if l1 < 1e-15 or l2_sq < 1e-15:
        return 0.0
    conv = np.convolve(h, h, mode='full')
    sup_conv = np.max(conv)
    if sup_conv < 1e-15:
        return 0.0
    return l2_sq / (l1 * sup_conv)


def _generate_sidon_set(n, max_size):
    """Greedy Sidon set construction in [0, n-1]."""
    sidon = [0]
    used_sums = {0}
    for candidate in range(1, n):
        valid = True
        for i in sidon:
            s = i + candidate
            if s in used_sums:
                valid = False
                break
        if valid:
            sidon.append(candidate)
            used_sums.add(candidate + candidate)
            for i in sidon[:-1]:
                used_sums.add(i + candidate)
        if len(sidon) >= max_size:
            break
    return sidon


def _multiplicative_update(h, eta, max_iters=5000, deadline=None, patience=80):
    """
    Iterative multiplicative update to maximize ||h||_2^2 / (||h||_1 * sup(h*h)).
    Fix ||h||_1 = 1, maximize ||h||_2^2 / sup(h*h).
    At each step, reduce h_i where it contributes most to the peak autoconvolution lag.
    """
    h = np.asarray(h, dtype=float).copy()
    s = np.sum(h)
    if s < 1e-15:
        return h, 0.0
    h /= s

    best_h = h.copy()
    best_ratio = _compute_ratio(h)
    stale = 0

    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break

        conv = np.convolve(h, h, mode='full')
        sup_conv = np.max(conv)
        if sup_conv < 1e-15:
            break

        # Find the lag(s) that achieve the maximum
        # Use all lags within a small tolerance of the max for smoother gradients
        k_star = np.argmax(conv)

        # Compute gradient of conv[k_star] w.r.t. h
        # conv[k] = sum_j h[j] * h[k-j] (with h=0 outside [0,N-1])
        # d/dh[i] conv[k] = h[k-i] + h[i-k] (where indices are valid)
        grad_h = np.zeros(N)
        n_star = k_star
        for i in range(N):
            idx1 = n_star - i
            idx2 = i - n_star
            val = 0.0
            if 0 <= idx1 < N:
                val += h[idx1]
            if 0 <= idx2 < N:
                val += h[idx2]
            grad_h[i] = val

        # Also consider the second-highest lag for smoother updates
        # Find top 2 lags
        conv_sorted = np.sort(conv)[::-1]
        if len(conv_sorted) >= 2 and conv_sorted[1] > 0.9 * conv_sorted[0]:
            # There's a near-tie, use weighted gradient
            k_second = np.where(conv == conv_sorted[1])[0][0]
            if k_second != k_star:
                grad_h2 = np.zeros(N)
                for i in range(N):
                    idx1 = k_second - i
                    idx2 = i - k_second
                    val = 0.0
                    if 0 <= idx1 < N:
                        val += h[idx1]
                    if 0 <= idx2 < N:
                        val += h[idx2]
                    grad_h2[i] = val
                grad_h = 0.7 * grad_h + 0.3 * grad_h2

        # Multiplicative update: reduce h where gradient is high
        # h_i <- h_i * exp(-eta * grad_h_i)
        update = np.exp(-eta * grad_h)
        h = h * update

        # Renormalize to sum=1
        s = np.sum(h)
        if s < 1e-15:
            break
        h /= s

        # Evaluate
        r_new = _compute_ratio(h)
        if r_new > best_ratio:
            best_ratio = r_new
            best_h = h.copy()
            stale = 0
        else:
            stale += 1

        if stale > patience:
            # Decay eta and reset patience
            eta *= 0.9
            if eta < 1e-5:
                break
            stale = 0

    return best_h, best_ratio


def _refine_with_local_search(h, deadline):
    """
    Fine-tune using coordinate ascent on individual heights.
    For each coordinate, try to find the optimal height given others fixed.
    """
    h = np.asarray(h, dtype=float).copy()
    s = np.sum(h)
    if s < 1e-15:
        return h, 0.0
    h /= s

    best_h = h.copy()
    best_ratio = _compute_ratio(h)
    eta = 0.05

    for sweep in range(200):
        if time.time() > deadline:
            break

        improved = False
        for i in range(N):
            if time.time() > deadline:
                break

            # Try varying h[i] while keeping others fixed
            # We want to maximize ||h||_2^2 / (||h||_1 * sup(h*h))
            # With h[j] fixed for j != i, let x = h[i]
            # ||h||_2^2 = x^2 + S2 where S2 = sum_{j!=i} h[j]^2
            # ||h||_1 = x + S1 where S1 = sum_{j!=i} h[j]
            # sup(h*h) depends on x in a complex way
            # Use small perturbations
            h_old = h[i]
            for factor in [0.5, 0.7, 0.85, 1.0, 1.15, 1.3, 1.5]:
                h[i] = h_old * factor
                if h[i] < 0:
                    h[i] = 0.0
                r = _compute_ratio(h)
                if r > best_ratio:
                    best_ratio = r
                    best_h = h.copy()
                    improved = True
            # Reset to best found for this coordinate
            h[i] = best_h[i]

        if not improved:
            break

    return best_h, best_ratio


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    best_h = np.zeros(N)
    best_ratio = 0.0

    np.random.seed(12345)

    # Generate diverse initializations
    initializations = []

    # 1. Sidon sets of various sizes
    for size in [6, 7, 8, 9, 10, 12, 14]:
        sidon_set = _generate_sidon_set(N, size)
        h_s = np.zeros(N)
        for idx in sidon_set:
            h_s[idx] = 1.0
        initializations.append(h_s)

    # 2. Sparse random patterns (Sidon-inspired)
    for trial in range(15):
        h_sparse = np.zeros(N)
        num_ones = np.random.randint(4, 16)
        indices = np.random.choice(N, num_ones, replace=False)
        h_sparse[indices] = np.random.uniform(0.5, 2.0, num_ones)
        initializations.append(h_sparse)

    # 3. Alternating sparse
    h_alt = np.zeros(N)
    h_alt[::2] = 1.0
    initializations.append(h_alt)

    # 4. Every 3
    h_sp3 = np.zeros(N)
    h_sp3[::3] = 1.0
    initializations.append(h_sp3)

    # 5. Uniform
    initializations.append(np.ones(N))

    # 6. Triangular
    t = np.linspace(-1, 1, N)
    initializations.append(np.maximum(1 - t**2, 0.01))

    # 7. Bimodal
    initializations.append(np.maximum(1.0 + 2.0*np.exp(-((t-0.6)**2)/0.1) + 2.0*np.exp(-((t+0.6)**2)/0.1), 0.05))

    # 8. Gaussian
    initializations.append(np.exp(-t**2/0.3))

    # 9. Delta (single spike)
    h_delta = np.zeros(N)
    h_delta[0] = 1.0
    initializations.append(h_delta)

    # 10. Two deltas
    h_2d = np.zeros(N)
    h_2d[0] = 1.0
    h_2d[25] = 1.0
    initializations.append(h_2d)

    # Run multiplicative update from each initialization
    for h0 in initializations:
        if time.time() > deadline - 10:
            break

        h0 = np.maximum(h0, 0)
        s0 = np.sum(h0)
        if s0 < 1e-15:
            continue
        h0 = h0 / s0

        # Check initial ratio
        r0 = _compute_ratio(h0)
        if r0 > best_ratio:
            best_ratio = r0
            best_h = h0.copy()

        # Run multiplicative update with different eta values
        for eta in [0.1, 0.05, 0.02]:
            if time.time() > deadline - 10:
                break

            h_ref, r_ref = _multiplicative_update(
                h0, eta, max_iters=3000,
                deadline=deadline - 10, patience=60
            )
            if r_ref > best_ratio:
                best_ratio = r_ref
                best_h = h_ref.copy()

    # Local refinement on the best solution
    if time.time() < deadline - 15:
        h_ref, r_ref = _refine_with_local_search(best_h, deadline - 10)
        if r_ref > best_ratio:
            best_ratio = r_ref
            best_h = h_ref.copy()

    # One more round of multiplicative update from the refined solution
    if time.time() < deadline - 15:
        for eta in [0.03, 0.01]:
            if time.time() > deadline - 10:
                break
            h_ref, r_ref = _multiplicative_update(
                best_h, eta, max_iters=2000,
                deadline=deadline - 10, patience=40
            )
            if r_ref > best_ratio:
                best_ratio = r_ref
                best_h = h_ref.copy()

    # Final local refinement
    if time.time() < deadline - 5:
        h_ref, r_ref = _refine_with_local_search(best_h, deadline - 2)
        if r_ref > best_ratio:
            best_ratio = r_ref
            best_h = h_ref.copy()

    # Ensure nonnegative
    best_h = np.maximum(best_h, 0)

    return best_h.tolist()
