import numpy as np
import time
import random

N = 50

def _ratio(h):
    h = np.maximum(h, 0.0)
    s = np.sum(h)
    if s < 1e-15:
        return 0.0
    l2_sq = np.sum(h ** 2)
    conv = np.convolve(h, h, mode='full')
    sup_conv = np.max(conv)
    if sup_conv < 1e-15:
        return 0.0
    return l2_sq / (s * sup_conv)

def _reweight(h, eta, n_steps=80):
    """Iterative multiplicative updates: reduce h_i proportional to its
    contribution to the current autoconvolution peak, renormalize to L1=1."""
    h = np.maximum(h, 0.0).copy()
    s = np.sum(h)
    if s < 1e-15:
        return h
    h = h / s
    best_h = h.copy()
    best_r = _ratio(h)
    for _ in range(n_steps):
        conv = np.convolve(h, h, mode='full')
        k = int(np.argmax(conv))
        # Gradient of conv[k] w.r.t. h[i]:
        #   d conv[k]/d h[i] = h[k-i] (if valid) + h[i-k] (if valid)
        grad = np.zeros(N)
        for i in range(N):
            a = k - i
            b = i - k
            v = 0.0
            if 0 <= a < N:
                v += h[a]
            if 0 <= b < N:
                v += h[b]
            grad[i] = v
        h = h * np.exp(-eta * grad)
        s = np.sum(h)
        if s < 1e-15:
            break
        h = h / s
        r = _ratio(h)
        if r > best_r:
            best_r = r
            best_h = h.copy()
    return best_h

def _random_h():
    """Random nonnegative vector with L1=1."""
    h = np.abs(np.random.randn(N))
    s = np.sum(h)
    if s < 1e-15:
        h[0] = 1.0
        return h
    return h / s

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # 15s safety margin under 150s limit

    random.seed(42)
    np.random.seed(42)

    best_h = np.zeros(N)
    best_ratio = 0.0

    # --- Phase 1: Global search via reweighting from diverse starts ---
    # Try multiple random initializations; reweighting from each converges
    # to a different local optimum. Keep the best.
    n_starts = 60
    for it in range(n_starts):
        if time.time() > deadline:
            break

        # Vary the "spikiness" of the initialization
        if it == 0:
            h0 = np.ones(N) / N
        elif it == 1:
            h0 = np.zeros(N)
            h0[0] = 1.0
        elif it == 2:
            # Sparse: 8 random positions
            idx = np.random.choice(N, size=8, replace=False)
            h0 = np.zeros(N)
            h0[idx] = 1.0 / 8
        elif it == 3:
            # Very sparse: 5 positions
            idx = np.random.choice(N, size=5, replace=False)
            h0 = np.zeros(N)
            h0[idx] = 1.0 / 5
        else:
            # Random with varying sparsity
            sparsity = np.random.choice([0.1, 0.2, 0.3, 0.5, 1.0])
            mask = np.random.rand(N) < sparsity
            if not np.any(mask):
                mask[0] = True
            h0 = np.where(mask, np.abs(np.random.randn(N)), 0.0)
            s = np.sum(h0)
            if s < 1e-15:
                h0[0] = 1.0
            else:
                h0 = h0 / s

        # Run reweighting with a few different eta values
        for eta in [0.05, 0.1, 0.2]:
            if time.time() > deadline:
                break
            h_new = _reweight(h0, eta, n_steps=100)
            r = _ratio(h_new)
            if r > best_ratio:
                best_ratio = r
                best_h = h_new.copy()

    # --- Phase 2: Support refinement (from current program's strategy) ---
    # Use the best continuous solution to seed a binary support search,
    # then re-optimize heights on that support.
    if time.time() < deadline - 20.0:
        # Identify support from best_h (positions with significant mass)
        threshold = np.max(best_h) * 0.1
        support = np.where(best_h > threshold)[0]
        if len(support) < 2:
            support = np.where(best_h > 0)[0]
        if len(support) < 2:
            support = np.array([0, N // 2])

        # Binary support search via simulated annealing
        current_support = set(support.tolist())
        current_h = np.zeros(N)
        for i in current_support:
            current_h[i] = 1.0
        current_ratio = _ratio(current_h)

        T = 0.1
        for i in range(3000):
            if time.time() > deadline - 10.0:
                break

            move_type = random.random()
            new_support = set(current_support)

            if move_type < 0.3:
                # Add
                zeros = [j for j in range(N) if j not in new_support]
                if zeros:
                    new_support.add(random.choice(zeros))
            elif move_type < 0.6:
                # Remove
                if len(new_support) > 2:
                    new_support.remove(random.choice(list(new_support)))
            else:
                # Swap
                if new_support:
                    ones = list(new_support)
                    zeros = [j for j in range(N) if j not in new_support]
                    if zeros:
                        out = random.choice(ones)
                        new_support.discard(out)
                        new_support.add(random.choice(zeros))

            new_h = np.zeros(N)
            for j in new_support:
                new_h[j] = 1.0
            new_ratio = _ratio(new_h)

            if new_ratio > current_ratio:
                current_support = new_support
                current_h = new_h
                current_ratio = new_ratio
                if current_ratio > best_ratio:
                    best_ratio = current_ratio
                    best_h = current_h.copy()
            else:
                delta = current_ratio - new_ratio
                if random.random() < np.exp(-delta / T):
                    current_support = new_support
                    current_h = new_h
                    current_ratio = new_ratio

            T = max(T * 0.99, 0.01)

        # If the binary support gave a better ratio, re-optimize heights
        # on that support via reweighting
        if current_ratio > best_ratio:
            best_ratio = current_ratio
            best_h = current_h.copy()

        # Continuous reweighting starting from the best binary solution
        if time.time() < deadline - 5.0:
            h_new = _reweight(best_h, 0.05, n_steps=200)
            r = _ratio(h_new)
            if r > best_ratio:
                best_ratio = r
                best_h = h_new.copy()

    # --- Phase 3: Final polish ---
    if time.time() < deadline - 2.0:
        h_new = _reweight(best_h, 0.02, n_steps=100)
        r = _ratio(h_new)
        if r > best_ratio:
            best_ratio = r
            best_h = h_new.copy()

    # Ensure nonnegativity and finite values
    best_h = np.maximum(best_h, 0.0)
    best_h[~np.isfinite(best_h)] = 0.0

    return list(best_h)
