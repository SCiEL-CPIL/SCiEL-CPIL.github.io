import numpy as np
import time
import sys

N = 50

def _ratio(h):
    """Compute ||h||_2^2 / (||h||_1 * ||h||_inf * ||h*h||_inf)"""
    h = np.asarray(h, dtype=float)
    l2_sq = np.sum(h**2)
    l1 = np.sum(h)
    l_inf = np.max(h)
    if l1 < 1e-15 or l_inf < 1e-15:
        return 0.0
    # autoconvolution
    h_conv = np.convolve(h, h, mode='full')
    max_conv = np.max(h_conv)
    if max_conv < 1e-15:
        return 0.0
    return (l2_sq) / (l1 * l_inf * max_conv)

def _spike_ratio(indices):
    """
    For a set of spike indices (each with height 1), compute the ratio.
    ||h||_2^2 = k, ||h||_1 = k, ||h||_inf = 1
    So ratio = k / max_conv
    """
    if len(indices) == 0:
        return 0.0
    h = np.zeros(N)
    for idx in indices:
        h[idx] = 1.0
    h_conv = np.convolve(h, h, mode='full')
    max_conv = np.max(h_conv)
    if max_conv < 1e-15:
        return 0.0
    return len(indices) / max_conv

def _sidon_set(max_n):
    """
    Construct a Sidon set in [0, N-1] greedily.
    A Sidon set has distinct pairwise sums, so no two pairs collide.
    """
    S = set()
    used_sums = set()
    for i in range(max_n):
        ok = True
        for j in S:
            s = i + j
            if s in used_sums:
                ok = False
                break
        if ok:
            for j in S:
                used_sums.add(i + j)
            S.add(i)
    return sorted(S)

def _random_sidon_like(n_spikes, rng, max_trials=2000):
    """
    Try to build a Sidon set of size n_spikes within [0, N-1].
    """
    for _ in range(max_trials):
        S = []
        used_sums = set()
        while len(S) < n_spikes:
            candidates = list(range(N))
            # Try to find a number that can be added
            found = False
            for _ in range(N):
                # Shuffle candidates for diversity
                rng.shuffle(candidates)
                for c in candidates:
                    if c in S:
                        continue
                    can_add = True
                    new_sums = []
                    for s in S:
                        sm = c + s
                        if sm in used_sums:
                            can_add = False
                            break
                        new_sums.append(sm)
                    if can_add:
                        S.append(c)
                        used_sums.update(new_sums)
                        found = True
                        break
                if found:
                    break
            if not found:
                break
        if len(S) == n_spikes:
            return S
    return None

def _refined_optimize(indices, deadline):
    """
    Refine the spike pattern by allowing fractional heights around each spike.
    This lets the optimizer tune each height to balance L2 vs convolution peak.
    """
    from scipy.optimize import minimize

    k = len(indices)
    if k == 0:
        return np.zeros(N), 0.0

    # Start with all heights = 1
    x0 = np.ones(k)

    def _obj(x):
        if np.any(x < -1e-12):
            return 1e10
        h = np.zeros(N)
        for i, idx in enumerate(indices):
            h[idx] = max(0.0, x[i])
        l2_sq = np.sum(h**2)
        l1 = np.sum(h)
        l_inf = np.max(h)
        if l1 < 1e-15 or l_inf < 1e-15:
            return 1e10
        h_conv = np.convolve(h, h, mode='full')
        max_conv = np.max(h_conv)
        if max_conv < 1e-15:
            return 1e10
        r = l2_sq / (l1 * l_inf * max_conv)
        return -r

    try:
        res = minimize(_obj, x0, method='L-BFGS-B',
                       bounds=[(0.0, 2.0)] * k,
                       options={'maxiter': 500, 'ftol': 1e-10})
        x_opt = np.clip(res.x, 0.0, None)
        h = np.zeros(N)
        for i, idx in enumerate(indices):
            h[idx] = x_opt[i]
        score = _ratio(h)
        return h, score
    except Exception:
        h = np.zeros(N)
        for idx in indices:
            h[idx] = 1.0
        return h, _spike_ratio(indices)

def _simulated_annealing(indices, deadline):
    """
    Simulated annealing on the set of spike positions.
    Moves: add a spike, remove a spike, swap a spike position.
    """
    rng = np.random.RandomState(42)
    current = list(indices)
    current_score = _spike_ratio(current)
    best = list(current)
    best_score = current_score

    T = 1.0
    T_min = 1e-6
    cooling = 0.995

    while time.time() < deadline and T > T_min:
        move_type = rng.random()
        if move_type < 0.4:
            candidates = [i for i in range(N) if i not in current]
            if not candidates:
                T *= cooling
                continue
            new_idx = rng.choice(candidates)
            neighbor = current + [new_idx]
        elif move_type < 0.7:
            if len(current) <= 2:
                T *= cooling
                continue
            remove_idx = rng.choice(len(current))
            neighbor = [x for j, x in enumerate(current) if j != remove_idx]
        else:
            if not current:
                T *= cooling
                continue
            remove_idx = rng.choice(len(current))
            candidates = [i for i in range(N) if i not in current]
            if not candidates:
                T *= cooling
                continue
            new_idx = rng.choice(candidates)
            neighbor = [x for j, x in enumerate(current) if j != remove_idx]
            neighbor.append(new_idx)

        neighbor_score = _spike_ratio(neighbor)
        delta = neighbor_score - current_score

        if delta > 0 or rng.random() < np.exp(delta / T):
            current = neighbor
            current_score = neighbor_score
            if current_score > best_score:
                best = list(current)
                best_score = current_score

        T *= cooling

    return best, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    # ---- Phase 1: Build a good spike set ----

    # Start from a Sidon set
    sidon = _sidon_set(N)
    # Take up to ~15 largest possible
    sidon_sub = sidon[:min(15, len(sidon))]
    if len(sidon_sub) < 2:
        sidon_sub = [0, 1]

    best_indices = list(sidon_sub)
    best_score = _spike_ratio(best_indices)

    # Try random Sidon-like sets of varying sizes
    rng = np.random.RandomState(123)
    for k in range(3, min(20, N + 1)):
        if time.time() > deadline - 20:
            break
        trial = _random_sidon_like(k, rng, max_trials=300)
        if trial is not None:
            s = _spike_ratio(trial)
            if s > best_score:
                best_indices = trial
                best_score = s

    # Greedy expansion from best_indices
    current = list(best_indices)
    current_score = best_score
    while time.time() < deadline - 15:
        best_add = None
        best_add_score = current_score
        # Precompute convolution of current spikes
        h_cur = np.zeros(N)
        for idx in current:
            h_cur[idx] = 1.0
        conv_cur = np.convolve(h_cur, h_cur, mode='full')
        for i in range(N):
            if i in current:
                continue
            # Fast update: conv_new = conv_cur + 2*h_cur shifted by i + spike at 2i
            # For now, just compute full conv for correctness
            h_new = np.zeros(N)
            for idx in current:
                h_new[idx] = 1.0
            h_new[i] = 1.0
            conv_new = np.convolve(h_new, h_new, mode='full')
            s = len(current) + 1 / np.max(conv_new)
            if s > best_add_score:
                best_add_score = s
                best_add = i
        if best_add is None or best_add_score <= current_score:
            break
        current.append(best_add)
        current_score = best_add_score
    if current_score > best_score:
        best_indices = current
        best_score = current_score

    # Simulated annealing
    if time.time() < deadline - 10:
        sa_result, sa_score = _simulated_annealing(best_indices, deadline - 10)
        if sa_score > best_score:
            best_indices = sa_result
            best_score = sa_score

    # Local search: try removing each spike
    if time.time() < deadline - 5:
        cur = list(best_indices)
        cur_score = best_score
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for i in range(len(cur)):
                trial = [x for j, x in enumerate(cur) if j != i]
                if not trial:
                    continue
                ts = _spike_ratio(trial)
                if ts > cur_score:
                    cur = trial
                    cur_score = ts
                    improved = True
                    break
        if cur_score > best_score:
            best_indices = cur
            best_score = cur_score

    # ---- Phase 2: Fractional-height refinement ----
    if time.time() < deadline - 2:
        h_ref, ref_score = _refined_optimize(best_indices, deadline - 2)
        if ref_score > best_score:
            # Normalize so max is 1 (problem allows any nonnegative heights)
            if np.max(h_ref) > 1e-15:
                h_out = h_ref / np.max(h_ref)
            else:
                h_out = np.zeros(N)
                h_out[0] = 1.0
            # Verify
            final_score = _ratio(h_out)
            return list(h_out)

    # Fallback: pure spikes
    h = np.zeros(N)
    for idx in best_indices:
        h[idx] = 1.0
    if np.max(h) < 1e-15:
        h[0] = 1.0
    return list(h)
