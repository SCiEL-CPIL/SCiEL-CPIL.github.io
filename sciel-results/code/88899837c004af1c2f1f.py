"""Erdos minimum overlap: Barker-style bang-bang + continuous bisection refinement.

We minimize max_k ov[k] where ov[k]=sum_i h[i](1-h[(i+k)%N]) = N/2 - C[k],
C being the circular autocorrelation.  Min max ov <=> max min C.

Two complementary mechanisms:
  * Bang-bang (Barker) search: flip *pairs* to keep sum=N/2 exactly;
    recompute overlap via FFT when needed, use cheap heuristic for acceptance.
  * Continuous refinement: soft-max surrogate + projected gradient descent
    with adaptive step size for smooth optimization of max overlap.
"""
import numpy as np
import time

N = 100
HALF = N / 2.0
DEADLINE = 142.0  # safety margin before 150s kill


def _time_left(t0):
    return DEADLINE - (time.time() - t0)


def _overlap_vector(h):
    """ov[k] = sum_i h[i]*(1-h[(i+k)%N]) via FFT cross-correlation.
    
    Note: ov[k] = sum_i h[i] - sum_i h[i]*h[(i+k)%N] = N/2 - C[k]
    where C[k] = sum_i h[i]*h[(i+k)%N] is circular autocorrelation.
    """
    h = np.asarray(h, dtype=np.float64)
    H = np.fft.fft(h)
    # C[k] = sum_i h[i]*h[(i+k)%N] = IFFT(|H|^2)[k]
    C = np.fft.ifft(np.conj(H) * H).real
    ov = HALF - C
    return ov


def _max_overlap(h):
    return float(np.max(_overlap_vector(h)))


def _project(h):
    """Project to [0,1]^N with sum exactly N/2."""
    h = np.clip(np.asarray(h, dtype=np.float64), 0.0, 1.0)
    for _ in range(40):
        diff = HALF - np.sum(h)
        if abs(diff) < 1e-14:
            break
        if diff > 0:
            mask = h < 1.0 - 1e-15
            if not np.any(mask):
                break
            h[mask] += diff / np.sum(mask)
        else:
            mask = h > 1e-15
            if not np.any(mask):
                break
            h[mask] += diff / np.sum(mask)
        h = np.clip(h, 0.0, 1.0)
    diff = HALF - np.sum(h)
    if abs(diff) > 1e-14:
        idx = np.argsort(1.0 - h) if diff > 0 else np.argsort(h)
        for i in idx:
            if diff > 0:
                add = min(diff, 1.0 - h[i])
                h[i] += add
                diff -= add
            else:
                sub = min(-diff, h[i])
                h[i] -= sub
                diff += sub
            if abs(diff) < 1e-14:
                break
    return h


def _barker_init(rng):
    """Random bang-bang (0/1) with exactly N/2 ones."""
    x = np.zeros(N)
    ones = N // 2
    idx = rng.permutation(N)[:ones]
    x[idx] = 1.0
    return x


def _barker_search(x, t0, max_iter=4000000):
    """Flip-pair local search on bang-bang x minimizing max overlap.
    
    Uses periodic FFT recomputation (every K flips) with cheap heuristic
    acceptance in between. Sum is preserved exactly by flipping pairs.
    """
    x = np.asarray(x, dtype=np.float64).copy()
    ov = _overlap_vector(x)
    cur = float(np.max(ov))
    rng = np.random.default_rng(123)
    it = 0
    stalls = 0
    recomputed = 0
    K = 10  # recompute every K accepted flips
    while it < max_iter and _time_left(t0) > 12:
        it += 1
        a = int(rng.integers(N))
        b = int(rng.integers(N))
        if a == b:
            continue
        # Save state
        xa, xb = x[a], x[b]
        # Apply flip pair
        x[a] = 1.0 - xa
        x[b] = 1.0 - xb
        # Recompute overlap periodically
        if recomputed >= K or it % 50 == 0:
            ov = _overlap_vector(x)
            new = float(np.max(ov))
            recomputed = 0
            if new < cur - 1e-12:
                cur = new
                stalls = 0
            else:
                # Revert
                x[a] = xa
                x[b] = xb
                ov = _overlap_vector(x)
                stalls += 1
                if stalls > 100000:
                    break
            continue
        # Cheap heuristic: use variance of C as proxy
        # When x is bang-bang, C[k] = sum_i x[i]*x[(i+k)%N]
        # Flipping a pair changes C in a structured way
        # Use a simple acceptance: accept if random, reject if not
        # (This is a hill-climbing with occasional random moves)
        # Actually, let's just recompute less frequently but still check
        # Use a subsample check
        if it % 5 == 0:
            ov = _overlap_vector(x)
            new = float(np.max(ov))
            if new < cur - 1e-12:
                cur = new
                stalls = 0
                recomputed = 0
            else:
                x[a] = xa
                x[b] = xb
                ov = _overlap_vector(x)
                stalls += 1
                if stalls > 100000:
                    break
        else:
            # Assume improvement, count as recomputed
            recomputed += 1
    return x, cur


def _barker_multi(x, t0):
    """Run Barker search from a few random perturbations, keep the best."""
    best_x = x.copy()
    best_v = _max_overlap(x)
    rng = np.random.default_rng(7)
    # perturb: flip a handful of pairs
    cand = x.copy()
    for _ in range(6):
        a = int(rng.integers(N)); b = int(rng.integers(N))
        if a != b:
            cand[a] = 1.0 - cand[a]; cand[b] = 1.0 - cand[b]
    xs, v = _barker_search(cand, t0, max_iter=1500000)
    if v < best_v:
        best_x, best_v = xs, v
    return best_x, best_v


def _project_t(h, t):
    """Water-filling projection: max(0, min(1, h - t)) adjusted to sum N/2.

    For t in the active range, sum is strictly decreasing in t, so we can
    bisection on t to hit sum=N/2 exactly."""
    h = np.asarray(h, dtype=np.float64)
    def sum_at(t):
        return np.sum(np.clip(h - t, 0.0, 1.0))
    lo, hi = -1.0, 1.0
    # ensure bracket
    if sum_at(lo) < HALF:
        lo = -2.0
    if sum_at(hi) > HALF:
        hi = 2.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if sum_at(mid) > HALF:
            lo = mid
        else:
            hi = mid
    t = 0.5 * (lo + hi)
    out = np.clip(h - t, 0.0, 1.0)
    # exact fix
    diff = HALF - np.sum(out)
    if abs(diff) > 1e-14:
        if diff > 0:
            mask = out < 1.0 - 1e-15
            if np.any(mask):
                out[mask] += diff / np.sum(mask)
        else:
            mask = out > 1e-15
            if np.any(mask):
                out[mask] += diff / np.sum(mask)
        out = np.clip(out, 0.0, 1.0)
    return out


def _soft_max(ov, temperature=0.05):
    """Smooth approximation of max(ov) using log-sum-exp."""
    scaled = ov / temperature
    max_val = np.max(scaled)
    if max_val > 500:
        # Overflow protection
        return temperature * (max_val + np.log(np.sum(np.exp(scaled - max_val))))
    return temperature * (max_val + np.log(np.sum(np.exp(scaled - max_val))))


def _soft_max_grad(ov, temperature=0.05):
    """Gradient of soft-max w.r.t. input."""
    scaled = ov / temperature
    max_val = np.max(scaled)
    exp_scaled = np.exp(scaled - max_val)
    weights = exp_scaled / np.sum(exp_scaled)
    return weights / temperature


def _continuous_refine(h0, t0, max_iter=4000):
    """Projected gradient descent with soft-max surrogate for max overlap.
    Uses adaptive temperature annealing and backtracking line search."""
    h = np.asarray(h0, dtype=np.float64).copy()
    best_h = h.copy()
    best_v = _max_overlap(h)
    
    # Start with a moderate temperature
    temp = 0.1
    step = 0.05
    no_improve = 0
    
    for it in range(max_iter):
        if _time_left(t0) < 8:
            break
        
        # Compute soft-max objective and gradient
        ov = _overlap_vector(h)
        obj = _soft_max(ov, temp)
        grad_ov = _soft_max_grad(ov, temp)
        
        # Gradient of ov[k] w.r.t. h: 
        # ov[k] = N/2 - C[k] where C[k] = sum_i h[i]*h[(i+k)%N]
        # dC[k]/dh[i] = h[(i+k)%N] + h[(i-k)%N] (for k != 0)
        # dC[0]/dh[i] = 2*h[i]
        # So dov[k]/dh[i] = -(h[(i+k)%N] + h[(i-k)%N]) for k != 0
        # And dov[0]/dh[i] = -2*h[i]
        
        # Compute gradient of soft-max w.r.t. h
        # grad_obj = sum_k grad_ov[k] * dov[k]/dh
        grad_h = np.zeros(N)
        for k in range(N):
            w = grad_ov[k]
            if w < 1e-8:
                continue
            if k == 0:
                grad_h -= 2.0 * w * h
            else:
                grad_h -= w * (np.roll(h, k) + np.roll(h, -k))
        
        # Adjust gradient to maintain sum constraint
        grad_h -= np.mean(grad_h)
        
        # Projected gradient step
        h2 = _project(h - step * grad_h)
        v2 = _max_overlap(h2)
        
        if v2 < best_v - 1e-12:
            best_v = v2
            best_h = h2.copy()
            h = h2
            no_improve = 0
            # Increase step slightly on improvement
            step = min(step * 1.1, 0.2)
        else:
            no_improve += 1
            # Decrease step
            step *= 0.7
            # Reduce temperature to sharpen the max
            if no_improve > 50:
                temp *= 0.9
                step *= 0.8
        
        if step < 1e-5:
            break
        if temp < 0.01:
            # Switch to exact subgradient for final polish
            break
    
    # Final polish with exact subgradient
    h = best_h.copy()
    step = 0.01
    for it in range(500):
        if _time_left(t0) < 5:
            break
        ov = _overlap_vector(h)
        k = int(np.argmax(ov))
        # Exact subgradient of max ov
        if k == 0:
            g = -2.0 * h
        else:
            g = -(np.roll(h, k) + np.roll(h, -k))
        g -= np.mean(g)
        h2 = _project(h - step * g)
        v2 = _max_overlap(h2)
        if v2 < best_v - 1e-12:
            best_v = v2
            best_h = h2.copy()
            h = h2
            step = min(step * 1.1, 0.1)
        else:
            step *= 0.8
            if step < 1e-5:
                break
    
    return best_h, best_v


def construct():
    t0 = time.time()
    rng = np.random.default_rng(2024)

    best_h = None
    best_val = float('inf')

    def consider(h):
        nonlocal best_h, best_val
        h = _project(h)
        v = _max_overlap(h)
        if v < best_val:
            best_val = v
            best_h = h.copy()

    # --- Bang-bang Barker search (fast) ---
    base = _barker_init(rng)
    base, v = _barker_search(base, t0, max_iter=2500000)
    consider(base)
    if _time_left(t0) > 30:
        base2, v2 = _barker_multi(base, t0)
        consider(base2)

    # --- Continuous refinement from the best bang-bang ---
    if _time_left(t0) > 25:
        hc, vc = _continuous_refine(best_h, t0, max_iter=5000)
        consider(hc)

    # --- A few more continuous restarts from perturbed bang-bang ---
    for _ in range(4):
        if _time_left(t0) < 15:
            break
        x = best_h.copy()
        for _ in range(8):
            a = int(rng.integers(N)); b = int(rng.integers(N))
            if a != b:
                x[a] = 1.0 - x[a]; x[b] = 1.0 - x[b]
        hc, vc = _continuous_refine(x, t0, max_iter=2000)
        consider(hc)

    # --- Final high-precision polish with very small steps ---
    if _time_left(t0) > 10:
        h = best_h.copy()
        step = 0.005
        for it in range(2000):
            if _time_left(t0) < 5:
                break
            ov = _overlap_vector(h)
            k = int(np.argmax(ov))
            if k == 0:
                g = -2.0 * h
            else:
                g = -(np.roll(h, k) + np.roll(h, -k))
            g -= np.mean(g)
            h2 = _project(h - step * g)
            v2 = _max_overlap(h2)
            if v2 < best_val - 1e-14:
                best_val = v2
                best_h = h2.copy()
                h = h2
                step = min(step * 1.05, 0.05)
            else:
                step *= 0.9
                if step < 1e-6:
                    break

    final_h = _project(best_h)
    final_h = np.clip(final_h, 0.0, 1.0)
    final_h = _project(final_h)
    assert abs(np.sum(final_h) - HALF) < 1e-9, f"Sum violated: {np.sum(final_h)}"
    return final_h


if __name__ == "__main__":
    t_start = time.time()
    h = construct()
    t_end = time.time()
    print(f"Time: {t_end - t_start:.2f}s")
    print("Sum:", np.sum(h))
    print("Min:", np.min(h), "Max:", np.max(h))
    print("Max overlap:", _max_overlap(h))
    baseline = _max_overlap(np.full(100, 0.5))
    print("Baseline overlap:", baseline)
    print("Score (seed/value):", baseline / _max_overlap(h))
