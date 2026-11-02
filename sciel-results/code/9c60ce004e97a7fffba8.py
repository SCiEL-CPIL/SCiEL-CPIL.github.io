"""
Third autocorrelation inequality: minimize C3 = max_k |c_k| for a signed step
function with 400 equal intervals on [-1/4, 1/4].

c_k = D * sum_j h_j h_{j+k}, D = 1/100.
Minimizing max_k |c_k| <=> minimizing max_k |(h*h)_k| since D is constant.
h_j in {-1, +1}.

STRUCTURAL CHANGE (vs. ECA-seed + single-bit local search):
  Use block-flip SIMULATED ANNEALING with an incremental autoconvolution
  update. Each move flips a contiguous block of B signs; the convolution
  changes only over a window of width (2N-1)+2B-1, so the update is O(B)
  rather than O(N). Block flips of size 2..128 let the search make
  coordinated sign changes that escape the single-bit local minima that
  trap greedy coordinate descent. A multi-phase cooling schedule plus a
  final local-refinement pass are run within an explicit wall-clock guard.
"""
import time
import numpy as np

N = 400
L = 2 * N - 1          # length of full convolution
DEADLINE = 150.0
MARGIN = 8.0
RNG_SEED = 42


def _max_abs_conv(h):
    """Exact max over all lags of |(h*h)_k| for h in {+-1}."""
    c = np.convolve(h, h, mode='full')
    return int(np.max(np.abs(c)))


def _init_conv(h):
    """Full autoconvolution of h (length L)."""
    return np.convolve(h, h, mode='full')


def _apply_block_flip(h, c, start, length, h_val):
    """Flip h[start:start+length] to h_val, updating c incrementally.
    Returns (new_max_abs, changed) where changed is True if any bit flipped.
    """
    end = start + length
    if end > N:
        end = N
        length = end - start
    if length <= 0:
        return int(np.max(np.abs(c))), False

    # Affected convolution lags: k in [start-(N-1), end-1+(N-1)]
    lo = start - (N - 1)
    hi = end - 1 + (N - 1)
    if lo < 0:
        lo = 0
    if hi >= L:
        hi = L - 1

    # For each affected lag k, contribution involves h[k-N+1 : k+1] intersect block.
    # delta_k = 2*h_val * (sum of h_j over j in block AND j in [k-N+1, k])
    # We build the per-lag overlap count of the block with the convolution window.
    # block indices: [start, end)
    # window for lag k: [k-N+1, k]
    # overlap length = max(0, min(end, k+1) - max(start, k-N+1))
    ks = np.arange(lo, hi + 1, dtype=np.int64)
    win_lo = ks - (N - 1)
    win_hi = ks
    ov = np.minimum(end, win_hi + 1) - np.maximum(start, win_lo)
    ov = np.maximum(ov, 0)
    # delta_k = 2*h_val*ov_k ; new_c = c + delta
    new_c = c[lo:hi + 1] + (2 * h_val * ov).astype(c.dtype)
    new_max = int(np.max(np.abs(new_c)))

    # Apply to h
    h[start:end] = h_val
    c[lo:hi + 1] = new_c
    return new_max, True


def _unapply_block_flip(h, c, start, length, h_val, old_val):
    """Revert a block flip: set h[start:start+length] back to old_val.
    (h currently holds h_val on that block.)"""
    end = start + length
    if end > N:
        end = N
        length = end - start
    if length <= 0:
        return int(np.max(np.abs(c)))
    lo = start - (N - 1)
    hi = end - 1 + (N - 1)
    if lo < 0:
        lo = 0
    if hi >= L:
        hi = L - 1
    ks = np.arange(lo, hi + 1, dtype=np.int64)
    win_lo = ks - (N - 1)
    win_hi = ks
    ov = np.minimum(end, win_hi + 1) - np.maximum(start, win_lo)
    ov = np.maximum(ov, 0)
    new_c = c[lo:hi + 1] + (2 * old_val * ov).astype(c.dtype)
    h[start:end] = old_val
    c[lo:hi + 1] = new_c
    return int(np.max(np.abs(new_c)))


def _anneal(h, c, t0, limit, rng):
    """Block-flip simulated annealing. Returns (h, best_val, best_h)."""
    cur = int(np.max(np.abs(c)))
    best = cur
    best_h = h.copy()

    # Temperature schedule (multi-phase). We tune by wall-clock, not fixed iters.
    # Start hot enough that most block flips are accepted, cool geometrically.
    # Energy scale: max_abs_conv ranges ~ a few hundred. Use T in same units.
    T = 40.0
    T_min = 0.4
    # cooling: recompute per-iteration; we cool by factor per "warmup" block
    cool_every = 4000      # iterations between temperature reductions
    cool_factor = 0.94

    # Block size distribution: mix of small and large
    # P(B) roughly decreasing; sample from a few candidate sizes
    block_sizes = np.array([2, 3, 4, 5, 6, 8, 10, 14, 20, 30, 48, 80, 128], dtype=np.int64)
    # weights: favor smaller but allow large
    block_weights = np.array([10, 8, 7, 6, 5, 4, 3, 2, 1.5, 1.0, 0.6, 0.3, 0.15])
    block_weights = block_weights / block_weights.sum()
    block_choice = rng.choice(len(block_sizes), p=block_weights)

    it = 0
    while time.time() < limit:
        # occasionally re-sample block size to keep diversity
        if it % 64 == 0:
            block_choice = rng.choice(len(block_sizes), p=block_weights)
        B = int(block_sizes[block_choice])

        # random start
        max_start = N - B
        if max_start < 0:
            start = 0
            B = N
        else:
            start = int(rng.integers(0, max_start + 1))
            if start + B > N:
                B = N - start

        # current block value (all bits in block are equal in our representation? No.)
        # We flip the block to the *opposite* of its current majority to change more.
        # Simpler: pick target = -h[start] (flip relative to first bit) -> not uniform.
        # Better: target = +1 or -1 chosen so that it differs from current block value.
        # Since block bits may be mixed, compute current block sum.
        block_sum = int(np.sum(h[start:start + B]))
        # If block_sum >= 0, majority is +1 -> target -1; else target +1.
        if block_sum >= 0:
            target = -1
            old_val = 1
        else:
            target = 1
            old_val = -1

        new_max, changed = _apply_block_flip(h, c, start, B, target)
        if not changed:
            it += 1
            continue
        delta = new_max - cur
        if delta <= 0 or (T > T_min and rng.random() < np.exp(-delta / T)):
            cur = new_max
            if cur < best:
                best = cur
                best_h = h.copy()
        else:
            _unapply_block_flip(h, c, start, B, target, old_val)

        it += 1
        if it % cool_every == 0:
            T *= cool_factor
            if T < T_min:
                T = T_min

    return h, best, best_h


def _local_refine(h, c, limit):
    """Final single-bit local refinement (fast incremental)."""
    cur = int(np.max(np.abs(c)))
    best = cur
    best_h = h.copy()
    rng = np.random.default_rng(12345)

    # Repeated sweeps until no improvement or time up.
    while time.time() < limit:
        improved = False
        order = rng.permutation(N)
        for i in order:
            lo = i - (N - 1)
            hi = i + (N - 1)
            if lo < 0:
                lo = 0
            if hi >= L:
                hi = L - 1
            if lo > hi:
                continue
            jlo = lo - i
            jhi = hi - i
            delta = 2 * int(h[i]) * h[jlo:jhi + 1]
            new_c = c[lo:hi + 1] + delta
            new_max = int(np.max(np.abs(new_c)))
            if new_max < cur:
                c[lo:hi + 1] = new_c
                h[i] = -h[i]
                cur = new_max
                improved = True
                if cur < best:
                    best = cur
                    best_h = h.copy()
        if not improved:
            break
    return h, best, best_h


def _construct():
    t0 = time.time()
    limit = DEADLINE - MARGIN
    rng = np.random.default_rng(RNG_SEED)

    best_val = None
    best_h = None

    # Allocate time budget across a few independent annealing runs.
    # We do a few restarts with different seeds; each gets a time slice.
    # Keep at least some time for the final local refine.
    n_runs = 4
    # Reserve 25% of the time for final refinement of the best so far.
    anneal_budget = (limit - t0) * 0.70
    per_run = anneal_budget / n_runs

    for r in range(n_runs):
        run_limit = t0 + per_run * (r + 1)
        if run_limit > limit - 0.15 * (limit - t0):
            run_limit = limit - 0.15 * (limit - t0)
        if time.time() >= run_limit:
            break

        # Fresh random start (mix in a little structure: alternating blocks)
        h = rng.choice(np.array([-1, 1], dtype=np.int64), size=N)
        # Optionally seed with a low-frequency pattern to give structure
        if r % 2 == 1:
            # add a mild sinusoidal bias
            idx = np.arange(N)
            bias = np.sign(np.sin(2 * np.pi * idx / 80.0 + rng.random() * 6.28))
            h = np.where(rng.random(N) < 0.5, bias, h)
        c = _init_conv(h)

        h_a, best_r, best_h_r = _anneal(h, c, t0, run_limit, rng)
        if best_val is None or best_r < best_val:
            best_val = best_r
            best_h = best_h_r.copy()

    # Final local refinement with remaining time.
    if best_h is not None and time.time() < limit:
        c = _init_conv(best_h)
        _, best_ref, best_h_ref = _local_refine(best_h, c, limit)
        if best_ref < best_val:
            best_val = best_ref
            best_h = best_h_ref.copy()

    return best_h


def construct():
    return _construct()


# ---- local self-test (not called by evaluator) ----
if __name__ == '__main__':
    t = time.time()
    h = construct()
    print("time", time.time() - t, "N", len(h), "max_abs_conv", _max_abs_conv(h))
