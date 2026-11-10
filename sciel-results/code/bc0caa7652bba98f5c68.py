"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

N = 256

def _max_abs_ap_sum(signs):
    """Compute max |signed sum| over all arithmetic progressions in [0, n-1]."""
    n = len(signs)
    max_disc = 1
    for d in range(1, n):
        for r in range(d):
            idx = list(range(r, n, d))
            if len(idx) < 2:
                continue
            csum = np.cumsum(signs[idx])
            m = int(np.max(np.abs(csum)))
            if m > max_disc:
                max_disc = m
    return max_disc

def _compute_deltas(signs, flip):
    """For flipping position `flip`, compute the change in |sum| for every AP containing flip.
    Returns (indices, old_abs, new_abs) where old_abs is |S| before flip, new_abs after."""
    n = len(signs)
    f = flip
    idx_list = []
    old_list = []
    new_list = []
    for d in range(1, n):
        # APs with step d that contain f: start s = f - k*d, k>=0, s>=0
        # length L: s + (L-1)*d <= n-1
        k = 0
        while f - k * d >= 0:
            s = f - k * d
            # max length
            L = (n - 1 - s) // d + 1
            # prefix sums of the AP signs[s], signs[s+d], ...
            # We need S_t = sum_{j=0}^{t-1} signs[s + j*d] for t=1..L
            # f is at position t0 = k (0-indexed) in this AP, so prefix length t0+1 = k+1
            seq = signs[s:s + L * d: d]
            csum = np.cumsum(seq)
            t0 = k  # 0-based index of f in seq
            # For each prefix length t in [t0+1, L], the sum includes f
            # old sum = csum[t-1], new sum = csum[t-1] - 2*signs[f]
            for t in range(t0 + 1, L + 1):
                old_s = csum[t - 1]
                new_s = old_s - 2 * signs[f]
                idx_list.append((s, d, t))
                old_list.append(abs(old_s))
                new_list.append(abs(new_s))
            k += 1
    return idx_list, old_list, new_list

def _current_max_abs(signs, idx_list):
    """Compute max |sum| over given APs (list of (s,d,t))."""
    m = 1
    for (s, d, t) in idx_list:
        seq = signs[s:s + t * d: d]
        val = int(np.sum(seq))
        a = abs(val)
        if a > m:
            m = a
    return m

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # safety margin before 60s

    rng = np.random.default_rng(171)
    signs = rng.choice([-1, 1], size=N).astype(np.int64)

    best_signs = signs.copy()
    best_disc = _max_abs_ap_sum(best_signs)

    # Precompute the list of all APs (s, d, t) with t>=2
    all_aps = []
    for d in range(1, N):
        for s in range(d):
            L = (N - 1 - s) // d + 1
            for t in range(2, L + 1):
                all_aps.append((s, d, t))

    # Build reverse map: for each position f, which APs contain it
    # AP (s,d,t) contains positions s, s+d, ..., s+(t-1)*d
    ap_by_pos = [[] for _ in range(N)]
    for i, (s, d, t) in enumerate(all_aps):
        for j in range(t):
            pos = s + j * d
            ap_by_pos[pos].append(i)

    # Current |sum| for each AP
    ap_sums = np.zeros(len(all_aps), dtype=np.int64)
    for i, (s, d, t) in enumerate(all_aps):
        seq = signs[s:s + t * d: d]
        ap_sums[i] = int(np.sum(seq))

    # Count how many APs have |sum| == current max (to know if max drops)
    # We'll track max via a simple recount when needed.

    def get_max():
        return int(np.max(np.abs(ap_sums)))

    cur_max = get_max()
    if cur_max < best_disc:
        best_disc = cur_max
        best_signs = signs.copy()

    no_improve = 0
    max_no_improve = 4000

    while time.time() < deadline:
        if no_improve > max_no_improve:
            # restart with a new random perturbation
            for _ in range(8):
                f = rng.integers(0, N)
                signs[f] = -signs[f]
                # update ap_sums for APs containing f
                for ai in ap_by_pos[f]:
                    ap_sums[ai] -= 2 * signs[f]
            no_improve = 0
            cur_max = get_max()
            if cur_max < best_disc:
                best_disc = cur_max
                best_signs = signs.copy()
            continue

        # Pick a flip that most reduces the number of "hot" APs or reduces max
        # Greedy: try random positions, accept if it doesn't increase max
        f = int(rng.integers(0, N))
        old_sign = signs[f]
        # Compute new max if we flip f
        old_sums = ap_sums.copy()
        signs[f] = -old_sign
        for ai in ap_by_pos[f]:
            ap_sums[ai] -= 2 * old_sign
        new_max = get_max()
        if new_max <= cur_max:
            # accept (keep if equal or better)
            cur_max = new_max
            no_improve = 0
            if new_max < best_disc:
                best_disc = new_max
                best_signs = signs.copy()
        else:
            # revert
            signs[f] = old_sign
            for ai in ap_by_pos[f]:
                ap_sums[ai] += 2 * old_sign
            no_improve += 1

    # Final: return best_signs
    result = []
    for v in best_signs:
        result.append(int(v))
    return result
# EVOLVE-BLOCK-END
