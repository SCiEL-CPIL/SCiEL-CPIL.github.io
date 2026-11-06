# EVOLVE-BLOCK-START
import time
import numpy as np

N = 256
DEADLINE = 80.0

def _max_ap_sum(s):
    """Return (max_abs_sum, list_of_worst_(start,step)) for signs s (1-indexed, length N)."""
    pref = np.zeros(N + 1, dtype=np.int64)
    np.cumsum(s, out=pref[1:])
    best = 0
    worst = []
    for step in range(1, N + 1):
        for start in range(1, N + 1, step):
            end = N - ((N - start) % step) + 1  # last index >= start with same residue
            length = (end - start) // step + 1
            if length < 2:
                continue
            # all contiguous sub-segments within this residue class
            p = pref[start:end + 1:step]          # length+1 values
            seg = p[1:] - p[:-1]                  # prefix of sub-sums
            mx = int(seg.max()); mn = int(seg.min())
            if mx > best:
                best = mx; worst = [(start, step)]
            elif mx == best:
                worst.append((start, step))
            if -mn > best:
                best = -mn; worst = [(start, step)]
            elif -mn == best:
                worst.append((start, step))
    return best, worst

def _init_balanced(seed=171):
    """Greedy balanced-threshold construction: at each index, pick the sign that keeps
    every AP ending here within the current global maximum."""
    rng = np.random.default_rng(seed)
    s = np.zeros(N, dtype=np.int64)
    cur = 1
    for i in range(N):
        best_sign = 1
        best_bad = None
        for sign in (1, -1):
            s[i] = sign
            bad = 0
            for step in range(1, i + 1):
                for start in range(1, i + 1, step):
                    end = i - ((i - start) % step) + 1
                    length = (i - start) // step + 1
                    if length < 2:
                        continue
                    sm = int(s[start - 1:i + 1:step].sum())
                    if abs(sm) > bad:
                        bad = abs(sm)
            if best_bad is None or bad < best_bad:
                best_bad = bad
                best_sign = sign
            if best_bad == 0:
                break
        s[i] = best_sign
        if best_bad > cur:
            cur = best_bad
    return s.tolist()

def _do_flip(s, idx, sign):
    s[idx] = sign

def _try_flip(s, idx, sign):
    """Return delta in max AP sum if flipping s[idx] to sign (sign in {-1,1})."""
    old = s[idx]
    if old == sign:
        return 0
    s[idx] = sign
    val, _ = _max_ap_sum(s)
    s[idx] = old
    return val

def _try_pair(s, i, si, j, sj):
    old_i, old_j = s[i], s[j]
    if (old_i == si) and (old_j == sj):
        return 0
    s[i] = si; s[j] = sj
    val, _ = _max_ap_sum(s)
    s[i] = old_i; s[j] = old_j
    return val

def construct():
    t0 = time.time()
    # Strong initial construction
    s = _init_balanced(171)
    best_val, _ = _max_ap_sum(s)
    best_s = s[:]
    best_val = min(best_val, 1)  # singleton bound is 1

    rng = np.random.default_rng(171)

    while time.time() - t0 < DEADLINE - 6.0:
        # Evaluate worst APs
        cur_val, worst = _max_ap_sum(best_s)

        # --- 1-flip moves targeting worst APs ---
        improved = False
        # Build candidate index list: positions in worst APs + random
        cand = set()
        for (st, sp) in worst[:64]:
            idx = st
            while idx <= N:
                cand.add(idx - 1)
                idx += sp
        for _ in range(40):
            cand.add(int(rng.integers(0, N)))
        cand = list(cand)
        rng.shuffle(cand)

        for idx in cand:
            if time.time() - t0 > DEADLINE - 6.0:
                break
            for sign in (1, -1):
                if best_s[idx] == sign:
                    continue
                delta = _try_flip(best_s, idx, sign)
                if delta < cur_val:
                    _do_flip(best_s, idx, sign)
                    cur_val = delta
                    best_val = min(best_val, cur_val)
                    improved = True
                    break
            if improved:
                break

        # --- 2-flip moves if still stuck ---
        if not improved and time.time() - t0 < DEADLINE - 6.0:
            # Try random pairs from candidate set
            for _ in range(200):
                if time.time() - t0 > DEADLINE - 6.0:
                    break
                i = int(rng.integers(0, N))
                j = int(rng.integers(0, N))
                if i == j:
                    continue
                si = 1 if best_s[i] == -1 else -1
                sj = 1 if best_s[j] == -1 else -1
                delta = _try_pair(best_s, i, si, j, sj)
                if delta < cur_val:
                    _do_flip(best_s, i, si)
                    _do_flip(best_s, j, sj)
                    cur_val = delta
                    best_val = min(best_val, cur_val)
                    improved = True
                    break

        # If no improvement in this round, do a small random restart perturbation
        if not improved:
            # Try a few random flips to escape local minimum
            for _ in range(5):
                if time.time() - t0 > DEADLINE - 6.0:
                    break
                idx = int(rng.integers(0, N))
                sign = 1 if best_s[idx] == -1 else -1
                old = best_s[idx]
                best_s[idx] = sign
                val, _ = _max_ap_sum(best_s)
                if val < best_val:
                    best_val = val
                else:
                    best_s[idx] = old

    return [int(x) for x in best_s]
# EVOLVE-BLOCK-END
