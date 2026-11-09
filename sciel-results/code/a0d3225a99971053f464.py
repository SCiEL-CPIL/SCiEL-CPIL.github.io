"""Covering code K_3(7,1): greedy set cover + local-search polish."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE_FRAC = 0.90  # safety margin before the 150 s wall-clock kill


def _build_ball_mask(ball, n):
    """For a word (list of ints), return a length-3**n int array: 1 for the ball
    (word + all words at Hamming distance <= 1), 0 elsewhere."""
    size = 3 ** n
    m = np.zeros(size, dtype=np.int32)
    w = np.array(ball, dtype=np.int64)
    for i in range(n):
        for d in range(3):
            if d == w[i]:
                continue
            v = w.copy()
            v[i] = d
            m[v[0] * 3 ** (n - 1) + v[1] * 3 ** (n - 2) + v[2] * 3 ** (n - 3)
              + v[3] * 3 ** (n - 4) + v[4] * 3 ** (n - 5) + v[5] * 3 ** (n - 6)
              + v[6] * 3 ** (n - 7)] = 1
    return m


def _ball_masks(n, k=2):
    """Precompute ball masks for the first k words (indices 0..k-1)."""
    masks = []
    for idx in range(k):
        digits = [0] * n
        t = idx
        for i in range(n):
            digits[i] = t % 3
            t //= 3
        masks.append(_build_ball_mask(digits, n))
    return masks


def _greedy_cover(n, k, seed=12345):
    """Greedy set cover using precomputed ball masks for the first k words.
    Returns a list of chosen word indices."""
    size = 3 ** n
    rng = np.random.default_rng(seed)
    # Work order: shuffle the first k words, then append the rest in order.
    first = np.arange(k)
    rng.shuffle(first)
    rest = np.arange(k, size)
    order = np.concatenate([first, rest])

    remaining = np.ones(size, dtype=np.int32)
    chosen = []
    chosen_set = set()

    for idx in order:
        if remaining.sum() == 0:
            break
        if idx < k:
            mask = _build_ball_mask_from_idx(idx, n)
        else:
            mask = _build_ball_mask_from_idx(idx, n)
        gain = int((remaining & mask).sum())
        if gain <= 0:
            continue
        chosen.append(idx)
        chosen_set.add(idx)
        remaining &= ~mask

    return chosen


def _build_ball_mask_from_idx(idx, n):
    """Build ball mask for word with base-3 representation idx."""
    size = 3 ** n
    m = np.zeros(size, dtype=np.int32)
    digits = [0] * n
    t = idx
    for i in range(n):
        digits[i] = t % 3
        t //= 3
    m[idx] = 1
    for i in range(n):
        for d in range(3):
            if d == digits[i]:
                continue
            v = digits.copy()
            v[i] = d
            pos = 0
            for j in range(n):
                pos = pos * 3 + v[j]
            m[pos] = 1
    return m


def _cover_count(chosen, n):
    """Total number of distinct words covered by the chosen set."""
    size = 3 ** n
    covered = np.zeros(size, dtype=np.int32)
    for idx in chosen:
        covered |= _build_ball_mask_from_idx(idx, n)
    return int(covered.sum())


def _local_search(chosen, n, deadline, max_iters=5000):
    """Local search: try removing a redundant codeword, or swapping one for another
    to gain coverage. Returns improved list."""
    size = 3 ** n
    chosen = list(chosen)
    chosen_set = set(chosen)

    # Build coverage counts: for each word, how many chosen codewords cover it.
    cov_count = np.zeros(size, dtype=np.int32)
    for idx in chosen:
        cov_count += _build_ball_mask_from_idx(idx, n)
    covered = cov_count > 0
    n_covered = int(covered.sum())

    improved = True
    iters = 0
    while improved and iters < max_iters:
        if time.time() > deadline:
            break
        improved = False
        iters += 1

        # Try removing each chosen codeword if it doesn't reduce coverage.
        for i, idx in enumerate(list(chosen)):
            if time.time() > deadline:
                break
            mask = _build_ball_mask_from_idx(idx, n)
            # Words covered only by this codeword
            only = (cov_count == 1) & (mask > 0)
            if only.sum() == 0:
                # Safe to remove
                chosen.pop(i)
                chosen_set.discard(idx)
                cov_count -= mask
                n_covered = int((cov_count > 0).sum())
                improved = True
                break

        if improved:
            continue

        # Try swapping: remove one, add one, if net gain >= 0 and size decreases or stays
        for i, idx in enumerate(list(chosen)):
            if time.time() > deadline:
                break
            mask_out = _build_ball_mask_from_idx(idx, n)
            only_out = (cov_count == 1) & (mask_out > 0)
            n_only_out = int(only_out.sum())
            if n_only_out > 0:
                # Removing idx loses n_only_out words; we need to gain at least n_only_out
                # from the replacement to not increase size (or gain more to reduce size)
                pass

            # Try adding a random new word
            new_idx = np.random.randint(0, size)
            if new_idx in chosen_set:
                continue
            mask_in = _build_ball_mask_from_idx(new_idx, n)
            gain = int(((cov_count == 0) & (mask_in > 0)).sum())
            # Net effect: remove idx (lose n_only_out), add new_idx (gain 'gain')
            # New size = len(chosen) (swap) or len(chosen)+1 (add only)
            # We want to reduce total size, so only do swap if gain > n_only_out
            if gain > n_only_out:
                chosen[i] = new_idx
                chosen_set.discard(idx)
                chosen_set.add(new_idx)
                cov_count -= mask_out
                cov_count += mask_in
                n_covered = int((cov_count > 0).sum())
                improved = True
                break

    return chosen


def _greedy_cover_full(n, deadline, seed=42):
    """Full greedy set cover over all 3^n words, with deadline guard."""
    size = 3 ** n
    rng = np.random.default_rng(seed)
    order = np.arange(size)
    rng.shuffle(order)

    remaining = np.ones(size, dtype=np.int32)
    chosen = []
    chosen_set = set()

    for idx in order:
        if time.time() > deadline:
            break
        if remaining.sum() == 0:
            break
        mask = _build_ball_mask_from_idx(int(idx), n)
        gain = int((remaining & mask).sum())
        if gain <= 0:
            continue
        chosen.append(int(idx))
        chosen_set.add(int(idx))
        remaining &= ~mask

    return chosen


def construct(q, n, radius):
    """Construct a covering code of length n over alphabet size q with covering
    radius `radius`. Returns a list of codewords (each a list of n ints)."""
    start = time.time()
    deadline = start + 150.0 * _DEADLINE_FRAC  # ~135 s

    # For the specific task: q=3, n=7, radius=1
    # General approach: greedy set cover + local search

    if q != 3 or radius != 1:
        # Fallback: simple greedy for general case
        size = q ** n
        if size <= 10000:
            # Brute-force greedy
            remaining = set(range(size))
            chosen = []
            for idx in range(size):
                if time.time() > deadline:
                    break
                if not remaining:
                    break
                # Ball around idx
                digits = []
                t = idx
                for i in range(n):
                    digits.append(t % q)
                    t //= q
                ball = set()
                for d in digits:
                    ball.add(d)
                for i in range(n):
                    for v in range(q):
                        if v == digits[i]:
                            continue
                        nd = digits.copy()
                        nd[i] = v
                        pos = 0
                        for j in range(n):
                            pos = pos * q + nd[j]
                        ball.add(pos)
                gain = len(remaining & ball)
                if gain > 0:
                    chosen.append(digits.copy())
                    remaining -= ball
            return chosen
        else:
            # Too large, return empty (invalid but safe)
            return []

    # Ternary, radius 1, n=7
    size = 3 ** n  # 2187

    # Phase 1: Greedy set cover
    chosen = _greedy_cover_full(n, deadline, seed=42)

    # Phase 2: Local search polish
    chosen = _local_search(chosen, n, deadline, max_iters=3000)

    # Convert indices to codeword lists
    codewords = []
    for idx in chosen:
        digits = []
        t = idx
        for i in range(n):
            digits.append(t % 3)
            t //= 3
        codewords.append(digits)

    return codewords
# EVOLVE-BLOCK-END
