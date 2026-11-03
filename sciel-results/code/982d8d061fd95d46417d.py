import time
import itertools
import numpy as np

def _seed_code(q, n, radius):
    # Direct sum of ternary tetracode (9 words length 4, perfect radius-1)
    # with every word on the remaining n-4 coordinates: 9 * 3**(n-4) words.
    tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
    return [h + list(tail) for h in tetracode
            for tail in itertools.product(range(q), repeat=n - 4)]

def _to_matrix(code, n):
    return np.array(code, dtype=np.int8)

def _is_covering(mat, q, n, radius):
    # Returns True if every word of {0..q-1}^n is within Hamming distance <= radius of mat.
    if mat.shape[0] == 0:
        return False
    # mat: (m, n) int8. Enumerate all q**n words.
    total = q ** n
    # Build full space as (total, n) int8 via odometer.
    space = np.zeros((total, n), dtype=np.int8)
    # fill using base-q digits
    for i in range(n):
        base = q ** (n - 1 - i)
        space[:, i] = (np.arange(total) // base) % q
    # digit-wise equality: (m, total, n)
    eq = (mat[:, None, :] == space[None, :, :])
    # number of mismatches per (codeword, word)
    mism = (~eq).sum(axis=2)  # (m, total)
    # covered if any codeword within radius
    return bool((mism <= radius).any(axis=0).all())

def _covered_mask(mat, q, n, radius):
    # Returns boolean array of length q**n: True if word is covered.
    total = q ** n
    space = np.zeros((total, n), dtype=np.int8)
    for i in range(n):
        base = q ** (n - 1 - i)
        space[:, i] = (np.arange(total) // base) % q
    if mat.shape[0] == 0:
        return np.zeros(total, dtype=bool)
    eq = (mat[:, None, :] == space[None, :, :])
    mism = (~eq).sum(axis=2)
    return (mism <= radius).any(axis=0)

def _swap_trial_gain(mat, q, n, radius, pos, new_val):
    # mat: (m, n). Try setting mat[:, pos] = new_val (broadcast). Return number of
    # previously-uncovered words that become covered, minus number of previously-covered
    # words that become uncovered. (Net gain.)
    # Compute mismatches before and after for the swapped matrix.
    total = q ** n
    space = np.zeros((total, n), dtype=np.int8)
    for i in range(n):
        base = q ** (n - 1 - i)
        space[:, i] = (np.arange(total) // base) % q

    def cov(m):
        eq = (m[:, None, :] == space[None, :, :])
        mism = (~eq).sum(axis=2)
        return (mism <= radius).any(axis=0)

    before = cov(mat)
    m2 = mat.copy()
    m2[:, pos] = new_val
    after = cov(m2)
    return int(after.sum() - before.sum())

def _greedy_reduce(mat, q, n, radius, deadline):
    # Repeatedly remove a codeword if the code remains a valid covering.
    m, n_ = mat.shape
    order = np.arange(m)
    changed = True
    while changed:
        changed = False
        if time.time() > deadline:
            break
        for i in order:
            if time.time() > deadline:
                break
            if i >= mat.shape[0]:
                continue
            sub = np.delete(mat, i, axis=0)
            if _is_covering(sub, q, n_, radius):
                mat = sub
                order = np.arange(mat.shape[0])
                changed = True
                break
    return mat

def construct(q, n, radius):
    deadline = time.time() + 100.0  # safety margin before 120s limit
    base = _seed_code(q, n, radius)
    mat = _to_matrix(base, n)

    # Ensure we have a valid starting covering (the direct sum is valid by construction
    # for radius >= 1 and n >= 4; but verify and if not, keep as is).
    best_mat = mat.copy()
    best_size = mat.shape[0]

    # Precompute candidate swap positions: (codeword_index, position, new_value != current)
    # We'll iterate over positions and values.

    # Hill climbing with swap moves
    def try_improve(m, deadline):
        improved = True
        while improved:
            improved = False
            if time.time() > deadline:
                break
            m_rows = m.shape[0]
            # iterate codewords and positions
            for ci in range(m_rows):
                if time.time() > deadline:
                    break
                for pos in range(n):
                    cur = m[ci, pos]
                    for nv in range(q):
                        if nv == cur:
                            continue
                        m2 = m.copy()
                        m2[ci, pos] = nv
                        # net gain: does this reduce total uncovered? We want to keep
                        # covering valid, so we require no word becomes uncovered.
                        # Compute covered before/after.
                        total = q ** n
                        space = np.zeros((total, n), dtype=np.int8)
                        for i in range(n):
                            base_ = q ** (n - 1 - i)
                            space[:, i] = (np.arange(total) // base_) % q
                        def cov(mm):
                            eq = (mm[:, None, :] == space[None, :, :])
                            mism = (~eq).sum(axis=2)
                            return (mism <= radius).any(axis=0)
                        before = cov(m)
                        after = cov(m2)
                        # accept if after covers everything (valid) and size same -> no gain in size,
                        # but we want to enable later reductions. Accept if valid and
                        # (uncovered count unchanged i.e. still full cover) -> then try reductions.
                        # Simpler: accept if after is a full cover (valid). This keeps validity.
                        if after.all():
                            # keep this move; it may enable a reduction
                            m = m2
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
        return m

    # Because a pure swap keeps size constant, the real gain comes from reduction.
    # Strategy: do a few random restarts: perturb the seed, then reduce.
    rng = np.random.default_rng(0)

    for restart in range(4):
        if time.time() > deadline - 20:
            break
        if restart == 0:
            m = mat.copy()
        else:
            # random perturbation: change a few random codeword/position/value
            m = mat.copy()
            m_rows = m.shape[0]
            for _ in range(5):
                if m_rows == 0:
                    break
                ci = int(rng.integers(0, m_rows))
                pos = int(rng.integers(0, n))
                nv = int(rng.integers(0, q))
                m[ci, pos] = nv
        # hill climb
        m = try_improve(m, deadline)
        if time.time() > deadline - 15:
            break
        # greedy reduce
        m_red = _greedy_reduce(m, q, n, radius, deadline)
        if m_red.shape[0] < best_size:
            best_size = m_red.shape[0]
            best_mat = m_red.copy()
        if time.time() > deadline - 10:
            break

    # Final safety: ensure best_mat is a valid covering; if not, fall back to seed.
    if not _is_covering(best_mat, q, n, radius):
        best_mat = mat
    return best_mat.tolist()
