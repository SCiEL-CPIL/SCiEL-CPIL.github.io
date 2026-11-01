"""Football pool K_3(6,1): greedy covering-code search + exact local-repair polish."""
import time
import itertools
import numpy as np

# Precompute: for a 2-bit-per-symbol packed word (q<=4, n<=16), the number of
# differing symbols vs mask 0. 65536 entries, built once.
_LUT = None


def _get_lut():
    global _LUT
    if _LUT is None:
        base = np.arange(65536, dtype=np.uint16)
        lut = np.zeros(65536, dtype=np.uint8)
        for s in range(16):
            m = 1 << (2 * s)
            lut += ((base & m) != 0).astype(np.uint8)
        _LUT = lut
    return _LUT


def _build_dist_lut(q, n):
    """dist[a,b] = Hamming distance between packed words a,b (values 0..q-1)."""
    lut = _get_lut()
    dist = lut.copy()
    for i in range(1, q):
        m = i << 2
        dist += lut[(np.arange(65536, dtype=np.uint16) & m) >> 2]
    # zero out distances to values >= q (never used) so they don't pollute counts
    for i in range(q, 4):
        m = i << 2
        dist[lut[(np.arange(65536, dtype=np.uint16) & m) >> 2] != 0] = 255
    return dist


def _to_packed(arr):
    v = np.asarray(arr, dtype=np.uint16)
    acc = np.zeros_like(v, dtype=np.uint16)
    for s in range(v.shape[1]):
        acc |= (v[:, s] << (2 * s))
    return acc


def construct(q, n, radius):
    start = time.time()
    deadline = start + 110.0  # safety margin under the 150s kill

    total = q ** n
    if total > 2 ** 16:
        # Fallback for out-of-range sizes: direct-sum / trivial construction.
        return _fallback(q, n, radius)

    dist = _build_dist_lut(q, n)
    allw = np.arange(total, dtype=np.uint16)
    # per-codeword distance vector (shape (total, total))
    D = dist[allw[:, None], allw[None, :]]

    # masks for the distance<=radius lookup
    masks = []
    for d in range(radius + 1):
        c = 0
        for comb in itertools.combinations(range(n), d):
            m = 0
            for s in comb:
                m |= 3 << (2 * s)
            c |= 1 << m
        masks.append(c)
    # count of codewords within distance<=radius of each word
    # cnt[word] = number of i with masks[dist[i,word]]
    cnt = np.zeros(total, dtype=np.int32)
    for i in range(total):
        cnt += np.array([masks[x] for x in D[i]], dtype=np.int32)

    # greedy seed selection
    selected = []
    covered = np.zeros(total, dtype=bool)
    remaining = total
    while remaining > 0:
        if time.time() > deadline - 30:  # reserve time for polish/return
            break
        uncovered = ~covered
        gains = cnt[uncovered]  # for each candidate, # uncovered words it covers
        best = int(np.argmax(gains))
        if gains[best] <= 0:
            break
        cand = int(np.where(uncovered)[0][best])
        selected.append(cand)
        selmask = (D[cand] <= radius)
        new = selmask & uncovered
        covered |= selmask
        remaining -= int(new.sum())
        cnt[new] = 0  # no longer needed for selection

    # exact local-repair polish: drop codewords whose residue is already covered
    def coverage_of(idx_list):
        cov = np.zeros(total, dtype=bool)
        for idx in idx_list:
            cov |= (D[idx] <= radius)
        return cov

    # verify current set covers everything
    cur_cov = coverage_of(selected)
    if not cur_cov.all():
        # not covered (shouldn't happen if remaining==0); add missing greedily
        while not cur_cov.all() and time.time() < deadline - 5:
            uncovered = ~cur_cov
            gains = cnt[uncovered]
            best = int(np.argmax(gains))
            if gains[best] <= 0:
                break
            cand = int(np.where(uncovered)[0][best])
            selected.append(cand)
            nm = (D[cand] <= radius) & uncovered
            cur_cov |= nm
            cnt[nm] = 0

    # polish: try removing each codeword
    changed = True
    while changed and time.time() < deadline - 5:
        changed = False
        new_selected = []
        for idx in selected:
            others = [j for j in selected if j != idx]
            # residue of idx = words covered by idx but by no other
            others_cov = coverage_of(others)
            idx_cov = D[idx] <= radius
            if np.all(idx_cov & ~others_cov):
                # safe to drop
                continue
            new_selected.append(idx)
        if len(new_selected) < len(selected):
            selected = new_selected
            changed = True

    # final safety: ensure full coverage
    final_cov = coverage_of(selected)
    if not final_cov.all():
        while not final_cov.all() and time.time() < deadline - 1:
            uncovered = ~final_cov
            gains = cnt[uncovered]
            best = int(np.argmax(gains))
            if gains[best] <= 0:
                break
            cand = int(np.where(uncovered)[0][best])
            selected.append(cand)
            final_cov |= (D[cand] <= radius) & uncovered

    # decode to lists of ints
    result = []
    for idx in selected:
        row = []
        for s in range(n):
            row.append((idx >> (2 * s)) & 3)
        result.append(row)
    return result


def _fallback(q, n, radius):
    # direct-sum of a small code when the space is too large for the packed kernel
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    if n >= 4:
        return [h + list(tail) for h in tetracode
                for tail in itertools.product(range(q), repeat=n - 4)]
    return [list(t) for t in itertools.product(range(q), repeat=n)]
