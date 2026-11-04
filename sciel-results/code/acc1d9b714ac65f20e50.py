"""Covering code K_3(6,1): greedy set-cover + local pruning with NumPy coverage."""
import time
import numpy as np


def construct(q, n, radius):
    t0 = time.time()
    deadline = 110.0  # safety margin before the 120s kill

    # Build the full space of q^n words as an (N, n) int array.
    base = np.array([list(np.base_repr(i, q).zfill(n)) for i in range(q ** n)], dtype=np.int8)
    N = base.shape[0]

    # Per-position one-hot "match" matrix: match[i, j] = 1 iff word i == codeword j at that position.
    # match_all = sum over positions = number of matching coordinates.
    # word i is within radius 1 of codeword j  <=>  match_all[i, j] >= n - 1.
    match_all = np.zeros((N, q ** n), dtype=np.int32)
    for pos in range(n):
        onehot = (base[:, pos:pos + 1] == np.arange(q, dtype=np.int8)[None, :])
        match_all += onehot.astype(np.int32)

    # Start from the known-good direct-sum code (upper bound) so we always have a valid cover.
    import itertools
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    ds = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    selected = [tuple(w) for w in ds]

    def covered_mask(sel):
        # boolean (N,) : word covered by at least one selected codeword
        if not sel:
            return np.zeros(N, dtype=bool)
        idx = np.array(sel, dtype=np.intp)
        return (match_all[:, idx] >= (n - 1)).any(axis=1)

    # Local pruning: drop any codeword whose removal keeps full coverage.
    def prune(sel):
        while True:
            if time.time() > deadline:
                break
            changed = False
            for i in range(len(sel)):
                rest = sel[:i] + sel[i + 1:]
                if covered_mask(rest).all():
                    sel = rest
                    changed = True
                    break
            if not changed:
                break
        return sel

    selected = prune(selected)

    # Greedy set cover: add codewords that maximise newly-covered words.
    covered = covered_mask(selected)
    while True:
        if time.time() > deadline:
            break
        if covered.all():
            break
        # For each candidate not yet selected, count how many uncovered words it covers.
        rem = ~covered
        gains = (match_all >= (n - 1))[rem, :].sum(axis=0)  # (q^n,)
        # mask out already-selected
        sel_arr = np.zeros(q ** n, dtype=bool)
        for s in selected:
            sel_arr[s] = True
        gains[sel_arr] = -1
        best = int(np.argmax(gains))
        if gains[best] <= 0:
            break
        selected.append(best)
        covered |= (match_all[:, best] >= (n - 1))

    # Final pruning to shed any redundancy introduced by the greedy phase.
    selected = prune(selected)

    return [list(w) for w in selected]
