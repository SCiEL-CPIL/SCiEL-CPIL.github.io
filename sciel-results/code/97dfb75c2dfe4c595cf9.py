"""Football pool / ternary covering code K_3(7,1) via greedy set cover with
coverage-count acceleration.  Starts from the tetracode direct sum (81 words)
and repeatedly: (a) removes any redundant codeword, (b) adds the uncovered
word whose sphere covers the most still-uncovered words."""
# EVOLVE-BLOCK-START
import time
import itertools
import random
from collections import Counter


def _sphere(words, q, n, r):
    """Yield each word at Hamming distance <= r from every word in `words`."""
    for w in words:
        yield w
        if r >= 1:
            for i in range(n):
                for v in range(q):
                    if v != w[i]:
                        yield tuple(w[:i] + (v,) + w[i + 1:])
        if r >= 2:
            for i in range(n):
                for j in range(i + 1, n):
                    for vi in range(q):
                        if vi == w[i]:
                            continue
                        for vj in range(q):
                            if vj == w[j]:
                                continue
                            yield tuple(w[:i] + (vi,) + w[i + 1:j] + (vj,) + w[j + 1:])


def _tetracode_direct_sum(q, n, r):
    """Direct sum of the ternary tetracode (9 words, length 4, radius 1 perfect)
    with the full space on the remaining n-4 coordinates.  Valid for q=3, n>=4, r=1."""
    tet = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    rest = n - 4
    out = []
    for h in tet:
        for tail in itertools.product(range(q), repeat=rest):
            out.append(tuple(h + list(tail)))
    return out


def _is_covered(idx, q, n, r):
    """True if word `idx` is within distance r of some chosen word (via coverage)."""
    # placeholder; actual check uses the coverage array in the main loop
    return False


def _coverage_array(initial, q, n, r):
    """Return a list of length q**n with coverage counts, plus an idx map."""
    total = q ** n
    cov = [0] * total
    # precompute idx for each word in initial
    for w in initial:
        base = 1
        idx = 0
        for d in range(n - 1, -1, -1):
            idx += w[d] * (q ** d)
        for s in _sphere([w], q, n, r):
            si = 0
            for d in range(n - 1, -1, -1):
                si = si * q + s[d]
            cov[si] += 1
    return cov


def _word_idx(w, q, n):
    idx = 0
    for d in range(n):
        idx = idx * q + w[d]
    return idx


def _idx_word(idx, q, n):
    w = [0] * n
    for d in range(n - 1, -1, -1):
        w[d] = idx % q
        idx //= q
    return tuple(w)


def _sphere_indices(w, q, n, r):
    """Yield integer indices of all words within distance r of w."""
    base = 1
    idx = 0
    for d in range(n - 1, -1, -1):
        idx += w[d] * (q ** d)
    yield idx
    if r >= 1:
        for i in range(n):
            pw = q ** (n - 1 - i)
            for v in range(q):
                if v != w[i]:
                    yield idx + (v - w[i]) * pw
    if r >= 2:
        for i in range(n):
            pw_i = q ** (n - 1 - i)
            for vi in range(q):
                if vi == w[i]:
                    continue
                for j in range(i + 1, n):
                    pw_j = q ** (n - 1 - j)
                    for vj in range(q):
                        if vj == w[j]:
                            continue
                        yield idx + (vi - w[i]) * pw_i + (vj - w[j]) * pw_j


def construct(q, n, radius):
    deadline = time.time() + 130.0  # 20 s safety margin under the 150 s kill
    rng = random.Random(20240527)

    # --- Initial solution: tetracode direct sum (valid when q==3, n>=4, radius==1) ---
    if q == 3 and n >= 4 and radius == 1:
        chosen = list(_tetracode_direct_sum(q, n, radius))
    else:
        # Fallback: greedy add from empty (covers the general interface)
        chosen = []

    chosen_set = set(chosen)
    total = q ** n

    # Build coverage counts
    cov = [0] * total
    for w in chosen:
        for si in _sphere_indices(w, q, n, radius):
            cov[si] += 1

    # Track which chosen word covers each position (for removal)
    # Use a dict: idx -> list of chosen indices covering it (we only need one)
    # To keep memory reasonable, store a single "witness" chosen-index per position.
    witness = [-1] * total
    for ci, w in enumerate(chosen):
        for si in _sphere_indices(w, q, n, radius):
            if witness[si] == -1:
                witness[si] = ci

    def try_remove():
        """Remove a redundant codeword if one exists; return True if removed."""
        # For each chosen word, check if every position it covers is covered by another.
        # We need, for each position s in sphere(w_i), whether witness[s] != i OR
        # some other chosen word also covers s.  Since witness stores only one,
        # we need a stronger check: count how many chosen words cover each position.
        # We already have cov[s]; removal is safe iff for all s in sphere(w_i), cov[s] >= 2.
        for ci, w in enumerate(chosen):
            ok = True
            for si in _sphere_indices(w, q, n, radius):
                if cov[si] < 2:
                    ok = False
                    break
            if ok:
                # Remove chosen[ci]
                for si in _sphere_indices(w, q, n, radius):
                    cov[si] -= 1
                    if witness[si] == ci:
                        witness[si] = -1
                del chosen[ci]
                # Rebuild witness for positions that lost their witness
                for si in _sphere_indices(w, q, n, radius):
                    if witness[si] == -1:
                        # find any remaining chosen word covering si
                        for cj, wj in enumerate(chosen):
                            if _within(wj, si, q, n, radius):
                                witness[si] = cj
                                break
                return True
        return False

    def _within(w, si, q, n, r):
        # check Hamming distance between w and word si
        sw = _idx_word(si, q, n)
        d = sum(1 for a, b in zip(w, sw) if a != b)
        return d <= r

    def greedy_add_step():
        """Add the word whose sphere covers the most uncovered positions."""
        best_gain = 0
        best_w = None
        # Candidate pool: all currently uncovered words
        for u in range(total):
            if cov[u] == 0:
                # gain = number of uncovered words in sphere(u)
                gain = 0
                for si in _sphere_indices(u, q, n, radius):
                    if cov[si] == 0:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_w = _idx_word(u, q, n)
                if time.time() > deadline:
                    return False
        if best_w is not None and best_gain > 0:
            for si in _sphere_indices(best_w, q, n, radius):
                cov[si] += 1
                if witness[si] == -1:
                    witness[si] = len(chosen)
            chosen.append(best_w)
            return True
        return False

    # --- Main loop: alternate removal and addition ---
    no_progress = 0
    while time.time() < deadline:
        # Pass 1: try to remove as many redundant words as possible
        removed_any = False
        while try_remove():
            removed_any = True
            if time.time() > deadline:
                break
        if removed_any:
            no_progress = 0
        else:
            # Pass 2: try to add a high-coverage word
            if greedy_add_step():
                no_progress = 0
            else:
                no_progress += 1
                if no_progress >= 3:
                    break

    # Final validity check and fix
    # Rebuild coverage from scratch to be safe
    cov2 = [0] * total
    for w in chosen:
        for si in _sphere_indices(w, q, n, radius):
            cov2[si] += 1
    # Add any missing words
    for u in range(total):
        if cov2[u] == 0:
            w = _idx_word(u, q, n)
            for si in _sphere_indices(w, q, n, radius):
                cov2[si] += 1
            chosen.append(w)
            if time.time() > deadline - 5:
                break

    # Deduplicate
    seen = set()
    unique = []
    for w in chosen:
        if w not in seen:
            seen.add(w)
            unique.append(list(w))
    return unique
# EVOLVE-BLOCK-END
