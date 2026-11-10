# EVOLVE-BLOCK-START
import time
import itertools
import numpy as np

_T1 = time.time()
_HARD = _T1 + 150.0


def _deadline(margin):
    return (time.time() + margin) < _HARD


def _fallback_base(q, n, radius):
    """Direct-sum of tetracode: 9 * q^(n-4) words, radius-1 for n>=4, q=3."""
    if q != 3 or n < 4:
        return [list(t) for t in itertools.product(range(q), repeat=n)]
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    tail_len = n - 4
    out = []
    for tail in itertools.product(range(q), repeat=tail_len):
        for h in tetracode:
            out.append(h + list(tail))
    return out


def _build_neighbors_np(q, n):
    """Return all_words (list of tuples), word_idx (dict), and nrm (list of lists of indices) using NumPy for speed."""
    N = q ** n
    # Generate all words as a 2D numpy array
    arr = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int8)
    word_idx = {tuple(arr[i]): i for i in range(N)}
    
    # Build neighbor indices
    nrm = [[] for _ in range(N)]
    for i in range(N):
        w = arr[i]
        nrm[i].append(i)
        for j in range(n):
            for d in (1, 2):
                b = w.copy()
                b[j] = (b[j] + d) % q
                nrm[i].append(word_idx[tuple(b)])
    return arr, word_idx, nrm


def _improve_remove_repair(base, q, n, radius):
    """Time-guarded remove-then-repair: shrink the code while keeping it a valid cover."""
    if radius < 1:
        return base
    N = q ** n
    all_words, word_idx, nrm = _build_neighbors_np(q, n)

    # map codewords to indices (dedupe)
    sel = []
    seen = set()
    for cw in base:
        i = word_idx[tuple(cw)]
        if i not in seen:
            seen.add(i)
            sel.append(i)

    # coverage count per word
    cov = np.zeros(N, dtype=np.int32)
    for ci in sel:
        for nb in nrm[ci]:
            cov[nb] += 1

    sel_set = set(sel)
    sel_pos = {ci: idx for idx, ci in enumerate(sel)}

    deadline_guard = _HARD - 10.0
    improved = True
    while improved and time.time() < deadline_guard:
        improved = False
        # find a removable codeword: after removal, at most 2 words become uncovered
        for ci in sel:
            # collect words that would drop to 0 if we remove ci
            uncover_list = []
            for nb in nrm[ci]:
                if cov[nb] == 1:
                    uncover_list.append(nb)
                    if len(uncover_list) > 2:
                        break
            if len(uncover_list) <= 2:
                # remove ci
                for nb in nrm[ci]:
                    cov[nb] -= 1
                sel.remove(ci)
                sel_set.discard(ci)
                improved = True
                # repair: for each uncovered word, add a codeword that covers it
                for uc in uncover_list:
                    if cov[uc] == 0:
                        # choose best repair: try uc itself first, then neighbors
                        best = uc
                        best_gain = 0
                        # score each candidate by how many currently-uncovered neighbors it covers
                        for cand in nrm[uc]:
                            gain = 0
                            for nb2 in nrm[cand]:
                                if cov[nb2] == 0:
                                    gain += 1
                            if gain > best_gain:
                                best_gain = gain
                                best = cand
                        if best not in sel_set:
                            sel.append(best)
                            sel_set.add(best)
                            for nb in nrm[best]:
                                cov[nb] += 1
                if not _deadline(10.0):
                    break
    return [list(all_words[i]) for i in sel]


def _greedy_set_cover(q, n, radius, initial_code=None):
    """Greedy set cover starting from initial_code or empty, trying to minimize codewords."""
    N = q ** n
    all_words, word_idx, nrm = _build_neighbors_np(q, n)
    
    # Start with initial code if provided
    if initial_code:
        sel = []
        seen = set()
        for cw in initial_code:
            i = word_idx[tuple(cw)]
            if i not in seen:
                seen.add(i)
                sel.append(i)
    else:
        sel = []
        seen = set()
    
    # coverage count per word
    cov = np.zeros(N, dtype=np.int32)
    for ci in sel:
        for nb in nrm[ci]:
            cov[nb] += 1
    
    uncovered = np.where(cov == 0)[0]
    
    deadline_guard = _HARD - 15.0
    while len(uncovered) > 0 and time.time() < deadline_guard:
        # Find the word that covers the most uncovered words
        best_word = -1
        best_count = 0
        
        # Sample candidates: prioritize words near uncovered words
        # Try a subset for speed
        candidates = list(uncovered[:min(200, len(uncovered))])
        # Also try some random samples
        if len(uncovered) > 200:
            idx = np.random.choice(len(uncovered), 200, replace=False)
            candidates.extend(uncovered[idx].tolist())
        
        for w in candidates:
            cnt = 0
            for nb in nrm[w]:
                if cov[nb] == 0:
                    cnt += 1
            if cnt > best_count:
                best_count = cnt
                best_word = w
        
        if best_word < 0 or best_count == 0:
            break
        
        sel.append(best_word)
        seen.add(best_word)
        for nb in nrm[best_word]:
            cov[nb] += 1
        
        uncovered = np.where(cov == 0)[0]
    
    return [list(all_words[i]) for i in sel]


def construct(q, n, radius):
    global _HARD
    _HARD = time.time() + 150.0
    if n < 0:
        return []
    if n == 0:
        return [[]]

    # --- Build a guaranteed-valid base code ---
    base = _fallback_base(q, n, radius)
    
    # For q=3, n=8, radius=1: try to get a better initial code via greedy set cover
    if q == 3 and n == 8 and radius == 1 and _deadline(100.0):
        # First, try greedy from empty
        greedy_result = _greedy_set_cover(3, 8, 1, initial_code=None)
        if len(greedy_result) < len(base):
            base = greedy_result
        
        # Also try greedy starting from tetracode structure
        tetracode_result = _greedy_set_cover(3, 8, 1, initial_code=base)
        if len(tetracode_result) < len(base):
            base = tetracode_result

    # --- Guarantee validity: cover all words (repair if base was not perfect) ---
    if radius >= 1:
        N = q ** n
        all_words, word_idx, nrm = _build_neighbors_np(q, n)
        cov = np.zeros(N, dtype=np.int32)
        sel = []
        seen = set()
        for cw in base:
            i = word_idx[tuple(cw)]
            if i not in seen:
                seen.add(i)
                sel.append(i)
                for nb in nrm[i]:
                    cov[nb] += 1
        uncovered = np.where(cov == 0)[0]
        # repair: add uncovered words as codewords
        while len(uncovered) > 0 and _deadline(30.0):
            uc = int(uncovered[0])
            if uc not in seen:
                seen.add(uc)
                sel.append(uc)
                for nb in nrm[uc]:
                    cov[nb] += 1
                # recompute remaining uncovered efficiently
                if len(uncovered) > 0:
                    uncovered = np.where(cov == 0)[0]

        base = [list(all_words[i]) for i in sel]

        # --- Optional improvement (time-guarded) ---
        if _deadline(60.0) and len(base) <= 20000:
            base = _improve_remove_repair(base, q, n, radius)

    return [list(c) for c in base]
# EVOLVE-BLOCK-END
