# EVOLVE-BLOCK-START
import time
import itertools
import random

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


def _build_neighbor_indices(q, n):
    """Return (all_words, word_idx, nrm) where nrm[i] is list of indices
    of words within Hamming distance 1 of all_words[i]."""
    N = q ** n
    all_words = list(itertools.product(range(q), repeat=n))
    word_idx = {w: i for i, w in enumerate(all_words)}
    nrm = []
    for w in all_words:
        nbrs = [word_idx[w]]
        for j in range(n):
            for d in (1, 2):
                b = list(w)
                b[j] = (b[j] + d) % q
                nbrs.append(word_idx[tuple(b)])
        nrm.append(nbrs)
    return all_words, word_idx, nrm, N


def _greedy_remove_fast(base, q, n, radius, seed=42):
    """Fast greedy removal using coverage counts. Returns list of codeword indices."""
    if radius < 1:
        return [tuple(c) for c in base]
    
    all_words, word_idx, nrm, N = _build_neighbor_indices(q, n)
    
    sel = []
    seen = set()
    for cw in base:
        i = word_idx[tuple(cw)]
        if i not in seen:
            seen.add(i)
            sel.append(i)
    
    cov = [0] * N
    for ci in sel:
        for nb in nrm[ci]:
            cov[nb] += 1
    
    sel_set = set(sel)
    rng = random.Random(seed)
    deadline_guard = _HARD - 20.0
    improved = True
    max_rounds = 50
    round_num = 0
    
    while improved and time.time() < deadline_guard and round_num < max_rounds:
        round_num += 1
        improved = False
        # Shuffle for diversity
        rnd_order = list(range(len(sel)))
        rng.shuffle(rnd_order)
        for idx in rnd_order:
            ci = sel[idx]
            removable = True
            for nb in nrm[ci]:
                if cov[nb] == 1:
                    removable = False
                    break
            if removable:
                for nb in nrm[ci]:
                    cov[nb] -= 1
                sel.remove(ci)
                sel_set.discard(ci)
                improved = True
        if not _deadline(15.0):
            break
    
    return [list(all_words[i]) for i in sel]


def _multi_pass_remove(base, q, n, radius):
    """Try multiple random seeds for greedy removal, keep the best result."""
    if radius < 1 or len(base) <= 20000:
        return _greedy_remove_fast(base, q, n, radius, seed=42)
    
    best_result = base
    best_len = len(base)
    
    seeds = [42, 123, 777, 2024, 999]
    for seed in seeds:
        if not _deadline(25.0):
            break
        result = _greedy_remove_fast(base, q, n, radius, seed=seed)
        if len(result) < best_len:
            best_len = len(result)
            best_result = result
    
    return best_result


def _swap_improve(base, q, n, radius):
    """Try swaps that enable further removals. Returns improved base."""
    if radius < 1 or len(base) > 50000:
        return base
    
    all_words, word_idx, nrm, N = _build_neighbor_indices(q, n)
    
    sel = []
    seen = set()
    for cw in base:
        i = word_idx[tuple(cw)]
        if i not in seen:
            seen.add(i)
            sel.append(i)
    
    cov = [0] * N
    for ci in sel:
        for nb in nrm[ci]:
            cov[nb] += 1
    
    sel_set = set(sel)
    deadline_guard = _HARD - 30.0
    improved = True
    max_rounds = 20
    
    while improved and time.time() < deadline_guard and max_rounds > 0:
        max_rounds -= 1
        improved = False
        for ci in list(sel):
            # Find words that would become uncovered
            would_uncover = []
            for nb in nrm[ci]:
                if cov[nb] == 1:
                    would_uncover.append(nb)
            
            if not would_uncover:
                # Safe removal
                for nb in nrm[ci]:
                    cov[nb] -= 1
                sel.remove(ci)
                sel_set.discard(ci)
                improved = True
                break
            
            if len(would_uncover) > 4:
                continue
            
            # Try to find a replacement that covers all would_uncover
            # and enables removing another codeword
            candidates = set()
            for uc in would_uncover:
                candidates.add(uc)
                for nb in nrm[uc]:
                    candidates.add(nb)
            
            best_cand = -1
            best_gain = 0
            
            for cand in candidates:
                if cand in sel_set:
                    continue
                cand_nrm = nrm[cand]
                covers_all = True
                for uc in would_uncover:
                    if uc not in cand_nrm:
                        covers_all = False
                        break
                if not covers_all:
                    continue
                
                # Calculate gain: how many words go from cov=1 to cov>=2
                # (making them potentially removable)
                gain = 0
                for nb in cand_nrm:
                    if cov[nb] == 1:
                        gain += 1
                # Subtract words that go from cov>=2 to cov+1 (less beneficial)
                for nb in nrm[ci]:
                    if nb in cand_nrm and cov[nb] >= 2:
                        gain -= 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_cand = cand
            
            if best_cand >= 0 and best_gain > 0:
                # Perform swap
                for nb in nrm[ci]:
                    cov[nb] -= 1
                sel.remove(ci)
                sel_set.discard(ci)
                for nb in nrm[best_cand]:
                    cov[nb] += 1
                sel.append(best_cand)
                sel_set.add(best_cand)
                improved = True
                break
        
        if not _deadline(20.0):
            break
    
    return [list(all_words[i]) for i in sel]


def _greedy_add(base, q, n, radius):
    """Ensure validity by adding codewords to cover any uncovered words."""
    if radius < 1:
        return base
    
    all_words, word_idx, nrm, N = _build_neighbor_indices(q, n)
    
    sel = []
    seen = set()
    for cw in base:
        i = word_idx[tuple(cw)]
        if i not in seen:
            seen.add(i)
            sel.append(i)
    
    cov = [0] * N
    for ci in sel:
        for nb in nrm[ci]:
            cov[nb] += 1
    
    uncovered = [i for i in range(N) if cov[i] == 0]
    if not uncovered:
        return [list(all_words[i]) for i in sel]
    
    sel_set = set(sel)
    deadline_guard = _HARD - 10.0
    while uncovered and time.time() < deadline_guard:
        best_i = -1
        best_cov = 0
        for i in uncovered:
            c = 0
            for nb in nrm[i]:
                if cov[nb] == 0:
                    c += 1
            if c > best_cov:
                best_cov = c
                best_i = i
        if best_i < 0:
            break
        sel.append(best_i)
        sel_set.add(best_i)
        for nb in nrm[best_i]:
            cov[nb] += 1
        uncovered = [i for i in uncovered if cov[i] == 0]
    
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
    
    if radius >= 1:
        # --- Guarantee validity: cover all words ---
        N = q ** n
        all_words, word_idx, nrm = _build_neighbor_indices(q, n)
        
        cov = [0] * N
        sel = []
        seen = set()
        for cw in base:
            i = word_idx[tuple(cw)]
            if i not in seen:
                seen.add(i)
                sel.append(i)
                for nb in nrm[i]:
                    cov[nb] += 1
        
        uncovered = [i for i in range(N) if cov[i] == 0]
        while uncovered and _deadline(20.0):
            uc = uncovered.pop()
            if uc not in seen:
                seen.add(uc)
                sel.append(uc)
                for nb in nrm[uc]:
                    cov[nb] += 1
                if uncovered:
                    uncovered = [i for i in uncovered if cov[i] == 0]
        
        base = [list(all_words[i]) for i in sel]
        
        # --- Multi-pass greedy removal with different seeds ---
        if _deadline(60.0) and len(base) <= 50000:
            base = _multi_pass_remove(base, q, n, radius)
        
        # --- Swap improvement ---
        if _deadline(30.0) and len(base) <= 50000:
            base = _swap_improve(base, q, n, radius)
        
        # --- Final greedy removal pass ---
        if _deadline(20.0) and len(base) <= 50000:
            base = _greedy_remove_fast(base, q, n, radius, seed=999)
        
        # --- Ensure validity after all modifications ---
        if _deadline(10.0) and len(base) <= 50000:
            base = _greedy_add(base, q, n, radius)
    
    return [list(c) for c in base]
# EVOLVE-BLOCK-END
