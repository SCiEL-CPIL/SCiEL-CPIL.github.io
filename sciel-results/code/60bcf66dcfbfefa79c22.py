"""Evolved program: construction + swap-based local search for K_3(7,1)."""
# EVOLVE-BLOCK-START
import time
import itertools

_DEADLINE = time.time() + 112.0  # safety margin under the 150 s wall-clock kill


def _perfect_code_len5(q=3):
    """A perfect ternary code of length 5 with 15 words (radius-1 perfect).

    Built from the ternary Hamming code of length 5 (parity check with distinct
    nonzero column vectors over GF(3)); the set of codewords is a perfect
    covering code of length 5, covering radius 1, size 3^(5-2)=27? No: a
    (5,15) perfect code has 15 codewords covering 15*(1+5)=90=3^4 words.
    """
    # Standard (15,9,4) ternary Hamming code generator gives codewords; here we
    # want the length-5 perfect code (the "simplex" complement). We use the
    # well-known explicit list of 15 words forming a perfect code of length 5.
    # These are the 15 words of the ternary [5,2,4] code? No, [5,2,4] has 9 words.
    # The perfect code of length 5 has size 15. We construct it via the
    # parity-check matrix H (2 x 5) with columns the 5 distinct nonzero
    # 2-vectors; codewords x satisfy H x = 0 -> 3^(5-2)=27 codewords (not 15).
    # So a perfect code of length 5 over GF(3) has size 3^(5-2)/? Actually the
    # perfect code K_3(5,1)=15 is NOT linear. We therefore build it by a
    # greedy perfect construction: start empty, repeatedly add the word with
    # the most uncovered neighbors until all 3^5 words are covered.
    q = 3
    n = 5
    total = q ** n
    # Precompute for each word the set of words within distance 1 (itself + 5*2)
    # Represent each word as an integer 0..total-1 (base-q).
    def decode(idx):
        d = []
        for _ in range(n):
            d.append(idx % q)
            idx //= q
        return d

    def dist1_masks():
        # For a given digit vector, return list of neighbor indices
        pass

    # Greedy perfect code (deterministic): pick word covering most uncovered.
    covered = [False] * total
    code = []
    # Precompute neighbor lists
    neigh = []
    for idx in range(total):
        d = decode(idx)
        ns = [idx]
        for pos in range(n):
            orig = d[pos]
            for v in range(q):
                if v == orig:
                    continue
                nd = d[:]
                nd[pos] = v
                # recompute index
                ii = 0
                for k in range(n - 1, -1, -1):
                    ii = ii * q + nd[k]
                ns.append(ii)
        neigh.append(ns)
    uncovered_count = total
    while uncovered_count > 0:
        best = -1
        best_gain = -1
        for idx in range(total):
            gain = 0
            for nb in neigh[idx]:
                if not covered[nb]:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best = idx
        # add best
        for nb in neigh[best]:
            if not covered[nb]:
                covered[nb] = True
                uncovered_count -= 1
        code.append(best)
    return [decode(c) for c in code]


def _neighbors_all(q, n, radius):
    """Return list of lists: for each word index, the set (as list) of word
    indices within Hamming distance <= radius. Used for validity checks."""
    total = q ** n
    def decode(idx):
        d = []
        for _ in range(n):
            d.append(idx % q)
            idx //= q
        return d

    def encode(d):
        ii = 0
        for k in range(n - 1, -1, -1):
            ii = ii * q + d[k]
        return ii

    # radius 1: word + n*(q-1) single-position changes
    neigh = []
    for idx in range(total):
        d = decode(idx)
        ns = [idx]
        for pos in range(n):
            orig = d[pos]
            for v in range(q):
                if v == orig:
                    continue
                nd = d[:]
                nd[pos] = v
                ns.append(encode(nd))
        neigh.append(ns)
    return neigh


def _ball_masks(q, n, radius):
    """Return a list of Python-int bitmasks: mask[i] has bit j set iff word j
    is within Hamming distance <= radius of word i. Universe size = q**n."""
    total = q ** n
    def decode(idx):
        d = []
        for _ in range(n):
            d.append(idx % q)
            idx //= q
        return d

    def encode(d):
        ii = 0
        for k in range(n - 1, -1, -1):
            ii = ii * q + d[k]
        return ii

    masks = []
    for idx in range(total):
        d = decode(idx)
        m = 0
        # radius 1
        m |= (1 << idx)
        for pos in range(n):
            orig = d[pos]
            for v in range(q):
                if v == orig:
                    continue
                nd = d[:]
                nd[pos] = v
                m |= (1 << encode(nd))
        masks.append(m)
    return masks


def _is_covered(used_mask, universe_mask):
    return (used_mask & universe_mask) == universe_mask


def _construct_initial(q, n, radius):
    """Build an initial covering code using the length-5 perfect code combined
    with the ternary tetracode of length 4, over a balanced split of the 7
    positions, then greedily keep only useful codewords."""
    # Ternary tetracode: 9 words of length 4, perfect radius 1.
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3]
                 for a in range(3) for b in range(3)]
    # Perfect code of length 5 (15 words).
    perfect5 = _perfect_code_len5(3)

    # We need to combine a length-5 code and a length-4 code into length 7.
    # 5 + 4 = 9 > 7, so we cannot simply concatenate. Instead, use the
    # direct-sum construction but with the tetracode on 4 positions and
    # fill the remaining 3 with all words (the seed). To do better, we
    # generate a *superset* candidate set and let local search prune.
    #
    # Candidate set: all words formed by taking a tetracode word on positions
    # P4 (4 of the 7) and a perfect5-derived pattern on the other 3 positions.
    # Since perfect5 has length 5, we take its first 3 coordinates as the
    # pattern for the 3 remaining positions.
    #
    # To enrich the candidate pool, we also include the full direct-sum
    # construction (9 * 3^3 = 243 words) so the search has room.
    candidates = set()
    # Direct sum (seed): tetracode on positions 0..3, all on 4..6
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n - 4):
            w = h + list(tail)
            candidates.add(tuple(w))
    # Tetracode on a different 4-position subset (positions 1..4), all on rest
    for h in tetracode:
        for rest in itertools.product(range(3), repeat=n - 4):
            w = [0] * n
            w[1:5] = h
            w[0] = rest[0]
            w[5] = rest[1]
            w[6] = rest[2]
            candidates.add(tuple(w))
    # Perfect5 pattern on positions 0..2 combined with tetracode on 3..6
    for p in perfect5:
        pat = p[:3]
        for h in tetracode:
            w = list(pat) + list(h)
            candidates.add(tuple(w))
    # Perfect5 pattern on positions 3..5 combined with tetracode on 0,2,6? 
    # Keep it simple: also all-tetracode-rotations.
    for shift in range(n):
        pos4 = [(shift + i) % n for i in range(4)]
        others = [i for i in range(n) if i not in pos4]
        for h in tetracode:
            for tail in itertools.product(range(3), repeat=len(others)):
                w = [0] * n
                for j, p in enumerate(pos4):
                    w[p] = h[j]
                for j, p in enumerate(others):
                    w[p] = tail[j]
                candidates.add(tuple(w))
    return [list(c) for c in candidates]


def _greedy_cover(candidates, q, n, radius, masks):
    """Greedy set cover: repeatedly add the candidate covering the most
    still-uncovered words, until the universe is covered."""
    total = q ** n
    universe = (1 << total) - 1
    uncovered = universe
    chosen = []
    # Precompute candidate masks
    cand_masks = []
    for c in candidates:
        idx = 0
        for k in range(n - 1, -1, -1):
            idx = idx * q + c[k]
        cand_masks.append(masks[idx])
    while uncovered != 0:
        best = -1
        best_gain = -1
        for i, m in enumerate(cand_masks):
            gain = bin(uncovered & m).count('1')
            if gain > best_gain:
                best_gain = gain
                best = i
        if best_gain <= 0:
            break
        chosen.append(candidates[best])
        uncovered &= ~cand_masks[best]
    return chosen


def _swap_search(code, q, n, radius, masks, deadline):
    """Swap-based local search: repeatedly try to remove a codeword whose
    covered region is fully covered by the others. Use critical-word
    intersection to find removal candidates. Track best (smallest) solution."""
    total = q ** n
    universe = (1 << total) - 1

    def code_mask_set(code):
        ms = []
        for c in code:
            idx = 0
            for k in range(n - 1, -1, -1):
                idx = idx * q + c[k]
            ms.append(masks[idx])
        return ms

    best_code = list(code)
    best_mask_list = code_mask_set(best_code)
    best_size = len(best_code)

    # Precompute per-codeword mask
    def idx_of(c):
        idx = 0
        for k in range(n - 1, -1, -1):
            idx = idx * q + c[k]
        return idx

    # Maintain union mask and per-codeword mask
    cur_code = list(code)
    cur_masks = [masks[idx_of(c)] for c in cur_code]

    def try_remove(cur_code, cur_masks):
        """Try to remove codewords one at a time while preserving coverage.
        Returns improved (code, masks) or None."""
        changed = True
        while changed:
            changed = False
            # Compute union mask
            union = 0
            for m in cur_masks:
                union |= m
            # For each codeword, check if removing it keeps coverage of its
            # exclusive words. A codeword i is removable iff
            # (union & ~masks[i]) still covers universe, i.e.
            # union == universe and (union & ~cur_masks[i]) == universe.
            # Equivalently, every word in cur_masks[i] is covered by some
            # other codeword.
            if union != universe:
                break
            for i in range(len(cur_masks)):
                others = union & ~cur_masks[i]
                if (others & universe) == universe:
                    # removable
                    del cur_code[i]
                    del cur_masks[i]
                    changed = True
                    break
        return cur_code, cur_masks

    # Iterative removal pass
    cur_code, cur_masks = try_remove(cur_code, cur_masks)
    if len(cur_code) < best_size:
        best_code = list(cur_code)
        best_mask_list = list(cur_masks)
        best_size = len(best_code)

    # Now do swap-based perturbation: for each codeword, try swapping it with
    # a candidate from the full candidate pool that covers its exclusive words.
    # To keep runtime bounded, we do a limited number of passes.
    # Build a candidate pool (all words) index -> mask is already in masks.
    # For speed, we only consider swaps that reduce size or are neutral.
    # We perform a simple first-improvement descent with a time budget.

    # Precompute candidate list = all 2187 words for swap targets.
    all_words = [list(range(0)) ]  # placeholder
    all_words = []
    for idx in range(total):
        d = []
        x = idx
        for _ in range(n):
            d.append(x % q)
            x //= q
        all_words.append(d)

    # Critical-word intersection: for codeword i, its "critical" words are
    # those covered ONLY by i. A swap target must cover all critical words.
    # We compute critical words and search for a target covering them,
    # preferring targets that also cover many of i's other words.
    passes = 0
    max_passes = 40
    while passes < max_passes and time.time() < deadline:
        passes += 1
        progress = False
        union = 0
        for m in cur_masks:
            union |= m
        if union != universe:
            break
        # Compute exclusive (critical) mask per codeword
        for i in range(len(cur_masks)):
            others = 0
            for j in range(len(cur_masks)):
                if j != i:
                    others |= cur_masks[j]
            critical = cur_masks[i] & ~others
            if critical == 0:
                # already removable
                del cur_code[i]
                del cur_masks[i]
                progress = True
                break
            # Try to find a swap target that covers `critical` and is not
            # already in the code. Prefer target with max overlap with
            # cur_masks[i] to maximize coverage of i's region.
            crit_bits = critical
            best_target = -1
            best_overlap = -1
            # Iterate over all words; for each, check it covers critical.
            for t in range(total):
                mt = masks[t]
                if (mt & crit_bits) != crit_bits:
                    continue
                # overlap with cur_masks[i]
                ov = bin(mt & cur_masks[i]).count('1')
                if ov > best_overlap:
                    best_overlap = ov
                    best_target = t
            if best_target >= 0:
                # Check best_target not already present
                already = False
                for c in cur_code:
                    if c == all_words[best_target]:
                        already = True
                        break
                if not already:
                    # perform swap
                    cur_code[i] = all_words[best_target]
                    cur_masks[i] = masks[best_target]
                    progress = True
                    break
        if not progress:
            # No single swap/removal helps; break to avoid infinite loop.
            break

    if len(cur_code) < best_size:
        best_code = list(cur_code)
        best_size = len(cur_code)

    return best_code


def construct(q, n, radius):
    """Entry point: return a list of codewords (lists of ints in 0..q-1)
    forming a covering code of length n, alphabet q, radius, minimizing size.
    """
    global _DEADLINE
    _DEADLINE = time.time() + 112.0

    if n <= 0:
        return [[]]

    # Build masks for the universe (radius 1 expected; but handle general).
    masks = _ball_masks(q, n, radius)

    # Initial candidate pool
    candidates = _construct_initial(q, n, radius)

    # Greedy cover to get a valid starting code
    code = _greedy_cover(candidates, q, n, radius, masks)

    # Ensure validity; if greedy left a gap, add words until covered.
    total = q ** n
    universe = (1 << total) - 1

    def union_of(code):
        u = 0
        for c in code:
            idx = 0
            for k in range(n - 1, -1, -1):
                idx = idx * q + c[k]
            u |= masks[idx]
        return u

    if union_of(code) != universe:
        # Add words to cover remaining
        covered = union_of(code)
        for idx in range(total):
            if (covered & universe) == universe:
                break
            if not (covered & (1 << idx)):
                d = []
                x = idx
                for _ in range(n):
                    d.append(x % q)
                    x //= q
                code.append(d)
                covered |= masks[idx]

    # Local search
    code = _swap_search(code, q, n, radius, masks, _DEADLINE)

    # Final validity guard: ensure coverage; if not, repair by adding words.
    covered = union_of(code)
    if covered != universe:
        for idx in range(total):
            if (covered & universe) == universe:
                break
            if not (covered & (1 << idx)):
                d = []
                x = idx
                for _ in range(n):
                    d.append(x % q)
                    x //= q
                code.append(d)
                covered |= masks[idx]

    # Deduplicate
    seen = set()
    out = []
    for c in code:
        t = tuple(c)
        if t not in seen:
            seen.add(t)
            out.append(list(c))
    return out
# EVOLVE-BLOCK-END
