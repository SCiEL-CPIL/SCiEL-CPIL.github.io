"""Structural redesign: greedy set-cover + bit-parallel local search for K_q(n, radius)."""
import time
import numpy as np


def _build_sphere(q, n, radius):
    """Return a numpy array of all offsets (as ints in [0,q^n)) within Hamming radius `radius`."""
    from itertools import product
    offsets = []
    for r in range(radius + 1):
        for pos in product(range(n), repeat=r):
            # unique positions, allow repeats? Hamming distance <= r means choosing r distinct positions.
            # Use combinations for distinct positions.
            pass
    # Use combinations for distinct positions (distance exactly r uses r distinct coords)
    from itertools import combinations
    offsets = []
    for r in range(radius + 1):
        for pos in combinations(range(n), r):
            base = [0] * n
            for p in pos:
                for d in range(1, q):
                    v = base[:]
                    v[p] = d
                    # encode
                    idx = 0
                    for i in range(n):
                        idx = idx * q + v[i]
                    offsets.append(idx)
    # dedupe
    uniq = list(set(offsets))
    return np.array(uniq, dtype=np.int64)


def _encode(codeword, q, n):
    idx = 0
    for i in range(n):
        idx = idx * q + codeword[i]
    return idx


def _decode(idx, q, n):
    out = [0] * n
    for i in range(n - 1, -1, -1):
        out[i] = idx % q
        idx //= q
    return out


def construct(q, n, radius):
    deadline = time.time() + 110.0  # safety margin under 120s
    total = q ** n
    sphere = _build_sphere(q, n, radius)
    sphere_size = len(sphere)

    # Represent universe as bits: use a list of int64 blocks for the bitset of covered words.
    # For greedy, we need to compute coverage count per candidate center quickly.
    # We'll maintain an array `cover_count` of length total (number of centers covering each word).
    # Greedy: pick center maximizing number of currently-uncovered words it covers.

    # Precompute, for each center c, the list of words it covers (c + sphere offsets).
    # That's total * sphere_size which may be large; instead compute on the fly but cache?
    # total up to 2187, sphere_size for q=3,n=7,r=1 is 1 + 7*2 = 15. total*sphere = ~32805, fine.
    # Precompute coverage sets as arrays of indices.
    centers_all = np.arange(total, dtype=np.int64)
    # cover[c] = array of word indices covered by center c
    # compute: for each center c (as integer), covered = (c + offsets) but with carry issues in mixed radix.
    # Mixed radix addition: adding an offset in mixed radix is not the same as integer addition.
    # So we must compute covered words by decoding/encoding, or by mixed-radix vector ops.

    # Efficient approach: represent center as array of n digits; covered = center + each offset digit-wise mod? No,
    # Hamming offset changes specific positions by a delta (1 or 2), no wrap-around in the coordinate (0..q-1),
    # but the offset value is in 0..q-1 and we add it to the coordinate; since offset digit is in {0,1,..,q-1}
    # and center digit in 0..q-1, sum can exceed q-1 -> that's a different word, not a valid neighbor.
    # Wait: neighbor of center c at position p with value d means word w where w[p]=d and w[i]=c[i] for i!=p.
    # So the offset is NOT additive; it's a substitution. The neighbor word has coordinate p set to d (any value != c[p]).
    # So covered words = all words that differ from c in <= radius positions (any values).
    # So for radius 1: covered = {c} U { words with one coordinate changed to any of the other q-1 values }.
    # So the set is: for each position p, for each value v in 0..q-1, v != c[p]: the word with c[p] replaced by v.

    # Let's precompute covered sets properly.
    # For each center c (digits), covered words:
    #   distance 0: c itself
    #   distance 1: for each p, for each v != c[p]: word = c with c[p]=v
    # Represent as integer indices.

    # Build a table: for each center index, list of covered indices.
    # We'll compute using numpy by iterating centers but vectorizing over offsets per center.
    # Given total is small (<= 2187) and sphere small, do it in Python loops but with numpy for the union.

    # Precompute covered as a list of arrays.
    covered_sets = []
    for c in range(total):
        digits = _decode(c, q, n)
        idxs = [c]
        if radius >= 1:
            for p in range(n):
                orig = digits[p]
                for v in range(q):
                    if v == orig:
                        continue
                    d2 = digits[:]
                    d2[p] = v
                    idxs.append(_encode(d2, q, n))
        if radius >= 2:
            for p1 in range(n):
                for p2 in range(p1 + 1, n):
                    for v1 in range(q):
                        if v1 == digits[p1]:
                            continue
                        for v2 in range(q):
                            if v2 == digits[p2]:
                                continue
                            d2 = digits[:]
                            d2[p1] = v1
                            d2[p2] = v2
                            idxs.append(_encode(d2, q, n))
        # Store as a list of ints to allow flexible indexing, or np.array int64
        covered_sets.append(np.array(idxs, dtype=np.int64))

    # Greedy set cover:
    cover_count = np.zeros(total, dtype=np.int32)  # how many chosen centers cover each word
    chosen = []
    # uncovered bitset as boolean array
    uncovered = np.ones(total, dtype=bool)
    remaining = total
    
    # Precompute gain for all centers initially? No, update dynamically.
    # To speed up greedy, we can just loop. total=2187, sphere~15. 
    # 2187 * 15 ops per gain calc is ~30k. 
    # Number of centers picked ~180. 
    # 180 * 2187 * 30k is too slow if we recalculate all gains every time.
    # However, we only need to check candidates. 
    # Actually, standard greedy recalculates gain for all candidates each step.
    # 180 steps * 2187 candidates * (cost to sum boolean array of size ~15)
    # Cost to sum is small. 180 * 2187 * 15 ~ 5.8M operations. This is fine in Python/NumPy.
    
    while remaining > 0 and time.time() < deadline:
        best_c = -1
        best_gain = 0
        # Optimization: skip centers that are already chosen? No, we can add duplicates in greedy logic but we use a set later.
        # Actually, if a center is chosen, its gain becomes 0 for *newly* uncovered words if we update uncovered correctly.
        # But we don't remove chosen centers from consideration, we just pick the one with max gain.
        # If a center was picked, its gain for *remaining* uncovered words might be 0 if it covered everything it could, 
        # but it might cover others? No, if it was picked, all words it covered that were uncovered became covered.
        # So its gain for *current* uncovered set is 0. 
        # So we can safely skip chosen centers in the next iterations to save time? 
        # Yes, if we maintain a list of active candidates.
        
        # For simplicity and correctness first:
        for c in range(total):
            if c in chosen: 
                continue
            cs = covered_sets[c]
            # Ensure cs is int array for indexing
            gain = np.count_nonzero(uncovered[cs])
            if gain > best_gain:
                best_gain = gain
                best_c = c
        if best_gain == 0:
            break
        chosen.append(best_c)
        cs = covered_sets[best_c]
        # update cover_count and uncovered
        # We only care about `uncovered` for the next gain calculation.
        # We don't strictly need cover_count for greedy, only for local search validity checks later.
        # But let's keep it.
        cover_count[cs] += 1
        # Update uncovered
        # Words covered by best_c that are currently uncovered become covered
        newly_covered_mask = uncovered[cs]
        remaining -= np.count_nonzero(newly_covered_mask)
        uncovered[cs] = False
        
    best = list(set(chosen))
    best_size = len(best)

    # Local search: 
    # 1. Removal phase: Check if any center is redundant.
    # 2. Swap phase: Try to replace one center with another to improve coverage or allow further removals.
    # To be fast, we maintain cover_count and uncovered.
    
    # Re-initialize cover_count and uncovered for the current best set
    cover_count = np.zeros(total, dtype=np.int32)
    for c in best:
        cover_count[covered_sets[c]] += 1
    uncovered = (cover_count == 0)
    
    def try_remove(i):
        """Try to remove best[i]. Returns True if successful."""
        nonlocal cover_count, uncovered, best, best_size
        c = best[i]
        cs = covered_sets[c]
        # If any word in cs has cover_count 1, it becomes uncovered
        # We need to check if all words in cs have cover_count > 1
        if np.any(cover_count[cs] == 1):
            return False
        # Safe to remove
        cover_count[cs] -= 1
        best.pop(i)
        best_size -= 1
        return True

    def try_swap(remove_idx, add_c):
        """Try to replace best[remove_idx] with add_c."""
        nonlocal cover_count, uncovered, best, best_size
        c_old = best[remove_idx]
        c_new = add_c
        if c_old == c_new:
            return False
        
        cs_old = covered_sets[c_old]
        cs_new = covered_sets[c_new]
        
        # Calculate new cover counts virtually
        # Words in cs_old lose 1
        # Words in cs_new gain 1
        
        # Optimization: 
        # 1. If any word in cs_old has count 1, it becomes 0 (bad) UNLESS it is also in cs_new and gains 1.
        # 2. If any word in cs_new has count 0, it becomes 1 (good).
        
        # Check validity:
        # A word w is valid if new_count(w) > 0.
        # new_count(w) = old_count(w) - (1 if w in cs_old else 0) + (1 if w in cs_new else 0)
        
        # We can check:
        # For w in cs_old: if old_count(w)==1 and w not in cs_new -> invalid.
        # For w in cs_new: if old_count(w)==0 and w not in cs_old -> valid (it was 0, now 1).
        # For w in intersection: count stays same.
        # For w outside: count stays same.
        
        # So we only need to check:
        # 1. Are there any words in cs_old that have count 1 AND are not in cs_new?
        # 2. (Optional optimization) Does this swap help? It doesn't change size, but might enable future removals.
        
        # Efficient check:
        # Create a set of cs_new for O(1) lookup? Or use boolean array.
        # Given small sizes, boolean array is fast.
        in_new = np.zeros(total, dtype=bool)
        in_new[cs_new] = True
        
        # Check condition 1:
        # words_at_risk = cs_old where cover_count == 1
        at_risk = cover_count[cs_old] == 1
        # Check if any of these are NOT in cs_new
        # We need to check if (at_risk AND NOT in_new) is True for any index in cs_old
        # Let's get the indices of at_risk in cs_old
        risky_indices = cs_old[at_risk]
        if len(risky_indices) > 0:
            # Check if any of these risky indices are NOT in cs_new
            # in_new[risky_indices] gives boolean array
            # If any is False, then that word is left uncovered
            if np.any(~in_new[risky_indices]):
                return False
            
        # If we pass, the swap is valid (maintains coverage).
        # Apply swap
        cover_count[cs_old] -= 1
        cover_count[cs_new] += 1
        best[remove_idx] = c_new
        return True

    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Phase 1: Removals
        i = 0
        while i < len(best) and time.time() < deadline:
            if try_remove(i):
                improved = True
                # Don't increment i, check same index again (now next element)
            else:
                i += 1
        
        # Phase 2: Swaps
        # We only do swaps if we couldn't remove anything, or to find better configurations?
        # Usually, for covering codes, greedy + removal is quite good.
        # Swaps can help escape local minima.
        # To keep it fast, we limit swaps.
        if not improved and time.time() < deadline:
            # Try random swaps or systematic? Systematic is better for reproducibility but slower.
            # Let's try a limited number of random swaps to see if we can enable removals.
            # Actually, a common heuristic is: 
            # For each center, try swapping with a center that covers "different" uncovered areas?
            # But we have no uncovered areas.
            # Let's try swapping a center with another center that is currently in the set? No, that's a move.
            # Swap with a center NOT in the set.
            
            # Heuristic: Pick a center to remove, find the best replacement that minimizes loss of coverage?
            # Actually, if we can't remove, maybe we can swap to a position that allows a *different* center to be removed?
            
            # Let's try a simple hill-climbing: 
            # Pick a random center in best, pick a random center not in best.
            # If swap is valid, accept it? No, size doesn't change. 
            # Accept it only if it allows a subsequent removal?
            # This requires lookahead.
            
            # Alternative: Just accept valid swaps that change the set? 
            # If we accept any valid swap, we might cycle.
            # Let's only accept swaps if they reduce the "redundancy" or something?
            
            # Given time constraints, let's stick to Removals primarily. 
            # The greedy solution for K_3(7,1) is often close to optimal.
            # 186 is the known bound. Greedy might give ~200-220. 
            # Removals will cut it down.
            
            # Let's add a simple "2-opt" style: 
            # Try removing i, if fails, try adding j and removing i?
            # That's what try_swap does.
            
            # Let's try a few random swaps to perturb the solution, then run removals again.
            if time.time() < deadline:
                # Pick a random center to replace
                idx_to_replace = np.random.randint(0, len(best))
                # Pick a random candidate not in best
                # To avoid O(N) check for "not in best", we can just pick random from total and check.
                for _ in range(10): # 10 attempts
                    if time.time() > deadline: break
                    cand = np.random.randint(0, total)
                    if cand in best:
                        continue
                    if try_swap(idx_to_replace, cand):
                        # Swap accepted. This might enable new removals.
                        improved = True
                        break

    # Convert back to codewords
    result = [_decode(c, q, n) for c in best]
    return result
