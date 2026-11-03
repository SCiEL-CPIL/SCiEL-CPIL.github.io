"""Football pool / ternary covering code K_3(6,1) — time-bounded local search."""
import time
import random
import itertools
import sys

# ---------- helpers ----------

def _all_words(q, n):
    """Yield every word in {0..q-1}^n as a tuple."""
    for t in itertools.product(range(q), repeat=n):
        yield t


def _ball_indices(word, q, n, radius=1):
    """Return the list of words within Hamming distance <= radius of word."""
    idx = [word]
    for i in range(n):
        orig = word[i]
        for v in range(q):
            if v == orig:
                continue
            w = list(word)
            w[i] = v
            idx.append(tuple(w))
    return idx


def _build_index_map(q, n):
    """Map each word tuple to its integer index 0..q^n-1."""
    m = {}
    idx = 0
    for t in itertools.product(range(q), repeat=n):
        m[t] = idx
        idx += 1
    return m


def _word_to_idx(word, q, n):
    idx = 0
    for v in word:
        idx = idx * q + v
    return idx


def _idx_to_word(idx, q, n):
    w = [0] * n
    for i in range(n - 1, -1, -1):
        w[i] = idx % q
        idx //= q
    return tuple(w)


# ---------- initial construction ----------

def _initial_construction(q, n, radius):
    """Direct-sum of ternary tetracode with all tails. Returns list of codeword tuples."""
    if n < 4:
        # For very short codes, just use all words (trivially valid)
        return list(itertools.product(range(q), repeat=n))
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    result = []
    for h in tetracode:
        for tail in itertools.product(range(q), repeat=n - 4):
            result.append(tuple(h + list(tail)))
    return result


# ---------- coverage bookkeeping ----------

def _compute_coverage(codewords, q, n, radius):
    """Return a list cover_count where cover_count[i] = number of codewords within dist<=radius of word i."""
    total = q ** n
    cover_count = [0] * total
    for cw in codewords:
        for w in _ball_indices(cw, q, n, radius):
            cover_count[_word_to_idx(w, q, n)] += 1
    return cover_count


def _is_covered(cover_count, q, n, radius):
    """Check if every word is covered (cover_count[i] >= 1 for all i)."""
    return all(c >= 1 for c in cover_count)


# ---------- local search moves ----------

def _try_remove(codewords, cover_count, q, n, radius, idx_map):
    """Try to remove a redundant codeword. Returns (new_codewords, new_cover_count, removed) or None."""
    total = q ** n
    # Find codewords whose removal keeps coverage intact
    # A codeword is redundant if for every word in its ball, cover_count[word] >= 2
    # (since removing this codeword reduces each by 1)
    for i, cw in enumerate(codewords):
        ball = _ball_indices(cw, q, n, radius)
        redundant = True
        for w in ball:
            wi = idx_map[w]
            if cover_count[wi] <= 1:
                redundant = False
                break
        if redundant:
            # Remove it
            new_cover_count = cover_count[:]
            for w in ball:
                wi = idx_map[w]
                new_cover_count[wi] -= 1
            new_codewords = codewords[:i] + codewords[i+1:]
            return new_codewords, new_cover_count, cw
    return None


def _try_add(codewords, cover_count, q, n, radius, idx_map, deadline):
    """Try to add a codeword that covers the most uncovered words. Returns (new_codewords, new_cover_count) or None."""
    total = q ** n
    # Find uncovered words
    uncovered = [i for i in range(total) if cover_count[i] == 0]
    if not uncovered:
        return None
    
    # Find the word with the most uncovered neighbors (i.e., a candidate codeword
    # that would cover the most currently-uncovered words)
    best_cw = None
    best_gain = 0
    # Sample candidate positions: uncovered words themselves and their neighbors
    candidates = set()
    for ui in uncovered:
        w = _idx_to_word(ui, q, n)
        candidates.add(w)
        for nb in _ball_indices(w, q, n, radius):
            candidates.add(nb)
    
    # Limit candidates for speed
    if len(candidates) > 200:
        candidates = set(random.sample(list(candidates), 200))
    
    for cw in candidates:
        gain = 0
        for w in _ball_indices(cw, q, n, radius):
            wi = idx_map[w]
            if cover_count[wi] == 0:
                gain += 1
        if gain > best_gain:
            best_gain = gain
            best_cw = cw
        if time.time() > deadline:
            break
    
    if best_cw is not None and best_gain > 0:
        new_cover_count = cover_count[:]
        for w in _ball_indices(best_cw, q, n, radius):
            wi = idx_map[w]
            new_cover_count[wi] += 1
        new_codewords = codewords + [best_cw]
        return new_codewords, new_cover_count
    return None


def _try_swap(codewords, cover_count, q, n, radius, idx_map, deadline):
    """Try to swap a codeword with another that covers more uncovered words. Returns (new_codewords, new_cover_count) or None."""
    total = q ** n
    uncovered = [i for i in range(total) if cover_count[i] == 0]
    if not uncovered:
        return None
    
    # For each existing codeword, check if replacing it with a better one helps
    # We look for codewords that are "wasteful" (cover few uncovered words)
    best_swap = None
    best_improvement = 0
    
    # Sample a subset of codewords to consider for removal
    n_sample = min(len(codewords), 20)
    sample_idx = random.sample(range(len(codewords)), n_sample)
    
    for si in sample_idx:
        cw = codewords[si]
        # Compute how many uncovered words this codeword covers
        covered_uncovered = 0
        for w in _ball_indices(cw, q, n, radius):
            wi = idx_map[w]
            if cover_count[wi] == 1:  # only covered by this codeword
                covered_uncovered += 1
        # If we remove it, these become uncovered (if not covered by others)
        # Actually, we need: words where cover_count == 1 (only this cw covers them)
        # Those would become uncovered if we remove cw.
        # We want a replacement that covers those words plus more.
        
        # Find candidate replacement: a word that covers many of the "critical" words
        critical_words = []
        for w in _ball_indices(cw, q, n, radius):
            wi = idx_map[w]
            if cover_count[wi] == 1:
                critical_words.append(w)
        
        if not critical_words:
            continue
        
        # Try replacing with a word near the critical words
        candidates = set()
        for cw_word in critical_words[:5]:
            candidates.add(cw_word)
            for nb in _ball_indices(cw_word, q, n, radius):
                candidates.add(nb)
        
        if len(candidates) > 50:
            candidates = set(random.sample(list(candidates), 50))
        
        for new_cw in candidates:
            if new_cw == cw:
                continue
            # Simulate: remove cw, add new_cw
            # Check if all words remain covered
            ok = True
            # Words that were only covered by cw
            for w in _ball_indices(cw, q, n, radius):
                wi = idx_map[w]
                if cover_count[wi] == 1:
                    # This word would lose its only cover
                    # Check if new_cw covers it
                    new_ball = _ball_indices(new_cw, q, n, radius)
                    if w not in new_ball:
                        ok = False
                        break
            if ok:
                # This swap is valid; it doesn't increase size but might help future removals
                # For now, we only do swaps that maintain size
                pass
    
    return None


# ---------- main search ----------

def _local_search(codewords, q, n, radius, deadline):
    """Iterative local search: remove redundant, add to cover gaps, repeat."""
    idx_map = _build_index_map(q, n)
    total = q ** n
    cover_count = _compute_coverage(codewords, q, n, radius)
    
    best_size = len(codewords)
    best_codewords = codewords[:]
    
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() < deadline:
        improved = False
        
        # Phase 1: Try to remove redundant codewords (repeatedly)
        while time.time() < deadline:
            result = _try_remove(codewords, cover_count, q, n, radius, idx_map)
            if result is None:
                break
            codewords, cover_count, _ = result
            improved = True
            if len(codewords) < best_size:
                best_size = len(codewords)
                best_codewords = codewords[:]
        
        # Phase 2: If not fully covered, add codewords
        if not _is_covered(cover_count, q, n, radius):
            while time.time() < deadline and not _is_covered(cover_count, q, n, radius):
                result = _try_add(codewords, cover_count, q, n, radius, idx_map, deadline)
                if result is None:
                    break
                codewords, cover_count = result
                improved = True
        
        # Phase 3: After adding, try removing again (the added codeword might make others redundant)
        if improved:
            while time.time() < deadline:
                result = _try_remove(codewords, cover_count, q, n, radius, idx_map)
                if result is None:
                    break
                codewords, cover_count, _ = result
                improved = True
                if len(codewords) < best_size:
                    best_size = len(codewords)
                    best_codewords = codewords[:]
        
        if not improved:
            no_improve_count += 1
            if no_improve_count >= max_no_improve:
                break
        else:
            no_improve_count = 0
    
    # Return the best solution found
    return best_codewords


# ---------- entry point ----------

def construct(q, n, radius):
    """Construct a covering code for {0..q-1}^n with covering radius.
    
    Returns a list of codewords, each a list of n integers in 0..q-1.
    """
    start_time = time.time()
    deadline = start_time + 110  # 10-second safety margin before 120s limit
    
    # Step 1: Initial construction
    codewords = _initial_construction(q, n, radius)
    
    # Step 2: Local search
    codewords = _local_search(codewords, q, n, radius, deadline)
    
    # Step 3: Final verification and cleanup
    # Ensure all codewords are valid and distinct
    seen = set()
    valid_codewords = []
    for cw in codewords:
        if cw not in seen:
            seen.add(cw)
            valid_codewords.append(cw)
    
    # Convert to lists of ints
    result = [list(cw) for cw in valid_codewords]
    
    return result
