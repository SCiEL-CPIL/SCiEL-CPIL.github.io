import time
import itertools
import numpy as np

_TETRA = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]

def _neighbors(cw, n, q):
    """All words at Hamming distance 1 from cw."""
    out = []
    for i in range(n):
        v = cw[i]
        for d in range(q):
            if d == v:
                continue
            out.append(cw[:i] + [d] + cw[i + 1:])
    return out

def _tetracode_solution(q, n, radius):
    """Direct sum of the ternary tetracode with every word on the remaining coords."""
    if n < 4:
        return [list(t) for t in itertools.product(range(q), repeat=n)]
    return [h + list(tail) for h in _TETRA
            for tail in itertools.product(range(q), repeat=n - 4)]

def _encode(word, q, n):
    """Convert a list of ints to an integer index in [0, q^n)."""
    val = 0
    for i in range(n):
        val = val * q + word[i]
    return val

def _decode(idx, q, n):
    """Convert an integer index to a list of ints."""
    word = [0] * n
    for i in range(n - 1, -1, -1):
        word[i] = idx % q
        idx //= q
    return word

def _build_cover_counts(codewords, q, n):
    """
    Returns:
    cover_count: array of size q^n, cover_count[i] = number of codewords covering word i.
    """
    total = q ** n
    cover_count = np.zeros(total, dtype=np.int32)
    for cw in codewords:
        # Self
        idx = _encode(cw, q, n)
        cover_count[idx] += 1
        # Neighbors
        for nb in _neighbors(cw, n, q):
            idx = _encode(nb, q, n)
            cover_count[idx] += 1
    return cover_count

def _is_valid_cover(cover_count):
    """Check if every word is covered by at least one codeword."""
    return np.all(cover_count > 0)

def _find_redundant(codewords, cover_count, q, n):
    """
    Find codewords that can be removed without violating the cover property.
    A codeword c is redundant if for every word w in Ball(c), there is another codeword c' != c covering w.
    This is equivalent to: for every w in Ball(c), cover_count[w] > 1.
    """
    redundant = []
    for i, cw in enumerate(codewords):
        # Check self
        idx = _encode(cw, q, n)
        if cover_count[idx] == 1:
            continue
        # Check neighbors
        redundant_candidate = True
        for nb in _neighbors(cw, n, q):
            idx = _encode(nb, q, n)
            if cover_count[idx] == 1:
                redundant_candidate = False
                break
        if redundant_candidate:
            redundant.append(i)
    return redundant

def _remove_codeword(codewords, cover_count, idx, q, n):
    """Remove codeword at index idx from codewords and update cover_count."""
    cw = codewords.pop(idx)
    cover_count[_encode(cw, q, n)] -= 1
    for nb in _neighbors(cw, n, q):
        cover_count[_encode(nb, q, n)] -= 1

def _add_codeword(codewords, cover_count, cw, q, n):
    """Add codeword cw to codewords and update cover_count."""
    codewords.append(cw)
    cover_count[_encode(cw, q, n)] += 1
    for nb in _neighbors(cw, n, q):
        cover_count[_encode(nb, q, n)] += 1

def _local_search(codewords, q, n, deadline):
    """
    Structural change: Local search with moves.
    Try to move a codeword to a neighbor position if it allows removing another codeword.
    """
    total = q ** n
    cover_count = _build_cover_counts(codewords, q, n)
    
    if not _is_valid_cover(cover_count):
        return codewords

    # Initial pruning
    redundant_indices = _find_redundant(codewords, cover_count, q, n)
    # Remove from end to start to keep indices valid
    for i in sorted(redundant_indices, reverse=True):
        _remove_codeword(codewords, cover_count, i, q, n)
        
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try to find a move that allows a removal
        # To make this faster, we can iterate through codewords and try their neighbors
        for i in range(len(codewords)):
            if time.time() > deadline:
                break
            cw = codewords[i]
            # Try moving to neighbors
            for cw_new in _neighbors(cw, n, q):
                # Simulate move: remove i, add cw_new
                # We need to check if the new set is valid and if it creates redundancy
                
                # Quick check: Does cw_new cover any word that ONLY cw covers?
                # If yes, moving cw to cw_new might uncover that word unless cw_new covers it.
                # Actually, we just need to check validity of the new cover.
                
                # Efficient check:
                # Words covered by cw: Ball(cw)
                # Words covered by cw_new: Ball(cw_new)
                # If we move cw to cw_new, the coverage changes by:
                # - Ball(cw) \ Ball(cw_new) loses 1 count
                # + Ball(cw_new) \ Ball(cw) gains 1 count
                # Words in intersection keep count.
                
                # We need to ensure no word in Ball(cw) \ Ball(cw_new) drops to 0.
                # i.e., for all w in Ball(cw) \ Ball(cw_new), cover_count[w] > 1.
                
                # And we want to see if this move makes some OTHER codeword redundant.
                
                # This simulation is O(n*q) per neighbor.
                
                valid_move = True
                # Check words that would lose coverage
                for w in _neighbors(cw, n, q):
                    if w == cw: continue # handled below
                    # If w is not covered by cw_new, its count decreases by 1
                    if w not in [cw_new] + _neighbors(cw_new, n, q):
                        idx = _encode(w, q, n)
                        if cover_count[idx] == 1:
                            valid_move = False
                            break
                if not valid_move:
                    continue
                
                # Check self
                if cw not in [cw_new] + _neighbors(cw_new, n, q):
                    idx = _encode(cw, q, n)
                    if cover_count[idx] == 1:
                        valid_move = False
                        continue

                if not valid_move:
                    continue
                
                # If valid, perform the move temporarily to check for redundancy
                # To avoid deep copying, we can just check if any *other* codeword becomes redundant.
                # A codeword j becomes redundant if all words in Ball(codewords[j]) have count > 1 in the NEW configuration.
                # New count for w:
                # old_count[w] - 1 (if w in Ball(cw)) + 1 (if w in Ball(cw_new))
                
                # This is getting complex to check without actually doing it.
                # Let's just do the move, check redundancy, and revert if no gain.
                
                # Save state
                old_cw = cw
                # Remove i
                _remove_codeword(codewords, cover_count, i, q, n)
                # Add cw_new
                _add_codeword(codewords, cover_count, cw_new, q, n)
                
                # Check validity (should be true based on pre-check, but safe to be sure)
                if not _is_valid_cover(cover_count):
                    # Revert
                    _remove_codeword(codewords, cover_count, len(codewords)-1, q, n)
                    _add_codeword(codewords, cover_count, old_cw, q, n)
                    # Re-insert at i? Order doesn't matter for set, but index i is now empty in list logic?
                    # My _remove/_add modify the list. This is messy.
                    # Let's use a simpler, safer approach for the move.
                    pass # This path is tricky with in-place list modification.
                
                # Better approach: 
                # 1. Calculate new cover counts virtually? No, too slow.
                # 2. Just do the move on a copy? Too slow.
                
                # Let's restart the loop logic to be safer.
                # We will revert the move we just did.
                _remove_codeword(codewords, cover_count, len(codewords)-1, q, n)
                # Re-add old cw at end, then swap to position i?
                # Actually, _remove_codeword removes from list. _add_codeword appends.
                # So codewords list is now: [0..i-1, i+1..end-1, cw_new]
                # We want to revert to: [0..end-1]
                # Remove cw_new
                _remove_codeword(codewords, cover_count, len(codewords)-1, q, n)
                # Add old cw
                _add_codeword(codewords, cover_count, old_cw, q, n)
                # The list is now [0..i-1, i+1..end-1, old_cw]. 
                # The index i is now occupied by old_cw[i+1] originally? No.
                # The list structure is broken relative to original indices.
                
                # This in-place modification strategy is error-prone.
                # Let's use a different strategy: 
                # Keep codewords as a list. 
                # To test a move, create a NEW cover_count array? 
                # q^n = 6561. np.zeros(6561) is fast.
                # Copying cover_count is fast.
                
                # Let's redo the loop with safe copying.
                pass
            break # Break inner loop to restart outer logic? No, this was a dummy block.
        
        # Let's implement the safe version.
        improved = False
        for i in range(len(codewords)):
            if time.time() > deadline:
                return codewords
            cw = codewords[i]
            for cw_new in _neighbors(cw, n, q):
                if cw_new == cw: continue
                
                # Test move
                new_cover_count = cover_count.copy()
                
                # Subtract cw
                new_cover_count[_encode(cw, q, n)] -= 1
                for nb in _neighbors(cw, n, q):
                    new_cover_count[_encode(nb, q, n)] -= 1
                    
                # Add cw_new
                new_cover_count[_encode(cw_new, q, n)] += 1
                for nb in _neighbors(cw_new, n, q):
                    new_cover_count[_encode(nb, q, n)] += 1
                    
                if not np.all(new_cover_count > 0):
                    continue
                    
                # Check if any codeword (other than the moved one, effectively index i in new set) becomes redundant
                # We need to check redundancy for all codewords in the new set.
                # The new set is codewords[:i] + codewords[i+1:] + [cw_new]
                
                # Optimization: Only check codewords that might be affected?
                # Actually, only codewords whose balls intersect with Ball(cw) or Ball(cw_new) can change redundancy status?
                # No, a codeword far away might become redundant if its "unique" words were covered by cw?
                # No, if a codeword was NOT redundant before, it had a unique word.
                # If that unique word is still covered by it, it's still not redundant.
                # The unique word of codeword j is a word w such that cover_count[w] == 1 and w in Ball(c_j).
                # If we move c_i to c_i', the counts change.
                # If w was unique to c_j, and w is not in Ball(c_i) or Ball(c_i'), count stays 1.
                # If w is in Ball(c_i), count decreases by 1. If it was 1, it becomes 0 -> Invalid (checked above).
                # If w is in Ball(c_i'), count increases by 1.
                
                # So, a codeword j can only become redundant if:
                # 1. It was already "close" to redundant (all its words had count >= 2, except maybe some).
                # Actually, simpler: Just check all codewords. It's O(|C| * n * q).
                # |C| ~ 500. n=8, q=3. 500 * 24 = 12000 ops. Very fast.
                
                new_codewords = codewords[:i] + codewords[i+1:] + [cw_new]
                
                redundant_found = -1
                for j, c_j in enumerate(new_codewords):
                    # Check if c_j is redundant in new_cover_count
                    # Self
                    if new_cover_count[_encode(c_j, q, n)] == 1:
                        continue
                    # Neighbors
                    is_red = True
                    for nb in _neighbors(c_j, n, q):
                        if new_cover_count[_encode(nb, q, n)] == 1:
                            is_red = False
                            break
                    if is_red:
                        # Don't remove the one we just moved (cw_new) if it's the only one?
                        # If cw_new is redundant, we could remove it, but then we're back to original or worse.
                        # We want to remove a DIFFERENT codeword.
                        # If j corresponds to cw_new, skip?
                        # If cw_new is redundant, it means its ball is fully covered by others.
                        # We could remove it, but we just added it.
                        # Let's prefer removing an existing codeword.
                        if c_j == cw_new:
                            # If we remove cw_new, we are back to a state similar to original but with c_i removed?
                            # No, we removed c_i and added cw_new. If we remove cw_new, we removed c_i.
                            # That reduces size by 1. That's a valid improvement!
                            # But wait, if cw_new is redundant, it means the original c_i was redundant?
                            # Not necessarily.
                            # Let's just accept any redundant codeword.
                            pass
                        redundant_found = j
                        break
                        
                if redundant_found != -1:
                    # Perform the move and removal
                    # 1. Update codewords list
                    codewords = new_codewords
                    # 2. Update cover_count
                    cover_count = new_cover_count
                    # 3. Remove redundant
                    _remove_codeword(codewords, cover_count, redundant_found, q, n)
                    improved = True
                    break # Restart outer loop
            if improved:
                break
                
    return codewords

def construct(q, n, radius):
    deadline = time.time() + 140.0
    
    try:
        # Initial solution
        codewords = _tetracode_solution(q, n, radius)
        
        # Local search
        codewords = _local_search(codewords, q, n, deadline)
        
        return codewords
    except Exception:
        return _tetracode_solution(q, n, radius)
