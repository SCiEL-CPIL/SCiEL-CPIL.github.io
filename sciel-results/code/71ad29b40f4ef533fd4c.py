"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import numpy as np


def construct(q, n, radius):
    """
    Greedy set cover with vectorized incidence matrix and greedy removal
    for ternary covering code K_3(8,1).
    Returns a list of codewords (each a list of n ints in 0..q-1).
    """
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s kill

    # Generate all words in {0,1,2}^n
    all_words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.uint8)
    N = len(all_words)  # 6561

    # Build word-to-index mapping
    word_to_idx = {}
    for idx, w in enumerate(all_words):
        word_to_idx[tuple(w)] = idx

    # Precompute cover sets: for each word, the indices of words it covers
    covers = []
    for idx in range(N):
        w = all_words[idx]
        cover_set = [idx]
        for i in range(n):
            for alt in range(q):
                if alt == w[i]:
                    continue
                nw = list(w)
                nw[i] = alt
                cover_set.append(word_to_idx[tuple(nw)])
        covers.append(cover_set)

    covers_np = [np.array(c, dtype=np.int32) for c in covers]

    # Build incidence matrix: M[j, w] = 1 if word w is in covers[j]
    # Shape: (N, N) - 6561 x 6561, ~43M entries, ~170MB as float32 or 43MB as bool
    M = np.zeros((N, N), dtype=np.bool_)
    for j in range(N):
        M[j, covers_np[j]] = True

    # Phase 1: Greedy set cover using vectorized scoring
    covered = np.zeros(N, dtype=bool)
    selected_indices = []
    selected_set = set()

    # Track which candidates are still active (not yet selected)
    active = np.ones(N, dtype=bool)

    while not covered.all():
        if time.time() > deadline - 10.0:
            break

        # Score for each active candidate: number of uncovered words it covers
        # score[j] = sum over w of M[j,w] AND (not covered[w])
        # Vectorized: M @ (~covered) gives count of uncovered words per codeword
        uncovered_mask = ~covered  # shape (N,)
        scores = M @ uncovered_mask.astype(np.int32)  # shape (N,)
        scores[~active] = -1  # deactivate already-selected candidates

        best_idx = int(np.argmax(scores))
        if scores[best_idx] <= 0:
            break

        selected_indices.append(best_idx)
        selected_set.add(best_idx)
        active[best_idx] = False

        # Update covered
        covered[covers_np[best_idx]] = True

    # Phase 2: Greedy removal - try to remove codewords that are redundant
    # A codeword c is redundant if every word it covers is also covered by another selected codeword
    # We iterate: for each selected codeword, check if removing it still leaves all words covered
    # If yes, remove it. Repeat until no more removals possible.

    def is_redundant(c_idx, current_selected):
        """Check if codeword c_idx is redundant given the set current_selected (which includes c_idx)."""
        # For each word w in covers[c_idx], check if there's another selected codeword that covers w
        # Vectorized: for each w in covers[c_idx], check if any other selected codeword covers w
        cover_words = covers_np[c_idx]
        # For each word w, count how many selected codewords (other than c_idx) cover w
        # M[selected, w] for all selected, then check if any > 1 (including c_idx itself)
        # Actually: if for all w in cover_words, the count of selected codewords covering w is >= 2,
        # then c_idx is redundant.
        # counts[w] = sum over s in selected of M[s, w]
        # c_idx is redundant if min(counts[cover_words]) >= 2
        
        sel_arr = np.array([s for s in current_selected if s != c_idx], dtype=np.int32)
        if len(sel_arr) == 0:
            return False
        # For each word in cover_words, count how many of sel_arr cover it
        # M[sel_arr, cover_words] is a (len(sel_arr), len(cover_words)) matrix
        # We want: for each column (word), is the sum >= 1?
        sub = M[np.ix_(sel_arr, cover_words)]  # (len(sel_arr), len(cover_words))
        col_sums = sub.sum(axis=0)  # (len(cover_words),)
        return bool(np.all(col_sums >= 1))

    # Iterate removal
    changed = True
    while changed and time.time() < deadline - 5.0:
        changed = False
        for c_idx in list(selected_indices):
            if time.time() > deadline - 5.0:
                break
            if is_redundant(c_idx, selected_indices):
                selected_indices.remove(c_idx)
                selected_set.discard(c_idx)
                changed = True

    # Phase 3: If we still haven't covered everything (shouldn't happen), do more greedy
    if not covered.all():
        # Recompute covered from current selected
        covered = np.zeros(N, dtype=bool)
        for c_idx in selected_indices:
            covered[covers_np[c_idx]] = True
        
        if not covered.all():
            active = np.ones(N, dtype=bool)
            for c_idx in selected_indices:
                active[c_idx] = False
            
            while not covered.all() and time.time() < deadline:
                uncovered_mask = ~covered
                scores = M @ uncovered_mask.astype(np.int32)
                scores[~active] = -1
                best_idx = int(np.argmax(scores))
                if scores[best_idx] <= 0:
                    break
                selected_indices.append(best_idx)
                active[best_idx] = False
                covered[covers_np[best_idx]] = True

    # Final safety: if still not covered, use direct sum construction
    if not covered.all():
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        fallback = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        return fallback

    # Convert to list of lists
    return [all_words[idx].tolist() for idx in selected_indices]
# EVOLVE-BLOCK-END
