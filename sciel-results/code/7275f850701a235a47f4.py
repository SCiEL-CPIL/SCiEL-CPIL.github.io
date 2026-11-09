"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import numpy as np


def construct(q, n, radius):
    """
    Greedy set cover with vectorized incidence matrix and 1-exchange local search
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
    cover_sizes = np.array([len(c) for c in covers], dtype=np.int32)

    # Build incidence matrix: M[j, w] = 1 if word w is in covers[j]
    M = np.zeros((N, N), dtype=np.bool_)
    for j in range(N):
        M[j, covers_np[j]] = True

    # Phase 1: Greedy set cover using vectorized scoring
    covered = np.zeros(N, dtype=bool)
    selected_indices = []
    selected_set = set()
    active = np.ones(N, dtype=bool)

    while not covered.all():
        if time.time() > deadline - 15.0:
            break

        # Score for each active candidate: number of uncovered words it covers
        uncovered_mask = ~covered
        scores = M @ uncovered_mask.astype(np.int32)
        scores[~active] = -1

        best_idx = int(np.argmax(scores))
        if scores[best_idx] <= 0:
            break

        selected_indices.append(best_idx)
        selected_set.add(best_idx)
        active[best_idx] = False
        covered[covers_np[best_idx]] = True

    # Phase 2: 1-exchange local search (reduce + swap)
    # First, try to remove redundant codewords
    def try_remove(c_idx, current_selected):
        """Check if codeword c_idx can be removed while maintaining coverage."""
        cover_words = covers_np[c_idx]
        sel_arr = np.array([s for s in current_selected if s != c_idx], dtype=np.int32)
        if len(sel_arr) == 0:
            return False
        sub = M[np.ix_(sel_arr, cover_words)]
        col_sums = sub.sum(axis=0)
        return bool(np.all(col_sums >= 1))

    # Iterative removal
    changed = True
    while changed and time.time() < deadline - 10.0:
        changed = False
        for c_idx in list(selected_indices):
            if time.time() > deadline - 10.0:
                break
            if try_remove(c_idx, selected_indices):
                selected_indices.remove(c_idx)
                selected_set.discard(c_idx)
                changed = True

    # Phase 3: 1-exchange swap search
    # For each selected codeword, try to swap it with an unselected codeword
    # if the swap reduces the total number of codewords or maintains coverage with fewer
    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        current_covered = np.zeros(N, dtype=bool)
        for c_idx in selected_indices:
            current_covered[covers_np[c_idx]] = True

        # Find critical words: words covered by exactly one selected codeword
        coverage_count = np.zeros(N, dtype=np.int32)
        for c_idx in selected_indices:
            coverage_count[covers_np[c_idx]] += 1
        critical_words = np.where(coverage_count == 1)[0]

        if len(critical_words) == 0:
            break

        # For each selected codeword, try to swap with an unselected one
        for c_idx in list(selected_indices):
            if time.time() > deadline - 5.0:
                break

            # Words that would become uncovered if we remove c_idx
            would_uncover = covers_np[c_idx] & (coverage_count == 1)
            if len(would_uncover) == 0:
                continue

            # Find unselected codewords that cover all words in would_uncover
            # Score: number of words in would_uncover that the candidate covers
            candidates = np.where(active)[0]
            if len(candidates) == 0:
                break

            # For each candidate, check if it covers all would_uncover words
            # M[candidates, would_uncover] should be all True
            sub = M[np.ix_(candidates, would_uncover)]
            covers_all = np.all(sub, axis=1)
            good_candidates = candidates[covers_all]

            if len(good_candidates) > 0:
                # Pick the one that covers the most additional uncovered words
                # (to potentially enable future removals)
                additional_cover = M[good_candidates] @ (~current_covered).astype(np.int32)
                best_swap = good_candidates[np.argmax(additional_cover)]

                # Perform the swap
                selected_indices.remove(c_idx)
                selected_set.discard(c_idx)
                selected_indices.append(best_swap)
                selected_set.add(best_swap)
                active[c_idx] = True
                active[best_swap] = False
                improved = True
                break

    # Phase 4: Final greedy if still not covered
    if not covered.all():
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

    return [all_words[idx].tolist() for idx in selected_indices]
# EVOLVE-BLOCK-END
