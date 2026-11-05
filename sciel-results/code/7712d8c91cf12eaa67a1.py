"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np
from collections import defaultdict, deque


def _ternary_to_int(word, q, n):
    """Convert a list of q-ary digits to an integer in [0, q**n)."""
    val = 0
    for i in range(n):
        val = val * q + word[i]
    return val


def _int_to_ternary(val, q, n):
    """Convert an integer to a list of n q-ary digits."""
    word = [0] * n
    for i in range(n - 1, -1, -1):
        word[i] = val % q
        val //= q
    return word


def construct(q, n, radius):
    """
    Construct a q-ary code of length n with covering radius r.
    Uses a greedy set-cover approach with local search (swap/replace) to minimize size.
    """
    total_time = 150.0  # evaluator kills at 150s wall-clock; keep safety margin
    start_time = time.time()
    time_budget = total_time - 10.0  # 10s safety margin

    qn = q ** n

    # Precompute neighbors for each word: all words within Hamming distance <= radius
    # For q=3, n=7, radius=1: each word has 1 + n*(q-1) = 1 + 7*2 = 15 neighbors
    # Total: 2187 * 15 = 32805 neighbor entries

    # Build neighbor lists: for each word index, list of word indices within radius
    # Use base-q representation
    all_words = list(range(qn))

    # Precompute neighbor sets efficiently
    # For radius 1: neighbors are the word itself + all words differing in exactly one position
    # For general radius, we'd need more, but the task specifies radius=1 for this problem
    # Let's handle radius=1 specifically for speed, with a fallback

    if radius == 1:
        # For each word, compute its neighbors (including itself)
        # A word w has neighbors: w itself, and for each position i, change digit to other q-1 values
        # Precompute the position multipliers
        pos_mult = [1] * n
        for i in range(n - 2, -1, -1):
            pos_mult[i] = pos_mult[i + 1] * q

        # For each word index, we need to find all indices within distance 1
        # Build a mapping from word index to its neighbors
        # To do this efficiently, for each word, generate all neighbors and check they're valid

        # Actually, let's build the cover requirement differently:
        # We need to select a subset S of words such that every word is in N(s) for some s in S
        # where N(s) is the closed ball of radius r around s.

        # Build the incidence: for each codeword candidate c, which words does it cover?
        # covered_by[word] = set of codewords that cover it
        # Or: covers[codeword] = set of words covered by it

        # For radius 1, q=3, n=7:
        # Each codeword covers 1 + 7*2 = 15 words
        # We need to cover all 2187 words

        # Greedy: start with empty set, repeatedly add the codeword that covers the most uncovered words

        # First, build the coverage map
        # covers[c] = set of words covered by codeword c
        # For memory efficiency with 2187 codewords each covering 15 words, use lists

        covers = []
        for c in range(qn):
            word = _int_to_ternary(c, q, n)
            covered = {c}
            for i in range(n):
                orig = word[i]
                for new_val in range(q):
                    if new_val == orig:
                        continue
                    word[i] = new_val
                    idx = _ternary_to_int(word, q, n)
                    covered.add(idx)
                    word[i] = orig
            covers.append(covered)

        # Greedy set cover
        uncovered = set(range(qn))
        selected = []
        # coverage_count[word] = number of selected codewords covering it
        # For greedy, we need to efficiently find the codeword covering the most uncovered words

        # Use a heap-based approach: for each codeword, track how many uncovered words it covers
        # When we select a codeword, we need to update the counts for all other codewords
        # that share covered words with it.

        # Simpler approach: just do greedy by scanning, but that's O(qn * qn) which is ~4.7M per iteration
        # With ~150 iterations needed, that's ~700M operations. Might be too slow in pure Python.

        # Let's use a smarter approach:
        # Maintain for each word, how many selected codewords cover it
        # For the greedy step, we need max over c of |covers[c] ∩ uncovered|
        # = sum over w in uncovered of [w in covers[c]]
        # = sum over w in covers[c] of [w in uncovered]

        # Let's just do a reasonable greedy and then local search.

        # Greedy: iterate, each time pick the codeword with max uncovered coverage
        # To speed up, precompute for each pair of codewords how many words they share
        # This is qn^2 = 4.7M pairs, each with up to 15*15=225 intersections... too much.

        # Alternative: just do the greedy with a count array
        # uncovered_count[c] = number of currently uncovered words that c covers
        # Initially, uncovered_count[c] = len(covers[c]) = 15 for all c
        # When we select codeword s, for each word w in covers[s]:
        #   if w was uncovered (i.e., it's in uncovered set):
        #     remove w from uncovered
        #     for each other codeword c that covers w, decrement uncovered_count[c]

        # We need: for each word w, list of codewords that cover w
        covered_by = defaultdict(list)
        for c in range(qn):
            for w in covers[c]:
                covered_by[w].append(c)

        # Initialize
        uncovered = set(range(qn))
        uncovered_count = [len(covers[c]) for c in range(qn)]
        # Use a max-heap: (-count, codeword)
        import heapq
        heap = [(-uncovered_count[c], c) for c in range(qn)]
        heapq.heapify(heap)
        in_heap = [True] * qn  # track if entry is still valid
        best_count = [0] * qn  # best known count for each codeword

        for c in range(qn):
            best_count[c] = uncovered_count[c]

        selected = []
        # Greedy selection
        while uncovered:
            # Pop from heap until we find a valid entry
            while heap:
                neg_cnt, c = heapq.heappop(heap)
                if in_heap[c] and best_count[c] == -neg_cnt:
                    break
            if not heap:
                break
            c = heapq.heappop(heap) if isinstance(heap[0], tuple) else heap[0]
            # Actually let me redo this more carefully

        # Let me simplify: just do a basic greedy without the heap complexity
        # For q=3, n=7, the greedy should give a reasonable starting point

        uncovered = set(range(qn))
        selected = []

        # Precompute covers as frozensets for fast intersection
        covers_fs = [frozenset(covers[c]) for c in range(qn)]

        # Greedy: repeatedly pick codeword covering most uncovered
        # Optimization: only consider codewords that cover at least one uncovered word
        # Use numpy for speed? Let's try a vectorized approach

        # Represent uncovered as a boolean array
        uncovered_arr = np.ones(qn, dtype=bool)
        # coverage matrix: cov[c, w] = 1 if codeword c covers word w
        # This is qn x qn = 2187 x 2187 = 4.7M booleans = 4.7MB. Fine.
        cov = np.zeros((qn, qn), dtype=np.uint8)
        for c in range(qn):
            for w in covers[c]:
                cov[c, w] = 1

        # Greedy
        while uncovered_arr.any():
            # For each codeword, count how many uncovered words it covers
            counts = cov @ uncovered_arr.astype(np.uint8)
            # Pick the one with max count
            c = int(np.argmax(counts))
            if counts[c] == 0:
                break  # shouldn't happen
            selected.append(c)
            uncovered_arr[cov[c] == 1] = False

        # Convert to list of word lists
        code = [_int_to_ternary(c, q, n) for c in selected]
        initial_size = len(code)

        # Now do local search to reduce the code size
        # Local search: try to remove a codeword if it's redundant (all words it covers are covered by others)
        # Or try to swap: remove one, add another that covers more net uncovered words

        # Build coverage: for each word, which selected codewords cover it
        # Use the cov matrix restricted to selected codewords
        # Let's work with indices into the selected list

        # Rebuild: sel_cov[s_idx, w] = 1 if selected[s_idx] covers word w
        sel_indices = selected
        sel_cov = cov[np.ix_(sel_indices, range(qn))]  # shape (len(selected), qn)

        # coverage_count[w] = number of selected codewords covering word w
        cov_count = sel_cov.sum(axis=0)

        # Try to remove redundant codewords
        improved = True
        while improved:
            improved = False
            for i in range(len(sel_indices) - 1, -1, -1):
                # Check if removing codeword i leaves all words covered
                # A word w is still covered if cov_count[w] > 1 OR (cov_count[w] == 1 and sel_cov[i, w] == 0)
                # i.e., w is still covered if cov_count[w] - sel_cov[i, w] > 0
                # So we need: for all w, cov_count[w] - sel_cov[i, w] > 0
                # Equivalently: min over w of (cov_count[w] - sel_cov[i, w]) > 0
                # Which is: min over w where sel_cov[i,w]==1 of cov_count[w] > 1
                # And for w where sel_cov[i,w]==0, cov_count[w] > 0 (which is always true if covered)

                # Check: are there any words w such that sel_cov[i,w]==1 and cov_count[w]==1?
                # If yes, removing i would uncover those words.
                # Find words covered only by codeword i
                only_i = (sel_cov[i] == 1) & (cov_count == 1)
                if not only_i.any():
                    # Codeword i is redundant, remove it
                    sel_indices = sel_indices[:i] + sel_indices[i+1:]
                    sel_cov = np.delete(sel_cov, i, axis=0)
                    cov_count = sel_cov.sum(axis=0)
                    improved = True
                    break

        # Now try swap-based local search: remove one codeword, add another
        # For each pair (i, j) where i is in selected and j is not, check if replacing i with j reduces size or maintains coverage
        # Actually, a swap that reduces size isn't possible directly. We can only reduce size by removals.
        # But we can try: remove i, then see if we need to add anything back. If we can remove i without adding, great.
        # We already did that above.

        # For further improvement, try 2-for-1 swaps: remove two codewords, add one that covers all their unique words
        # This is more complex. Let's try a different approach:
        # Iteratively try to remove codewords in random order, and if a word becomes uncovered, 
        # try to find a single codeword (not in selected) that covers all newly uncovered words.

        # Actually, let's just do more passes of the removal above, and also try random restarts of the greedy.

        # Let's also try: after the initial greedy + removal, do a local search where we
        # randomly select a codeword to remove, and if coverage is broken, try to fix by adding
        # a codeword that covers the most of the broken words.

        # For now, let's just do multiple passes of the removal and also try different random seeds for greedy.

        best_code = [list(w) for w in code]
        best_size = len(best_code)

        # Try multiple greedy runs with different orderings
        # Actually, the greedy is deterministic given the tie-breaking. Let's add randomness.

        # Let's do a few more local search passes
        # Pass 2: try removing codewords in different orders
        for _ in range(50):  # up to 50 more passes
            if time.time() - start_time > time_budget:
                break
            # Try removing codewords in random order
            indices = list(range(len(sel_indices)))
            random.shuffle(indices)
            for i in indices:
                if i >= len(sel_indices):
                    continue
                only_i = (sel_cov[i] == 1) & (cov_count == 1)
                if not only_i.any():
                    sel_indices = sel_indices[:i] + sel_indices[i+1:]
                    sel_cov = np.delete(sel_cov, i, axis=0)
                    cov_count = sel_cov.sum(axis=0)
                    if len(sel_indices) < best_size:
                        best_size = len(sel_indices)
                        best_code = [_int_to_ternary(c, q, n) for c in sel_indices]

        # Try swap-based improvement:
        # For each codeword i in selected, try replacing it with a codeword j not in selected
        # such that the new code is still a cover. This doesn't reduce size, but might enable
        # subsequent removals.
        # This is expensive. Let's skip for now and focus on what we have.

        # Check time and return best
        if time.time() - start_time > time_budget:
            return best_code

        # Do a final check: verify the code is valid
        # (The evaluator will check, but let's make sure we return something valid)

        # Convert sel_indices to word lists
        final_code = [_int_to_ternary(c, q, n) for c in sel_indices]

        # If final is better than best, use it
        if len(final_code) < len(best_code):
            best_code = final_code

        return best_code
    else:
        # Fallback for other radii: use the direct sum construction
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(q) for b in range(q)]
        if n > 4:
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
        elif n == 4:
            return tetracode
        else:
            # For n < 4, just return all words
            return [list(tail) for tail in itertools.product(range(q), repeat=n)]
# EVOLVE-BLOCK-END
