"""Rewritten covering-code search: vectorized greedy + kick/prune local search."""
import time
import itertools
import numpy as np

def construct(q, n, radius):
    start = time.time()
    DEADLINE = start + 145.0

    if q < 2 or n < 1 or radius < 0:
        return [[0] * n]
    if radius >= n:
        return [[0] * n]

    total = q ** n
    # Precompute all words as int array
    words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int8).reshape(total, n)
    words_i = words.astype(np.int32)

    # Precompute distance (<= radius) from every word to every other word:
    # diff matrix shape (total, total, n) is 6561^2*8 ~ 344M entries ~ 1.4GB int8 -> too big.
    # Instead compute per-word neighbor index lists lazily but vectorized via broadcasting
    # in chunks to keep memory bounded.
    # We'll build a boolean coverage matrix incrementally.

    # Coverage tracking: covered[word_idx] is True if within radius of some chosen codeword.
    covered = np.zeros(total, dtype=bool)
    chosen = []  # list of int word indices
    chosen_set = set()

    # For speed, precompute for each position the "distance contribution".
    # We compute neighbors of a word w: words that differ in <= radius positions.
    # Vectorized: given a candidate c (length n), compute Hamming distance to all words:
    def dists_to_all(c):
        # c: length-n int array
        # number of equal positions
        eq = (words_i == c).sum(axis=1)
        return n - eq

    # Greedy: pick codeword maximizing number of newly covered points.
    # To evaluate all candidates at once would be total x total work. We do it in a
    # smarter way: maintain for each word the number of chosen codewords that cover it,
    # and the "marginal gain" of adding candidate c is the number of currently-uncovered
    # words within radius of c.
    #
    # Efficient greedy: precompute, for each word, the list of words within radius
    # (its closed ball). Then marginal gain of c = sum over uncovered in ball(c).
    # We can precompute the balls as index arrays.

    # Build balls: for each word w, indices of all words within radius.
    # Do it in chunks over w to bound memory.
    # ball_indices: list of int arrays (one per word)
    ball_indices = [None] * total
    CH = 256
    for s in range(0, total, CH):
        e = min(s + CH, total)
        cand = words_i[s:e]                      # (CH, n)
        # distance from each cand row to all words: (CH, total)
        # eq positions: (CH, total)
        eq = (cand[:, None, :] == words_i[None, :, :]).sum(axis=2)
        d = n - eq
        within = d <= radius                      # (CH, total) bool
        # for each row, get indices where True
        for i in range(within.shape[0]):
            ball_indices[s + i] = np.nonzero(within[i])[0]

    # uncovered count helper
    # We maintain an array uncovered_mask (bool) and for a candidate c, gain = uncovered_mask[ball(c)].sum()

    uncovered_mask = covered.copy()
    uncovered_mask[:] = True  # start: nothing covered

    def ball_gain(c_idx):
        return uncovered_mask[ball_indices[c_idx]].sum()

    # Initial greedy: pick best candidate, update, repeat until all covered.
    # To speed up, evaluate candidates in a batched fashion:
    # gain for all candidates = for each c, sum of uncovered over ball(c).
    # This is total x (avg ball size). Ball size for q=3,n=8,r=1 is 1 + 8*2 = 17.
    # So total work per full pass = total * 17 ~ 111k, times ~486 picks = ~54M -> fine.

    # We'll do a full scan each pick (simple but ok for ~500 picks).
    while True:
        if time.time() > DEADLINE - 20.0:
            break
        # compute gains for all candidates not already chosen
        gains = np.zeros(total, dtype=np.int32)
        # vectorize: for each c, gain = sum uncovered over ball(c)
        # Do in chunks over c
        for s in range(0, total, 128):
            e = min(s + 128, total)
            for i in range(s, e):
                gains[i] = uncovered_mask[ball_indices[i]].sum()
        # exclude already chosen (gain 0 effectively since covered)
        for c in chosen:
            gains[c] = -1
        best = int(np.argmax(gains))
        if gains[best] <= 0:
            # all covered
            break
        chosen.append(best)
        chosen_set.add(best)
        # update uncovered_mask: mark ball(best) as covered
        uncovered_mask[ball_indices[best]] = False
        if not uncovered_mask.any():
            break

    # Convert to word lists
    def idx_to_word(i):
        return words[i].tolist()

    def word_to_idx(w):
        # w: list of ints
        return int(np.ravel_multi_index(tuple(w), (q,) * n))

    current = chosen[:]

    # Local search: kick (add a random word) then prune (remove any codeword whose
    # removal still leaves everything covered). Repeat.
    def is_fully_covered(codeword_idxs):
        # build covered array
        cov = np.zeros(total, dtype=bool)
        for c in codeword_idxs:
            cov[ball_indices[c]] = True
        return bool(cov.all())

    def prune(codeword_idxs):
        # try to remove each codeword if still covered
        changed = True
        cur = list(codeword_idxs)
        while changed:
            changed = False
            # For each candidate removal, check coverage
            # Build coverage counts: for each word, how many chosen codewords cover it.
            counts = np.zeros(total, dtype=np.int32)
            for c in cur:
                counts[ball_indices[c]] += 1
            # any word with count 0 -> not covered (shouldn't happen if covered)
            # A codeword c can be removed if for all w in ball(c), counts[w] >= 2
            # i.e., min counts over ball(c) >= 2
            for idx, c in enumerate(cur):
                if counts[ball_indices[c]].min() >= 2:
                    # remove
                    for w in ball_indices[c]:
                        counts[w] -= 1
                    cur.pop(idx)
                    changed = True
                    break
        return cur

    # Kick: add a random word (prefer one that covers many uncovered? but we're covered,
    # so kick is to enable a different, smaller cover). Add a random word then prune.
    rng = np.random.default_rng(0)
    best_size = len(current)
    best_solution = current[:]

    # Precompute random candidate pool
    all_idx = np.arange(total)

    kick_count = 0
    while time.time() < DEADLINE - 5.0:
        kick_count += 1
        # Kick: add a random codeword
        cand = int(rng.choice(all_idx))
        trial = current[:] + [cand]
        # Prune to minimal
        trial = prune(trial)
        if len(trial) < best_size:
            best_size = len(trial)
            best_solution = trial[:]
            current = trial[:]
        # Occasionally accept equal-size different solutions to escape
        elif len(trial) == best_size and rng.random() < 0.3:
            current = trial[:]
        if time.time() > DEADLINE - 5.0:
            break

    # Final: ensure best_solution is a valid cover; if not, fall back to current
    result = best_solution if is_fully_covered(best_solution) else current
    # Deduplicate
    seen = set()
    final = []
    for c in result:
        if c not in seen:
            seen.add(c)
            final.append(c)
    return [idx_to_word(c) for c in final]
