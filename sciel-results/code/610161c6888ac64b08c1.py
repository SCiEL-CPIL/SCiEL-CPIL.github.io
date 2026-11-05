"""Model-and-optimize approach for ternary covering code K_3(6,1).

Strategy:
1. Start from the direct-sum construction (81 words) as a baseline.
2. Fast NumPy-vectorized coverage counting for greedy removal.
3. Strong local search: greedy removal, single-swap, 2-exchange, and
   simulated-annealing-style add-then-remove to escape local minima.
4. Aggressive time management with a safety margin before the 150s limit.
"""
import time
import random
import numpy as np


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin: finish by 140s, limit is 150s

    def now():
        return time.time()

    # ---------- Problem setup ----------
    # For q=3, n=6, radius=1 the space is 3^6 = 729 words.
    total_words = q ** n

    def word_to_index(w):
        idx = 0
        for i in range(n):
            idx = idx * q + w[i]
        return idx

    def index_to_word(idx):
        w = [0] * n
        for i in range(n - 1, -1, -1):
            w[i] = idx % q
            idx //= q
        return tuple(w)

    # All words as an array of shape (total_words, n)
    all_words_array = np.empty((total_words, n), dtype=np.int32)
    for idx in range(total_words):
        all_words_array[idx] = index_to_word(idx)

    # Each codeword covers 1 + n*(q-1) words (itself + one-position changes)
    words_per_codeword = 1 + n * (q - 1)

    # Precompute coverage: covered_all[ci, k] = word index covered by codeword ci at slot k
    covered_all = np.zeros((total_words, words_per_codeword), dtype=np.int32)
    for ci in range(total_words):
        c = all_words_array[ci]
        covered_all[ci, 0] = ci
        slot = 1
        for pos in range(n):
            orig = c[pos]
            for val in range(q):
                if val == orig:
                    continue
                w = c.tolist()
                w[pos] = val
                covered_all[ci, slot] = word_to_index(tuple(w))
                slot += 1

    # ---------- Initial code: direct sum of tetracode with tails ----------
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(q) for b in range(q)]
    initial_indices = []
    for h in tetracode:
        base = word_to_index(tuple(h) + (0,) * (n - 4))
        # tails vary the last (n-4) positions
        tail_len = n - 4
        for t in range(q ** tail_len):
            # build tail
            tail = [0] * tail_len
            x = t
            for i in range(tail_len - 1, -1, -1):
                tail[i] = x % q
                x //= q
            full = tuple(h) + tuple(tail)
            initial_indices.append(word_to_index(full))
    initial_indices = np.array(sorted(set(initial_indices)), dtype=np.int32)

    # ---------- Coverage bookkeeping ----------
    # cov_count[wi] = number of active codewords covering word wi
    cov_count = np.zeros(total_words, dtype=np.int32)
    active = np.zeros(total_words, dtype=bool)
    active[initial_indices] = True
    for ci in initial_indices:
        cov_count[covered_all[ci]] += 1

    current_size = int(active.sum())

    # For each word, the list of codewords that cover it (for swap search)
    word_coverers = [[] for _ in range(total_words)]
    for ci in range(total_words):
        for wi in covered_all[ci]:
            word_coverers[wi].append(ci)

    # ---------- Fast helpers ----------
    def is_redundant(ci):
        # ci redundant iff every word it covers has cov_count > 1
        covered = covered_all[ci]
        return bool(np.all(cov_count[covered] > 1))

    def do_remove(ci):
        active[ci] = False
        cov_count[covered_all[ci]] -= 1
        return 1

    def do_add(ci):
        active[ci] = True
        cov_count[covered_all[ci]] += 1
        return 1

    def greedy_remove():
        """Remove redundant codewords until none remain. Returns count removed."""
        nonlocal current_size
        removed = 0
        order = list(range(total_words))
        # shuffle for diversity
        random.shuffle(order)
        for ci in order:
            if now() > deadline:
                break
            if not active[ci]:
                continue
            if is_redundant(ci):
                do_remove(ci)
                removed += 1
                current_size -= 1
        return removed

    # Initial greedy removal
    greedy_remove()

    # Track best solution found
    best_size = current_size
    best_active = active.copy()
    best_cov = cov_count.copy()

    # ---------- Local search ----------
    # We alternate between:
    #  (A) greedy removal
    #  (B) single-swap: remove a non-redundant codeword, add one covering its orphaned words
    #  (C) 2-exchange: remove two, add two (net same or better)
    #  (D) add-then-remove (simulated annealing style): add a random codeword,
    #      then greedily remove as many as possible; keep if net improvement.
    #
    # We keep the best (minimum size) valid configuration seen so far.

    def orphaned_words(ci):
        """Words that would become uncovered if we remove ci (cov_count == 1)."""
        covered = covered_all[ci]
        return covered[cov_count[covered] == 1]

    # Main search loop with a time budget
    no_improve = 0
    max_no_improve = 200000

    while now() < deadline - 5.0 and no_improve < max_no_improve:
        # (A) Greedy removal first
        r = greedy_remove()
        if r > 0:
            no_improve = 0
            if current_size < best_size:
                best_size = current_size
                best_active = active.copy()
                best_cov = cov_count.copy()

        # Get active list
        active_list = np.where(active)[0].tolist()
        if not active_list:
            break
        inactive_list = np.where(~active)[0].tolist()
        if not inactive_list:
            # nothing to add; only removals possible
            r = greedy_remove()
            if r > 0:
                no_improve = 0
                if current_size < best_size:
                    best_size = current_size
                    best_active = active.copy()
                    best_cov = cov_count.copy()
            else:
                no_improve += 1
            continue

        # (B) Single-swap
        improved = False
        ci = random.choice(active_list)
        orph = orphaned_words(ci)
        if len(orph) == 0:
            # redundant; greedy should have caught it, but handle
            do_remove(ci)
            current_size -= 1
            improved = True
        else:
            # find an inactive codeword covering all orph
            # orph is small (<= words_per_codeword). Candidate codewords must
            # cover each orphan; intersect candidate sets.
            cand = None
            for w in orph:
                wc = word_coverers[w]
                if cand is None:
                    cand = wc
                else:
                    # intersect
                    wc_set = set(wc)
                    cand = [x for x in cand if x in wc_set]
                if not cand:
                    break
            if cand:
                # pick a candidate that is inactive
                chosen = None
                for cj in cand:
                    if not active[cj]:
                        chosen = cj
                        break
                if chosen is None:
                    # all candidates active; try a random inactive as fallback swap
                    # (may not cover all orph -> skip)
                    pass
                else:
                    do_remove(ci)
                    do_add(chosen)
                    # size unchanged; try to remove more
                    r = greedy_remove()
                    if r > 0:
                        improved = True
                    # else no net change; still a valid move (no harm)

        if improved:
            no_improve = 0
            if current_size < best_size:
                best_size = current_size
                best_active = active.copy()
                best_cov = cov_count.copy()
        else:
            no_improve += 1

        # (C) 2-exchange: remove two codewords, add two codewords.
        #    Net size change can be 0, +1, -1. We look for -1 (net removal).
        if now() < deadline - 5.0 and random.random() < 0.4:
            if len(active_list) >= 2:
                ci1, ci2 = random.sample(active_list, 2)
                # orphans if we remove both
                orph1 = orphaned_words(ci1)
                orph2 = orphaned_words(ci2)
                # combine (unique)
                orph_all = list(set(orph1.tolist()) | set(orph2.tolist()))
                if len(orph_all) <= 2 * words_per_codeword:
                    # find two inactive codewords whose union covers orph_all
                    # candidate set: codewords covering at least one orphan
                    cand_union = set()
                    for w in orph_all:
                        cand_union.update(word_coverers[w])
                    cand_union = [c for c in cand_union if not active[c]]
                    if len(cand_union) >= 2:
                        # try pairs (limited)
                        found = False
                        num_pairs = min(30, len(cand_union))
                        pairs = random.sample(cand_union, num_pairs)
                        for a in range(len(pairs)):
                            for b in range(a + 1, len(pairs)):
                                ca, cb = pairs[a], pairs[b]
                                cov_a = set(covered_all[ca].tolist())
                                cov_b = set(covered_all[cb].tolist())
                                if all(w in cov_a or w in cov_b for w in orph_all):
                                    do_remove(ci1)
                                    do_remove(ci2)
                                    do_add(ca)
                                    do_add(cb)
                                    current_size += 0  # net 0
                                    # now try to remove more
                                    r = greedy_remove()
                                    if r > 0:
                                        no_improve = 0
                                        if current_size < best_size:
                                            best_size = current_size
                                            best_active = active.copy()
                                            best_cov = cov_count.copy()
                                    found = True
                                    break
                            if found:
                                break
                        if not found:
                            no_improve += 1

        # (D) add-then-remove (escape local minima)
        if now() < deadline - 5.0 and random.random() < 0.3:
            cj = random.choice(inactive_list)
            do_add(cj)
            current_size += 1
            # greedily remove as many as possible
            r = greedy_remove()
            # net change = +1 - r
            if current_size < best_size:
                no_improve = 0
                best_size = current_size
                best_active = active.copy()
                best_cov = cov_count.copy()
            else:
                # revert: we don't have a clean undo; but the config is still
                # valid (superset of a valid config is valid). Keep it if it
                # doesn't hurt future search; but to control size, if it grew
                # beyond best, revert by removing the added one if redundant.
                if is_redundant(cj):
                    do_remove(cj)
                    current_size -= 1
                no_improve += 1

    # ---------- Final polish: restore best, then heavy greedy removal ----------
    if now() < deadline - 2.0:
        active = best_active.copy()
        cov_count = best_cov.copy()
        current_size = best_size
        # multiple rounds of greedy removal with different random orders
        for _ in range(200):
            if now() > deadline - 2.0:
                break
            r = greedy_remove()
            if r == 0:
                break
            if current_size < best_size:
                best_size = current_size
                best_active = active.copy()
                best_cov = cov_count.copy()

    # ---------- Build final code from best_active ----------
    final_indices = np.where(best_active)[0]
    final_code = [all_words_array[i].tolist() for i in final_indices]

    # ---------- Verify coverage; fallback if invalid ----------
    if now() < deadline - 0.5:
        covered_set = set()
        for c in final_code:
            ci = word_to_index(tuple(c))
            covered_set.update(covered_all[ci].tolist())
        if len(covered_set) < total_words:
            # fallback: rebuild from initial and greedy only
            active = np.zeros(total_words, dtype=bool)
            active[initial_indices] = True
            cov_count = np.zeros(total_words, dtype=np.int32)
            for ci in initial_indices:
                cov_count[covered_all[ci]] += 1
            current_size = int(active.sum())
            for _ in range(500):
                removed_any = False
                order = list(range(total_words))
                random.shuffle(order)
                for ci in order:
                    if not active[ci]:
                        continue
                    if is_redundant(ci):
                        active[ci] = False
                        cov_count[covered_all[ci]] -= 1
                        removed_any = True
                if not removed_any:
                    break
            final_indices = np.where(active)[0]
            final_code = [all_words_array[i].tolist() for i in final_indices]

    return final_code
