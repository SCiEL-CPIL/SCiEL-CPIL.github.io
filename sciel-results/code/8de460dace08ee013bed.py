"""Structural rewrite: coverage-count vector + systematic 2-opt exchange + aggressive deletion."""
# EVOLVE-BLOCK-START
import time
import random
import itertools

def construct(q, n, radius):
    if q != 3 or n != 7 or radius != 1:
        return _fallback(q, n, radius)

    start_time = time.time()
    DEADLINE = 135  # safety margin

    p3 = [1] * (n + 1)
    for i in range(1, n + 1):
        p3[i] = p3[i - 1] * 3
    total = 3 ** n  # 2187

    # Precompute neighbors for each codeword index (16 words: self + 15 distance-1)
    # neighbors[c] = list of 16 word indices
    neighbors = [[c] for c in range(total)]
    for c in range(total):
        w = []
        tmp = c
        for i in range(n):
            w.append(tmp % 3)
            tmp //= 3
        for pos in range(n):
            for delta in (1, 2):
                nw = w[:]
                nw[pos] = (nw[pos] + delta) % 3
                ni = 0
                for i in range(n):
                    ni += nw[i] * p3[i]
                neighbors[c].append(ni)

    def idx_to_word(c):
        w = []
        tmp = c
        for i in range(n):
            w.append(tmp % 3)
            tmp //= 3
        return w

    def time_ok():
        return time.time() < DEADLINE

    # Greedy: return list of codeword indices
    def greedy(seed):
        rng = random.Random(seed)
        # uncovered mask as a Python int bitset
        uncovered = (1 << total) - 1
        chosen = []
        chosen_set = set()
        while uncovered and time_ok():
            best_c = -1
            best_cnt = -1
            # Randomize scan order for diversity
            order = list(range(total))
            rng.shuffle(order)
            for c in order:
                if c in chosen_set:
                    continue
                # count overlap with uncovered
                # We need to count bits in (neighbors[c] & uncovered)
                # Use the precomputed neighbor list
                cnt = 0
                for nb in neighbors[c]:
                    if (uncovered >> nb) & 1:
                        cnt += 1
                if cnt > best_cnt:
                    best_cnt = cnt
                    best_c = c
            if best_c < 0:
                break
            chosen.append(best_c)
            chosen_set.add(best_c)
            for nb in neighbors[best_c]:
                uncovered &= ~(1 << nb)
        return chosen

    # Coverage count: cov[w] = number of chosen codewords that cover word w
    def compute_cov(chosen):
        cov = [0] * total
        for c in chosen:
            for nb in neighbors[c]:
                cov[nb] += 1
        return cov

    def can_remove(c, cov):
        for nb in neighbors[c]:
            if cov[nb] <= 1:
                return False
        return True

    def remove_codeword(c, cov, chosen, chosen_set):
        for nb in neighbors[c]:
            cov[nb] -= 1
        chosen.remove(c)
        chosen_set.discard(c)

    def add_codeword(c, cov, chosen, chosen_set):
        for nb in neighbors[c]:
            cov[nb] += 1
        chosen.append(c)
        chosen_set.add(c)

    def aggressive_deletion(chosen, cov, rng):
        changed = True
        while changed and time_ok():
            changed = False
            # Randomize deletion order
            order = list(range(len(chosen)))
            rng.shuffle(order)
            for i in order:
                if i >= len(chosen):
                    continue
                c = chosen[i]
                if can_remove(c, cov):
                    for nb in neighbors[c]:
                        cov[nb] -= 1
                    chosen.pop(i)
                    changed = True
        return changed

    def exchange_search(chosen, cov, chosen_set, rng, max_rounds=50):
        """Try all (remove c_old, add c_new) exchanges. Accept if coverage preserved."""
        improved = False
        for rnd in range(max_rounds):
            if not time_ok():
                break
            made_move = False
            # For each current codeword, try replacing with a candidate
            for i in range(len(chosen)):
                if not time_ok():
                    break
                c_old = chosen[i]
                # Find words uniquely covered by c_old
                # A swap is valid iff every word w with cov[w]==1 and w in neighbors[c_old]
                # is also in neighbors[c_new].
                # Equivalently: for all w in neighbors[c_old] \ neighbors[c_new], cov[w] >= 2.
                # We'll check this for each candidate.
                # Precompute the "critical" set: words in neighbors[c_old] with cov==1
                critical = set()
                for nb in neighbors[c_old]:
                    if cov[nb] == 1:
                        critical.add(nb)
                if not critical:
                    # c_old is fully redundant — but aggressive_deletion should have caught it.
                    # Still, try to remove.
                    if can_remove(c_old, cov):
                        for nb in neighbors[c_old]:
                            cov[nb] -= 1
                        chosen.pop(i)
                        made_move = True
                        improved = True
                        break
                    continue

                # Try candidates: sample a few + some structured ones
                # Use a small random sample plus a few deterministic candidates
                candidates = []
                # Random samples
                for _ in range(8):
                    c_new = rng.randint(0, total - 1)
                    if c_new not in chosen_set:
                        candidates.append(c_new)
                # Also try neighbors of c_old that aren't already chosen
                for nb in neighbors[c_old]:
                    if nb not in chosen_set:
                        candidates.append(nb)

                for c_new in candidates:
                    if c_new in chosen_set:
                        continue
                    # Check validity: all critical words must be covered by c_new
                    valid = True
                    for w in critical:
                        # Is w in neighbors[c_new]?
                        if not any(nb2 == w for nb2 in neighbors[c_new]):
                            valid = False
                            break
                    if valid:
                        # Perform swap
                        for nb in neighbors[c_old]:
                            cov[nb] -= 1
                        for nb in neighbors[c_new]:
                            cov[nb] += 1
                        chosen[i] = c_new
                        chosen_set.discard(c_old)
                        chosen_set.add(c_new)
                        made_move = True
                        improved = True
                        break
                if made_move:
                    break
            if not made_move:
                break
        return improved

    # Two-move exchange: remove two, add two (to escape size-stuck local minima)
    def two_move_search(chosen, cov, chosen_set, rng, max_rounds=20):
        improved = False
        for rnd in range(max_rounds):
            if not time_ok():
                break
            made_move = False
            if len(chosen) < 2:
                break
            # Pick two random codewords to remove
            i, j = rng.sample(range(len(chosen)), 2)
            c1, c2 = chosen[i], chosen[j]
            # Critical words: covered only by c1 or c2 (cov==1 and in their neighbor sets)
            critical = set()
            for nb in neighbors[c1]:
                if cov[nb] == 1:
                    critical.add(nb)
            for nb in neighbors[c2]:
                if cov[nb] == 1 and nb not in critical:
                    critical.add(nb)
            if not critical:
                # Both redundant
                for nb in neighbors[c1]:
                    cov[nb] -= 1
                for nb in neighbors[c2]:
                    cov[nb] -= 1
                chosen.pop(max(i, j))
                chosen.pop(min(i, j))
                made_move = True
                improved = True
                break
            # Try two candidates
            cands = []
            for _ in range(6):
                c = rng.randint(0, total - 1)
                if c not in chosen_set:
                    cands.append(c)
            for c_a in cands[:3]:
                for c_b in cands[3:6]:
                    if c_a in chosen_set or c_b in chosen_set or c_a == c_b:
                        continue
                    # Check: all critical words covered by c_a or c_b
                    valid = True
                    for w in critical:
                        if not (any(nb2 == w for nb2 in neighbors[c_a]) or
                                any(nb2 == w for nb2 in neighbors[c_b])):
                            valid = False
                            break
                    if valid:
                        for nb in neighbors[c1]:
                            cov[nb] -= 1
                        for nb in neighbors[c2]:
                            cov[nb] -= 1
                        for nb in neighbors[c_a]:
                            cov[nb] += 1
                        for nb in neighbors[c_b]:
                            cov[nb] += 1
                        chosen[i] = c_a
                        chosen[j] = c_b
                        chosen_set.discard(c1)
                        chosen_set.discard(c2)
                        chosen_set.add(c_a)
                        chosen_set.add(c_b)
                        made_move = True
                        improved = True
                        break
                if made_move:
                    break
            if made_move:
                break
            if not made_move and rnd > 0:
                # Try less aggressive: single removal if possible
                break
        return improved

    best_result = None
    best_size = float('inf')

    # Phase 1: Multiple greedy restarts with local refinement
    for seed in range(30):
        if not time_ok():
            break
        chosen = greedy(seed)
        cov = compute_cov(chosen)
        chosen_set = set(chosen)

        # Interleave deletion and exchange
        for it in range(10):
            if not time_ok():
                break
            aggressive_deletion(chosen, cov, random)
            if len(chosen) < best_size:
                best_size = len(chosen)
                best_result = list(chosen)
            exchange_search(chosen, cov, chosen_set, random, max_rounds=30)
            aggressive_deletion(chosen, cov, random)
            if len(chosen) < best_size:
                best_size = len(chosen)
                best_result = list(chosen)
            # Occasional two-move
            two_move_search(chosen, cov, chosen_set, random, max_rounds=5)
            aggressive_deletion(chosen, cov, random)
            if len(chosen) < best_size:
                best_size = len(chosen)
                best_result = list(chosen)

        if best_size <= 156:
            break

    # Phase 2: If we have time, do more intensive search on the best solution found
    if best_result is not None and time_ok():
        chosen = list(best_result)
        cov = compute_cov(chosen)
        chosen_set = set(chosen)
        for it in range(50):
            if not time_ok():
                break
            aggressive_deletion(chosen, cov, random)
            exchange_search(chosen, cov, chosen_set, random, max_rounds=50)
            aggressive_deletion(chosen, cov, random)
            two_move_search(chosen, cov, chosen_set, random, max_rounds=10)
            aggressive_deletion(chosen, cov, random)
            if len(chosen) < best_size:
                best_size = len(chosen)
                best_result = list(chosen)

    if best_result is None:
        best_result = greedy(0)

    result = [idx_to_word(c) for c in best_result]
    return result


def _fallback(q, n, radius):
    # Direct product / brute-force fallback for non-target parameters
    if n == 0:
        return [[]]
    # Greedy fallback
    p = [1] * (n + 1)
    for i in range(1, n + 1):
        p[i] = p[i - 1] * q
    total = q ** n

    def word_to_idx(w):
        idx = 0
        for i in range(n):
            idx += w[i] * p[i]
        return idx

    # For small total, brute-force greedy
    if total <= 10000:
        all_words = [list(w) for w in itertools.product(range(q), repeat=n)]
        uncovered = set(all_words)
        chosen = []
        while uncovered:
            best_w = None
            best_cnt = -1
            for w in all_words:
                cnt = 0
                for u in uncovered:
                    if sum(1 for a, b in zip(w, u) if a != b) <= radius:
                        cnt += 1
                if cnt > best_cnt:
                    best_cnt = cnt
                    best_w = w
            if best_w is None:
                break
            chosen.append(best_w)
            to_remove = []
            for u in uncovered:
                if sum(1 for a, b in zip(best_w, u) if a != b) <= radius:
                    to_remove.append(u)
            for u in to_remove:
                uncovered.discard(u)
        return chosen
    # Very large: return all words (valid but worst case)
    return [list(w) for w in itertools.product(range(q), repeat=n)]
# EVOLVE-BLOCK-END
