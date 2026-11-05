import time
import random
import itertools
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140

    if q != 3 or n != 7 or radius != 1:
        import itertools
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    total_words = q ** n
    words = []
    for i in range(total_words):
        w = []
        x = i
        for _ in range(n):
            w.append(x % q)
            x //= q
        w.reverse()
        words.append(tuple(w))

    powers = [1] * n
    for i in range(n-2, -1, -1):
        powers[i] = powers[i+1] * q

    def get_idx(w):
        idx = 0
        for i in range(n):
            idx += w[i] * powers[i]
        return idx

    # Precompute neighbors as sorted lists (each codeword covers 1 + n*(q-1) = 15 words)
    neighbors = []
    for i, w in enumerate(words):
        covered = [i]
        for pos in range(n):
            for val in range(q):
                if val == w[pos]:
                    continue
                new_w = list(w)
                new_w[pos] = val
                covered.append(get_idx(new_w))
        neighbors.append(covered)

    # --- Phase 1: Deterministic greedy set cover ---
    # Maintain uncovered_count[j] = number of selected codewords covering word j
    uncovered_count = np.zeros(total_words, dtype=np.int32)
    selected = []
    in_code = np.zeros(total_words, dtype=bool)

    # For efficient gain computation, we track for each candidate how many of its neighbors are uncovered
    # gain[c] = number of j in neighbors[c] with uncovered_count[j] == 0
    gain = np.zeros(total_words, dtype=np.int32)
    for c in range(total_words):
        gain[c] = len(neighbors[c])  # initially all uncovered

    while True:
        if time.time() > deadline - 30:
            break
        if uncovered_count.min() > 0:
            break

        # Find candidate with max gain (not already in code)
        best_gain = -1
        best_cand = -1
        for c in range(total_words):
            if in_code[c]:
                continue
            g = gain[c]
            if g > best_gain:
                best_gain = g
                best_cand = c

        if best_cand == -1 or best_gain <= 0:
            break

        selected.append(best_cand)
        in_code[best_cand] = True

        # Update uncovered_count and gains
        for j in neighbors[best_cand]:
            was_uncovered = (uncovered_count[j] == 0)
            uncovered_count[j] += 1
            if was_uncovered:
                # This word just became covered; decrement gain for all candidates covering j
                # We need to find which candidates cover j. Precompute reverse mapping.
                pass

        # Recompute gains for affected candidates (those sharing neighbors with best_cand)
        # Actually, let's just recompute gains for all candidates each step - with 2187 candidates
        # and 15 neighbors each, that's 2187*15 = 32805 ops per step, and we do ~150 steps.
        # Total: ~5M ops. Acceptable.
        for c in range(total_words):
            if in_code[c]:
                gain[c] = 0
            else:
                g = 0
                for j in neighbors[c]:
                    if uncovered_count[j] == 0:
                        g += 1
                gain[c] = g

    # --- Phase 2: Iterated 1-exchange local search ---
    # Try to remove a codeword and add another to reduce size while maintaining coverage.
    # A codeword c is "removable" if every word it covers is also covered by at least one other codeword.
    # After removal, if coverage breaks, we add the best replacement.

    def is_removable(idx_in_selected, c):
        """Check if removing codeword c still covers all words."""
        for j in neighbors[c]:
            count = 0
            for k in range(len(selected)):
                if k == idx_in_selected:
                    continue
                if j in neighbors[selected[k]]:
                    count += 1
                    break
            if count == 0:
                return False
        return True

    def coverage_complete():
        """Check if current selection covers all words."""
        covered = np.zeros(total_words, dtype=bool)
        for c in selected:
            for j in neighbors[c]:
                covered[j] = True
        return covered.all()

    best_size = len(selected)
    best_selected = selected[:]

    # Local search: try each removal, then find best addition if needed
    no_improve_rounds = 0
    max_no_improve = 200

    while no_improve_rounds < max_no_improve:
        if time.time() > deadline - 10:
            break

        improved = False
        no_improve_rounds += 1

        # Try removing each codeword
        for i in range(len(selected) - 1, -1, -1):
            if time.time() > deadline - 10:
                break

            c = selected[i]

            # Quick check: is c redundant?
            redundant = True
            for j in neighbors[c]:
                found = False
                for k in range(len(selected)):
                    if k == i:
                        continue
                    if j in neighbors[selected[k]]:
                        found = True
                        break
                if not found:
                    redundant = False
                    break

            if redundant:
                # Remove c, size decreases by 1
                selected.pop(i)
                in_code[c] = False
                improved = True
                break  # Restart scan from beginning

            # Try 1-exchange: remove c, add some d not in code
            # Only worth trying if we can find a d that covers all words c uniquely covers
            # and doesn't add too much overhead
            unique_to_c = set()
            for j in neighbors[c]:
                count = 0
                for k in range(len(selected)):
                    if k == i:
                        continue
                    if j in neighbors[selected[k]]:
                        count += 1
                        break
                if count == 0:
                    unique_to_c.add(j)

            if len(unique_to_c) == 0:
                # Should have been caught as redundant above
                continue

            # Find best d that covers all of unique_to_c
            best_d = -1
            best_d_gain = -1
            for d in range(total_words):
                if in_code[d]:
                    continue
                # Check d covers all unique_to_c
                d_neighbors_set = set(neighbors[d])
                if unique_to_c.issubset(d_neighbors_set):
                    # Gain: how many new words d covers that aren't already covered
                    # (excluding the ones c was covering)
                    new_gain = 0
                    for j in neighbors[d]:
                        if j not in set(neighbors[c]):
                            # Check if j is covered by other selected
                            covered_by_other = False
                            for k in range(len(selected)):
                                if k == i:
                                    continue
                                if j in neighbors[selected[k]]:
                                    covered_by_other = True
                                    break
                            if not covered_by_other:
                                new_gain += 1
                    if new_gain > best_d_gain:
                        best_d_gain = new_gain
                        best_d = d

            if best_d != -1:
                # Exchange: remove c, add best_d
                selected[i] = best_d
                in_code[c] = False
                in_code[best_d] = True
                improved = True
                break  # Restart scan

        if not improved:
            break

        # Update best
        if len(selected) < best_size:
            best_size = len(selected)
            best_selected = selected[:]
            no_improve_rounds = 0

    selected = best_selected

    result = []
    for idx in selected:
        result.append(list(words[idx]))

    return result
