"""Rewritten: Greedy set-cover + iterative redundancy elimination for K_3(7,1)."""
import time
import random
from itertools import product
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin under 120s

    # For the specific task: q=3, n=7, radius=1
    # Total space: 3^7 = 2187 words
    # Ball size: 1 + n*(q-1) = 1 + 7*2 = 15 words covered per codeword
    # Lower bound: ceil(2187/15) = 146, best known 186, seed 189

    if q != 3 or n != 7 or radius != 1:
        # Fallback: direct sum construction
        tetracode = [[a, b, (a + b) % 3, (a + 2*b) % 3] for a in range(3) for b in range(3)]
        if n >= 4:
            return [h + list(tail) for h in tetracode for tail in product(range(3), repeat=n-4)]
        else:
            return [list(c) for c in product(range(3), repeat=n)]

    total_words = 3 ** n  # 2187

    # Precompute all words as integers 0..2186 (base-3 representation)
    all_words = []
    word_to_idx = {}
    for w in product(range(3), repeat=n):
        idx = 0
        for d in w:
            idx = idx * 3 + d
        all_words.append(w)
        word_to_idx[idx] = w

    # Precompute Hamming distance-1 neighbors for each word
    # For each word, its ball = itself + all words at distance 1
    ball = {}
    for i, w in enumerate(all_words):
        neighbors = [i]
        for pos in range(n):
            orig = w[pos]
            for new_val in range(3):
                if new_val == orig:
                    continue
                w2 = list(w)
                w2[pos] = new_val
                idx2 = 0
                for d in w2:
                    idx2 = idx2 * 3 + d
                neighbors.append(idx2)
        ball[i] = set(neighbors)

    # Precompute: for each codeword candidate (word index), which words it covers
    # Use the ball sets above

    # Strategy: Start with a good initial covering, then greedily remove redundant words
    # Initial: Use the direct sum (189 words) as a starting point

    tetracode = [[a, b, (a + b) % 3, (a + 2*b) % 3] for a in range(3) for b in range(3)]
    initial_codewords = []
    for h in tetracode:
        for tail in product(range(3), repeat=n-4):
            cw = h + list(tail)
            idx = 0
            for d in cw:
                idx = idx * 3 + d
            initial_codewords.append(idx)

    # Convert to set for fast lookup
    codeword_set = set(initial_codewords)

    # Compute coverage: for each word, how many codewords cover it
    coverage_count = np.zeros(total_words, dtype=np.int32)
    for cw_idx in codeword_set:
        for covered in ball[cw_idx]:
            coverage_count[covered] += 1

    # Iteratively remove redundant codewords
    # A codeword is redundant if all words in its ball are covered by at least 2 codewords
    # (i.e., coverage_count[w] >= 2 for all w in ball[cw])
    # More precisely: after removing cw, every word in ball[cw] must still be covered
    # i.e., coverage_count[w] >= 2 for all w in ball[cw]

    removed = 0
    while time.time() < deadline - 5.0:
        removed_this_round = 0
        # Try to remove codewords in random order
        candidates = list(codeword_set)
        random.shuffle(candidates)

        for cw_idx in candidates:
            if cw_idx not in codeword_set:
                continue
            if time.time() > deadline - 5.0:
                break

            # Check if removing cw_idx is safe
            safe = True
            for w in ball[cw_idx]:
                if coverage_count[w] <= 1:
                    safe = False
                    break

            if safe:
                # Remove it
                codeword_set.remove(cw_idx)
                for w in ball[cw_idx]:
                    coverage_count[w] -= 1
                removed_this_round += 1
                removed += 1

        if removed_this_round == 0:
            break  # No more removable codewords

    # Now try to add codewords to enable more removals (swap strategy)
    # This is more complex; for now, try a different initial seed and repeat

    # Try multiple random perturbations
    best_size = len(codeword_set)
    best_codewords = set(codeword_set)

    # Try: start from scratch with greedy covering
    # Greedy: repeatedly pick the word that covers the most uncovered words
    uncovered = set(range(total_words))
    greedy_codewords = set()

    if time.time() < deadline - 10.0:
        # Greedy set cover
        # For each candidate word, compute how many uncovered words it covers
        # Pick the one with max coverage, add to codewords, update uncovered
        # Repeat until all covered

        # Precompute for speed: for each word, the list of words it covers
        # We already have ball[]

        # Use a priority approach
        while uncovered and time.time() < deadline - 10.0:
            best_cw = None
            best_count = -1
            # Sample a subset of candidates to save time
            # If uncovered is large, sample; otherwise check all
            if len(uncovered) > 500:
                candidates = random.sample(uncovered, min(200, len(uncovered)))
            else:
                candidates = list(uncovered)

            for cw in candidates:
                # Count how many uncovered words are in ball[cw]
                count = 0
                for w in ball[cw]:
                    if w in uncovered:
                        count += 1
                        if count > best_count:
                            pass  # We'll compute fully
                # Full count
                count = sum(1 for w in ball[cw] if w in uncovered)
                if count > best_count:
                    best_count = count
                    best_cw = cw

            if best_cw is None or best_count == 0:
                break

            greedy_codewords.add(best_cw)
            for w in ball[best_cw]:
                uncovered.discard(w)

        # Now apply redundancy removal to greedy result
        if len(greedy_codewords) < best_size:
            coverage_count2 = np.zeros(total_words, dtype=np.int32)
            for cw_idx in greedy_codewords:
                for covered in ball[cw_idx]:
                    coverage_count2[covered] += 1

            # Remove redundant
            while time.time() < deadline - 5.0:
                removed_this_round = 0
                candidates = list(greedy_codewords)
                random.shuffle(candidates)
                for cw_idx in candidates:
                    if cw_idx not in greedy_codewords:
                        continue
                    if time.time() > deadline - 5.0:
                        break
                    safe = True
                    for w in ball[cw_idx]:
                        if coverage_count2[w] <= 1:
                            safe = False
                            break
                    if safe:
                        greedy_codewords.remove(cw_idx)
                        for w in ball[cw_idx]:
                            coverage_count2[w] -= 1
                        removed_this_round += 1
                if removed_this_round == 0:
                    break

            if len(greedy_codewords) < best_size:
                best_size = len(greedy_codewords)
                best_codewords = set(greedy_codewords)

    # Final: verify coverage and return
    # Check that best_codewords covers everything
    covered = set()
    for cw in best_codewords:
        covered.update(ball[cw])

    if len(covered) < total_words:
        # Fallback to the direct sum if greedy failed
        codeword_set = set(initial_codewords)
        best_codewords = codeword_set

    # Convert back to list of lists
    result = []
    for cw_idx in best_codewords:
        w = word_to_idx[cw_idx]
        result.append(list(w))

    return result
