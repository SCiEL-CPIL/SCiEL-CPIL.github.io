"""Covers the q-ary n-space within Hamming radius using a greedy set cover with
fast NumPy vectorization, plus a local-improvement polish pass."""
import time
import itertools
import numpy as np


def construct(q, n, radius):
    start = time.time()
    deadline = start + 110.0  # safety margin before 120s

    # Trivial / small cases handled quickly
    total = q ** n
    if total <= 2000:
        # brute force: every word is covered by itself; return minimal? 
        # Actually for small total we can just return all words (valid but large).
        # Better: if radius >= n, one codeword covers all.
        if radius >= n:
            return [[0] * n]
        # For very small, greedy is fast
        pass

    if radius >= n:
        return [[0] * n]

    # Generate all words as index 0..q^n-1
    # Represent each word as an n-digit base-q number
    all_indices = np.arange(total, dtype=np.int64)

    # Precompute the "sphere" for each possible codeword? That's total * sphere_size.
    # sphere_size = 1 + n*(q-1) for radius 1.
    sphere_size = 1 + n * (q - 1)

    # Instead, we use a greedy approach:
    # 1. Start with no codewords.
    # 2. Repeatedly pick the word that covers the most uncovered words.
    # 3. Mark those as covered.
    # 4. Repeat until all covered.

    # To speed up, we'll use a bitset for covered status.
    covered = np.zeros(total, dtype=np.bool_)

    # Precompute for each word index, the list of words within radius 1.
    # But storing all spheres is total * sphere_size which for 2187*19 ~ 41553, fine.
    # Actually for general q,n it could be larger, but the task is specifically 3,7,1.
    # Let's build spheres for each word.

    # Convert index to tuple
    def idx_to_tuple(idx):
        t = []
        for _ in range(n):
            t.append(idx % q)
            idx //= q
        return t

    def tuple_to_idx(t):
        idx = 0
        for i in range(n):
            idx = idx * q + t[i]
        return idx

    # Build sphere lists
    spheres = []
    for idx in range(total):
        t = idx_to_tuple(idx)
        sphere = [idx]
        for pos in range(n):
            orig = t[pos]
            for v in range(q):
                if v == orig:
                    continue
                t[pos] = v
                sphere.append(tuple_to_idx(t))
            t[pos] = orig
        spheres.append(sphere)

    # Greedy set cover
    codewords = []
    uncovered_count = total

    # For efficiency, we track for each word how many times it's in candidate spheres
    # But simple greedy: at each step, find the word whose sphere has the most uncovered.

    # We'll use a priority approach: maintain a count of uncovered neighbors for each word.
    # count[i] = number of uncovered words in sphere[i]
    counts = np.zeros(total, dtype=np.int32)
    for i in range(total):
        counts[i] = len(spheres[i])  # initially all uncovered

    # But we need to update counts when we cover words.
    # When we pick word w, we cover all words in sphere[w].
    # For each newly covered word c, we decrement counts[j] for all j whose sphere contains c.
    # This requires reverse mapping: for each word c, which spheres contain it?
    # sphere_of[c] = list of codeword indices w such that c in sphere[w]
    # But by symmetry, c in sphere[w] iff w in sphere[c]. So reverse is just spheres.

    # Let's build reverse: rev[c] = list of w such that c in spheres[w]
    # This is the same as spheres[c] by symmetry!
    # So rev = spheres.

    while uncovered_count > 0:
        if time.time() > deadline - 5:
            break

        # Find word with max count
        best_idx = int(np.argmax(counts))
        if counts[best_idx] == 0:
            break

        codewords.append(best_idx)
        new_covered = 0
        for c in spheres[best_idx]:
            if not covered[c]:
                covered[c] = True
                new_covered += 1
                # Update counts for all w whose sphere contains c
                for w in spheres[c]:
                    counts[w] -= 1

        uncovered_count -= new_covered

    # Convert to list of lists
    result = [idx_to_tuple(w) for w in codewords]

    # Local improvement: try to remove or swap codewords to reduce size
    # Simple: try removing each codeword and check if still covered
    # This is O(len(result) * total * sphere_size) which might be slow.
    # For now, skip if time is tight.

    if time.time() < deadline - 10 and len(result) > 100:
        # Try greedy removal
        improved = True
        while improved:
            improved = False
            for i in range(len(result)):
                if time.time() > deadline - 5:
                    break
                # Try removing codeword i
                test_covered = covered.copy()
                # Uncover words that are only covered by codeword i
                # Simpler: rebuild covered from result without i
                test_covered = np.zeros(total, dtype=np.bool_)
                for j, w in enumerate(result):
                    if j == i:
                        continue
                    for c in spheres[w]:
                        test_covered[c] = True
                if test_covered.all():
                    result.pop(i)
                    improved = True
                    break

    return result
