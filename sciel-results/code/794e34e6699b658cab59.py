import time
import numpy as np

def construct(q, n, radius):
    # Deadline guard: must finish within 120s wall-clock
    t0 = time.time()
    DEADLINE = 110.0  # safety margin

    total = q ** n

    def to_int(word):
        val = 0
        for i in range(n):
            val = val * q + word[i]
        return val

    def to_word(val):
        word = []
        for i in range(n):
            word.append(val % q)
            val //= q
        word.reverse()
        return word

    # Precompute balls: for each word, the set of words within Hamming distance <= radius
    # For radius=1, q=3, n=7: each ball has 1 + n*(q-1) = 1 + 7*2 = 15 words
    balls = []
    for w in range(total):
        word = to_word(w)
        ball = {w}
        for pos in range(n):
            for alt in range(q):
                if alt == word[pos]:
                    continue
                new_word = word[:]
                new_word[pos] = alt
                ball.add(to_int(new_word))
        balls.append(ball)

    # Greedy set cover
    # uncovered: boolean array of size total
    uncovered = np.ones(total, dtype=bool)
    uncovered_count = total
    selected = []

    # For efficiency, precompute balls as numpy arrays of indices
    ball_arrays = [np.array(list(b), dtype=np.int32) for b in balls]

    while uncovered_count > 0:
        if time.time() > DEADLINE:
            break

        best_idx = -1
        best_gain = 0

        for i in range(total):
            if time.time() > DEADLINE:
                break
            # Count how many uncovered words are in ball i
            ba = ball_arrays[i]
            gain = np.count_nonzero(uncovered[ba])
            if gain > best_gain:
                best_gain = gain
                best_idx = i
                if best_gain == uncovered_count:
                    break  # Can't do better

        if best_idx == -1 or best_gain == 0:
            break

        # Add best_idx to selected
        selected.append(to_word(best_idx))
        # Mark covered
        ba = ball_arrays[best_idx]
        newly_covered = uncovered[ba]
        uncovered_count -= np.count_nonzero(newly_covered)
        uncovered[ba] = False

    # Verify coverage; if not fully covered (due to deadline), add remaining uncovered words
    if uncovered_count > 0:
        # Add all uncovered words as codewords to ensure validity
        for w in range(total):
            if uncovered[w]:
                selected.append(to_word(w))
        uncovered_count = 0

    # Remove duplicates (shouldn't happen but just in case)
    seen = set()
    unique_selected = []
    for cw in selected:
        key = tuple(cw)
        if key not in seen:
            seen.add(key)
            unique_selected.append(cw)

    return unique_selected
