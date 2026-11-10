"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random


def construct(q, n, radius):
    deadline = time.time() + 110  # Safety margin before 120s limit

    # Fallback for non-standard parameters
    if q != 3 or n != 7 or radius != 1:
        if q == 3 and n >= 4 and radius == 1:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        # Generic fallback: all words (valid but large)
        return [list(t) for t in itertools.product(range(q), repeat=n)]

    # Specific case: q=3, n=7, radius=1
    total_words = 3 ** 7  # 2187

    # Precompute all words as tuples
    all_words = [tuple(w) for w in itertools.product(range(3), repeat=7)]
    word_index = {w: i for i, w in enumerate(all_words)}

    # For each word, precompute its ball (all words within distance 1)
    # Ball size: 1 + 7*2 = 15
    balls = {}
    for w in all_words:
        ball = set()
        ball.add(w)
        for pos in range(7):
            for val in range(3):
                if val != w[pos]:
                    new_w = list(w)
                    new_w[pos] = val
                    ball.add(tuple(new_w))
        balls[w] = ball

    # Reverse: for each word, which balls contain it (i.e., which codewords cover it)
    # We'll use a different approach: maintain coverage count per word

    # Greedy set cover
    covered = [0] * total_words  # coverage count for each word
    chosen = []

    # Initial greedy: pick word covering most uncovered
    # Use a priority approach
    uncovered_count = total_words

    while uncovered_count > 0 and time.time() < deadline - 30:
        best_word = None
        best_gain = 0
        for w in all_words:
            gain = sum(1 for v in balls[w] if covered[word_index[v]] == 0)
            if gain > best_gain:
                best_gain = gain
                best_word = w
                if best_gain == 15:  # Maximum possible
                    break
        if best_gain == 0:
            break
        chosen.append(best_word)
        for v in balls[best_word]:
            idx = word_index[v]
            covered[idx] += 1
            if covered[idx] == 1:
                uncovered_count -= 1

    # Local optimization: try to remove redundant codewords
    # Then try swaps: remove one, add another that covers what was lost
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        # Try removing each codeword
        for i in range(len(chosen)):
            if time.time() > deadline - 10:
                break
            # Check if codeword i is redundant
            ball_i = balls[chosen[i]]
            redundant = True
            for v in ball_i:
                idx = word_index[v]
                if covered[idx] <= 1:
                    redundant = False
                    break
            if redundant:
                # Remove it
                for v in ball_i:
                    covered[word_index[v]] -= 1
                chosen.pop(i)
                improved = True
                break

        # Try swaps: for each codeword, try replacing with another
        if not improved:
            for i in range(len(chosen)):
                if time.time() > deadline - 10:
                    break
                # Temporarily remove codeword i
                ball_i = balls[chosen[i]]
                lost = []
                for v in ball_i:
                    idx = word_index[v]
                    if covered[idx] == 1:
                        lost.append(idx)
                if not lost:
                    # Redundant, remove
                    for v in ball_i:
                        covered[word_index[v]] -= 1
                    chosen.pop(i)
                    improved = True
                    break
                # Try to find a replacement that covers all lost words
                # and adds new coverage
                best_replace = None
                best_new_gain = 0
                for w in all_words:
                    if w in chosen:
                        continue
                    ball_w = balls[w]
                    # Check if ball_w covers all lost
                    covers_all = all(covered[idx] >= 1 or idx in ball_w for idx in lost)
                    if not covers_all:
                        continue
                    # Count new coverage
                    new_gain = sum(1 for v in ball_w if covered[word_index[v]] == 0 and word_index[v] not in ball_i)
                    if new_gain > best_new_gain:
                        best_new_gain = new_gain
                        best_replace = w
                if best_replace is not None:
                    # Perform swap
                    for v in ball_i:
                        covered[word_index[v]] -= 1
                    chosen[i] = best_replace
                    for v in balls[best_replace]:
                        covered[word_index[v]] += 1
                    improved = True
                    break

    # Convert to lists
    return [list(w) for w in chosen]
# EVOLVE-BLOCK-END
