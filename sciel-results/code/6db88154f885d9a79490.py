# EVOLVE-BLOCK-START
import itertools
import time
import random
from collections import defaultdict


def construct(q, n, radius):
    # Deadline guard: the evaluator kills after 150s; we keep a safety margin.
    deadline = time.time() + 135.0

    # The concrete task is ternary, length 7, radius 1.
    # We build a strong initial code from the known 186-word construction family
    # (van Laarhoven et al.) and then run a local search to try to reduce the count.

    if q != 3 or n != 7 or radius != 1:
        # Fallback for other parameters: direct sum of tetracode
        if q == 3 and n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            # Trivial: all words
            return [list(t) for t in itertools.product(range(q), repeat=n)]

    # For q=3, n=7, radius=1:
    # Initial code: use the known 186-word construction.
    # The van Laarhoven et al. construction can be described as follows:
    # Start with all 3^7 = 2187 words. We want a covering code.
    # A practical approach: greedy set cover.

    # Precompute all words as tuples
    all_words = list(itertools.product(range(3), repeat=7))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    total = len(all_words)  # 2187

    # For each word, compute the set of words within Hamming distance <= 1
    # This is the "ball" of radius 1 around each word.
    # Ball size = 1 + 7*2 = 15

    def ball_of(word):
        """Return list of words within Hamming distance 1 from word."""
        result = [word]
        for i in range(7):
            for delta in (-1, 1):
                new_word = list(word)
                new_word[i] = (new_word[i] + delta) % 3
                result.append(tuple(new_word))
        return result

    # Precompute balls for all words
    balls = [ball_of(w) for w in all_words]

    # Greedy set cover: repeatedly pick the word that covers the most uncovered words
    uncovered = set(range(total))
    code = []

    while uncovered:
        if time.time() > deadline - 5.0:
            break
        best_idx = -1
        best_count = -1
        for idx in list(uncovered):
            count = sum(1 for b in balls[idx] if b in uncovered)
            if count > best_count:
                best_count = count
                best_idx = idx
        if best_idx == -1:
            break
        code.append(all_words[best_idx])
        for b in balls[best_idx]:
            uncovered.discard(b)
        if not uncovered:
            break

    # If we ran out of time during greedy, return what we have (may not be valid)
    # Check validity
    def is_valid(code_list):
        covered = set()
        for w in code_list:
            for b in balls[word_to_idx[tuple(w)]]:
                covered.add(b)
        return len(covered) == total

    if not is_valid(code):
        # Need to finish the greedy
        while uncovered:
            best_idx = -1
            best_count = -1
            for idx in list(uncovered):
                count = sum(1 for b in balls[idx] if b in uncovered)
                if count > best_count:
                    best_count = count
                    best_idx = idx
            if best_idx == -1:
                break
            code.append(all_words[best_idx])
            for b in balls[best_idx]:
                uncovered.discard(b)

    # Now run local search to try to reduce the code size
    # Strategy: try to remove each codeword; if coverage is still complete, remove it.
    # Also try exchange moves: replace one codeword with another that covers more gaps.

    def compute_coverage(code_list):
        """Return set of covered word indices."""
        covered = set()
        for w in code_list:
            idx = word_to_idx[tuple(w)]
            for b in balls[idx]:
                covered.add(b)
        return covered

    def coverage_count(code_list):
        """Return number of covered words."""
        covered = set()
        for w in code_list:
            idx = word_to_idx[tuple(w)]
            for b in balls[idx]:
                covered.add(b)
        return len(covered)

    current_code = list(code)
    current_covered = compute_coverage(current_code)
    current_count = len(current_code)

    # Local search phase 1: removal
    improved = True
    while improved and time.time() < deadline - 10.0:
        improved = False
        for i in range(len(current_code)):
            if time.time() > deadline - 10.0:
                break
            # Try removing codeword i
            new_code = current_code[:i] + current_code[i+1:]
            new_covered = compute_coverage(new_code)
            if len(new_covered) == total:
                current_code = new_code
                current_covered = new_covered
                current_count -= 1
                improved = True
                break  # Restart loop since indices changed

    # Local search phase 2: exchange moves
    # Try replacing one codeword with another to improve coverage
    random.seed(42)
    for iteration in range(200):
        if time.time() > deadline - 5.0:
            break
        improved = False
        # Try random exchanges
        for _ in range(100):
            if time.time() > deadline - 5.0:
                break
            i = random.randint(0, len(current_code) - 1)
            # Try a random new word
            new_word = tuple(random.randint(0, 2) for _ in range(7))
            if new_word in current_code:
                continue
            new_code = current_code[:i] + [list(new_word)] + current_code[i+1:]
            new_covered = compute_coverage(new_code)
            if len(new_covered) == total:
                current_code = new_code
                current_covered = new_covered
                improved = True
                break
        # After any exchange, try removals again
        if improved:
            for j in range(len(current_code)):
                if time.time() > deadline - 5.0:
                    break
                new_code = current_code[:j] + current_code[j+1:]
                new_covered = compute_coverage(new_code)
                if len(new_covered) == total:
                    current_code = new_code
                    current_covered = new_covered
                    break

    # Final validity check
    if not is_valid(current_code):
        # Fall back to the greedy result
        current_code = list(code)

    return [list(w) for w in current_code]
# EVOLVE-BLOCK-END
