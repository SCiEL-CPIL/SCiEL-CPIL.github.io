"""Ternary covering code K_3(8,1) via randomized local search."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s kill

    # For the specific case q=3, n=8, radius=1
    # We need to cover all 3^8 = 6561 words with spheres of radius 1.
    # Each codeword covers 1 + n*(q-1) = 1 + 8*2 = 17 words.
    # Lower bound: ceil(6561 / 17) = 386.
    # Best known: 486.

    # Strategy:
    # 1. Generate a random initial covering using a greedy approach.
    # 2. Use local search to minimize the number of codewords.
    #    Moves: remove a codeword if it doesn't break coverage, or swap.

    # Precompute all words and their neighbors for fast coverage checking
    # A word is covered if it's in the code or differs in exactly 1 position from a codeword.

    # We represent the code as a set of integer indices (0 to q^n - 1)
    # and maintain a coverage count array.

    total_words = q ** n
    # neighbors[i] = list of words within distance <= 1 from word i
    # Actually, we need: for each codeword c, which words does it cover?
    # And for each word w, which codewords cover it?

    # Let's precompute the "ball" of radius 1 around each word.
    # ball[i] = set of words within distance <= 1 from i
    # This is 1 + n*(q-1) words.

    # To save memory and time, let's work with numpy arrays.
    # We'll maintain:
    # - code: a list of codeword indices
    # - coverage: an array of length total_words, coverage[w] = number of codewords that cover w

    # Precompute balls
    # For speed, let's precompute for each word, the list of words it covers (itself + neighbors)
    # Actually, it's better to precompute for each word w, the list of codeword positions that would cover w.
    # But that's the same thing.

    # Let's precompute: for each index i, the list of indices in its ball of radius 1.
    # ball[i] = [i] + [neighbors]

    # Convert index to tuple and back
    def idx_to_tuple(idx):
        t = []
        for _ in range(n):
            t.append(idx % q)
            idx //= q
        return tuple(reversed(t))

    def tuple_to_idx(t):
        idx = 0
        for i in range(n):
            idx = idx * q + t[i]
        return idx

    # Precompute balls
    balls = [None] * total_words
    for i in range(total_words):
        t = idx_to_tuple(i)
        ball = [i]
        for pos in range(n):
            for val in range(q):
                if val == t[pos]:
                    continue
                new_t = list(t)
                new_t[pos] = val
                ball.append(tuple_to_idx(tuple(new_t)))
        balls[i] = ball

    # Greedy initial solution
    # Start with empty code, add codewords to cover uncovered words.
    # To make it faster, we can start with a known good structure or random.
    # Let's use a random greedy: repeatedly pick a random uncovered word, add a codeword that covers the most uncovered words.
    # But that's slow. Let's just add random codewords until all covered.

    # Actually, a better greedy:
    # uncovered = set of all words
    # while uncovered:
    #   pick a random word w from uncovered
    #   find the codeword c (from all possible) that covers the most words in uncovered
    #   add c to code
    #   remove covered words from uncovered
    # This is O(total_words^2) which is too slow for 6561.

    # Alternative: Start with a large random set, then remove.
    # Or: Use the tetracode construction as a starting point (486 words) and try to reduce.

    # Let's start with the tetracode construction: 486 words.
    import itertools
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_code = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n - 4):
            word = h + list(tail)
            initial_code.append(tuple_to_idx(tuple(word)))

    code = list(set(initial_code))
    random.shuffle(code)

    # Compute initial coverage
    coverage = np.zeros(total_words, dtype=np.int32)
    for c in code:
        for w in balls[c]:
            coverage[w] += 1

    # Local search: try to remove codewords
    # A codeword can be removed if all words it covers are also covered by other codewords.
    # i.e., for all w in balls[c], coverage[w] > 1 (before removing c).

    # We'll do multiple passes, trying to remove codewords.
    # Also, we can try swapping: remove c1, add c2, if it reduces the total or keeps it same but allows further removals.
    # For now, let's just do removal.

    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Shuffle code to randomize order
        random.shuffle(code)
        for i in range(len(code) - 1, -1, -1):
            if time.time() > deadline:
                break
            c = code[i]
            # Check if c can be removed
            can_remove = True
            for w in balls[c]:
                if coverage[w] <= 1:
                    can_remove = False
                    break
            if can_remove:
                # Remove c
                for w in balls[c]:
                    coverage[w] -= 1
                code.pop(i)
                improved = True

    # Now try swaps: replace c with c' if it allows removing more codewords
    # This is more complex. Let's try a simpler swap:
    # Pick a codeword c, pick a random new codeword c', remove c, add c', check if we can remove more.
    # This is expensive. Let's try a different approach.

    # Try to add codewords that allow removal of multiple existing codewords.
    # This is a 2-opt style move.

    # For time efficiency, let's stick with removal for now and see how we do.
    # The tetracode gives 486. Can we do better with removal?

    # Let's also try: start with a random set of size ~400, and use a repair heuristic to make it valid, then minimize.

    # Actually, let's try a different initial solution: random set of size 400, then repair.
    # But repair is hard. Let's stick with the tetracode start.

    # Let's try to do some random restarts with different initial solutions.
    # But time is limited.

    # Return the best solution found
    # Convert indices back to tuples
    result = [list(idx_to_tuple(c)) for c in code]
    return result

# EVOLVE-BLOCK-END
