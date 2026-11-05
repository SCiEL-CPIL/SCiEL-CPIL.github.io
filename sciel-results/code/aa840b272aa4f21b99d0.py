"""Greedy set-cover with precomputed Hamming balls, randomized restarts, and a time guard."""
# EVOLVE-BLOCK-START
import time
import random
import itertools


def _build_balls(q, n, radius):
    """For each word id in [0, q^n), return the set of word ids within Hamming distance <= radius."""
    total = q ** n
    # Precompute digit tuples for each id to speed up distance checks.
    tuples = [tuple(map(int, divmod(i, q) and [i // q ** p % q for p in range(n)])) for i in range(total)]
    # Simpler: build tuples via base-q expansion.
    tuples = []
    for i in range(total):
        x = i
        d = [0] * n
        for p in range(n):
            d[p] = x % q
            x //= q
        tuples.append(tuple(d))

    balls = []
    for a in range(total):
        ta = tuples[a]
        s = set()
        for b in range(total):
            tb = tuples[b]
            dist = sum(1 for i in range(n) if ta[i] != tb[i])
            if dist <= radius:
                s.add(b)
        balls.append(s)
    return balls


def _greedy(balls, seed=None):
    """Greedy set cover over word ids 0..len(balls)-1. Returns list of chosen codeword ids."""
    rng = random.Random(seed)
    total = len(balls)
    uncovered = set(range(total))
    chosen = []
    while uncovered:
        best = -1
        best_gain = 0
        best_rand = None
        for c in range(total):
            gain = len(balls[c] & uncovered)
            if gain > best_gain or (gain == best_gain and gain > 0):
                # tie-break randomly among equal-max gains to diversify
                if gain > best_gain:
                    best = c
                    best_gain = gain
                    best_rand = rng.random()
                else:
                    # equal gain: keep with some probability
                    if rng.random() < 0.5:
                        best = c
                        best_rand = rng.random()
        if best_gain == 0:
            break
        chosen.append(best)
        uncovered -= balls[best]
    return chosen


def construct(q, n, radius):
    deadline = time.time() + 145  # safety margin before 150s kill
    total = q ** n
    balls = _build_balls(q, n, radius)

    # Initial deterministic greedy (seed 0)
    best_chosen = _greedy(balls, seed=0)
    best_len = len(best_chosen)

    # Randomized restarts
    seed = 1
    while time.time() < deadline:
        chosen = _greedy(balls, seed=seed)
        if len(chosen) < best_len:
            best_len = len(chosen)
            best_chosen = chosen
        seed += 1

    # Convert chosen ids back to codeword lists
    def id_to_word(wid):
        d = [0] * n
        x = wid
        for p in range(n):
            d[p] = x % q
            x //= q
        return d

    return [id_to_word(c) for c in best_chosen]
# EVOLVE-BLOCK-END
