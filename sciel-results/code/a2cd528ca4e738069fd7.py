"""Construct the largest known 18-bit binary code with min distance 6."""
import time
import random

N = 18
MIN_D = 6
DEADLINE_MARGIN = 5.0  # seconds before the 90s kill


def _hamming(a, b):
    return (a ^ b).bit_count()


def _valid_pairwise(words, min_d):
    n = len(words)
    for i in range(n):
        wi = words[i]
        for j in range(i + 1, n):
            if _hamming(wi, words[j]) < min_d:
                return False
    return True


def _greedy_add(words, min_d, rng, max_rounds=200, batch_size=512, deadline=None):
    """Try to add random words to the code, keeping only those at distance >= min_d from all existing."""
    n0 = len(words)
    for _ in range(max_rounds):
        if deadline is not None and time.time() > deadline:
            break
        added_this_round = 0
        for _ in range(batch_size):
            if deadline is not None and time.time() > deadline:
                break
            w = rng.getrandbits(N)
            ok = True
            for existing in words:
                if _hamming(w, existing) < min_d:
                    ok = False
                    break
            if ok:
                words.append(w)
                added_this_round += 1
        if added_this_round == 0:
            # No progress in a full batch; do a few more targeted tries
            for _ in range(batch_size):
                if deadline is not None and time.time() > deadline:
                    break
                w = rng.getrandbits(N)
                ok = True
                for existing in words:
                    if _hamming(w, existing) < min_d:
                        ok = False
                        break
                if ok:
                    words.append(w)
                    break
    return len(words) - n0


def _reed_muller_extended():
    """First-order Reed-Muller RM(1,4) extended to 18 bits: 32 words, distance 8."""
    words = []
    for slope in range(16):
        for intercept in range(2):
            word = 0
            for point in range(16):
                bit = ((slope & point).bit_count() + intercept) % 2
                word |= bit << point
            words.append(word)
    return words


def construct():
    start = time.time()
    deadline = start + (90.0 - DEADLINE_MARGIN)

    # Start with the Reed-Muller seed (32 words, distance 8)
    words = _reed_muller_extended()

    # Greedy random expansion with multiple seeds
    rng = random.Random(12345)
    added = _greedy_add(words, MIN_D, rng, max_rounds=300, batch_size=256, deadline=deadline)

    # Try a few more random restarts of greedy if time permits
    for seed in range(10):
        if time.time() > deadline:
            break
        rng2 = random.Random(1000 + seed)
        words2 = list(words)
        _greedy_add(words2, MIN_D, rng2, max_rounds=100, batch_size=128, deadline=deadline)
        if len(words2) > len(words):
            words = words2

    # Final verification
    if not _valid_pairwise(words, MIN_D):
        # Fallback to the safe Reed-Muller code
        words = _reed_muller_extended()

    return words
