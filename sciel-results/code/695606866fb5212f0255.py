import time
import random
import numpy as np
from itertools import combinations

N = 17
MIN_D = 6
DEADLINE = 82.0
TOTAL_WORDS = 1 << N  # 131072

# Precompute all flip masks for distances 0..5
_FLIP_MASKS = []
for _d in range(MIN_D):
    if _d == 0:
        _FLIP_MASKS.append((0,))
    else:
        _masks = []
        for _bits in combinations(range(N), _d):
            _mask = 0
            for _b in _bits:
                _mask |= (1 << _b)
            _masks.append(_mask)
        _FLIP_MASKS.append(tuple(_masks))

def _mark_blocked(blocked, w):
    for masks in _FLIP_MASKS:
        for mask in masks:
            blocked[w ^ mask] = 1

def _greedy_sieve(seed=42):
    blocked = np.zeros(TOTAL_WORDS, dtype=np.uint8)
    words = []
    words.append(0)
    _mark_blocked(blocked, 0)
    rng = np.random.default_rng(seed)
    order = rng.permutation(TOTAL_WORDS)
    for idx in order:
        w = int(idx)
        if blocked[w] == 0:
            words.append(w)
            _mark_blocked(blocked, w)
    return words

def construct():
    start_time = time.time()
    best_words = []
    for seed in range(20):
        if time.time() - start_time > DEADLINE - 5:
            break
        words = _greedy_sieve(seed)
        if len(words) > len(best_words):
            best_words = words
    # Verify
    for i in range(len(best_words)):
        for j in range(i + 1, len(best_words)):
            if (best_words[i] ^ best_words[j]).bit_count() < MIN_D:
                raise ValueError(f"Distance violation: {i},{j}")
    return best_words
