"""Rewritten program: incremental forbidden-set (conflict mask) + batched random search."""
import time
import numpy as np
from itertools import combinations

N = 18
MIN_DISTANCE = 6
SIZE = 1 << N


def _build_offsets():
    offs = [0]
    for k in range(1, MIN_DISTANCE):
        for comb in combinations(range(N), k):
            mask = 0
            for b in comb:
                mask |= 1 << b
            offs.append(mask)
    return np.array(offs, dtype=np.int64)


OFFSETS = _build_offsets()


def _rm32():
    words = []
    for slope in range(16):
        for intercept in range(2):
            word = 0
            for point in range(16):
                bit = ((slope & point).bit_count() + intercept) % 2
                word |= bit << point
            words.append(word)
    return words


def _initial_forbidden(code_words):
    forbidden = np.zeros(SIZE, dtype=bool)
    code_arr = np.array(code_words, dtype=np.int64)
    chunk = 256
    for i in range(0, len(code_arr), chunk):
        block = code_arr[i:i + chunk]
        idx = (block[:, None] ^ OFFSETS[None, :]).ravel()
        forbidden[idx] = True
    return forbidden


def _add_conflicts(forbidden, new_words):
    if len(new_words) == 0:
        return
    new_arr = np.array(new_words, dtype=np.int64)
    idx = (new_arr[:, None] ^ OFFSETS[None, :]).ravel()
    forbidden[idx] = True


def construct():
    start = time.time()
    DEADLINE = 85.0

    best_code = _rm32()
    best_len = len(best_code)

    def try_search(initial_code, seed):
        nonlocal best_code, best_len
        rng = np.random.default_rng(seed)
        code = list(initial_code)
        forbidden = _initial_forbidden(code)
        while time.time() - start < DEADLINE:
            batch_size = 20000
            cands = rng.integers(0, SIZE, size=batch_size, dtype=np.int64)
            cands = np.unique(cands)
            ok = ~forbidden[cands]
            good = cands[ok]
            if len(good) > 0:
                _add_conflicts(forbidden, good.tolist())
                code.extend(good.tolist())
                if len(code) > best_len:
                    best_len = len(code)
                    best_code = list(code)
            if len(code) > 300:
                break
        return len(code)

    seed = 12345
    while time.time() - start < DEADLINE:
        try_search(best_code, seed)
        seed += 1

    return best_code
