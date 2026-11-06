import time
import random
import numpy as np

N = 17
MIN_DISTANCE = 6
DEADLINE = 82.0

_TABLE8 = np.array([bin(i).count('1') for i in range(256)], dtype=np.int32)

def _pc17(x):
    x = x.astype(np.uint32)
    a = (x & 0xFF).astype(np.int64)
    b = ((x >> 8) & 0xFF).astype(np.int64)
    c = ((x >> 16) & 0xFF).astype(np.int64)
    return (_TABLE8[a] + _TABLE8[b] + _TABLE8[c]).astype(np.int32)

def _build_reference():
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
    deadline = start + DEADLINE
    rng = random.Random(42)

    code_set = set(_build_reference())
    code_arr = np.array(list(code_set), dtype=np.uint32)

    all_words = list(range(1 << N))
    all_words.sort(key=lambda w: w.bit_count())
    candidates = [w for w in all_words if w not in code_set]

    # Greedy phase
    for cand in candidates:
        if time.time() > deadline:
            break
        if cand in code_set:
            continue
        if _pc17(code_arr ^ cand).min() >= MIN_DISTANCE:
            code_set.add(cand)
            code_arr = np.array(list(code_set), dtype=np.uint32)

    # Local search phase
    for _ in range(10):
        if time.time() > deadline:
            break
        code_list = list(code_set)
        rng.shuffle(code_list)
        improved = False
        for i in range(len(code_list)):
            if time.time() > deadline:
                break
            out_w = code_list[i]
            code_set.remove(out_w)
            code_arr = np.array(list(code_set), dtype=np.uint32)
            added = False
            for cand in candidates:
                if cand in code_set:
                    continue
                if _pc17(code_arr ^ cand).min() >= MIN_DISTANCE:
                    code_set.add(cand)
                    code_arr = np.array(list(code_set), dtype=np.uint32)
                    code_list[i] = cand
                    added = True
                    improved = True
                    break
            if not added:
                code_set.add(out_w)
                code_arr = np.array(list(code_set), dtype=np.uint32)
                code_list[i] = out_w
        if not improved:
            break

    return list(code_set)
